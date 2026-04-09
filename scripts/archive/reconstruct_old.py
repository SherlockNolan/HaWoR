"""
reconstruct.py
--------------
核心重建类，从 demo.py 提取和改编。

支持两种使用方式：
  1. Pipeline 方式（外部调用）：
       from reconstruct import HaWoRReconstructor
       rec = HaWoRReconstructor(checkpoint=..., infiller_weight=...)
       result = rec.run(video_path, output_dir=..., rendering=True)

  2. 命令行方式：
       python reconstruct.py --video_path example/video_0.mp4 --output_dir ./results
       python reconstruct.py --video_path example/factory001_worker001_00000.mp4 --output_dir ./results --rendering --vis_mode cam

注意由于服务器没有显示器，而渲染依赖于显示器（受制于python包aitviewer），如果要渲染必须如下调整
export MGLW_WINDOW=moderngl_window.context.headless.Window
export PYOPENGL_PLATFORM=egl
xvfb-run -a python reconstruct_old.py --video_path example/video_0.mp4 --output_dir ./results --rendering --vis_mode cam


"""

import argparse
import sys
import os

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_infiller_plain, hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam


# ---------------------------------------------------------------------------
# 面片常量（与 demo.py 保持一致）
# ---------------------------------------------------------------------------
_FACES_NEW = np.array([
    [92, 38, 234],
    [234, 38, 239],
    [38, 122, 239],
    [239, 122, 279],
    [122, 118, 279],
    [279, 118, 215],
    [118, 117, 215],
    [215, 117, 214],
    [117, 119, 214],
    [214, 119, 121],
    [119, 120, 121],
    [121, 120, 78],
    [120, 108, 78],
    [78, 108, 79],
])

# 绕 X 轴旋转 180° 的矩阵（坐标系对齐用）
_R_X = torch.tensor([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=torch.float32)


def _build_faces():
    """构建右手 / 左手的 face 数组。"""
    faces_base = get_mano_faces()
    faces_right = np.concatenate([faces_base, _FACES_NEW], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_right, faces_left


def _build_hand_dicts(pred_trans, pred_rot, pred_hand_pose, pred_betas,
                      vis_start, vis_end, faces_right, faces_left):
    """
    用 MANO 模型前向推理，得到双手的顶点字典。

    返回 (right_dict, left_dict)，其中每个 dict 包含：
        - 'vertices': (1, T, N, 3) Tensor
        - 'faces': np.ndarray
    """
    hand2idx = {"right": 1, "left": 0}

    # 右手
    hi = hand2idx["right"]
    pred_glob_r = run_mano(
        pred_trans[hi:hi+1, vis_start:vis_end],
        pred_rot[hi:hi+1, vis_start:vis_end],
        pred_hand_pose[hi:hi+1, vis_start:vis_end],
        betas=pred_betas[hi:hi+1, vis_start:vis_end],
    )
    right_dict = {
        "vertices": pred_glob_r["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "faces": faces_right,
    }

    # 左手
    hi = hand2idx["left"]
    pred_glob_l = run_mano_left(
        pred_trans[hi:hi+1, vis_start:vis_end],
        pred_rot[hi:hi+1, vis_start:vis_end],
        pred_hand_pose[hi:hi+1, vis_start:vis_end],
        betas=pred_betas[hi:hi+1, vis_start:vis_end],
    )
    left_dict = {
        "vertices": pred_glob_l["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "faces": faces_left,
    }

    return right_dict, left_dict


def _apply_coord_transform(right_dict, left_dict,
                            R_c2w_sla_all, t_c2w_sla_all):
    """
    将双手顶点和相机位姿统一变换到渲染坐标系（与 demo.py 保持一致）。

    返回 (right_dict, left_dict, R_w2c_sla_all, t_w2c_sla_all,
           R_c2w_sla_all, t_c2w_sla_all)
    """
    R_x = _R_X

    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    left_dict["vertices"] = torch.einsum(
        "ij,btnj->btni", R_x, left_dict["vertices"].cpu()
    )
    right_dict["vertices"] = torch.einsum(
        "ij,btnj->btni", R_x, right_dict["vertices"].cpu()
    )

    return (right_dict, left_dict,
            R_w2c_sla_all, t_w2c_sla_all,
            R_c2w_sla_all, t_c2w_sla_all)


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class HaWoRReconstructor:
    """
    HaWoR 重建 pipeline 的核心类。

    参数
    ----
    checkpoint : str
        HaWoR 模型权重路径。
    infiller_weight : str
        Infiller 模型权重路径。
    img_focal : float | None
        相机焦距，若为 None 则自动估计。
    """

    DEFAULT_CHECKPOINT = "./weights/hawor/checkpoints/hawor.ckpt"
    DEFAULT_INFILLER   = "./weights/hawor/checkpoints/infiller.pt"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        infiller_weight: str = DEFAULT_INFILLER,
        img_focal: float | None = None,
    ):
        self.checkpoint      = checkpoint
        self.infiller_weight = infiller_weight
        self.img_focal       = img_focal

    # ------------------------------------------------------------------
    # 内部辅助：把 args namespace 透传给各子模块
    # ------------------------------------------------------------------
    def _make_args(self, video_path: str):
        """构造一个兼容子模块签名的简单 namespace 对象。"""
        import types
        args = types.SimpleNamespace(
            video_path     = video_path,
            input_type     = "file",
            checkpoint     = self.checkpoint,
            infiller_weight= self.infiller_weight,
            img_focal      = self.img_focal,
        )
        return args

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------
    def run(
        self,
        video_path: str,
        output_dir: str | None = None,
        rendering: bool = False,
        vis_mode: str = "world",
    ) -> dict:
        """
        对单个视频执行完整重建 pipeline。

        参数
        ----
        video_path : str
            输入视频路径。
        output_dir : str | None
            输出目录。若为 None，则默认保存在与视频同名的子目录中。
        rendering : bool
            是否渲染并合成 mp4 视频。
        vis_mode : str
            渲染视角：'world' 或 'cam'。

        返回
        ----
        result : dict
            包含以下键：
            - 'pred_trans'    : Tensor (2, T, 3)  — 双手平移
            - 'pred_rot'      : Tensor (2, T, 3)  — 双手根方向（轴角）
            - 'pred_hand_pose': Tensor (2, T, 45) — 双手姿态（轴角展开）
            - 'pred_betas'    : Tensor (2, T, 10) — 形状参数
            - 'pred_valid'    : Tensor (2, T)     — 有效帧掩码
            - 'right_dict'    : dict              — 右手顶点 & 面片（变换后）
            - 'left_dict'     : dict              — 左手顶点 & 面片（变换后）
            - 'R_c2w'         : Tensor (T, 3, 3)  — 相机→世界旋转
            - 't_c2w'         : Tensor (T, 3)     — 相机→世界平移
            - 'R_w2c'         : Tensor (T, 3, 3)  — 世界→相机旋转
            - 't_w2c'         : Tensor (T, 3)     — 世界→相机平移
            - 'img_focal'     : float             — 使用的焦距
            - 'imgfiles'      : list[str]         — 帧图像路径列表
            - 'seq_folder'    : str               — 序列工作目录
            - 'rendered_video': str | None        — 渲染视频路径（仅 rendering=True 时有值）
        """
        args = self._make_args(video_path)

        # ── Step 1: 检测 & 追踪 ─────────────────────────────────────────
        print("[HaWoR] Step 1/4 — Detect & Track")
        start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

        # 若外部指定了 output_dir，则将 seq_folder 重定向
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            # 子模块内部仍用 seq_folder 写中间文件；
            # 最终结果和渲染视频将拷贝/存到 output_dir
            effective_output = output_dir
        else:
            effective_output = seq_folder

        # ── Step 2: HaWoR 运动估计 ──────────────────────────────────────
        print("[HaWoR] Step 2/4 — Motion Estimation")
        frame_chunks_all, img_focal = hawor_motion_estimation(
            args, start_idx, end_idx, seq_folder
        )

        # ── Step 3: SLAM ─────────────────────────────────────────────────
        print("[HaWoR] Step 3/4 — SLAM")
        slam_path = os.path.join(
            seq_folder,
            f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
        )
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = \
            load_slam_cam(slam_path)

        # ── Step 4: Infiller ─────────────────────────────────────────────
        print("[HaWoR] Step 4/4 — Infiller")
        # pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        #     hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
            hawor_infiller_plain(args, start_idx, end_idx, frame_chunks_all)

        # ── 构建双手网格字典 ─────────────────────────────────────────────
        faces_right, faces_left = _build_faces()
        vis_start = 0
        vis_end   = pred_trans.shape[1] - 1

        right_dict, left_dict = _build_hand_dicts(
            pred_trans, pred_rot, pred_hand_pose, pred_betas,
            vis_start, vis_end, faces_right, faces_left
        )

        # ── 坐标系变换 ───────────────────────────────────────────────────
        (right_dict, left_dict,
         R_w2c_sla_all, t_w2c_sla_all,
         R_c2w_sla_all, t_c2w_sla_all) = _apply_coord_transform(
            right_dict, left_dict, R_c2w_sla_all, t_c2w_sla_all
        )

        # ── 整理返回结果 ─────────────────────────────────────────────────
        result = dict(
            pred_trans     = pred_trans,
            pred_rot       = pred_rot,
            pred_hand_pose = pred_hand_pose,
            pred_betas     = pred_betas,
            pred_valid     = pred_valid,
            right_dict     = right_dict,
            left_dict      = left_dict,
            R_c2w          = R_c2w_sla_all,
            t_c2w          = t_c2w_sla_all,
            R_w2c          = R_w2c_sla_all,
            t_w2c          = t_w2c_sla_all,
            img_focal      = img_focal,
            imgfiles       = imgfiles,
            seq_folder     = seq_folder,
            rendered_video = None,
        )

        # ── 可选：渲染 mp4 ───────────────────────────────────────────────
        if rendering:
            rendered_video = self._render(
                result      = result,
                vis_start   = vis_start,
                vis_end     = vis_end,
                output_dir  = effective_output,
                vis_mode    = vis_mode,
                video_path  = video_path,
            )
            result["rendered_video"] = rendered_video
            if rendered_video:
                print(f"[HaWoR] Rendered video saved to: {rendered_video}")

        return result

    # ------------------------------------------------------------------
    # 渲染（可选）
    # ------------------------------------------------------------------
    def _render(
        self,
        result: dict,
        vis_start: int,
        vis_end: int,
        output_dir: str,
        vis_mode: str,
        video_path: str = "",
    ) -> str | None:
        """
        调用公共渲染函数 render_hand_results，返回生成的 mp4 路径（失败则返回 None）。
        """
        from lib.vis.run_vis2 import render_hand_results

        video_stem = os.path.splitext(os.path.basename(video_path))[0] if video_path else "output"
        image_names = list(result["imgfiles"][vis_start:vis_end])

        print(f"[HaWoR] Rendering frames {vis_start} → {vis_end}  (mode={vis_mode})")

        return render_hand_results(
            left_dict    = result["left_dict"],
            right_dict   = result["right_dict"],
            image_names  = image_names,
            img_focal    = result["img_focal"],
            output_dir   = output_dir,
            vis_start    = vis_start,
            vis_end      = vis_end,
            vis_mode     = vis_mode,
            R_c2w        = result["R_c2w"],
            t_c2w        = result["t_c2w"],
            R_w2c        = result["R_w2c"],
            t_w2c        = result["t_w2c"],
            video_stem   = video_stem,
        )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HaWoR — Hand-in-World Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default='results',
        help="输出目录，默认在./results下面",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=HaWoRReconstructor.DEFAULT_CHECKPOINT,
        help="HaWoR 模型权重路径",
    )
    parser.add_argument(
        "--infiller_weight", type=str,
        default=HaWoRReconstructor.DEFAULT_INFILLER,
        help="Infiller 模型权重路径",
    )
    parser.add_argument(
        "--img_focal", type=float, default=None,
        help="相机焦距（像素），不提供则自动估计",
    )
    parser.add_argument(
        "--rendering", action="store_true",
        help="开启渲染并合成 mp4（默认关闭）",
    )
    parser.add_argument(
        "--vis_mode", type=str, default="world",
        choices=["world", "cam"],
        help="渲染视角：world（世界坐标）或 cam（相机坐标）",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    reconstructor = HaWoRReconstructor(
        checkpoint      = args.checkpoint,
        infiller_weight = args.infiller_weight,
        img_focal       = args.img_focal,
    )

    result = reconstructor.run(
        video_path = args.video_path,
        output_dir = args.output_dir,
        rendering  = args.rendering,
        vis_mode   = args.vis_mode,
    )

    print("\n=== Reconstruction complete ===")
    print(f"  seq_folder    : {result['seq_folder']}")
    print(f"  img_focal     : {result['img_focal']}")
    print(f"  pred_trans    : {result['pred_trans'].shape}")
    print(f"  pred_rot      : {result['pred_rot'].shape}")
    print(f"  pred_hand_pose: {result['pred_hand_pose'].shape}")
    print(f"  pred_betas    : {result['pred_betas'].shape}")
    if result["rendered_video"]:
        print(f"  rendered_video: {result['rendered_video']}")


if __name__ == "__main__":
    main()
