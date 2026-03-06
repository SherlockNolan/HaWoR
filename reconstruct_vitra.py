"""
reconstruct_vitra.py
--------------------
基于 VITRA 改进版 pipeline（hawor_pipeline_vitra.py）的重建与渲染脚本。

VITRA 对 HaWoR 检测/追踪/运动估计部分进行了改进，本脚本通过调用
HaworPipeline 完成重建，并复用公共渲染函数 render_hand_results 输出视频。

支持两种使用方式：

  1. Pipeline 方式（外部调用）：
       from reconstruct_vitra import VitraReconstructor
       rec = VitraReconstructor(model_path=..., detector_path=...)
       result = rec.run(video_path, output_dir=..., rendering=True)

  2. 命令行方式：
       python reconstruct_vitra.py --video_path example/video_0.mp4 --output_dir ./results
       python reconstruct_vitra.py --video_path example/video_0.mp4 --output_dir ./results --rendering --vis_mode cam

注意：渲染依赖显示器（aitviewer），无显示器时请在服务器端如下设置：
  export MGLW_WINDOW=moderngl_window.context.headless.Window
  export PYOPENGL_PLATFORM=egl
  xvfb-run -a python reconstruct_vitra.py --video_path example/video_0.mp4 --output_dir ./results --rendering
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from hawor_pipeline_vitra import HaworPipeline
from lib.vis.run_vis2 import render_hand_results


# ---------------------------------------------------------------------------
# 面片常量（与 reconstruct.py / demo.py 保持一致）
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

# 绕 X 轴旋转 180° 的矩阵（坐标系对齐用，与原版保持一致）
_R_X = torch.tensor([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=torch.float32)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _build_faces():
    """构建右手 / 左手的 face 数组。"""
    faces_base = get_mano_faces()
    faces_right = np.concatenate([faces_base, _FACES_NEW], axis=0)
    faces_left  = faces_right[:, [0, 2, 1]]
    return faces_right, faces_left


def _load_video_frames(video_path: str) -> tuple[list, float]:
    """
    从视频文件中读取所有帧（BGR numpy 数组列表）及焦距估算值。

    参数
    ----
    video_path : str
        视频文件路径。

    返回
    ----
    tuple:
        - frames (list[np.ndarray]): BGR 帧列表
        - img_focal (float): 估算焦距（max(H, W) * 1.2）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"视频中未读取到任何帧：{video_path}")

    H, W = frames[0].shape[:2]
    img_focal = max(H, W) * 1.2
    return frames, img_focal


def _recon_results_to_tensors(
    recon_results: dict,
    num_frames: int,
    faces_right: np.ndarray,
    faces_left: np.ndarray,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    """
    将 HaworPipeline.recon() 返回的逐帧字典转换为 MANO 前向输入所需的 Tensor，
    并运行 MANO 得到双手网格字典。

    HaworPipeline 输出格式（每帧）：
        result['beta']          : (10,)        形状参数
        result['hand_pose']     : (15, 3, 3)   手部关节旋转矩阵
        result['global_orient'] : (1, 3, 3)    根节点旋转矩阵
        result['transl']        : (3,)         平移向量

    参数
    ----
    recon_results : dict
        HaworPipeline.recon() 返回值，包含 'left' 和 'right' 两个子字典。
    num_frames : int
        视频总帧数。
    faces_right : np.ndarray
        右手面片数组。
    faces_left : np.ndarray
        左手面片数组。

    返回
    ----
    right_dict : dict   右手网格字典 {'vertices': (1,T,N,3), 'faces': ...}
    left_dict  : dict   左手网格字典 {'vertices': (1,T,N,3), 'faces': ...}
    valid_right : Tensor (T,)  右手有效帧掩码
    valid_left  : Tensor (T,)  左手有效帧掩码
    """
    def _unpack_side(side_results: dict, num_frames: int, is_left: bool):
        """
        将一侧手的逐帧字典整理为 (1, T, ...) 格式的 Tensor。
        返回 (trans, root_orient_aa, hand_pose_aa, betas, valid_mask)
        """
        T = num_frames
        trans       = torch.zeros(1, T, 3)
        root_orient = torch.zeros(1, T, 3)      # 轴角
        hand_pose   = torch.zeros(1, T, 15, 3)  # 轴角
        betas       = torch.zeros(1, T, 10)
        valid       = torch.zeros(T, dtype=torch.bool)

        for fid, res in side_results.items():
            if fid < 0 or fid >= T:
                continue
            # transl : (3,)
            trans[0, fid] = torch.tensor(res['transl'], dtype=torch.float32)

            # global_orient : (1, 3, 3) 旋转矩阵 → 轴角
            go_mat = torch.tensor(res['global_orient'], dtype=torch.float32)  # (1,3,3) or (3,3)
            if go_mat.ndim == 2:
                go_mat = go_mat.unsqueeze(0)  # (1,3,3)
            root_orient[0, fid] = rotation_matrix_to_angle_axis(go_mat.unsqueeze(0))[0, 0]  # (3,)

            # hand_pose : (15, 3, 3) 旋转矩阵 → 轴角
            hp_mat = torch.tensor(res['hand_pose'], dtype=torch.float32)  # (15,3,3)
            root_orient_aa_local = rotation_matrix_to_angle_axis(hp_mat.unsqueeze(0))  # (1,15,3)
            hand_pose[0, fid] = root_orient_aa_local[0]  # (15,3)

            # betas : (10,)
            betas[0, fid] = torch.tensor(res['beta'], dtype=torch.float32)

            valid[fid] = True

        return trans, root_orient, hand_pose, betas, valid

    # ── 右手 ─────────────────────────────────────────────────────────────
    right_side = recon_results.get('right', {})
    r_trans, r_root, r_pose, r_betas, r_valid = _unpack_side(right_side, num_frames, is_left=False)

    # ── 左手 ─────────────────────────────────────────────────────────────
    left_side = recon_results.get('left', {})
    l_trans, l_root, l_pose, l_betas, l_valid = _unpack_side(left_side, num_frames, is_left=True)

    # ── 运行 MANO ─────────────────────────────────────────────────────────
    pred_glob_r = run_mano(r_trans, r_root, r_pose, betas=r_betas)
    right_dict = {
        "vertices": pred_glob_r["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "faces":    faces_right,
    }

    pred_glob_l = run_mano_left(l_trans, l_root, l_pose, betas=l_betas)
    left_dict = {
        "vertices": pred_glob_l["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "faces":    faces_left,
    }

    return right_dict, left_dict, r_valid, l_valid


def _apply_coord_transform(right_dict: dict, left_dict: dict) -> tuple[dict, dict]:
    """
    对双手顶点施加绕 X 轴 180° 的旋转（与 reconstruct.py / demo.py 保持一致）。
    在没有 SLAM 轨迹的纯视觉估计场景下，仅对顶点变换。
    """
    R_x = _R_X
    left_dict["vertices"]  = torch.einsum("ij,btnj->btni", R_x, left_dict["vertices"].cpu())
    right_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, right_dict["vertices"].cpu())
    return right_dict, left_dict


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class VitraReconstructor:
    """
    基于 VITRA 改进 pipeline 的手部重建类。

    与 HaWoRReconstructor 的差异：
    - 直接接受视频帧列表（BGR numpy 数组），由 HaworPipeline 完成检测+追踪+运动估计
    - 无 SLAM 步骤（VITRA pipeline 暂不包含 SLAM），相机位姿设为 Identity
    - 无 Infiller 步骤（VITRA pipeline 已内置平滑/插值）
    - 渲染默认使用 cam 模式（无相机轨迹时 world 模式意义不大）

    参数
    ----
    model_path : str
        HaWoR 模型权重路径（.ckpt）。
    detector_path : str
        手部检测器 YOLO 权重路径（.pt）。
    device : torch.device
        推理设备。
    """

    DEFAULT_MODEL_PATH    = "./weights/hawor/checkpoints/hawor.ckpt"
    DEFAULT_DETECTOR_PATH = "./weights/external/detector.pt"

    def __init__(
        self,
        model_path:    str          = DEFAULT_MODEL_PATH,
        detector_path: str          = DEFAULT_DETECTOR_PATH,
        device:        torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.pipeline = HaworPipeline(
            model_path    = model_path,
            detector_path = detector_path,
            device        = device,
        )

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------
    def run(
        self,
        video_path:   str,
        output_dir:   str | None = None,
        img_focal:    float | None = None,
        thresh:       float = 0.2,
        single_image: bool  = False,
        rendering:    bool  = False,
        vis_mode:     str   = "cam",
    ) -> dict:
        """
        对单个视频执行完整 VITRA 重建 pipeline。

        参数
        ----
        video_path : str
            输入视频路径。
        output_dir : str | None
            输出目录。若为 None，则在视频同级目录下建立同名子目录。
        img_focal : float | None
            相机焦距（像素）。若为 None，则自动估算为 max(H, W) * 1.2。
        thresh : float
            手部检测置信度阈值。
        single_image : bool
            是否以单帧模式运行（仅需单帧有效检测）。
        rendering : bool
            是否渲染并合成 mp4 视频。
        vis_mode : str
            渲染视角：'world' 或 'cam'。
            注意：world 模式依赖相机轨迹，VITRA pipeline 无 SLAM 故默认使用 cam。

        返回
        ----
        result : dict
            包含以下键：
            - 'right_dict'     : dict              — 右手顶点 & 面片（变换后）
            - 'left_dict'      : dict              — 左手顶点 & 面片（变换后）
            - 'valid_right'    : Tensor (T,)       — 右手有效帧掩码
            - 'valid_left'     : Tensor (T,)       — 左手有效帧掩码
            - 'img_focal'      : float             — 使用的焦距
            - 'frames'         : list[np.ndarray]  — BGR 帧列表
            - 'num_frames'     : int               — 总帧数
            - 'seq_folder'     : str               — 输出目录
            - 'recon_results'  : dict              — HaworPipeline 原始输出（'left'/'right'）
            - 'rendered_video' : str | None        — 渲染视频路径（仅 rendering=True 时有值）
        """
        # ── Step 0: 确定输出目录 ─────────────────────────────────────────
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(video_path)), video_stem
            )
        os.makedirs(output_dir, exist_ok=True)

        # ── Step 1: 读取视频帧 ───────────────────────────────────────────
        print("[VITRA] Step 1/3 — Loading video frames")
        frames, focal_estimated = _load_video_frames(video_path)
        num_frames = len(frames)
        if img_focal is None:
            img_focal = focal_estimated
        print(f"[VITRA]   frames={num_frames}, img_focal={img_focal:.1f}")

        # ── Step 2: 检测 + 追踪 + 运动估计 ──────────────────────────────
        print("[VITRA] Step 2/3 — HaworPipeline (detect + track + motion estimation)")
        recon_results = self.pipeline.recon(
            images       = frames,
            img_focal    = img_focal,
            thresh       = thresh,
            single_image = single_image,
        )

        # ── Step 3: 构建双手网格 ─────────────────────────────────────────
        print("[VITRA] Step 3/3 — Building hand meshes (MANO forward pass)")
        faces_right, faces_left = _build_faces()
        right_dict, left_dict, valid_right, valid_left = _recon_results_to_tensors(
            recon_results, num_frames, faces_right, faces_left
        )

        # 坐标系对齐（绕 X 轴 180°）
        right_dict, left_dict = _apply_coord_transform(right_dict, left_dict)

        # ── 整理返回结果 ─────────────────────────────────────────────────
        result = dict(
            right_dict    = right_dict,
            left_dict     = left_dict,
            valid_right   = valid_right,
            valid_left    = valid_left,
            img_focal     = img_focal,
            frames        = frames,
            num_frames    = num_frames,
            seq_folder    = output_dir,
            recon_results = recon_results,
            rendered_video= None,
        )

        # ── 可选：渲染 mp4 ───────────────────────────────────────────────
        if rendering:
            rendered_video = self._render(
                result     = result,
                vis_start  = 0,
                vis_end    = num_frames,
                output_dir = output_dir,
                vis_mode   = vis_mode,
                video_stem = video_stem,
                video_path = video_path,
            )
            result["rendered_video"] = rendered_video
            if rendered_video:
                print(f"[VITRA] Rendered video saved to: {rendered_video}")
            else:
                print("[VITRA] Rendering did not produce an output file.")

        return result

    # ------------------------------------------------------------------
    # 渲染（可选）
    # ------------------------------------------------------------------
    def _render(
        self,
        result:     dict,
        vis_start:  int,
        vis_end:    int,
        output_dir: str,
        vis_mode:   str,
        video_stem: str,
        video_path: str = "",
    ) -> str | None:
        """
        调用公共渲染函数 render_hand_results。

        VITRA pipeline 无 SLAM，因此：
        - world 模式：相机位姿设为 Identity（相机始终在原点）
        - cam 模式：相机位姿同样设为 Identity（手部顶点已在相机坐标系中）
        """
        T = vis_end - vis_start

        # 构造 Identity 相机轨迹
        R_identity = torch.eye(3).unsqueeze(0).expand(T, -1, -1)  # (T,3,3)
        t_zero     = torch.zeros(T, 3)                            # (T,3)

        # 为渲染生成图像路径列表（将 BGR 帧临时写到磁盘）
        frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        frames = result["frames"]
        image_names = []
        for i in range(vis_start, vis_end):
            frame_path = os.path.join(frame_dir, f"{i:06d}.jpg")
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frames[i])
            image_names.append(frame_path)

        print(f"[VITRA] Rendering frames {vis_start} → {vis_end}  (mode={vis_mode})")

        return render_hand_results(
            left_dict   = result["left_dict"],
            right_dict  = result["right_dict"],
            image_names = image_names,
            img_focal   = result["img_focal"],
            output_dir  = output_dir,
            vis_start   = vis_start,
            vis_end     = vis_end,
            vis_mode    = vis_mode,
            R_c2w       = R_identity,
            t_c2w       = t_zero,
            R_w2c       = R_identity,
            t_w2c       = t_zero,
            video_stem  = video_stem,
        )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VITRA 改进版 HaWoR Pipeline — 手部 3D 重建与渲染",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results_vitra",
        help="输出目录",
    )
    parser.add_argument(
        "--model_path", type=str,
        default=VitraReconstructor.DEFAULT_MODEL_PATH,
        help="HaWoR 模型权重路径（.ckpt）",
    )
    parser.add_argument(
        "--detector_path", type=str,
        default=VitraReconstructor.DEFAULT_DETECTOR_PATH,
        help="手部检测器 YOLO 权重路径（.pt）",
    )
    parser.add_argument(
        "--img_focal", type=float, default=None,
        help="相机焦距（像素），不提供则自动估算为 max(H,W)*1.2",
    )
    parser.add_argument(
        "--thresh", type=float, default=0.2,
        help="手部检测置信度阈值",
    )
    parser.add_argument(
        "--single_image", action="store_true",
        help="以单帧模式运行（仅需单帧有效检测）",
    )
    parser.add_argument(
        "--rendering", action="store_true",
        help="开启渲染并合成 mp4（默认关闭）",
    )
    parser.add_argument(
        "--vis_mode", type=str, default="cam",
        choices=["world", "cam"],
        help="渲染视角：world（世界坐标）或 cam（相机坐标，默认）",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[VITRA] CUDA 不可用，已自动切换至 CPU")

    reconstructor = VitraReconstructor(
        model_path    = args.model_path,
        detector_path = args.detector_path,
        device        = device,
    )

    result = reconstructor.run(
        video_path   = args.video_path,
        output_dir   = args.output_dir,
        img_focal    = args.img_focal,
        thresh       = args.thresh,
        single_image = args.single_image,
        rendering    = args.rendering,
        vis_mode     = args.vis_mode,
    )

    print("\n=== VITRA Reconstruction complete ===")
    print(f"  seq_folder  : {result['seq_folder']}")
    print(f"  img_focal   : {result['img_focal']:.1f}")
    print(f"  num_frames  : {result['num_frames']}")
    right_valid_cnt = result["valid_right"].sum().item()
    left_valid_cnt  = result["valid_left"].sum().item()
    print(f"  right valid : {right_valid_cnt} / {result['num_frames']} frames")
    print(f"  left  valid : {left_valid_cnt} / {result['num_frames']} frames")
    if result["rendered_video"]:
        print(f"  rendered    : {result['rendered_video']}")


if __name__ == "__main__":
    main()
