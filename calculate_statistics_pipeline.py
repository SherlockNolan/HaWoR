"""
calculate_statistics_pipeline.py

使用 HaWoRPipeline 统计手部坐标统计量（支持 cam / world 两种模式）。

与 calculate_statistics.py 的区别
----------------------------------
- 原版通过脚本分步调用 detect_track_video / hawor_motion_estimation / hawor_slam /
  hawor_infiller，需要读写磁盘中间文件。
- 本版改用 HaWoRPipeline.reconstruct()，一次调用完成全流程，无需中间文件。

使用方式:
    python calculate_statistics_pipeline.py \
        --video_path example/video_0.mp4 \
        --checkpoint ./weights/hawor/checkpoints/hawor.ckpt \
        --infiller_weight ./weights/hawor/checkpoints/infiller.pt \
        --vis_mode world          # cam | world
        [--img_focal 600]         # 可选，焦距
        [--output_json stats.json] # 可选，保存统计结果到 JSON
        [--use_joints]             # 可选，同时统计关节点坐标
        [--no_smooth]              # 可选，关闭抖动平滑
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from lib.pipeline.HaWoRPipeline import HaWoRConfig, HaWoRPipeline
from hawor.utils.process import run_mano, run_mano_left


# ─────────────────────────────────────────────────────────────────────────────
# 统计工具函数（与 calculate_statistics.py 完全相同，便于独立运行）
# ─────────────────────────────────────────────────────────────────────────────

def compute_coord_stats(arr: np.ndarray, label: str) -> dict:
    """
    计算坐标数组的各项统计量。

    Parameters
    ----------
    arr : np.ndarray
        任意形状的浮点数组，最后一维通常是 3（x/y/z）。
    label : str
        用于打印的描述标签。

    Returns
    -------
    dict
        包含各项统计量的字典。
    """
    flat = arr.reshape(-1, arr.shape[-1])  # (N, 3)

    stats = {
        "shape": list(arr.shape),
        "num_samples": int(flat.shape[0]),
        "per_axis": {},
        "overall": {},
    }

    axis_names = ["x", "y", "z"]
    for i, axis in enumerate(axis_names):
        col = flat[:, i]
        stats["per_axis"][axis] = {
            "min":    float(col.min()),
            "max":    float(col.max()),
            "mean":   float(col.mean()),
            "median": float(np.median(col)),
            "std":    float(col.std()),
            "var":    float(col.var()),
            "range":  float(col.max() - col.min()),
            "p5":     float(np.percentile(col, 5)),
            "p25":    float(np.percentile(col, 25)),
            "p75":    float(np.percentile(col, 75)),
            "p95":    float(np.percentile(col, 95)),
        }

    all_vals = flat.ravel()
    stats["overall"] = {
        "min":    float(all_vals.min()),
        "max":    float(all_vals.max()),
        "mean":   float(all_vals.mean()),
        "median": float(np.median(all_vals)),
        "std":    float(all_vals.std()),
        "var":    float(all_vals.var()),
        "range":  float(all_vals.max() - all_vals.min()),
    }

    origin = flat[0:1, :]
    dists = np.linalg.norm(flat - origin, axis=-1)
    stats["displacement_from_first_frame"] = {
        "min":  float(dists.min()),
        "max":  float(dists.max()),
        "mean": float(dists.mean()),
        "std":  float(dists.std()),
    }

    return stats


def print_stats(stats: dict, label: str):
    """格式化打印统计结果。"""
    print(f"\n{'='*60}")
    print(f"  {label}  (shape={stats['shape']}, samples={stats['num_samples']})")
    print(f"{'='*60}")

    print(f"  {'指标':<12} {'x':>10} {'y':>10} {'z':>10}")
    print(f"  {'-'*46}")
    metric_keys = ["min", "max", "mean", "median", "std", "var", "range",
                   "p5", "p25", "p75", "p95"]
    metric_labels = {
        "min":    "最小值",
        "max":    "最大值",
        "mean":   "平均值",
        "median": "中位数",
        "std":    "标准差",
        "var":    "方差",
        "range":  "极差",
        "p5":     "5%分位",
        "p25":    "25%分位",
        "p75":    "75%分位",
        "p95":    "95%分位",
    }
    for key in metric_keys:
        x = stats["per_axis"]["x"][key]
        y = stats["per_axis"]["y"][key]
        z = stats["per_axis"]["z"][key]
        print(f"  {metric_labels[key]:<12} {x:>10.4f} {y:>10.4f} {z:>10.4f}")

    o = stats["overall"]
    print(f"\n  [整体（xyz 合并）]")
    print(f"    最小值={o['min']:.4f}  最大值={o['max']:.4f}  极差={o['range']:.4f}")
    print(f"    平均值={o['mean']:.4f}  中位数={o['median']:.4f}")
    print(f"    标准差={o['std']:.4f}  方差={o['var']:.4f}")

    d = stats["displacement_from_first_frame"]
    print(f"\n  [相对首帧位移幅度（欧式距离）]")
    print(f"    min={d['min']:.4f}  max={d['max']:.4f}  "
          f"mean={d['mean']:.4f}  std={d['std']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 坐标变换
# ─────────────────────────────────────────────────────────────────────────────

def transform_verts_cam(verts: torch.Tensor,
                        R_w2c: torch.Tensor,
                        t_w2c: torch.Tensor) -> torch.Tensor:
    """world 空间顶点 → camera 空间顶点。verts: (T, N, 3)"""
    return torch.einsum("tij,tnj->tni", R_w2c, verts) + t_w2c[:, None, :]


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="使用 HaWoRPipeline 统计手部坐标统计量"
    )
    parser.add_argument("--video_path",      type=str,   default="example/video_0.mp4")
    parser.add_argument("--img_focal",       type=float, default=None)
    parser.add_argument("--checkpoint",      type=str,
                        default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str,
                        default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--metric_3D_path",  type=str,
                        default="thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth")
    parser.add_argument("--vis_mode",        type=str,   default="world",
                        help="cam | world")
    parser.add_argument("--output_json",     type=str,   default=None,
                        help="若指定，则将统计结果保存为 JSON 文件")
    parser.add_argument("--use_joints",      action="store_true",
                        help="同时统计 MANO 关节点坐标（默认统计顶点坐标）")
    parser.add_argument("--no_smooth",       action="store_true",
                        help="关闭抖动平滑（smooth_hands 和 smooth_camera 均置 False）")
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    # ── 1. 构建 pipeline 并运行重建 ─────────────────────────────────────────
    cfg = HaWoRConfig(
        checkpoint=args.checkpoint,
        infiller_weight=args.infiller_weight,
        metric_3D_path=args.metric_3D_path,
        verbose=args.verbose,
        smooth_hands=not args.no_smooth,
        smooth_camera=not args.no_smooth,
    )
    pipeline = HaWoRPipeline(cfg)

    result = pipeline.reconstruct(
        video_path=args.video_path,
        image_focal=args.img_focal,
        rendering=False,
    )

    # ── 2. 从 result 中取出所需张量 ──────────────────────────────────────────
    pred_trans     = result["pred_trans"]      # (2, T, 3)
    pred_rot       = result["pred_rot"]        # (2, T, 3)
    pred_hand_pose = result["pred_hand_pose"]  # (2, T, 45)
    pred_betas     = result["pred_betas"]      # (2, T, 10)
    R_w2c          = result["R_w2c"]           # (T, 3, 3)  已经过 R_x 变换
    t_w2c          = result["t_w2c"]           # (T, 3)     已经过 R_x 变换

    T = pred_trans.shape[1]
    vis_start = 0
    vis_end = T - 1

    # ── 3. MANO 前向推理，获取顶点 & 关节点 ─────────────────────────────────
    #   注意：reconstruct() 内部 _build_hand_dicts 已做了 MANO 推理并存入
    #   right_dict/left_dict，但其 vertices 经过了 R_x 变换（world 坐标系）。
    #   为了同时拿到 joints，这里单独再推一次（开销小，仅 CPU 前向）。
    pred_glob_r = run_mano(
        pred_trans[1:2, vis_start:vis_end],
        pred_rot[1:2, vis_start:vis_end],
        pred_hand_pose[1:2, vis_start:vis_end],
        betas=pred_betas[1:2, vis_start:vis_end],
    )
    pred_glob_l = run_mano_left(
        pred_trans[0:1, vis_start:vis_end],
        pred_rot[0:1, vis_start:vis_end],
        pred_hand_pose[0:1, vis_start:vis_end],
        betas=pred_betas[0:1, vis_start:vis_end],
    )

    right_verts  = pred_glob_r["vertices"][0]   # (T, N, 3)  MANO 原始输出（未翻转）
    right_joints = pred_glob_r["joints"][0]     # (T, J, 3)
    left_verts   = pred_glob_l["vertices"][0]
    left_joints  = pred_glob_l["joints"][0]

    # ── 4. 坐标系变换 ────────────────────────────────────────────────────────
    #   reconstruct() 内的 _apply_coord_transform 对 vertices 施加了 R_x 翻转，
    #   使其与 R_w2c / t_w2c 处于同一坐标系。这里对 MANO 原始顶点做同样的变换。
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)

    def to_world(v: torch.Tensor) -> torch.Tensor:
        """MANO 输出 → world 坐标（R_x 翻转）。v: (T, N, 3)"""
        return torch.einsum("ij,tnj->tni", R_x, v.cpu())

    right_verts_world  = to_world(right_verts)
    left_verts_world   = to_world(left_verts)
    right_joints_world = to_world(right_joints)
    left_joints_world  = to_world(left_joints)

    # world → cam
    R_w2c_slice = R_w2c[vis_start:vis_end]
    t_w2c_slice = t_w2c[vis_start:vis_end]

    right_verts_cam  = transform_verts_cam(right_verts_world,  R_w2c_slice, t_w2c_slice)
    left_verts_cam   = transform_verts_cam(left_verts_world,   R_w2c_slice, t_w2c_slice)
    right_joints_cam = transform_verts_cam(right_joints_world, R_w2c_slice, t_w2c_slice)
    left_joints_cam  = transform_verts_cam(left_joints_world,  R_w2c_slice, t_w2c_slice)

    # ── 5. 根据 vis_mode 选择统计目标 ────────────────────────────────────────
    if args.vis_mode == "world":
        rv = right_verts_world.numpy()
        lv = left_verts_world.numpy()
        rj = right_joints_world.numpy()
        lj = left_joints_world.numpy()
        space_label = "World 坐标系"
    else:
        rv = right_verts_cam.numpy()
        lv = left_verts_cam.numpy()
        rj = right_joints_cam.numpy()
        lj = left_joints_cam.numpy()
        space_label = "Camera 坐标系"

    rt = pred_trans[1, vis_start:vis_end].numpy()  # 右手根节点平移 (T, 3)
    lt = pred_trans[0, vis_start:vis_end].numpy()  # 左手根节点平移 (T, 3)

    print(f"\n★ 统计模式: {args.vis_mode.upper()}  →  {space_label}")
    print(f"  视频: {args.video_path}")
    print(f"  帧范围: {vis_start} ~ {vis_end}  (共 {vis_end - vis_start} 帧)")
    print(f"  平滑: {'关闭' if args.no_smooth else '开启'}")

    # ── 6. 计算并打印统计量 ──────────────────────────────────────────────────
    all_stats: dict = {}

    stats_rv = compute_coord_stats(rv, f"右手顶点 [{space_label}]")
    stats_lv = compute_coord_stats(lv, f"左手顶点 [{space_label}]")
    print_stats(stats_rv, f"右手顶点坐标  [{space_label}]")
    print_stats(stats_lv, f"左手顶点坐标  [{space_label}]")
    all_stats["right_vertices"] = stats_rv
    all_stats["left_vertices"]  = stats_lv

    stats_rt = compute_coord_stats(rt[:, np.newaxis, :], f"右手根节点平移 [{space_label}]")
    stats_lt = compute_coord_stats(lt[:, np.newaxis, :], f"左手根节点平移 [{space_label}]")
    print_stats(stats_rt, f"右手根节点平移  [{space_label}]")
    print_stats(stats_lt, f"左手根节点平移  [{space_label}]")
    all_stats["right_translation"] = stats_rt
    all_stats["left_translation"]  = stats_lt

    if args.use_joints:
        stats_rj = compute_coord_stats(rj, f"右手关节点 [{space_label}]")
        stats_lj = compute_coord_stats(lj, f"左手关节点 [{space_label}]")
        print_stats(stats_rj, f"右手关节点坐标  [{space_label}]")
        print_stats(stats_lj, f"左手关节点坐标  [{space_label}]")
        all_stats["right_joints"] = stats_rj
        all_stats["left_joints"]  = stats_lj

    # ── 7. 帧间速度统计 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  帧间速度统计（顶点质心）  [{space_label}]")
    print(f"{'='*60}")
    for hand_label, verts_arr in [("右手", rv), ("左手", lv)]:
        centroid   = verts_arr.mean(axis=1)
        frame_disp = np.linalg.norm(np.diff(centroid, axis=0), axis=-1)
        print(f"\n  {hand_label} 质心帧间位移 (m):")
        print(f"    min={frame_disp.min():.4f}  max={frame_disp.max():.4f}  "
              f"mean={frame_disp.mean():.4f}  std={frame_disp.std():.4f}")
        speed_key = f"{'right' if hand_label == '右手' else 'left'}_centroid_frame_displacement"
        all_stats[speed_key] = {
            "min":  float(frame_disp.min()),
            "max":  float(frame_disp.max()),
            "mean": float(frame_disp.mean()),
            "std":  float(frame_disp.std()),
        }

    # ── 8. 双手间距统计 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  双手质心间距统计  [{space_label}]")
    print(f"{'='*60}")
    r_centroid = rv.mean(axis=1)
    l_centroid = lv.mean(axis=1)
    inter_dist = np.linalg.norm(r_centroid - l_centroid, axis=-1)
    print(f"    min={inter_dist.min():.4f}  max={inter_dist.max():.4f}  "
          f"mean={inter_dist.mean():.4f}  std={inter_dist.std():.4f}  "
          f"range={inter_dist.max()-inter_dist.min():.4f}")
    all_stats["inter_hand_distance"] = {
        "min":    float(inter_dist.min()),
        "max":    float(inter_dist.max()),
        "mean":   float(inter_dist.mean()),
        "median": float(np.median(inter_dist)),
        "std":    float(inter_dist.std()),
        "var":    float(inter_dist.var()),
        "range":  float(inter_dist.max() - inter_dist.min()),
    }

    # ── 9. 元信息 & 保存 JSON ────────────────────────────────────────────────
    all_stats["meta"] = {
        "vis_mode":   args.vis_mode,
        "video_path": args.video_path,
        "vis_start":  vis_start,
        "vis_end":    vis_end,
        "num_frames": vis_end - vis_start,
        "space":      space_label,
        "smooth":     not args.no_smooth,
    }

    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    if args.output_json:
        out_path = args.output_json
    else:
        out_dir = os.path.join(os.path.dirname(args.video_path), video_stem)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"hand_stats_{args.vis_mode}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 统计结果已保存至: {out_path}")
    print("\n[完成]")


if __name__ == "__main__":
    main()
