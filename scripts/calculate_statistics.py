"""
calculate_statistics.py

统计 HaWoR 估计的手部坐标统计量（支持 cam / world 两种模式）。

使用方式:
    python calculate_statistics.py \
        --video_path example/video_0.mp4 \
        --checkpoint ./weights/hawor/checkpoints/hawor.ckpt \
        --infiller_weight ./weights/hawor/checkpoints/infiller.pt \
        --vis_mode world          # cam | world
        [--img_focal 600]         # 可选，焦距
        [--output_json stats.json] # 可选，保存统计结果到 JSON
        [--use_joints]             # 可选，同时统计关节点坐标（默认统计顶点坐标）
"""

import argparse
import os
import sys

import numpy as np
import torch
import joblib
import json
from natsort import natsorted
from glob import glob

sys.path.insert(0, os.path.dirname(__file__))

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam


# ─────────────────────────────────────────────────────────────────────────────
# 统计工具函数
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

    # 全坐标整体统计（把 x/y/z 全部展平在一起）
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

    # 每帧的位移幅度（欧式距离相对首帧）
    origin = flat[0:1, :]  # (1, 3)
    dists = np.linalg.norm(flat - origin, axis=-1)  # (N,)
    stats["displacement_from_first_frame"] = {
        "min":    float(dists.min()),
        "max":    float(dists.max()),
        "mean":   float(dists.mean()),
        "std":    float(dists.std()),
    }

    return stats


def print_stats(stats: dict, label: str):
    """格式化打印统计结果。"""
    print(f"\n{'='*60}")
    print(f"  {label}  (shape={stats['shape']}, samples={stats['num_samples']})")
    print(f"{'='*60}")

    print(f"  {'指标':<12} {'x':>10} {'y':>10} {'z':>10}")
    print(f"  {'-'*46}")
    metric_keys = ["min", "max", "mean", "median", "std", "var", "range", "p5", "p25", "p75", "p95"]
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
    print(f"    min={d['min']:.4f}  max={d['max']:.4f}  mean={d['mean']:.4f}  std={d['std']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 坐标变换（与 demo.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def apply_R_x_transform(verts: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor):
    """
    对顶点和相机姿态施加 demo.py 中的 R_x 翻转变换，返回 world 模式所用的坐标。

    Parameters
    ----------
    verts : torch.Tensor  (T, N, 3)
    R_c2w : torch.Tensor  (T, 3, 3)
    t_c2w : torch.Tensor  (T, 3)

    Returns
    -------
    verts_world, R_c2w_new, t_c2w_new
    """
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    verts_world = torch.einsum("ij,tnj->tni", R_x, verts.cpu())
    R_c2w_new = torch.einsum("ij,njk->nik", R_x, R_c2w)
    t_c2w_new = torch.einsum("ij,nj->ni", R_x, t_c2w)
    return verts_world, R_c2w_new, t_c2w_new


def transform_verts_cam(verts: torch.Tensor, R_w2c: torch.Tensor, t_w2c: torch.Tensor) -> torch.Tensor:
    """
    将 world 空间的顶点投影到相机坐标系。

    Parameters
    ----------
    verts  : (T, N, 3)  world 空间坐标
    R_w2c  : (T, 3, 3)
    t_w2c  : (T, 3)

    Returns
    -------
    verts_cam : (T, N, 3)
    """
    # verts_cam[t] = R_w2c[t] @ verts[t].T + t_w2c[t]
    verts_cam = torch.einsum("tij,tnj->tni", R_w2c, verts) + t_w2c[:, None, :]
    return verts_cam


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="统计 HaWoR 估计的手部坐标统计量")
    parser.add_argument("--img_focal",        type=float, default=None)
    parser.add_argument("--video_path",       type=str,   default="example/video_0.mp4")
    parser.add_argument("--input_type",       type=str,   default="file")
    parser.add_argument("--checkpoint",       type=str,   default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight",  type=str,   default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--vis_mode",         type=str,   default="world", help="cam | world")
    parser.add_argument("--output_json",      type=str,   default=None,
                        help="若指定，则将统计结果保存为 JSON 文件")
    parser.add_argument("--use_joints",       action="store_true",
                        help="同时统计 MANO 关节点坐标（默认统计网格顶点坐标）")
    args = parser.parse_args()

    # ── 1. 运行检测 / 跟踪 ──────────────────────────────────────────────────
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    # ── 2. 运行 HaWoR 运动估计 ──────────────────────────────────────────────
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    # ── 3. 运行 SLAM ────────────────────────────────────────────────────────
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    # ── 4. 运行 Infiller ────────────────────────────────────────────────────
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1

    # ── 5. MANO 前向推理，获取顶点 & 关节点 ─────────────────────────────────
    faces = get_mano_faces()
    faces_new = np.array([
        [92, 38, 234], [234, 38, 239], [38, 122, 239],
        [239, 122, 279], [122, 118, 279], [279, 118, 215],
        [118, 117, 215], [215, 117, 214], [117, 119, 214],
        [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79]
    ])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left  = faces_right[:, [0, 2, 1]]

    # 右手
    hand_idx = hand2idx["right"]
    pred_glob_r = run_mano(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end],
    )
    right_verts  = pred_glob_r["vertices"][0]   # (T, N, 3)
    right_joints = pred_glob_r["joints"][0]      # (T, J, 3)

    # 左手
    hand_idx = hand2idx["left"]
    pred_glob_l = run_mano_left(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end],
    )
    left_verts  = pred_glob_l["vertices"][0]    # (T, N, 3)
    left_joints = pred_glob_l["joints"][0]      # (T, J, 3)

    # ── 6. 坐标系变换（与 demo.py 一致）────────────────────────────────────
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()

    R_c2w_sla = R_c2w_sla_all[vis_start:vis_end]
    t_c2w_sla = t_c2w_sla_all[vis_start:vis_end]
    R_c2w_sla = torch.einsum("ij,njk->nik", R_x, R_c2w_sla)
    t_c2w_sla = torch.einsum("ij,nj->ni", R_x, t_c2w_sla)
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)

    # world 空间顶点（R_x 翻转）
    right_verts_world  = torch.einsum("ij,tnj->tni", R_x, right_verts.cpu())
    left_verts_world   = torch.einsum("ij,tnj->tni", R_x, left_verts.cpu())
    right_joints_world = torch.einsum("ij,tnj->tni", R_x, right_joints.cpu())
    left_joints_world  = torch.einsum("ij,tnj->tni", R_x, left_joints.cpu())

    # cam 空间坐标（world -> cam）
    right_verts_cam  = transform_verts_cam(right_verts_world,  R_w2c_sla, t_w2c_sla)
    left_verts_cam   = transform_verts_cam(left_verts_world,   R_w2c_sla, t_w2c_sla)
    right_joints_cam = transform_verts_cam(right_joints_world, R_w2c_sla, t_w2c_sla)
    left_joints_cam  = transform_verts_cam(left_joints_world,  R_w2c_sla, t_w2c_sla)

    # ── 7. 根据 vis_mode 选择统计的坐标空间 ─────────────────────────────────
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

    # pred_trans 也按 vis_mode 呈现
    rt = pred_trans[hand2idx["right"], vis_start:vis_end].numpy()  # (T, 3)
    lt = pred_trans[hand2idx["left"],  vis_start:vis_end].numpy()  # (T, 3)

    print(f"\n★ 统计模式: {args.vis_mode.upper()}  →  {space_label}")
    print(f"  视频: {args.video_path}")
    print(f"  帧范围: {vis_start} ~ {vis_end}  (共 {vis_end - vis_start} 帧)")

    # ── 8. 计算并打印统计量 ──────────────────────────────────────────────────
    all_stats = {}

    # 顶点坐标统计
    stats_rv = compute_coord_stats(rv, f"右手顶点 [{space_label}]")
    stats_lv = compute_coord_stats(lv, f"左手顶点 [{space_label}]")
    print_stats(stats_rv, f"右手顶点坐标  [{space_label}]")
    print_stats(stats_lv, f"左手顶点坐标  [{space_label}]")
    all_stats["right_vertices"] = stats_rv
    all_stats["left_vertices"]  = stats_lv

    # 平移坐标统计（根节点）
    stats_rt = compute_coord_stats(rt[:, np.newaxis, :], f"右手根节点平移 [{space_label}]")
    stats_lt = compute_coord_stats(lt[:, np.newaxis, :], f"左手根节点平移 [{space_label}]")
    print_stats(stats_rt, f"右手根节点平移  [{space_label}]")
    print_stats(stats_lt, f"左手根节点平移  [{space_label}]")
    all_stats["right_translation"] = stats_rt
    all_stats["left_translation"]  = stats_lt

    # 关节点坐标统计（可选）
    if args.use_joints:
        stats_rj = compute_coord_stats(rj, f"右手关节点 [{space_label}]")
        stats_lj = compute_coord_stats(lj, f"左手关节点 [{space_label}]")
        print_stats(stats_rj, f"右手关节点坐标  [{space_label}]")
        print_stats(stats_lj, f"左手关节点坐标  [{space_label}]")
        all_stats["right_joints"] = stats_rj
        all_stats["left_joints"]  = stats_lj

    # ── 9. 帧间速度统计（顶点质心的帧间位移） ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  帧间速度统计（顶点质心）  [{space_label}]")
    print(f"{'='*60}")
    for hand_label, verts_arr in [("右手", rv), ("左手", lv)]:
        centroid = verts_arr.mean(axis=1)          # (T, 3) 每帧质心
        frame_disp = np.linalg.norm(np.diff(centroid, axis=0), axis=-1)  # (T-1,)
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

    # ── 10. 双手间距统计 ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  双手质心间距统计  [{space_label}]")
    print(f"{'='*60}")
    r_centroid = rv.mean(axis=1)   # (T, 3)
    l_centroid = lv.mean(axis=1)   # (T, 3)
    inter_dist = np.linalg.norm(r_centroid - l_centroid, axis=-1)  # (T,)
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

    # ── 11. 保存 JSON ────────────────────────────────────────────────────────
    all_stats["meta"] = {
        "vis_mode":   args.vis_mode,
        "video_path": args.video_path,
        "vis_start":  vis_start,
        "vis_end":    vis_end,
        "num_frames": vis_end - vis_start,
        "space":      space_label,
    }

    if args.output_json:
        out_path = args.output_json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 统计结果已保存至: {out_path}")
    else:
        # 默认保存到 seq_folder 旁
        default_out = os.path.join(seq_folder, f"hand_stats_{args.vis_mode}.json")
        with open(default_out, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 统计结果已自动保存至: {default_out}")

    print("\n[完成]")


if __name__ == "__main__":
    main()
