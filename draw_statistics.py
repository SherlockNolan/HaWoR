"""
draw_statistics.py

绘制 HaWoR 估计的手部坐标随帧数变化的曲线图。
支持 cam / world 两种坐标系，以及绝对量 / 变化量（delta）两种展示方式。

使用方式:
    # world 坐标系，绘制绝对量
    python draw_statistics.py --vis_mode world --plot_type abs

    # cam 坐标系，绘制帧间变化量
    python draw_statistics.py --vis_mode cam --plot_type delta

    # 同时绘制两种模式、两种量（2×2 对比图）
    python draw_statistics.py --vis_mode both --plot_type both

    # 额外绘制关节点曲线、指定输出目录
    python draw_statistics.py --vis_mode world --plot_type abs --use_joints --output_dir ./plots
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # 无头模式，不弹窗，可改为 "TkAgg" 交互
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam


# ─────────────────────────────────────────────────────────────────────────────
# 坐标变换（与 demo.py / calculate_statistics.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────

def transform_verts_cam(verts: torch.Tensor,
                        R_w2c: torch.Tensor,
                        t_w2c: torch.Tensor) -> torch.Tensor:
    """world space vertexes → camera space vertexes。verts: (T, N, 3)"""
    return torch.einsum("tij,tnj->tni", R_w2c, verts) + t_w2c[:, None, :]


# ─────────────────────────────────────────────────────────────────────────────
# 绘图工具
# ─────────────────────────────────────────────────────────────────────────────

AXIS_COLORS = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db"}
HAND_STYLES = {
    "right": {"ls": "-",  "alpha": 0.85},
    "left":  {"ls": "--", "alpha": 0.70},
}


def _centroid(arr: np.ndarray) -> np.ndarray:
    """返回每帧的质心坐标。arr: (T, N, 3) → (T, 3)"""
    return arr.mean(axis=1)


def _delta(arr: np.ndarray) -> np.ndarray:
    """计算帧间差分（变化量）。arr: (T, 3) → (T-1, 3)"""
    return np.diff(arr, axis=0)


def _make_frames(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n)


def plot_abs_or_delta(
    ax: plt.Axes,
    centroid: np.ndarray,    # (T, 3)
    hand_label: str,         # "right" / "left"
    plot_type: str,          # "abs" / "delta"
    frame_start: int = 0,
    show_legend: bool = True,
):
    """在给定 Axes 上绘制单只手的质心绝对量或变化量曲线（x/y/z 三轴）。"""
    style = HAND_STYLES[hand_label]
    hand_cn = "right hand" if hand_label == "right" else "left hand"

    if plot_type == "abs":
        data = centroid                          # (T, 3)
        frames = _make_frames(len(data), frame_start)
        ylabel = "coord. (m)"
    else:
        data = _delta(centroid)                  # (T-1, 3)
        frames = _make_frames(len(data), frame_start)
        ylabel = "Δcoord. (m/frame)"

    for i, axis in enumerate(["x", "y", "z"]):
        ax.plot(
            frames, data[:, i],
            color=AXIS_COLORS[axis],
            ls=style["ls"],
            alpha=style["alpha"],
            linewidth=1.2,
            label=f"{hand_cn}-{axis}",
        )

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Frame number", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    if show_legend:
        ax.legend(fontsize=7, ncol=3, loc="upper right")


def plot_distance(
    ax: plt.Axes,
    r_centroid: np.ndarray,   # (T, 3)
    l_centroid: np.ndarray,   # (T, 3)
    plot_type: str,            # "abs" / "delta"
    frame_start: int = 0,
):
    """绘制双手质心间距（或帧间变化量）曲线。"""
    dist = np.linalg.norm(r_centroid - l_centroid, axis=-1)  # (T,)

    if plot_type == "abs":
        data = dist
        ylabel = "double hand distances (m)"
    else:
        data = np.diff(dist)
        ylabel = "Δ(double hand distances) (m/frame)"

    frames = _make_frames(len(data), frame_start)
    ax.plot(frames, data, color="#9b59b6", linewidth=1.3, label="double hand distances")
    ax.axhline(data.mean(), color="#9b59b6", ls=":", alpha=0.6,
               label=f"mean={data.mean():.3f}")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Fram Number", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")


def plot_norm(
    ax: plt.Axes,
    centroid: np.ndarray,    # (T, 3)
    hand_label: str,
    plot_type: str,           # "abs" / "delta"
    frame_start: int = 0,
):
    """绘制质心到原点的欧式距离（或帧间变化量）。"""
    norm = np.linalg.norm(centroid, axis=-1)  # (T,)
    hand_cn = "right hand" if hand_label == "right" else "left hand"
    color = "#e67e22" if hand_label == "right" else "#1abc9c"

    if plot_type == "abs":
        data = norm
        ylabel = "‖center of mass‖ (m)"
    else:
        data = np.diff(norm)
        ylabel = "Δ‖center of mass‖ (m/frame)"

    frames = _make_frames(len(data), frame_start)
    ax.plot(frames, data, color=color, linewidth=1.2, label=f"{hand_cn}-norm")


# ─────────────────────────────────────────────────────────────────────────────
# 核心绘图函数：单张图（固定 vis_mode + plot_type）
# ─────────────────────────────────────────────────────────────────────────────

def draw_single(
    r_centroid: np.ndarray,    # (T, 3)  右手质心
    l_centroid: np.ndarray,    # (T, 3)  左手质心
    r_joints_centroid: np.ndarray | None,   # (T, 3) 或 None
    l_joints_centroid: np.ndarray | None,
    vis_mode: str,             # "world" / "cam"
    plot_type: str,            # "abs" / "delta"
    frame_start: int,
    output_dir: str,
    tag: str = "",
):
    """
    绘制一张完整的统计图，包含：
      Row 0: 右手质心 x/y/z 曲线
      Row 1: 左手质心 x/y/z 曲线
      Row 2: 双手质心 x/y/z 叠加对比
      Row 3: ‖质心‖（到原点距离）+ 双手间距
      Row 4（可选）: 关节点质心 x/y/z
    """
    use_joints = (r_joints_centroid is not None)
    n_rows = 5 if use_joints else 4
    space_label = "World coord." if vis_mode == "world" else "Camera coord."
    type_label  = "absolute value" if plot_type == "abs" else "frame changes（Delta）"

    fig = plt.figure(figsize=(14, 3.2 * n_rows), dpi=120)
    fig.suptitle(
        f"hand coord. statistics  [{space_label}]  ─  {type_label}",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.55, wspace=0.35)

    # ── Row 0: 右手质心 xyz ─────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[0, :])
    ax_r.set_title("coord. of mass center (right)", fontsize=10)
    plot_abs_or_delta(ax_r, r_centroid, "right", plot_type, frame_start)

    # ── Row 1: 左手质心 xyz ─────────────────────────────────────────────────
    ax_l = fig.add_subplot(gs[1, :])
    ax_l.set_title("coord. of mass center (left)", fontsize=10)
    plot_abs_or_delta(ax_l, l_centroid, "left", plot_type, frame_start)

    # ── Row 2: 左右叠加对比（每轴单独子图）──────────────────────────────────
    for col_i, (axis, color) in enumerate(
        [("x", AXIS_COLORS["x"]), ("y", AXIS_COLORS["y"]), ("z", AXIS_COLORS["z"])]
    ):
        # 把 x/y/z 各放一个格子（横向三格→合并到两列：左x，右y+z）
        pass  # 用下面统一方式

    ax_cmp = fig.add_subplot(gs[2, :])
    ax_cmp.set_title("left/right hand center of mass comparison (x/y/z, bold lin e= right, dashed line = left)", fontsize=10)
    plot_abs_or_delta(ax_cmp, r_centroid, "right", plot_type, frame_start, show_legend=False)
    plot_abs_or_delta(ax_cmp, l_centroid, "left",  plot_type, frame_start, show_legend=True)

    # ── Row 3 左: ‖质心‖ 距离 ───────────────────────────────────────────────
    ax_norm = fig.add_subplot(gs[3, 0])
    ax_norm.set_title("质心到原点距离 ‖·‖", fontsize=10)
    plot_norm(ax_norm, r_centroid, "right", plot_type, frame_start)
    plot_norm(ax_norm, l_centroid, "left",  plot_type, frame_start)
    ax_norm.set_xlabel("帧序号", fontsize=9)
    ax_norm.set_ylabel("‖质心‖ (m)" if plot_type == "abs" else "Δ‖质心‖ (m/frame)", fontsize=9)
    ax_norm.tick_params(labelsize=8)
    ax_norm.grid(True, linestyle=":", alpha=0.5)
    ax_norm.legend(fontsize=7, loc="upper right")

    # ── Row 3 右: 双手间距 ───────────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[3, 1])
    ax_dist.set_title("双手质心间距", fontsize=10)
    plot_distance(ax_dist, r_centroid, l_centroid, plot_type, frame_start)

    # ── Row 4（可选）: 关节点质心 ────────────────────────────────────────────
    if use_joints:
        ax_rj = fig.add_subplot(gs[4, 0])
        ax_rj.set_title("右手关节点质心坐标", fontsize=10)
        plot_abs_or_delta(ax_rj, r_joints_centroid, "right", plot_type, frame_start)

        ax_lj = fig.add_subplot(gs[4, 1])
        ax_lj.set_title("左手关节点质心坐标", fontsize=10)
        plot_abs_or_delta(ax_lj, l_joints_centroid, "left", plot_type, frame_start)

    fname = f"hand_coord_{vis_mode}_{plot_type}{('_' + tag) if tag else ''}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 核心绘图函数：双模式对比图（world vs cam，同一 plot_type）
# ─────────────────────────────────────────────────────────────────────────────

def draw_comparison(
    data: dict,                # {"world": {...}, "cam": {...}}
    plot_type: str,            # "abs" / "delta"
    frame_start: int,
    output_dir: str,
):
    """左列=world，右列=cam，行=右手/左手/双手间距/‖质心‖，绘对比图。"""
    type_label = "绝对量" if plot_type == "abs" else "帧间变化量（Delta）"
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), dpi=110)
    fig.suptitle(f"World vs Camera 坐标系对比  ─  {type_label}",
                 fontsize=13, fontweight="bold")

    row_titles = ["右手质心 (xyz)", "左手质心 (xyz)", "质心到原点距离", "双手质心间距"]
    col_titles = ["World 坐标系", "Camera 坐标系"]

    for col_i, mode in enumerate(["world", "cam"]):
        d = data[mode]
        r_c = d["r_centroid"]
        l_c = d["l_centroid"]

        axes[0, col_i].set_title(f"{col_titles[col_i]}\n右手质心 (xyz)", fontsize=9)
        plot_abs_or_delta(axes[0, col_i], r_c, "right", plot_type, frame_start)

        axes[1, col_i].set_title(f"{col_titles[col_i]}\n左手质心 (xyz)", fontsize=9)
        plot_abs_or_delta(axes[1, col_i], l_c, "left", plot_type, frame_start)

        axes[2, col_i].set_title(f"{col_titles[col_i]}\n质心到原点距离", fontsize=9)
        plot_norm(axes[2, col_i], r_c, "right", plot_type, frame_start)
        plot_norm(axes[2, col_i], l_c, "left",  plot_type, frame_start)
        axes[2, col_i].set_xlabel("帧序号", fontsize=9)
        axes[2, col_i].set_ylabel("‖质心‖ (m)" if plot_type == "abs" else "Δ‖质心‖", fontsize=9)
        axes[2, col_i].tick_params(labelsize=8)
        axes[2, col_i].grid(True, linestyle=":", alpha=0.5)
        axes[2, col_i].legend(fontsize=7)

        axes[3, col_i].set_title(f"{col_titles[col_i]}\n双手质心间距", fontsize=9)
        plot_distance(axes[3, col_i], r_c, l_c, plot_type, frame_start)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = f"hand_coord_comparison_{plot_type}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="绘制 HaWoR 手部坐标统计曲线")
    parser.add_argument("--img_focal",       type=float, default=None)
    parser.add_argument("--video_path",      type=str,   default="example/video_0.mp4")
    parser.add_argument("--input_type",      type=str,   default="file")
    parser.add_argument("--checkpoint",      type=str,   default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str,   default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--vis_mode",        type=str,   default="world",
                        help="cam | world | both（both 生成双模式对比图）")
    parser.add_argument("--plot_type",       type=str,   default="abs",
                        help="abs | delta | both（both 同时生成两种图）")
    parser.add_argument("--use_joints",      action="store_true",
                        help="同时绘制 MANO 关节点质心曲线")
    parser.add_argument("--output_dir",      type=str,   default=None,
                        help="图片输出目录，默认保存在 seq_folder/plots/")
    parser.add_argument("--show",            action="store_true",
                        help="保存后用系统默认程序打开图片")
    args = parser.parse_args()

    # ── 1. 前置流程（与 demo.py 完全相同）───────────────────────────────────
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    # ── 2. MANO 前向推理 ─────────────────────────────────────────────────────
    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1

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

    right_verts  = pred_glob_r["vertices"][0]   # (T, N_v, 3)
    right_joints = pred_glob_r["joints"][0]      # (T, J, 3)
    left_verts   = pred_glob_l["vertices"][0]
    left_joints  = pred_glob_l["joints"][0]

    # ── 3. 坐标变换（与 demo.py 一致）───────────────────────────────────────
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()

    R_c2w_sla = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all[vis_start:vis_end])
    t_c2w_sla = torch.einsum("ij,nj->ni",  R_x, t_c2w_sla_all[vis_start:vis_end])
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)

    def to_world(v):
        return torch.einsum("ij,tnj->tni", R_x, v.cpu())

    right_verts_world  = to_world(right_verts)
    left_verts_world   = to_world(left_verts)
    right_joints_world = to_world(right_joints)
    left_joints_world  = to_world(left_joints)

    right_verts_cam  = transform_verts_cam(right_verts_world,  R_w2c_sla, t_w2c_sla)
    left_verts_cam   = transform_verts_cam(left_verts_world,   R_w2c_sla, t_w2c_sla)
    right_joints_cam = transform_verts_cam(right_joints_world, R_w2c_sla, t_w2c_sla)
    left_joints_cam  = transform_verts_cam(left_joints_world,  R_w2c_sla, t_w2c_sla)

    # ── 4. 计算质心 ──────────────────────────────────────────────────────────
    def np_centroid(t: torch.Tensor) -> np.ndarray:
        return t.numpy().mean(axis=1)      # (T, 3)

    data_all = {
        "world": {
            "r_centroid":        np_centroid(right_verts_world),
            "l_centroid":        np_centroid(left_verts_world),
            "r_joints_centroid": np_centroid(right_joints_world) if args.use_joints else None,
            "l_joints_centroid": np_centroid(left_joints_world)  if args.use_joints else None,
        },
        "cam": {
            "r_centroid":        np_centroid(right_verts_cam),
            "l_centroid":        np_centroid(left_verts_cam),
            "r_joints_centroid": np_centroid(right_joints_cam) if args.use_joints else None,
            "l_joints_centroid": np_centroid(left_joints_cam)  if args.use_joints else None,
        },
    }

    # ── 5. 输出目录 ──────────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(seq_folder, "plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n★ 输出目录: {out_dir}")

    # ── 6. 确定需要绘制的 vis_mode / plot_type 组合 ─────────────────────────
    modes = ["world", "cam"] if args.vis_mode == "both" else [args.vis_mode]
    ptypes = ["abs", "delta"]  if args.plot_type == "both" else [args.plot_type]

    saved_files = []

    # 单模式图
    for mode in modes:
        d = data_all[mode]
        for ptype in ptypes:
            print(f"\n  绘制: vis_mode={mode}  plot_type={ptype}")
            fpath = draw_single(
                r_centroid        = d["r_centroid"],
                l_centroid        = d["l_centroid"],
                r_joints_centroid = d["r_joints_centroid"],
                l_joints_centroid = d["l_joints_centroid"],
                vis_mode  = mode,
                plot_type = ptype,
                frame_start = vis_start,
                output_dir  = out_dir,
            )
            saved_files.append(fpath)

    # 双模式对比图（仅当 vis_mode == both）
    if args.vis_mode == "both":
        for ptype in ptypes:
            print(f"\n  绘制对比图: plot_type={ptype}")
            fpath = draw_comparison(data_all, ptype, vis_start, out_dir)
            saved_files.append(fpath)

    print(f"\n[完成] 共保存 {len(saved_files)} 张图片。")

    # ── 7. 可选：用系统程序打开 ─────────────────────────────────────────────
    if args.show:
        import subprocess
        for fp in saved_files:
            subprocess.Popen(["start", "", fp], shell=True)


if __name__ == "__main__":
    main()
