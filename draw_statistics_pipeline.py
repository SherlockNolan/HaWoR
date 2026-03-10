"""
draw_statistics_pipeline.py

使用 HaWoRPipeline 绘制手部坐标随帧数变化的曲线图。
支持 cam / world 两种坐标系，以及绝对量 / 变化量（delta）两种展示方式。

与 draw_statistics.py 的区别
-----------------------------
- 原版通过脚本分步调用 detect_track_video / hawor_motion_estimation / hawor_slam /
  hawor_infiller，需要读写磁盘中间文件。
- 本版改用 HaWoRPipeline.reconstruct()，一次调用完成全流程，无需中间文件。

使用方式:
    # world 坐标系，绘制绝对量
    python draw_statistics_pipeline.py --vis_mode world --plot_type abs

    # cam 坐标系，绘制帧间变化量
    python draw_statistics_pipeline.py --vis_mode cam --plot_type delta

    # 同时绘制两种模式、两种量（world/cam × abs/delta 共 4 张图 + 2 张对比图）
    python draw_statistics_pipeline.py --vis_mode both --plot_type both

    # 额外绘制关节点曲线、关闭平滑、指定输出目录
    python draw_statistics_pipeline.py --vis_mode world --plot_type abs \
        --use_joints --no_smooth --output_dir ./plots
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # 无头模式，不弹窗；需交互时改为 "TkAgg"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from lib.pipeline.HaWoRPipeline import HaWoRConfig, HaWoRPipeline
from hawor.utils.process import run_mano, run_mano_left


# ─────────────────────────────────────────────────────────────────────────────
# 绘图工具（与 draw_statistics.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

AXIS_COLORS = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db"}
HAND_STYLES = {
    "right": {"ls": "-",  "alpha": 0.85},
    "left":  {"ls": "--", "alpha": 0.70},
}


def _centroid(arr: np.ndarray) -> np.ndarray:
    """每帧质心。arr: (T, N, 3) → (T, 3)"""
    return arr.mean(axis=1)


def _delta(arr: np.ndarray) -> np.ndarray:
    """帧间差分。arr: (T, 3) → (T-1, 3)"""
    return np.diff(arr, axis=0)


def _make_frames(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n)


def plot_abs_or_delta(
    ax: plt.Axes,
    centroid: np.ndarray,
    hand_label: str,
    plot_type: str,
    frame_start: int = 0,
    show_legend: bool = True,
):
    """在给定 Axes 上绘制单只手的质心绝对量或变化量曲线（x/y/z 三轴）。"""
    style   = HAND_STYLES[hand_label]
    hand_cn = "right hand" if hand_label == "right" else "left hand"

    if plot_type == "abs":
        data   = centroid
        frames = _make_frames(len(data), frame_start)
        ylabel = "coord. (m)"
    else:
        data   = _delta(centroid)
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
    r_centroid: np.ndarray,
    l_centroid: np.ndarray,
    plot_type: str,
    frame_start: int = 0,
):
    """绘制双手质心间距（或帧间变化量）曲线。"""
    dist = np.linalg.norm(r_centroid - l_centroid, axis=-1)

    if plot_type == "abs":
        data   = dist
        ylabel = "double hand distances (m)"
    else:
        data   = np.diff(dist)
        ylabel = "Δ(double hand distances) (m/frame)"

    frames = _make_frames(len(data), frame_start)
    ax.plot(frames, data, color="#9b59b6", linewidth=1.3, label="double hand distances")
    ax.axhline(data.mean(), color="#9b59b6", ls=":", alpha=0.6,
               label=f"mean={data.mean():.3f}")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Frame number", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")


def plot_norm(
    ax: plt.Axes,
    centroid: np.ndarray,
    hand_label: str,
    plot_type: str,
    frame_start: int = 0,
):
    """绘制质心到原点的欧式距离（或帧间变化量）。"""
    norm    = np.linalg.norm(centroid, axis=-1)
    hand_cn = "right hand" if hand_label == "right" else "left hand"
    color   = "#e67e22" if hand_label == "right" else "#1abc9c"

    if plot_type == "abs":
        data   = norm
        ylabel = "‖center of mass‖ (m)"
    else:
        data   = np.diff(norm)
        ylabel = "Δ‖center of mass‖ (m/frame)"

    frames = _make_frames(len(data), frame_start)
    ax.plot(frames, data, color=color, linewidth=1.2, label=f"{hand_cn}-norm")


# ─────────────────────────────────────────────────────────────────────────────
# 核心绘图函数：单张图
# ─────────────────────────────────────────────────────────────────────────────

def draw_single(
    r_centroid: np.ndarray,
    l_centroid: np.ndarray,
    r_joints_centroid: np.ndarray | None,
    l_joints_centroid: np.ndarray | None,
    vis_mode: str,
    plot_type: str,
    frame_start: int,
    output_dir: str,
    tag: str = "",
) -> str:
    """
    绘制一张完整的统计图：
      Row 0 : 右手质心 x/y/z
      Row 1 : 左手质心 x/y/z
      Row 2 : 左右手叠加对比
      Row 3L: ‖质心‖（到原点距离）
      Row 3R: 双手间距
      Row 4 : 关节点质心（可选）
    """
    use_joints  = (r_joints_centroid is not None)
    n_rows      = 5 if use_joints else 4
    space_label = "World coord." if vis_mode == "world" else "Camera coord."
    type_label  = "absolute value" if plot_type == "abs" else "frame changes (Delta)"

    fig = plt.figure(figsize=(14, 3.2 * n_rows), dpi=120)
    fig.suptitle(
        f"Hand coord. statistics  [{space_label}]  ─  {type_label}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.55, wspace=0.35)

    # Row 0 : 右手质心
    ax_r = fig.add_subplot(gs[0, :])
    ax_r.set_title("coord. of mass center (right)", fontsize=10)
    plot_abs_or_delta(ax_r, r_centroid, "right", plot_type, frame_start)

    # Row 1 : 左手质心
    ax_l = fig.add_subplot(gs[1, :])
    ax_l.set_title("coord. of mass center (left)", fontsize=10)
    plot_abs_or_delta(ax_l, l_centroid, "left", plot_type, frame_start)

    # Row 2 : 左右叠加对比
    ax_cmp = fig.add_subplot(gs[2, :])
    ax_cmp.set_title(
        "left/right hand center of mass comparison "
        "(x/y/z, solid line = right, dashed line = left)",
        fontsize=10,
    )
    plot_abs_or_delta(ax_cmp, r_centroid, "right", plot_type, frame_start, show_legend=False)
    plot_abs_or_delta(ax_cmp, l_centroid, "left",  plot_type, frame_start, show_legend=True)

    # Row 3L : ‖质心‖
    ax_norm = fig.add_subplot(gs[3, 0])
    ax_norm.set_title("‖center of mass‖ (distance to origin)", fontsize=10)
    plot_norm(ax_norm, r_centroid, "right", plot_type, frame_start)
    plot_norm(ax_norm, l_centroid, "left",  plot_type, frame_start)
    ax_norm.set_xlabel("Frame number", fontsize=9)
    ax_norm.set_ylabel("‖centroid‖ (m)" if plot_type == "abs" else "Δ‖centroid‖ (m/frame)",
                        fontsize=9)
    ax_norm.tick_params(labelsize=8)
    ax_norm.grid(True, linestyle=":", alpha=0.5)
    ax_norm.legend(fontsize=7, loc="upper right")

    # Row 3R : 双手间距
    ax_dist = fig.add_subplot(gs[3, 1])
    ax_dist.set_title("double hand distances", fontsize=10)
    plot_distance(ax_dist, r_centroid, l_centroid, plot_type, frame_start)

    # Row 4 : 关节点（可选）
    if use_joints:
        ax_rj = fig.add_subplot(gs[4, 0])
        ax_rj.set_title("right hand joints centroid", fontsize=10)
        plot_abs_or_delta(ax_rj, r_joints_centroid, "right", plot_type, frame_start)

        ax_lj = fig.add_subplot(gs[4, 1])
        ax_lj.set_title("left hand joints centroid", fontsize=10)
        plot_abs_or_delta(ax_lj, l_joints_centroid, "left", plot_type, frame_start)

    fname    = f"hand_coord_{vis_mode}_{plot_type}{('_' + tag) if tag else ''}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 核心绘图函数：world vs cam 对比图
# ─────────────────────────────────────────────────────────────────────────────

def draw_comparison(
    data: dict,
    plot_type: str,
    frame_start: int,
    output_dir: str,
) -> str:
    """左列 = world，右列 = cam，行 = 右手/左手/‖质心‖/双手间距。"""
    type_label = "absolute value" if plot_type == "abs" else "frame changes (Delta)"
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), dpi=110)
    fig.suptitle(f"World vs Camera coord. comparison  ─  {type_label}",
                 fontsize=13, fontweight="bold")

    col_titles = ["World coord.", "Camera coord."]
    for col_i, mode in enumerate(["world", "cam"]):
        d   = data[mode]
        r_c = d["r_centroid"]
        l_c = d["l_centroid"]

        axes[0, col_i].set_title(f"{col_titles[col_i]}\nright hand centroid (xyz)", fontsize=9)
        plot_abs_or_delta(axes[0, col_i], r_c, "right", plot_type, frame_start)

        axes[1, col_i].set_title(f"{col_titles[col_i]}\nleft hand centroid (xyz)", fontsize=9)
        plot_abs_or_delta(axes[1, col_i], l_c, "left", plot_type, frame_start)

        axes[2, col_i].set_title(f"{col_titles[col_i]}\n‖centroid‖ (distance to origin)", fontsize=9)
        plot_norm(axes[2, col_i], r_c, "right", plot_type, frame_start)
        plot_norm(axes[2, col_i], l_c, "left",  plot_type, frame_start)
        axes[2, col_i].set_xlabel("Frame number", fontsize=9)
        axes[2, col_i].set_ylabel(
            "‖centroid‖ (m)" if plot_type == "abs" else "Δ‖centroid‖ (m/frame)", fontsize=9
        )
        axes[2, col_i].tick_params(labelsize=8)
        axes[2, col_i].grid(True, linestyle=":", alpha=0.5)
        axes[2, col_i].legend(fontsize=7)

        axes[3, col_i].set_title(f"{col_titles[col_i]}\ndouble hand distances", fontsize=9)
        plot_distance(axes[3, col_i], r_c, l_c, plot_type, frame_start)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname    = f"hand_coord_comparison_{plot_type}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 保存: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="使用 HaWoRPipeline 绘制手部坐标统计曲线"
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
                        help="cam | world | both")
    parser.add_argument("--plot_type",       type=str,   default="abs",
                        help="abs | delta | both")
    parser.add_argument("--use_joints",      action="store_true",
                        help="同时绘制 MANO 关节点质心曲线")
    parser.add_argument("--no_smooth",       action="store_true",
                        help="关闭抖动平滑")
    parser.add_argument("--output_dir",      type=str,   default=None,
                        help="图片输出目录，默认保存在视频同名子目录/plots/")
    parser.add_argument("--show",            action="store_true",
                        help="保存后用系统默认程序打开图片")
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    # ── 1. 构建 pipeline 并运行重建 ─────────────────────────────────────────
    cfg = HaWoRConfig(
        checkpoint=args.checkpoint,
        infiller_weight=args.infiller_weight,
        metric_3D_path=args.metric_3D_path,
        verbose=True,
        smooth_hands=not args.no_smooth,
        smooth_camera=not args.no_smooth,
    )
    pipeline = HaWoRPipeline(cfg)

    result = pipeline.reconstruct(
        video_path=args.video_path,
        image_focal=args.img_focal,
        rendering=False,
    )

    # ── 2. 从 result 取出张量 ────────────────────────────────────────────────
    pred_trans     = result["pred_trans"]      # (2, T, 3)
    pred_rot       = result["pred_rot"]        # (2, T, 3)
    pred_hand_pose = result["pred_hand_pose"]  # (2, T, 45)
    pred_betas     = result["pred_betas"]      # (2, T, 10)
    R_w2c          = result["R_w2c"]           # (T, 3, 3)
    t_w2c          = result["t_w2c"]           # (T, 3)

    T         = pred_trans.shape[1]
    vis_start = 0
    vis_end   = T - 1

    # ── 3. MANO 前向推理（获取顶点 & 关节点） ───────────────────────────────
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

    right_verts  = pred_glob_r["vertices"][0]   # (T, N, 3)
    right_joints = pred_glob_r["joints"][0]     # (T, J, 3)
    left_verts   = pred_glob_l["vertices"][0]
    left_joints  = pred_glob_l["joints"][0]

    # ── 4. 坐标变换（与 _apply_coord_transform 一致） ───────────────────────
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)

    def to_world(v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,tnj->tni", R_x, v.cpu())

    right_verts_world  = to_world(right_verts)
    left_verts_world   = to_world(left_verts)
    right_joints_world = to_world(right_joints)
    left_joints_world  = to_world(left_joints)

    R_w2c_s = R_w2c[vis_start:vis_end]
    t_w2c_s = t_w2c[vis_start:vis_end]

    def to_cam(v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tij,tnj->tni", R_w2c_s, v) + t_w2c_s[:, None, :]

    right_verts_cam  = to_cam(right_verts_world)
    left_verts_cam   = to_cam(left_verts_world)
    right_joints_cam = to_cam(right_joints_world)
    left_joints_cam  = to_cam(left_joints_world)

    # ── 5. 计算质心 ──────────────────────────────────────────────────────────
    def np_centroid(t: torch.Tensor) -> np.ndarray:
        return t.numpy().mean(axis=1)   # (T, 3)

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

    # ── 6. 输出目录 ──────────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
    else:
        video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
        out_dir = os.path.join(os.path.dirname(args.video_path), video_stem, "plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n★ 输出目录: {out_dir}")

    # ── 7. 确定绘制的 vis_mode / plot_type 组合 ─────────────────────────────
    modes  = ["world", "cam"] if args.vis_mode == "both" else [args.vis_mode]
    ptypes = ["abs", "delta"]  if args.plot_type == "both" else [args.plot_type]

    saved_files = []

    # 单模式图
    for mode in modes:
        d = data_all[mode]
        for ptype in ptypes:
            print(f"\n  绘制: vis_mode={mode}  plot_type={ptype}")
            fpath = draw_single(
                r_centroid=d["r_centroid"],
                l_centroid=d["l_centroid"],
                r_joints_centroid=d["r_joints_centroid"],
                l_joints_centroid=d["l_joints_centroid"],
                vis_mode=mode,
                plot_type=ptype,
                frame_start=vis_start,
                output_dir=out_dir,
            )
            saved_files.append(fpath)

    # 双模式对比图（仅当 vis_mode == both）
    if args.vis_mode == "both":
        for ptype in ptypes:
            print(f"\n  绘制对比图: plot_type={ptype}")
            fpath = draw_comparison(data_all, ptype, vis_start, out_dir)
            saved_files.append(fpath)

    print(f"\n[完成] 共保存 {len(saved_files)} 张图片。")

    # ── 8. 可选：用系统程序打开 ─────────────────────────────────────────────
    if args.show:
        import subprocess
        for fp in saved_files:
            subprocess.Popen(["start", "", fp], shell=True)


if __name__ == "__main__":
    main()
