"""
统计 output 文件夹下已处理的视频数量、worker 数量，并按每视频 2000 帧计算总时长。

数据集目录结构（与 reconstruct_egocentric_opt.py 一致）:
  output_root/
    factory_XXX/
      workers/
        worker_YYY/
          video1_hawor.pkl
          video2_hawor.pkl
          ...

# 用法（Linux）:
python scripts/count_video.py \
    --output /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/egocentric-10k-hawor-2000frames \
    --fps 30

# 默认 output 路径与 reconstruct_egocentric_opt.py 的 DEFAULT_OUTPUT_ROOT 一致
python scripts/count_video.py
"""

import os
import glob
import argparse
from collections import defaultdict

# 与 reconstruct_egocentric_opt.py 保持一致
DEFAULT_OUTPUT_ROOT = (
    "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/egocentric-10k-hawor-2000frames"
)
FRAMES_PER_VIDEO = 2000  # 每个视频截取的帧数


def count_videos(output_root: str, fps: int = 30):
    """扫描 output 目录，统计 pkl 文件（已处理视频）并汇总。"""
    if not os.path.isdir(output_root):
        print(f"错误: 目录不存在 '{output_root}'")
        return

    # 匹配所有 _hawor.pkl 文件
    pattern = os.path.join(output_root, "**", "*_hawor.pkl")
    pkl_files = glob.glob(pattern, recursive=True)
    pkl_files.sort()

    total_videos = len(pkl_files)
    if total_videos == 0:
        print(f"在 '{output_root}' 下未找到任何 *_hawor.pkl 文件。")
        return

    # 解析 factory / worker 结构
    # rel_path 示例: factory_001/workers/worker_002/video_hawor.pkl
    factory_map: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for pkl in pkl_files:
        try:
            rel = os.path.relpath(pkl, output_root)
        except Exception:
            rel = os.path.basename(pkl)
        parts = rel.split(os.sep)

        if len(parts) >= 3 and parts[1] == "workers":
            factory = parts[0]
            worker = parts[2]
            factory_map[factory][worker].append(pkl)
        else:
            # 不符合 factory/workers/worker 结构的文件，归入 "unknown"
            factory_map["unknown"]["unknown"].append(pkl)

    # 统计
    total_workers = sum(len(workers) for workers in factory_map.values())
    total_factories = len(factory_map)
    total_frames = total_videos * FRAMES_PER_VIDEO
    total_seconds = total_frames / fps
    total_hours = total_seconds / 3600

    # 打印结果
    print("=" * 60)
    print(f"Output 目录: {output_root}")
    print(f"FPS (假设):  {fps}")
    print(f"每视频帧数:  {FRAMES_PER_VIDEO}")
    print("=" * 60)
    print(f"Factory 总数:   {total_factories}")
    print(f"Worker  总数:   {total_workers}")
    print(f"视频 (pkl) 总数: {total_videos}")
    print(f"总帧数:         {total_frames:,}")
    print(f"总时长:         {total_hours:,.2f} 小时 ({total_seconds:,.0f} 秒)")
    print("=" * 60)

    # 每个 factory 的详细统计
    print("\n--- 各 Factory 详情 ---")
    for factory in sorted(factory_map.keys()):
        workers = factory_map[factory]
        vid_count = sum(len(v) for v in workers.values())
        worker_count = len(workers)
        frames = vid_count * FRAMES_PER_VIDEO
        hours = frames / fps / 3600
        print(f"  {factory}: {worker_count} workers, {vid_count} videos, {hours:,.2f} hours")

    print(f"\n--- Top 20 Workers（按视频数降序）---")
    all_workers = []
    for factory, workers in factory_map.items():
        for worker, vids in workers.items():
            all_workers.append((factory, worker, len(vids)))
    all_workers.sort(key=lambda x: x[2], reverse=True)
    for factory, worker, count in all_workers[:20]:
        print(f"  {factory}/{worker}: {count} videos")


def main():
    parser = argparse.ArgumentParser(
        description="统计 output 目录下已处理视频的数量与时长"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="输出根目录（默认与 reconstruct_egocentric_opt.py 一致）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率，用于将帧数换算为时长（默认 30）",
    )
    args = parser.parse_args()
    count_videos(args.output, args.fps)


if __name__ == "__main__":
    main()
