"""
HaWoR Pipeline 单视频测试脚本
针对单个视频执行完整 pipeline 生成 pkl，显示所有控制台输出

使用示例:
    python tests/test_single_video.py --video-path=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/HaWoR/example/video_0.mp4
.venv/bin/python tests/test_single_video.py --video-path=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/HaWoR/example/video_0.mp4
"""

import os
import sys
import pickle
import argparse
import time
import gc
import psutil
import torch
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig
from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints

TEST_OUTPUT_DIR = "./test_results"


def get_memory_usage_mb():
    """获取当前进程内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_mb():
    """获取 GPU 内存使用（MB），仅在 CUDA 可用时有效"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / 1024 / 1024
    return 0


def main():
    parser = argparse.ArgumentParser(description="HaWoR Pipeline 单视频测试")
    parser.add_argument("--video-path", type=str, required=True,
                        help="视频文件路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="运行设备: cpu, cuda, cuda:0 等 (默认: cuda:0)")
    parser.add_argument("--output", type=str, default=TEST_OUTPUT_DIR,
                        help="输出目录 (默认: ./test_results)")
    parser.add_argument("--tmp-dir", type=str, default=None,
                        help="临时文件目录")
    parser.add_argument("--save-origin", action="store_true",
                        help="同时保存原始 hawor 结果 dict")
    parser.add_argument("--frame-start", type=int, default=0,
                        help="起始帧索引 (默认: 0)")
    parser.add_argument("--frame-end", type=int, default=-1,
                        help="结束帧索引 (默认: -1)")
    parser.add_argument("--image-focal", type=float, default=1031.0,
                        help="相机焦距 (默认: 1031.0)")

    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return

    # 记录初始状态
    mem_before = get_memory_usage_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = get_gpu_memory_mb()

    print(f"\n{'='*60}")
    print(f"HaWoR Pipeline 单视频测试")
    print(f"{'='*60}")
    print(f"视频: {video_path}")
    print(f"设备: {args.device}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  - cuda:{i} = {torch.cuda.get_device_name(i)}")
    print(f"初始内存: {mem_before:.1f} MB")
    if torch.cuda.is_available():
        print(f"初始 GPU 内存: {gpu_mem_before:.1f} MB")

    # 构建输出路径
    video_name = os.path.basename(video_path).replace(".mp4", "")
    video_output_dir = os.path.join(args.output, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    pkl_path = os.path.join(video_output_dir, f"{video_name}_hawor.pkl")

    print(f"输出目录: {video_output_dir}")
    print(f"{'='*60}\n")

    # ========== 阶段1: Pipeline 初始化 ==========
    init_start = time.time()

    cfg = HaWoRConfig(
        verbose=True,
        device=args.device,
        tmp_dir=args.tmp_dir,
    )
    print(f"[CONFIG] device={cfg.device}, tmp_dir={cfg.tmp_dir}")

    pipe = HaWoRPipelineOpt(cfg)

    init_time = time.time() - init_start
    init_mem = get_memory_usage_mb()
    if torch.cuda.is_available():
        init_gpu_mem = get_gpu_memory_mb()

    print(f"\n[INIT] Pipeline 初始化完成")
    print(f"  - 初始化耗时: {init_time:.2f}s")
    print(f"  - 初始化后内存: {init_mem:.1f} MB (增量: +{init_mem - mem_before:.1f} MB)")
    if torch.cuda.is_available():
        print(f"  - 初始化后 GPU 内存: {init_gpu_mem:.1f} MB")

    # ========== 阶段2: 视频重建 ==========
    reconstruct_start = time.time()

    print(f"\n[START] 开始处理视频...")
    result_dict_origin = pipe.reconstruct(
        video_path,
        output_dir=video_output_dir,
        image_focal=args.image_focal,
        start_idx=args.frame_start,
        end_idx=args.frame_end,
        rendering=False,
    )

    reconstruct_time = time.time() - reconstruct_start
    reconstruct_mem = get_memory_usage_mb()
    if torch.cuda.is_available():
        reconstruct_gpu_mem = get_gpu_memory_mb()

    print(f"\n[CONVERT] 转换结果为 keypoints 格式...")

    # ========== 阶段3: 结果转换与保存 ==========
    convert_start = time.time()

    result_dict = {
        "original_result": convert_hawor_to_keypoints(result_dict_origin, video_path, use_smoothed=False),
        "smoothed_result": convert_hawor_to_keypoints(result_dict_origin, video_path, use_smoothed=True) if result_dict_origin.get("smoothed_result") is not None else None,
    }

    # 保存结果
    print(f"[SAVE] 保存结果到: {pkl_path}")
    with open(pkl_path, "wb") as f:
        pickle.dump(result_dict, f)

    origin_pkl_path = None
    if args.save_origin:
        origin_pkl_path = pkl_path.replace(".pkl", "_origin_dict.pkl")
        print(f"[SAVE] 保存原始结果到: {origin_pkl_path}")
        with open(origin_pkl_path, "wb") as f:
            pickle.dump(result_dict_origin, f)

    convert_time = time.time() - convert_start

    # ========== 统计信息 ==========
    total_time = init_time + reconstruct_time + convert_time
    mem_after = get_memory_usage_mb()
    if torch.cuda.is_available():
        gpu_mem_after = get_gpu_memory_mb()

    # 文件大小
    pkl_size = get_file_size_mb(pkl_path)
    origin_pkl_size = get_file_size_mb(origin_pkl_path) if origin_pkl_path else 0

    # 帧数统计
    num_frames = len(result_dict.get('original_result', []))
    avg_time_per_frame = total_time / num_frames if num_frames > 0 else 0

    print(f"\n{'='*60}")
    print(f"处理完成! 统计信息如下:")
    print(f"{'='*60}")
    print(f"\n[耗时统计]")
    print(f"  总耗时:           {total_time:.2f}s")
    print(f"  - Pipeline 初始化: {init_time:.2f}s ({init_time/total_time*100:.1f}%)")
    print(f"  - 视频重建:        {reconstruct_time:.2f}s ({reconstruct_time/total_time*100:.1f}%)")
    print(f"  - 结果转换与保存:  {convert_time:.2f}s ({convert_time/total_time*100:.1f}%)")
    print(f"\n[平均耗时]")
    print(f"  每帧平均耗时:     {avg_time_per_frame:.4f}s/frame ({avg_time_per_frame*1000:.2f}ms/frame)")

    print(f"\n[内存统计]")
    print(f"  初始内存:         {mem_before:.1f} MB")
    print(f"  初始化后内存:     {init_mem:.1f} MB (增量: +{init_mem - mem_before:.1f} MB)")
    print(f"  最终内存:         {mem_after:.1f} MB (总增量: +{mem_after - mem_before:.1f} MB)")
    if torch.cuda.is_available():
        print(f"  初始 GPU 内存:     {gpu_mem_before:.1f} MB")
        print(f"  最终 GPU 内存:     {gpu_mem_after:.1f} MB (总增量: +{gpu_mem_after - gpu_mem_before:.1f} MB)")
        print(f"  GPU 显存峰值:     {torch.cuda.max_memory_allocated()/1024/1024:.1f} MB")

    print(f"\n[文件大小]")
    print(f"  结果 pkl:         {pkl_size:.2f} MB")
    if origin_pkl_path:
        print(f"  原始 pkl:         {origin_pkl_size:.2f} MB")
        print(f"  总计:             {pkl_size + origin_pkl_size:.2f} MB")

    print(f"\n[输出信息]")
    print(f"  结果帧数:         {num_frames}")
    print(f"  输出路径:         {pkl_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
