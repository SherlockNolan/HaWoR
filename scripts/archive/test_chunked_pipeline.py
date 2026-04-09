"""
测试 HaWoRPipelineChunk 分段处理功能

使用示例:
    # 单视频测试
    python scripts/test_chunked_pipeline.py --video-path example/video_0.mp4

    # 使用小 chunk 测试
    python scripts/test_chunked_pipeline.py --video-path test_video.mp4 --chunk-size 500

    # 对比标准模式和分段模式
    python scripts/test_chunked_pipeline.py --video-path example/factory001_worker001_00000.mp4 --compare --chunk-size=400
"""

import argparse
import os
import sys
import torch
import psutil
import time

# 添加根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline.HaWoRPipeline import HaWoRPipeline, HaWoRConfig
from lib.pipeline.HaWoRPipelineChunk import HaWoRPipelineChunk


def get_memory_usage():
    """获取当前进程的内存使用（GB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def test_chunked_pipeline(video_path, chunk_size=1000, overlap_frames=120, compare=False):
    """
    测试分段处理 pipeline

    Args:
        video_path: 视频路径
        chunk_size: 分块大小
        overlap_frames: 重叠帧数
        compare: 是否对比标准模式和分段模式
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"测试 HaWoRPipeline 分段处理功能")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"视频: {video_path}")
    print(f"分块大小: {chunk_size}")
    print(f"重叠帧数: {overlap_frames}")
    print(f"{'='*60}\n")

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return

    # 测试分段处理模式
    print("\n[测试] HaWoRPipelineChunk (分段处理模式)")
    print("-" * 60)
    cfg = HaWoRConfig(verbose=True, device=device)
    pipeline_chunked = HaWoRPipelineChunk(cfg)

    mem_before = get_memory_usage()
    print(f"初始内存: {mem_before:.2f} GB")

    start_time = time.time()
    result_chunked = pipeline_chunked.reconstruct(
        video_path,
        output_dir="./results_chunked",
        chunk_size=chunk_size,
        overlap_frames=overlap_frames
    )
    chunked_time = time.time() - start_time
    mem_after = get_memory_usage()

    print(f"\n分段处理完成:")
    print(f"  用时: {chunked_time:.2f} 秒")
    print(f"  峰值内存增量: {mem_after - mem_before:.2f} GB")
    print(f"  最终内存: {mem_after:.2f} GB")

    # 验证结果
    pred_trans = result_chunked["pred_trans"]
    print(f"\n结果验证:")
    print(f"  pred_trans shape: {pred_trans.shape}")
    print(f"  总帧数: {pred_trans.shape[1]}")

    # 如果需要对比，运行标准模式
    if compare:
        print("\n" + "="*60)
        print("[对比] HaWoRPipeline (标准模式)")
        print("-" * 60)
        pipeline_standard = HaWoRPipeline(cfg)

        # 清理内存
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        mem_before_std = get_memory_usage()
        print(f"初始内存: {mem_before_std:.2f} GB")

        start_time_std = time.time()
        result_standard = pipeline_standard.reconstruct(
            video_path,
            output_dir="./results_standard"
        )
        standard_time = time.time() - start_time_std
        mem_after_std = get_memory_usage()

        print(f"\n标准模式完成:")
        print(f"  用时: {standard_time:.2f} 秒")
        print(f"  峰值内存增量: {mem_after_std - mem_before_std:.2f} GB")
        print(f"  最终内存: {mem_after_std:.2f} GB")

        # 对比结果
        print("\n" + "="*60)
        print("[对比结果]")
        print("-" * 60)
        print(f"  用时对比: 分段 {chunked_time:.2f}s vs 标准 {standard_time:.2f}s")
        print(f"  内存对比: 分段 {mem_after - mem_before:.2f}GB vs 标准 {mem_after_std - mem_before_std:.2f}GB")

        # 验证结果一致性
        print(f"\n  结果一致性检查:")
        pred_trans_std = result_standard["pred_trans"]
        pred_trans_chk = result_chunked["pred_trans"]

        if pred_trans_std.shape == pred_trans_chk.shape:
            diff = torch.abs(pred_trans_std - pred_trans_chk).max().item()
            print(f"    最大差异: {diff:.6f}")
            print(f"    结果形状一致: {pred_trans_std.shape}")
        else:
            print(f"    结果形状不同: 标准 {pred_trans_std.shape} vs 分段 {pred_trans_chk.shape}")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="测试 HaWoRPipeline 分段处理功能")
    parser.add_argument("--video-path", type=str, required=True, help="测试视频路径")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块大小（默认 1000）")
    parser.add_argument("--overlap-frames", type=int, default=120, help="重叠帧数（默认 120）")
    parser.add_argument("--compare", action="store_true", help="是否对比标准模式和分段模式")

    args = parser.parse_args()

    test_chunked_pipeline(
        video_path=args.video_path,
        chunk_size=args.chunk_size,
        overlap_frames=args.overlap_frames,
        compare=args.compare
    )


if __name__ == "__main__":
    main()
