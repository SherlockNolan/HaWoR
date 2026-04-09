"""
HaWoRPipeline vs HaWoRPipelineOpt 内存与性能对比测试脚本

测试两种 Pipeline 处理同一视频时的：
1. 内存峰值占用 (GPU & CPU)
2. 运行时间
3. 输出结果一致性

使用示例:
    # 运行对比测试
    python scripts/test_pipeline_opt.py --video_path /path/to/video.mp4

    # 仅测试 HaWoROpt
    python scripts/test_pipeline_opt.py --video_path /path/to/video.mp4 --mode opt

    # 仅测试 HaWoRPipeline (原始)
    python scripts/test_pipeline_opt.py --video_path /path/to/video.mp4 --mode original
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.pipeline.HaWoRPipeline import HaWoRPipeline, HaWoRConfig
from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoROptConfig


def get_memory_usage():
    """获取当前内存使用情况 (GB)"""
    import psutil

    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024**3  # GB

    gpu_mem = 0.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB

    return cpu_mem, gpu_mem


def reset_memory_stats():
    """重置内存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def compare_results(result_orig, result_opt, rtol=1e-3, atol=1e-2):
    """
    比较两个结果字典的差异。

    Returns:
        dict: 包含比较结果的字典
    """
    keys_to_compare = [
        'pred_trans', 'pred_rot', 'pred_hand_pose', 'pred_betas', 'pred_valid'
    ]

    comparison = {}

    for key in keys_to_compare:
        if key not in result_orig or key not in result_opt:
            comparison[key] = {"status": "missing", "message": "Key not found in both results"}
            continue

        orig = result_orig[key]
        opt = result_opt[key]

        # 转换为 tensor
        if not isinstance(orig, torch.Tensor):
            orig = torch.from_numpy(np.array(orig))
        if not isinstance(opt, torch.Tensor):
            opt = torch.from_numpy(np.array(opt))

        # 检查形状
        if orig.shape != opt.shape:
            comparison[key] = {
                "status": "shape_mismatch",
                "orig_shape": orig.shape,
                "opt_shape": opt.shape
            }
            continue

        # 检查数值差异
        diff = torch.abs(orig - opt)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        is_close = torch.allclose(orig, opt, rtol=rtol, atol=atol)

        comparison[key] = {
            "status": "ok" if is_close else "diff",
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "shapes": list(orig.shape)
        }

    return comparison


def test_pipeline(pipeline_class, config_class, video_path, output_dir, desc=""):
    """
    测试单个 pipeline 的内存和时间性能。

    Returns:
        dict: 包含测试结果的字典
    """
    print(f"\n{'='*60}")
    print(f"Testing: {desc}")
    print(f"{'='*60}")

    # 重置内存统计
    reset_memory_stats()

    # 创建 pipeline
    config = config_class(
        verbose=True,
        smooth_hands=False,  # 简化测试
        smooth_camera=False,
    )
    pipeline = pipeline_class(config)

    # 记录开始时间
    start_time = time.time()

    # 运行重建
    try:
        result = pipeline.reconstruct(
            video_path=video_path,
            output_dir=output_dir,
            rendering=False,
            use_progress_bar=True
        )

        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 获取内存使用
        cpu_mem, gpu_mem = get_memory_usage()

        # 获取 GPU 峰值内存
        if torch.cuda.is_available():
            gpu_peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        else:
            gpu_peak_mem = 0.0

        print(f"\n[{desc}] 完成!")
        print(f"  运行时间: {elapsed_time:.2f} 秒")
        print(f"  CPU 内存: {cpu_mem:.2f} GB")
        print(f"  GPU 峰值内存: {gpu_peak_mem:.2f} GB")

        return {
            "success": True,
            "elapsed_time": elapsed_time,
            "cpu_mem_gb": cpu_mem,
            "gpu_peak_mem_gb": gpu_peak_mem,
            "result": result,
            "error": None
        }

    except Exception as e:
        end_time = time.time()
        import traceback
        print(f"\n[{desc}] 失败!")
        print(f"  错误: {str(e)}")
        print(traceback.format_exc())

        return {
            "success": False,
            "elapsed_time": end_time - start_time,
            "cpu_mem_gb": get_memory_usage()[0],
            "gpu_peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "result": None,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='HaWoRPipeline vs HaWoRPipelineOpt 对比测试')
    parser.add_argument('--video_path', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='输出目录')
    parser.add_argument('--mode', type=str, default='both', choices=['both', 'original', 'opt'],
                        help='测试模式: both=对比测试, original=仅原始, opt=仅优化版')
    parser.add_argument('--chunk_size_slam', type=int, default=500,
                        help='SLAM 分块大小 (HaWoROpt)')
    parser.add_argument('--chunk_size_mano', type=int, default=300,
                        help='MANO 网格构建分块大小 (HaWoROpt)')

    args = parser.parse_args()

    # 检查视频文件
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在: {args.video_path}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'opt'), exist_ok=True)

    results = {}

    # 测试原始版本
    if args.mode in ['both', 'original']:
        reset_memory_stats()
        results['original'] = test_pipeline(
            HaWoRPipeline, HaWoRConfig,
            args.video_path,
            os.path.join(args.output_dir, 'original'),
            desc="HaWoRPipeline (原始)"
        )
        # 清理
        torch.cuda.empty_cache()
        gc.collect()

    # 测试优化版本
    if args.mode in ['both', 'opt']:
        reset_memory_stats()
        opt_config = HaWoROptConfig(
            verbose=True,
            smooth_hands=False,
            smooth_camera=False,
            chunk_size_slam=args.chunk_size_slam,
            chunk_size_mano=args.chunk_size_mano,
        )
        results['opt'] = test_pipeline(
            HaWoRPipelineOpt, HaWoROptConfig,
            args.video_path,
            os.path.join(args.output_dir, 'opt'),
            desc="HaWoRPipelineOpt (优化)"
        )
        # 清理
        torch.cuda.empty_cache()
        gc.collect()

    # 对比结果
    if args.mode == 'both' and results.get('original', {}).get('success') and results.get('opt', {}).get('success'):
        print("\n" + "="*60)
        print("对比分析")
        print("="*60)

        orig = results['original']
        opt = results['opt']

        # 性能对比
        print(f"\n性能对比:")
        print(f"  {'指标':<20} {'原始':<15} {'优化':<15} {'提升':<15}")
        print(f"  {'-'*60}")

        time_ratio = (orig['elapsed_time'] - opt['elapsed_time']) / orig['elapsed_time'] * 100
        print(f"  {'运行时间':<20} {orig['elapsed_time']:<15.2f} {opt['elapsed_time']:<15.2f} {time_ratio:>+10.1f}%")

        mem_ratio = (orig['gpu_peak_mem_gb'] - opt['gpu_peak_mem_gb']) / orig['gpu_peak_mem_gb'] * 100
        print(f"  {'GPU 峰值内存 (GB)':<20} {orig['gpu_peak_mem_gb']:<15.2f} {opt['gpu_peak_mem_gb']:<15.2f} {mem_ratio:>+10.1f}%")

        # 结果一致性检查
        print(f"\n结果一致性检查:")
        comparison = compare_results(orig['result'], opt['result'])

        all_ok = True
        for key, comp in comparison.items():
            status_icon = "✓" if comp["status"] == "ok" else "✗"
            if comp["status"] != "ok":
                all_ok = False
            print(f"  {status_icon} {key}: {comp}")

        if all_ok:
            print(f"\n结果一致性检查通过!")
        else:
            print(f"\n警告: 结果存在差异，请检查输出!")

    elif args.mode == 'both':
        print("\n由于部分测试失败，无法进行对比分析")

    # 打印总结
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == '__main__':
    main()
