"""
HaWoRPipeline 性能测试脚本

测试不同 end_idx (1k, 10k, 100k, 1m) 下的内存占用和运行时间。

Usage:
    python scripts/test_pipeline_performance.py --video_path /inspire/dataset/egocentric-10k/v20251211/factory_001/workers/worker_001/factory001_worker001_00000.mp4 \
        --output_dir ./results --end_indices="1000"
"""

import argparse
import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path

import psutil
import torch
import numpy as np


# 路径配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline.HaWoRPipeline import HaWoRPipeline, HaWoRConfig
from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt

def get_memory_usage_mb():
    """获取当前进程内存占用 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_mb():
    """获取当前 GPU 显存占用 (MB)，如果可用"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def format_memory(size_bytes):
    """将字节数格式化为可读字符串"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def get_dict_size_mb(d) -> float:
    """
    计算 dict 或 list 中所有张量/数组的总大小（MB）
    支持 torch.Tensor, np.ndarray, joblib.MappedSeries 等
    """
    total_bytes = 0
    items = d.values() if isinstance(d, dict) else d
    for v in items:
        if isinstance(v, torch.Tensor):
            total_bytes += v.element_size() * v.nelement()
        elif isinstance(v, np.ndarray):
            total_bytes += v.nbytes
        elif isinstance(v, (np.integer, np.floating)):
            total_bytes += v.itemsize
        elif hasattr(v, "nbytes"):  # 其他类似数组对象
            total_bytes += v.nbytes
        elif isinstance(v, dict):
            total_bytes += get_dict_size_mb(v) * 1024 * 1024  # 递归，已是 MB，转回 bytes
        elif isinstance(v, list):
            total_bytes += get_dict_size_mb(v) * 1024 * 1024  # 递归处理列表
    return total_bytes / 1024 / 1024


def init_pipeline(device: str):
    """
    初始化 pipeline（只调用一次）

    Returns:
        tuple: (pipe, init_mem_mb, init_gpu_mem_mb)
    """
    print(f"\n{'='*60}")
    print(f"  [Init] 初始化 Pipeline...")
    print(f"{'='*60}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    mem_before_init = get_memory_usage_mb()
    gpu_mem_before_init = get_gpu_memory_mb()
    time_start = time.time()

    cfg = HaWoRConfig(
        verbose=False,
        device=device,
        smooth_hands=True,
        smooth_camera=True,
    )
    pipe = HaWoRPipelineOpt(cfg)

    init_time = time.time() - time_start
    _, process_mem_after_init = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    process_mem_after_init = process_mem_after_init / 1024 / 1024
    gpu_mem_after_init = get_gpu_memory_mb()

    print(f"[Init] 完成!")
    print(f"  - 初始化耗时:   {init_time:.2f} 秒")
    print(f"  - 进程内存:     {mem_before_init:.2f} MB -> {process_mem_after_init:.2f} MB (增量: {process_mem_after_init - mem_before_init:.2f} MB)")
    print(f"  - GPU 显存:     {gpu_mem_before_init:.2f} MB -> {gpu_mem_after_init:.2f} MB")

    return pipe, {
        "init_time_sec": init_time,
        "mem_before_init_mb": mem_before_init,
        "mem_after_init_mb": process_mem_after_init,
        "gpu_mem_before_init_mb": gpu_mem_before_init,
        "gpu_mem_after_init_mb": gpu_mem_after_init,
    }


def warmup_pipeline(pipe, video_path: str, output_dir: str, image_focal: float):
    """
    预热 pipeline - 只跑第10帧

    Returns:
        dict: 预热阶段的内存和时间信息
    """
    print(f"\n{'='*60}")
    print(f"  [Warmup] 预热 Pipeline...")
    print(f"{'='*60}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warmup_output_dir = os.path.join(output_dir, "warmup")
    os.makedirs(warmup_output_dir, exist_ok=True)

    tracemalloc.start()
    mem_before_warmup = get_memory_usage_mb()
    gpu_mem_before_warmup = get_gpu_memory_mb()
    time_start = time.time()

    _ = pipe.reconstruct(
        video_path,
        output_dir=warmup_output_dir,
        start_idx=0,
        end_idx=10,  # 只处理第10帧
        image_focal=image_focal,
        use_progress_bar=False,
    )

    warmup_time = time.time() - time_start
    _, mem_after_warmup = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_warmup = mem_after_warmup / 1024 / 1024
    gpu_mem_after_warmup = get_gpu_memory_mb()

    print(f"[Warmup] 完成!")
    print(f"  - 预热耗时:     {warmup_time:.2f} 秒")
    print(f"  - 进程内存:     {mem_before_warmup:.2f} MB -> {mem_after_warmup:.2f} MB (增量: {mem_after_warmup - mem_before_warmup:.2f} MB)")
    print(f"  - GPU 显存:     {gpu_mem_before_warmup:.2f} MB -> {gpu_mem_after_warmup:.2f} MB")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "warmup_time_sec": warmup_time,
        "mem_before_warmup_mb": mem_before_warmup,
        "mem_after_warmup_mb": mem_after_warmup,
        "gpu_mem_before_warmup_mb": gpu_mem_before_warmup,
        "gpu_mem_after_warmup_mb": gpu_mem_after_warmup,
    }


def run_reconstruct(
    pipe,
    video_path: str,
    output_dir: str,
    end_idx: int,
    image_focal: float = 1031.0,
):
    """
    运行单次 reconstruct 并记录内存

    Args:
        pipe: 已初始化的 pipeline
        video_path: 输入视频路径
        output_dir: 输出目录
        end_idx: 结束帧索引
        image_focal: 焦距

    Returns:
        dict: 包含各项指标的字典
    """
    print(f"\n{'='*60}")
    print(f"  [Reconstruct] end_idx={end_idx:,}")
    print(f"{'='*60}")

    test_output_dir = os.path.join(output_dir, f"test_end{end_idx}")
    os.makedirs(test_output_dir, exist_ok=True)

    # 重置 GPU 内存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    mem_start = get_memory_usage_mb()
    gpu_mem_start = get_gpu_memory_mb()
    time_start = time.time()

    # 执行重建
    result_dict_origin = pipe.reconstruct(
        video_path,
        output_dir=test_output_dir,
        start_idx=0,
        end_idx=end_idx,
        image_focal=image_focal,
        use_progress_bar=True,
    )

    elapsed_time = time.time() - time_start
    process_mem_peak, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    process_mem_peak = process_mem_peak / 1024 / 1024
    process_mem_end = get_memory_usage_mb()
    gpu_mem_peak = get_gpu_memory_mb() if torch.cuda.is_available() else 0.0

    print(f"\n[Result] end_idx={end_idx:,}")
    print(f"  - 运行时间:     {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"  - 进程内存峰值: {process_mem_peak:.2f} MB")
    print(f"  - 进程内存结束: {process_mem_end:.2f} MB")
    print(f"  - GPU 显存峰值: {gpu_mem_peak:.2f} MB")

    # 转换结果为 keypoints
    print(f"[Convert] Converting to keypoints...")
    if globals().get("convert_hawor_to_keypoints") is None:
        from lib.pipeline.HaworToKeypointsAdapter import (
            convert_hawor_to_keypoints as _convert_fn,
        )
        globals()["convert_hawor_to_keypoints"] = _convert_fn
    _convert = globals()["convert_hawor_to_keypoints"]

    result_dict = dict()
    result_dict["original_result"] = _convert(
        result_dict_origin, video_path, use_smoothed=False
    )
    result_dict["smoothed_result"] = _convert(
        result_dict_origin, video_path, use_smoothed=True
    )

    # 记录 result 字典大小
    result_origin_size_mb = get_dict_size_mb(result_dict["original_result"])
    result_converted_size_mb = get_dict_size_mb(result_dict["smoothed_result"])

    print(f"  - 结果字典大小 (original):   {result_origin_size_mb:.2f} MB")
    print(f"  - 结果字典大小 (smoothed): {result_converted_size_mb:.2f} MB")

    # 不清理 pipe，只清理结果
    del result_dict_origin
    del result_dict
    gc.collect()

    return {
        "end_idx": end_idx,
        "elapsed_time_sec": elapsed_time,
        "process_mem_peak_mb": process_mem_peak,
        "process_mem_end_mb": process_mem_end,
        "gpu_mem_peak_mb": gpu_mem_peak,
        "result_origin_size_mb": result_origin_size_mb,
        "result_converted_size_mb": result_converted_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="HaWoRPipeline 性能测试")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="输出目录",
    )
    parser.add_argument(
        "--image_focal",
        type=float,
        default=1031.0,
        help="相机焦距",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备",
    )
    parser.add_argument(
        "--end_indices",
        type=str,
        default="3000, 10000, 50000",
        help="逗号分隔的 end_idx 值列表，如: 1000,10000,100000,1000000",
    )
    args = parser.parse_args()

    # 解析 end_idx 列表
    end_indices = [int(x.strip()) for x in args.end_indices.split(",")]

    print(f"{'='*60}")
    print(f"  HaWoRPipeline 性能测试")
    print(f"{'='*60}")
    print(f"  视频路径:   {args.video_path}")
    print(f"  输出目录:   {args.output_dir}")
    print(f"  焦距:       {args.image_focal}")
    print(f"  设备:       {args.device}")
    print(f"  测试索引:   {end_indices}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 pipeline（只一次）
    pipe, init_info = init_pipeline(device=args.device)

    # 预热（只一次）
    warmup_info = warmup_pipeline(
        pipe, video_path=args.video_path, output_dir=args.output_dir, image_focal=args.image_focal
    )

    # 复用 pipeline，测试不同 end_idx
    reconstruct_results: list[dict] = []
    for end_idx in end_indices:
        result = run_reconstruct(
            pipe,
            video_path=args.video_path,
            output_dir=args.output_dir,
            end_idx=end_idx,
            image_focal=args.image_focal,
        )
        reconstruct_results.append(result)

    results: dict[str, dict | list[dict]] = {
        "init": init_info,
        "warmup": warmup_info,
        "reconstruct": reconstruct_results,
    }

    # 清理
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 打印汇总表格
    print(f"\n{'='*60}")
    print(f"  汇总结果")
    print(f"{'='*60}")

    # 初始化和预热信息
    print(f"\n[Init & Warmup]")
    print(f"  初始化: {init_info['init_time_sec']:.2f}秒, 内存增量: {init_info['mem_after_init_mb'] - init_info['mem_before_init_mb']:.2f}MB")
    print(f"  预热:   {warmup_info['warmup_time_sec']:.2f}秒, 内存增量: {warmup_info['mem_after_warmup_mb'] - warmup_info['mem_before_warmup_mb']:.2f}MB")

    print(
        f"\n{'end_idx':>12} | {'时间(秒)':>10} | {'时间(分)':>8} | "
        f"{'进程内存MB':>12} | {'GPU显存MB':>10} | {'结果原始MB':>10} | {'结果转换MB':>10}"
    )
    print("-" * 90)
    for r in results["reconstruct"]:
        elapsed_sec = float(r["elapsed_time_sec"])
        print(
            f"{r['end_idx']:>12,} | "
            f"{elapsed_sec:>10.2f} | "
            f"{elapsed_sec/60:>8.2f} | "
            f"{float(r['process_mem_peak_mb']):>12.2f} | "
            f"{float(r['gpu_mem_peak_mb']):>10.2f} | "
            f"{float(r['result_origin_size_mb']):>10.2f} | "
            f"{float(r['result_converted_size_mb']):>10.2f}"
        )

    # 保存结果到文件
    import json

    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")


if __name__ == "__main__":
    main()
