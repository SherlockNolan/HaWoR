"""
HaWoR Pipeline 冒烟测试脚本
支持 --device 参数指定运行设备（cpu/cuda/cuda:0 等）

使用示例:
    # CPU 测试
    python tests/test_pipeline.py --device=cpu --test

    # 单 GPU 测试
    python tests/test_pipeline.py --device=cuda:0 --video-path=/path/to/video.mp4

    # 多 GPU 测试（指定设备列表）
    python tests/test_pipeline.py --device=cuda:0,cuda:1 --num-workers=2 --test
"""

import os
import sys
import glob
import pickle
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import multiprocessing
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig
from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints

# 测试数据路径
TEST_VIDEO_DIR = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/DATASET/Self"
TEST_OUTPUT_DIR = "./test_results"


def find_test_videos():
    """查找测试视频"""
    if not os.path.isdir(TEST_VIDEO_DIR):
        return []
    pattern = os.path.join(TEST_VIDEO_DIR, "**", "*.mp4")
    videos = glob.glob(pattern, recursive=True)
    videos.sort()
    return videos[:3]  # 最多取3个


def process_single_video(video_path: str, device: str, output_dir: str, tmp_dir: str = None, save_origin: bool = False):
    """处理单个视频，返回结果"""
    print(f"\n{'='*60}")
    print(f"处理视频: {video_path}")
    print(f"使用设备: {device}")
    print(f"{'='*60}")

    # 构建输出路径
    video_name = os.path.basename(video_path).replace(".mp4", "")
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    pkl_path = os.path.join(video_output_dir, f"{video_name}_hawor.pkl")

    # 检查是否已存在
    if os.path.exists(pkl_path):
        print(f"PKL 已存在，跳过: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data

    # 创建 pipeline
    cfg = HaWoRConfig(
        verbose=True,
        device=device,
        tmp_dir=tmp_dir,
    )
    pipe = HaWoRPipelineOpt(cfg)

    # 计时
    start_time = time.time()

    # 执行重建
    result_dict_origin = pipe.reconstruct(
        video_path,
        output_dir=video_output_dir,
        image_focal=1031,
        start_idx=0,
        end_idx=2000,  # 限制处理帧数用于测试
    )

    # 转换为 keypoints 格式
    result_dict = {
        "original_result": convert_hawor_to_keypoints(result_dict_origin, video_path, use_smoothed=False),
        "smoothed_result": convert_hawor_to_keypoints(result_dict_origin, video_path, use_smoothed=True) if result_dict_origin.get("smoothed_result") is not None else None,
    }

    # 保存结果
    with open(pkl_path, "wb") as f:
        pickle.dump(result_dict, f)

    if save_origin:
        origin_pkl_path = pkl_path.replace(".pkl", "_origin_dict.pkl")
        with open(origin_pkl_path, "wb") as f:
            pickle.dump(result_dict_origin, f)

    elapsed = time.time() - start_time
    print(f"\n完成! 耗时: {elapsed:.1f}s")
    print(f"输出目录: {video_output_dir}")

    return result_dict


def main():
    parser = argparse.ArgumentParser(description="HaWoR Pipeline 冒烟测试")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="运行设备: cpu, cuda, cuda:0, cuda:1 等 (默认: cuda:0)")
    parser.add_argument("--video-path", type=str, default=None,
                        help="指定单个视频路径")
    parser.add_argument("--test", action="store_true",
                        help="使用测试视频目录")
    parser.add_argument("--output", type=str, default=TEST_OUTPUT_DIR,
                        help="输出目录 (默认: ./test_results)")
    parser.add_argument("--tmp-dir", type=str, default=None,
                        help="临时文件目录")
    parser.add_argument("--save-origin", action="store_true",
                        help="同时保存原始 hawor 结果 dict")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="并行处理视频数 (仅多进程模式)")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"HaWoR Pipeline 冒烟测试")
    print(f"{'#'*60}")
    print(f"设备: {args.device}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - cuda:{i} = {torch.cuda.get_device_name(i)}")

    # 确定要处理的视频
    video_list = []
    if args.video_path:
        video_list = [args.video_path]
    elif args.test:
        video_list = find_test_videos()
        if not video_list:
            print(f"\n错误: 测试视频目录为空或不存在: {TEST_VIDEO_DIR}")
            print("使用 --video-path 指定视频路径")
            return
        print(f"\n找到 {len(video_list)} 个测试视频")
    else:
        print("\n错误: 请指定 --video-path 或使用 --test")
        return

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    print(f"输出目录: {args.output}")

    # 处理视频
    if args.num_workers <= 1:
        # 单进程模式
        for video_path in video_list:
            try:
                result = process_single_video(
                    video_path=video_path,
                    device=args.device,
                    output_dir=args.output,
                    tmp_dir=args.tmp_dir,
                    save_origin=args.save_origin,
                )
                print(f"\n结果包含 {len(result.get('original_result', []))} 帧")
            except Exception as e:
                print(f"\n处理失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        # 多进程模式（复用 reconstruct_egocentric_opt.py 的逻辑）
        print(f"\n多进程模式: {args.num_workers} workers")
        from scripts.reconstruct_egocentric_opt import process_multi_workers

        extra_args = {
            "dataset_root": os.path.dirname(video_list[0]) if len(video_list) == 1 else "",
            "output_root": args.output,
            "save_origin": args.save_origin,
            "frame_start_idx": 0,
            "frame_end_idx": 2000,
            "tmp_dir": args.tmp_dir,
        }

        process_multi_workers(args, torch.float16, video_list, extra_args)

    print(f"\n{'#'*60}")
    print("测试完成!")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
