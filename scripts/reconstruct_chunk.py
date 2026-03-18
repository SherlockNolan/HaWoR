"""
reconstruct.py
--------------
核心重建类，从 demo.py 提取和改编。

支持两种使用方式：
  1. Pipeline 方式（外部调用）：
       from reconstruct import HaWoRReconstructor
       rec = HaWoRReconstructor(checkpoint=..., infiller_weight=...)
       result = rec.run(video_path, output_dir=..., rendering=True)

  2. 命令行方式：
       python reconstruct.py --video_path example/video_0.mp4 --output_dir ./results
       python reconstruct.py --video_path example/factory001_worker001_00000.mp4 --output_dir ./results --rendering --vis_mode cam

注意由于服务器没有显示器，而渲染依赖于显示器（受制于python包aitviewer），如果要渲染必须如下调整
export MGLW_WINDOW=moderngl_window.context.headless.Window
export PYOPENGL_PLATFORM=egl
xvfb-run -a python scripts/reconstruct_chunk.py --video_path example/video_0.mp4 --output_dir ./results --rendering --vis_mode cam
xvfb-run -a python scripts/reconstruct.py --video_path /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0.mp4 --output_dir ./results --rendering --vis_mode cam


export MGLW_WINDOW=moderngl_window.context.headless.Window
export PYOPENGL_PLATFORM=egl
xvfb-run -a python scripts/reconstruct_chunk.py --video_path example/video_0.mp4 --output_dir ./results --rendering --vis_mode cam --chunk_size=60 --overlap_frames=10

export MGLW_WINDOW=moderngl_window.context.headless.Window
export PYOPENGL_PLATFORM=egl
xvfb-run -a python scripts/reconstruct_chunk.py --video_path example/factory001_worker001_00000.mp4 --output_dir ./results --rendering --vis_mode cam --chunk_size=500 --overlap_frames=10

"""
import argparse
import numpy as np
from glob import glob

from natsort import natsorted
import subprocess
import argparse
import sys
import os
import cv2
import torch
# 将当前脚本的父目录（即根目录）加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_infiller_plain, hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.pipeline.HaWoRPipeline import HaWoRPipeline, HaWoRConfig
from lib.pipeline.HaWoRPipelineChunk import HaWoRPipelineChunk
from lib.pipeline.tools import detect_track


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HaWoR — Hand-in-World Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default='results',
        help="输出目录，默认在./results下面",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="HaWoR 模型权重路径",
    )
    parser.add_argument(
        "--infiller_weight", type=str,
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Infiller 模型权重路径",
    )
    parser.add_argument(
        "--image_focal", type=float, default=None,
        help="相机焦距（像素），不提供则自动估计",
    )
    parser.add_argument(
        "--rendering", action="store_true",
        help="开启渲染并合成 mp4（默认关闭）",
    )
    parser.add_argument(
        "--vis_mode", type=str, default="world",
        choices=["world", "cam"],
        help="渲染视角：world（世界坐标）或 cam（相机坐标）",
    )
    parser.add_argument(
        "--progress-bar", action="store_true",
        help="显示重建各阶段的总体进度条（平均分配到 4 或 5 个阶段）",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1000,
        help="分段chunk大小",
    )
    parser.add_argument(
        "--overlap_frames", type=int, default=120,
        help="分段chunk大小",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    
    haworConfig = HaWoRConfig(
        checkpoint      = args.checkpoint,
        infiller_weight = args.infiller_weight,
        verbose         = True,
    )
    reconstructor = HaWoRPipelineChunk(haworConfig)
    
    import time

    start_time = time.perf_counter()

    # 这里放置你的代码

    result = reconstructor.reconstruct(
        video_path = args.video_path,
        output_dir = args.output_dir,
        rendering  = args.rendering,
        vis_mode   = args.vis_mode,
        image_focal= args.image_focal,
        use_progress_bar = args.progress_bar,
        chunk_size = args.chunk_size,
        overlap_frames = args.overlap_frames,
    )
    
    end_time = time.perf_counter()

    print("\n=== Reconstruction complete ===")
    print(f"消耗时间（不含初始化）: {end_time - start_time:.6f} 秒")
    print(f"  seq_folder    : {result['seq_folder']}") # 用于rendering的时候extract_frames的临时图片生成目录
    print(f"  img_focal     : {result['img_focal']}")
    print(f"  pred_trans    : {result['pred_trans'].shape}")
    print(f"  pred_rot      : {result['pred_rot'].shape}")
    print(f"  pred_hand_pose: {result['pred_hand_pose'].shape}")
    print(f"  pred_betas    : {result['pred_betas'].shape}")
    if result["rendered_video"]:
        print(f"  rendered_video: {result['rendered_video']}")


if __name__ == "__main__":
    main()
