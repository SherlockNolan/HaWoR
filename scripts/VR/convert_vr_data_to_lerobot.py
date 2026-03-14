"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format. for Rhos_VR_Egohands dataset

Example usage: uv run examples/xtrainer_real/convert_vr_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name> --task <task description>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
from PIL import Image
import numpy as np
import torch
import tqdm
import tyro
import json
import cv2
import re

CAMERA_NAMES = [
    'cam_high',
    # 'image_wrist_left_up',
    'cam_left_wrist',
    # 'image_wrist_right_up',
    'cam_right_wrist'
]
CAMERA_MAP = {
    'remote_0': 'cam_high',
    'leftdown':'cam_left_wrist',
    'rightdown':'cam_right_wrist'
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

def get_grouped_files(input_dir:Path, task_name=None) -> list:
    """扫描文件夹并将文件按时间戳(episode)分组
    
    返回格式:
    ```
    [
        {
            "task_name": "pull_drawer",
            "timestamp": "2023-01-01T12-00-00",
            "metadata": Path(...),
            "keypoints": Path(...),
            "videos": [
                {"id": "remote_0", "path": Path(...)},
                {"id": "local_0", "path": Path(...)},
                ...
            ]
        },
        ...
    ]
    """
    episodes = {}
    if task_name == None:
        task_name = input_dir.name
    
    print(f"Scanning {input_dir} for recordings...")
    for file_path in input_dir.glob("recording_*"):
        FILE_PATTERN = re.compile(r"recording_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_(.+)\.(.+)")
        match = FILE_PATTERN.match(file_path.name)
        if match:
            timestamp = match.group(1)
            file_type = match.group(2) # metadata, keypoints, leftup, remote_0 etc.
            ext = match.group(3)
            
            if timestamp not in episodes:
                episodes[timestamp] = {
                    "metadata": None,
                    "keypoints": None,
                    "videos": []
                }
            
            if file_type == "metadata":
                episodes[timestamp]["metadata"] = file_path
            elif file_type == "keypoints":
                episodes[timestamp]["keypoints"] = file_path
            else:
                # 视频文件，不要webm只要mp4
                if ext == "mp4":
                    episodes[timestamp]["videos"].append({
                        "id": file_type,
                        "path": file_path
                    })
    
    # 过滤掉不完整的episode，并改格式
    valid_episodes = []
    for ts, data in episodes.items():
        if data["metadata"] and data["videos"] and data['keypoints']:
            data['timestamp'] = ts
            data['task_name'] = task_name
            valid_episodes.append(data)
        else:
            print(f"Skipping incomplete episode {task_name}: {ts}")

    return valid_episodes

def get_interpolated_action(kp_timestamps, keypoints, current_time: float) -> np.ndarray:
    """使用线性插值获取当前时间点的关键点动作"""
    if len(kp_timestamps) == 0:
        return None
    # 对每一维度的关键点，使用 numpy 进行线性插值
    interpolated_action = np.zeros(keypoints.shape[1]) # 关键点每一维度
    for i in range(keypoints.shape[1]):
        interpolated_action[i] = np.interp(
            current_time, 
            kp_timestamps, 
            keypoints[:, i],
            left=keypoints[0, i],  # 时间小于0时取第一帧
            right=keypoints[-1, i] # 时间超出时取最后一帧
        )
    return interpolated_action


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # 左右手各6个点，每个点3维 (x,y,z)，共 36 维
    # motors = [f"kp_{i}_{axis}" for i in range(12) for axis in ['x', 'y', 'z']]
    # HACK:取左右手各前三个点，剩下的补 0 至 32 维
    motors = [f"kp_{i}_{axis}" for i in [0,1,2,6,7,8] for axis in ['x', 'y', 'z']] + [0.0] * 14
    cameras = CAMERA_NAMES

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=25,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_episode_data(sample: dict) -> list[dict]:
    """
    返回该 episode 的所有 frame 数据
    """
    task_name = sample['task_name']
    
    with open(sample['keypoints'], 'r') as f:
        keypoints_data = json.load(f)

    # 加载 keypoints 数据
    kp_timestamps = []
    keypoints = []
    for item in keypoints_data:
        t_rel = item['timestamp'] / 1000.0
        kp_timestamps.append(t_rel)
        kp_val = [point['position'][axis] for point in item['keypoints'] for axis in ['x', 'y', 'z']]
        # HACK:筛选左右手各前3个点，后补零至32维
        kp_val_filtered = kp_val[:9] + kp_val[18:27] + [0.0] * 14
        # keypoints.append(np.array(kp_val).flatten())
        keypoints.append(np.array(kp_val_filtered).flatten())
    keypoints = np.stack(keypoints)

    # 打开视频流
    caps = {}
    for video in sample['videos']:
        vid_id = video['id']
        # 只打开选定机位的视频
        if vid_id in CAMERA_MAP.keys():
            cap = cv2.VideoCapture(str(video['path']))
            if not cap.isOpened():
                continue
            caps[vid_id] = cap

    frames = []
    frame_idx = 0
    fps = 25
    
    while True:
        current_time = frame_idx / fps
        step_data = {"observation.images": {}}
        
        any_cap_ended = False
        for vid_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB 
                step_data["observation.images"][vid_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                any_cap_ended = True
                break
        
        if any_cap_ended:
            break

        # 插值获取 state 和 action
        state = get_interpolated_action(kp_timestamps, keypoints, current_time)
        action = get_interpolated_action(kp_timestamps, keypoints, current_time + 1.0 / fps)
        
        step_data["observation.state"] = torch.from_numpy(state).float()
        step_data["action"] = torch.from_numpy(action).float()
        step_data["task"] = task_name
        
        frames.append(step_data)
        frame_idx += 1

    for cap in caps.values():
        cap.release()
        
    return frames


def populate_dataset(
    dataset: LeRobotDataset,
    samples: list[dict],
    min_frames: int = 100
) -> LeRobotDataset:
    
    for sample in tqdm.tqdm(samples, desc="Populating Dataset"):
        episode_frames = load_raw_episode_data(sample)
        
        if len(episode_frames) < min_frames:
            continue

        for frame in episode_frames:
            payload = {
                "observation.state": frame["observation.state"],
                "action": frame["action"],
                "task": frame["task"],
            }
            # 展开图片
            for cam_name, img in frame["observation.images"].items():
                payload[f"observation.images.{CAMERA_MAP[cam_name]}"] = img
                
            # 2. 图像缩放处理
            target_size = (640, 480)  # PIL resize 接收 (宽, 高)
            for key in [f"observation.images.{cam}" for cam in CAMERA_MAP.values()]:
                if key in payload:
                    img_array = payload[key]
                    
                    # 将 numpy 数组转为 PIL Image 进行缩放
                    pil_img = Image.fromarray(img_array)
                    # 使用 LANCZOS 滤镜保证缩放后的图像质量，防止锯齿
                    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 再转回 numpy 数组存回 payload
                    payload[key] = np.array(pil_img)
            
            dataset.add_frame(payload)

        dataset.save_episode()

    return dataset


def port_human_data(
    raw_dir: Path,
    repo_id: str,
    task: str,
    fps: int = 25,
    min_frames: int = 50,
    mode: Literal["video", "image"] = "video",
):
    """
    主转换函数
    raw_dir: 存放该task的数据集的目录 /path/to/task
    """
    raw_dir = Path(raw_dir)
    samples = get_grouped_files(
        input_dir=raw_dir,
        task_name=raw_dir.name
    )

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type="human_hand",
        mode=mode
    )
    
    populate_dataset(dataset, samples, min_frames=min_frames)
    # dataset.consolidate() # 生成统计信息，这对于下游 train 脚本至关重要


if __name__ == "__main__":
    tyro.cli(port_human_data)
