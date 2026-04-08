# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HaWoR (World-Space Hand Motion Reconstruction from Egocentric Videos) is a CVPR 2025 Highlight paper implementation that reconstructs 3D hand motion from egocentric videos. The system combines multiple computer vision models:

- **DROID-SLAM**: Camera pose estimation and SLAM
- **Metric3D**: Depth estimation
- **HaWoR**: Hand motion reconstruction model
- **Infiller**: Transformer model for temporal consistency
- **MANO**: Parametric hand model for mesh generation

## 项目说明（写在前面）

目前主要开发HaWoRPipeline这个类，其它的都是论文里面的结果，只进行诸如进度条之类的修改，但是算法主体不改变。

意图：使用当前的HaWoRPipeline对数据集（主要是Egocentric-10K）进行处理，生成3dKeypoints用于后面的VLA大模型的预训练。之前主要编辑的文件为：

- `scripts/`文件夹下面的脚本
- `lib/pipeline`下面的 `HaWoRPipeline.py`等。

编写代码时提供必要的注释，遵守良好的设计模式原则。

思考使用英文。注释和最后的总结回答推荐使用中文。

编辑单个文件的时候不要写README.md，基本用法通过注释的方式写在编辑的代码文件的底端。不需要考虑环境问题，如有需要安装的python包则一并写在注释中。

## Egocentric-10k 数据集格式

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total Hours** | 10,000 |
| **Total Frames** | 1.08 billion |
| **Video Clips** | 192,900 |
| **Median Clip Length** | 180.0 seconds |
| **Mean Hours per Worker** | 4.68 |
| **Storage Size** | 16.4 TB |
| **Format** | H.265/MP4 |
| **Resolution** | 1080p (1920x1080) |
| **Frame Rate** | 30 fps |
| **Field of View** | 128° horizontal, 67° vertical |
| **Camera Type** | Monocular head-mounted |
| **Audio** | No |
| **Device** | Build AI Gen 1 |

### Camera Intrinsics

Each worker folder contains an `intrinsics.json` file with calibrated camera parameters.

The intrinsics use the **OpenCV fisheye model** (Kannala-Brandt equidistant projection) with 4 distortion coefficients (k1-k4). All values are calibrated for the 1920x1080 resolution.

Example `intrinsics.json`:
```json
{
  "model": "fisheye",
  "image_width": 1920,
  "image_height": 1080,
  "fx": 1030.59,
  "fy": 1032.82,
  "cx": 966.69,
  "cy": 539.69,
  "k1": -0.1166,
  "k2": -0.0236,
  "k3": 0.0694,
  "k4": -0.0463
}
```

### Dataset Structure

Egocentric-10K is structured in **[WebDataset format](https://huggingface.co/docs/hub/en/datasets-webdataset)**.

```
builddotai/Egocentric-10K/
├── factory_001/
│   └── workers/
│       ├── worker_001/
│       │   ├── intrinsics.json              # Camera intrinsics for this worker
│       │   ├── factory001_worker001_part00.tar  # Shard 0 (≤1GB)
│       │   └── factory001_worker001_part01.tar  # Shard 1 (if needed)
│       ├── worker_002/
│       │   ├── intrinsics.json
│       │   └── factory001_worker002_part00.tar
│       └── worker_011/
│           ├── intrinsics.json
│           └── factory001_worker011_part00.tar
│
├── factory_002/
│   └── workers/
│       ├── worker_001/
│       │   ├── intrinsics.json
│       │   └── factory002_worker001_part00.tar
│       └── ...
│
├── factory_003/
│   └── workers/
│       └── ...
│
└── ... (factories 001-085)
```

Each TAR file contains pairs of video and metadata files:
```
factory001_worker001_part00.tar
├── factory001_worker001_00001.mp4        # Video 1
├── factory001_worker001_00001.json       # Metadata for video 1
├── factory001_worker001_00002.mp4        # Video 2
├── factory001_worker001_00002.json       # Metadata for video 2
├── factory001_worker001_00003.mp4        # Video 3
├── factory001_worker001_00003.json       # Metadata for video 3
└── ...                                   # Additional video/metadata pairs
```

Each JSON metadata file has the following fields:
```json
{
  "factory_id": "factory_002",      // Unique identifier for the factory location
  "worker_id": "worker_002",        // Unique identifier for the worker within factory
  "video_index": 0,                 // Sequential index for videos from this worker
  "duration_sec": 1200.0,           // Video duration in seconds
  "width": 1920,                    // Video width in pixels
  "height": 1080,                   // Video height in pixels
  "fps": 30.0,                      // Frames per second
  "size_bytes": 599697350,          // File size in bytes
  "codec": "h265"                   // Video codec
}
```

### 输出格式

处理流程：`HaWoRPipelineOpt.reconstruct()` 输出原始 dict → `HaworToKeypointsAdapter.convert()` 转换为按帧排序的 list → pickle 保存为 `*_hawor.pkl`

最终保存的 `result_dict` 结构：

```python
result_dict = {
    # ---- 未平滑结果 ----
    "original_result": [
        {
            "frame_idx": int,           # 帧序号 (0-based)
            "hands": [                  # 该帧检测到的手（0-2只）
                {
                    "is_right": int,                # 1=右手, 0=左手
                    "pred_keypoints_3d": np.ndarray, # (21, 3) 相机坐标系下的3D关节点
                    "pred_vertices_3d": np.ndarray,  # (778, 3) 相机坐标系下的MANO顶点
                    "pred_keypoints_2d": np.ndarray, # (21, 2) 2D投影关键点（像素坐标）
                    "pred_cam_t_full": np.ndarray,   # (3,) 相机平移（相机坐标系下为0）
                    "scaled_focal_length": float,    # 投影使用的焦距
                    "mano_params": {
                        "trans": np.ndarray,      # (3,) MANO平移参数
                        "rot": np.ndarray,        # (3,) MANO根旋转（轴角）
                        "hand_pose": np.ndarray,  # (15, 3) MANO手部姿态（轴角）
                        "betas": np.ndarray,      # (10,) MANO形状参数
                    }
                },
                # ... 可能有第二只手
            ],
            "camera_pose": {
                "R_w2c": np.ndarray,  # (3, 3) 世界→相机旋转矩阵
                "t_w2c": np.ndarray,  # (3,) 世界→相机平移
                "R_c2w": np.ndarray,  # (3, 3) 相机→世界旋转矩阵
                "t_c2w": np.ndarray,  # (3,) 相机→世界平移
            }
        },
        # ... 每帧一个元素
    ],

    # ---- 平滑结果（若平滑启用且成功，否则为 None）----
    "smoothed_result": [
        # 结构与 original_result 相同
        # 若 smooth_hands/smooth_camera 未启用或平滑失败，则为 None
    ],
}
```

**数据来源说明：**
- `HaWoRPipelineOpt.reconstruct()` 返回的原始 dict 包含 `pred_trans(2,T,3)`、`pred_rot(2,T,3)`、`pred_hand_pose(2,T,45)`、`pred_betas(2,T,10)`、`pred_valid(2,T)` 以及相机位姿等张量
- `HaworToKeypointsAdapter` 逐帧将 MANO 参数前向推理为 3D/2D 关键点，并存储原始 MANO 参数
- 若 `--save-origin` 开启，还会额外保存 `*_origin_dict.pkl`（HaWoRPipeline 的原始输出，未经过 adapter 转换）


## Installation

**note:** 在当前的工作环境中，并没有同步虚拟环境，只是纯coding代码。最后手动上传到linux服务器上进行运行。服务器是linux，注释中给出的python脚本示例运行等命令均需要linux系统的语法
