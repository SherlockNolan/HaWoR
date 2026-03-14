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

意图：使用当前的HaWoRPipeline对数据集进行处理，生成3dKeypoints用于后面的VLA大模型的预训练。之前主要编辑的文件为：

- `scripts/`文件夹下面的脚本
- `lib/pipeline`下面的 `HaWoRPipeline.py`等。

编写代码时提供必要的注释，遵守良好的设计模式原则。

思考可用英文。最后的总结回答推荐使用中文。

## Installation

**note:** 在当前的工作环境中，并没有同步虚拟环境，只是纯coding代码。最后手动上传到linux服务器上进行运行。

## Common Commands

### Batch Processing

```bash
# Process all videos with 4 workers (multi-GPU parallel)
python scripts/reconstruct_egocentric.py --dataroot /path/to/dataset --output /path/to/output --num-workers 4

# Process with single worker (sequential)
python scripts/reconstruct_egocentric.py --dataroot /path/to/dataset --output /path/to/output --num-workers 1

# Check existing pkl files
python scripts/reconstruct_egocentric.py --dataroot /path/to/dataset --output /path/to/output --check

# Test mode (using test_video folder)
python scripts/reconstruct_egocentric.py --test --num-workers 4
```

## Architecture Overview

### Core Components

**`lib/pipeline/HaWoRPipeline.py`**: Main reconstruction pipeline with lazy video loading and multi-GPU support. Key classes:

- `HaWoRPipeline`: Main pipeline class for reconstructing hand motion from videos
- `HaWoRConfig`: Configuration dataclass for pipeline parameters
- `LazyVideoFrames`: Memory-efficient video frame loading

**`scripts/reconstruct_egocentric.py`**: Batch processing script with multi-worker support

- Custom multiprocessing implementation for GPU assignment
- Progress tracking with real-time updates per worker
- Interleaved video processing for balanced factory/worker distribution

### Data Flow

1. **Input**: Video frames → `detect_track_video()` extracts hand bounding boxes
2. **Motion Estimation**: `hawor_motion_estimation()` processes chunks with HaWoR model
3. **SLAM**: `hawor_slam()` runs DROID-SLAM for camera poses
4. **Infilling**: `hawor_infiller()` applies transformer for temporal consistency
5. **Output**: PKL files containing hand pose parameters (trans, rot, hand_pose, betas, R_c2w, t_c2w)

### Output Format

PKL files contain dictionary with keys:

- `pred_trans`: Translation parameters (2, N, 3) for left/right hands
- `pred_rot`: Rotation parameters (2, N, 3)
- `pred_hand_pose`: Hand pose parameters (2, N, 15)
- `pred_betas`: Shape parameters (2, N, 10)
- `R_c2w`: Camera-to-world rotation matrices (N, 3, 3)
- `t_c2w`: Camera-to-world translation vectors (N, 3)

### Important Paths

- Weights: `./weights/external/` and `./weights/hawor/`
- MANO models: `_DATA/data/mano/MANO_RIGHT.pkl`, `_DATA/data_left/mano_left/MANO_LEFT.pkl`
- Third-party: `thirdparty/DROID-SLAM/`, `thirdparty/Metric3D/`
- Core modules: `lib/core/`, `lib/pipeline/`, `lib/models/`

## Development Notes

### Multi-Worker Architecture

The system uses a custom multiprocessing implementation to assign specific GPUs to each worker process:

- Workers are initialized with specific GPU devices via round-robin assignment
- Each worker maintains its own `HaWoRPipeline` instance to avoid GIL limitations
- Progress is tracked through multiprocessing queues with per-worker progress bars
- Workers can process videos in parallel with true GPU acceleration

### MANO Model Usage

MANO parameters are converted to 3D meshes using utilities in `hawor/utils/process.py`:

- `run_mano()`: Process right hand parameters
- `run_mano_left()`: Process left hand parameters
- `get_mano_faces()`: Get mesh face indices

### Third-Party Dependencies

- **DROID-SLAM**: Must be installed from local submodule
- **Metric3D**: Requires specific weights in `thirdparty/Metric3D/weights/`
- **Chumpy**: Older dependency for optimization (install with --no-build-isolation)

## Testing

For testing with local video files:

```bash
# Place test videos in ./test_video/
python scripts/reconstruct_egocentric.py --test --num-workers 1

# Check generated pkl files
python scripts/reconstruct_egocentric.py --test --check
```
