# VR vs HaWoR 关键点对比工具

## 概述

这个工具用于对比 VR 原始数据的关键点和 HaWoR Pipeline 预测的关键点，在 3D 空间中可视化两种方法的结果差异。

## 功能特点

- ✅ 读取 VR 原始关键点数据（JSON 格式）
- ✅ 读取 HaWoR Pipeline 输出数据（PKL 格式）
- ✅ 自动时间戳对齐和插值处理
- ✅ 3D 空间中对比两种方法的关键点
- ✅ 区分颜色标识不同方法的关键点
- ✅ 计算误差统计和误差曲线
- ✅ 支持单帧详细对比和多帧概览对比

## 数据格式

### VR 数据格式（JSON）

```json
[
  {
    "timestamp": 13.099999994039536,
    "frameIndex": 42846,
    "keypointCount": 12,
    "keypoints": [
      {
        "position": {"x": -0.18, "y": 0.84, "z": 0.003},
        "confidence": 1
      },
      // ... 共12个关键点
    ],
    "cameraPosition": {"x": 0.02, "y": 1.08, "z": 0.02},
    "cameraRotation": {"x": -0.35, "y": -0.01, "z": 0.006, "w": -0.93},
    "textureSize": {"width": 1280, "height": 960}
  }
]
```

### HaWoR 数据格式（PKL）

HaWoR Pipeline 输出的 PKL 文件应该包含 `pred_keypoints_3d` 字段：

```python
{
    'pred_keypoints_3d': np.array([
        # 左手关键点 (T, 21, 3)
        # 右手关键点 (T, 21, 3)
    ]),
    'pred_valid': np.array([
        # 左手有效帧掩码 (T,)
        # 右手有效帧掩码 (T,)
    ])
}
```

## 使用方法

### 基本用法

```bash
# 对比完整数据
python scripts/VR/compare_keypoints_3d.py \
    --vr-json data/vr_keypoints.json \
    --hawor-pkl data/hawor_output.pkl \
    --output-dir results/comparison

# 对比指定帧
python scripts/VR/compare_keypoints_3d.py \
    --vr-json data/vr_keypoints.json \
    --hawor-pkl data/hawor_output.pkl \
    --frame-idx 100 \
    --interactive
```

### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--vr-json` | string | 是 | VR 关键点 JSON 文件路径 |
| `--hawor-pkl` | string | 是 | HaWoR Pipeline 输出 PKL 文件路径 |
| `--alignment-fps` | float | 否 | 对齐后的采样帧率（默认: 30.0） |
| `--duration` | float | 否 | 对齐时长（秒），如果为 None 则使用 VR 数据的时长 |
| `--output-dir` | string | 否 | 输出目录（默认: comparison_output） |
| `--frame-idx` | int | 否 | 绘制指定帧的单帧对比图，如果为 None 则绘制多帧 |
| `--interactive` | flag | 否 | 显示交互式图表 |

### 示例

```bash
# 1. 基础对比（自动生成多帧对比图）
python scripts/VR/compare_keypoints_3d.py \
    --vr-json recordings/recording_keypoints.json \
    --hawor-pkl hawor_output.pkl

# 2. 指定对齐时长（只对比前 5 秒）
python scripts/VR/compare_keypoints_3d.py \
    --vr-json recordings/recording_keypoints.json \
    --hawor-pkl hawor_output.pkl \
    --duration 5.0

# 3. 查看特定帧的详细对比
python scripts/VR/compare_keypoints_3d.py \
    --vr-json recordings/recording_keypoints.json \
    --hawor-pkl hawor_output.pkl \
    --frame-idx 50 \
    --interactive

# 4. 自定义输出目录和帧率
python scripts/VR/compare_keypoints_3d.py \
    --vr-json recordings/recording_keypoints.json \
    --hawor-pkl hawor_output.pkl \
    --output-dir my_results/comparison \
    --alignment-fps 25.0
```

## 输出说明

脚本会在输出目录中生成以下文件：

1. **多帧对比图** (`multi_frame_comparison.png`)
   - 显示多个帧的关键点对比
   - 蓝色点：VR 关键点（12 个）
   - 红色点：HaWoR 左手关键点（21 个）
   - 绿色点：HaWoR 右手关键点（21 个）

2. **单帧对比图** (`frame_{idx}_comparison.png`)
   - 如果指定了 `--frame-idx`，会生成该帧的详细对比图
   - 更大的图片，更清晰的标注
   - 包含关键点索引标签

3. **误差统计 CSV** (`errors.csv`)
   - 包含每帧的详细误差数据
   - 时间戳、左手误差、右手误差、平均误差

4. **误差曲线图** (`error_curves.png`)
   - 显示误差随时间的变化趋势
   - 包含左手、右手和总体误差曲线

## 可视化说明

### 关键点颜色标识

- 🔵 **蓝色圆点**：VR 原始关键点（12 个点）
  - 前 6 个点：左手
  - 后 6 个点：右手

- 🔴 **红色三角形**：HaWoR 左手关键点（21 个点）
  - 完整的手部关键点

- 🟢 **绿色三角形**：HaWoR 右手关键点（21 个点）
  - 完整的手部关键点

### 坐标系说明

- 所有关键点都转换到相机坐标系下
- Z 轴正方向为相机视线方向
- 3D 绘图时坐标轴比例相等，便于比较位置差异

### 误差计算

由于 VR 数据（12 点）和 HaWoR 数据（21 点）数量不同：

- VR 前 6 个点与 HaWoR 左手的前 6 个点进行对比
- VR 后 6 个点与 HaWoR 右手的前 6 个点进行对比
- 使用欧氏距离计算误差

## 技术细节

### 时间戳对齐

1. 提取 VR 数据的时间戳范围 `[t_start, t_end]`
2. 提取 HaWoR 数据的时间戳范围 `[h_start, h_end]`
3. 计算重叠时间范围 `[max(t_start, h_start), min(t_end, h_end)]`
4. 在重叠范围内按指定帧率采样
5. 使用线性插值获取每个采样点的关键点位置

### 坐标系转换

VR 数据在世界坐标系中，需要转换到相机坐标系：

```
P_camera = R * (P_world - camera_position)
```

其中 R 是从四元数转换得到的旋转矩阵。

### 线性插值

对每个关键点的 x, y, z 坐标分别进行线性插值：

```python
interpolated_value = np.interp(target_timestamp, timestamps, keypoint_values)
```

## 注意事项

1. **数据完整性**：确保 VR 和 HaWoR 数据有足够的时间重叠
2. **坐标系一致性**：HaWoR 数据需要在相机坐标系中
3. **关键点数量**：VR 数据应有 12 个关键点，HaWoR 应有 21 个关键点每只手
4. **时间戳单位**：确保时间戳单位一致（秒）

## 故障排除

### 问题：没有时间重叠

**错误信息**：`VR 和 HaWoR 数据没有时间重叠`

**解决方案**：
- 检查 VR 和 HaWoR 数据的时间戳范围
- 确认视频对应关系是否正确
- 尝试调整 `--duration` 参数

### 问题：PKL 文件中没有 pred_keypoints_3d

**错误信息**：`PKL 文件中没有 pred_keypoints_3d，需要使用 MANO 模型转换`

**解决方案**：
- 确保 HaWoRPipeline 输出中包含关键点数据
- 检查 `HaworToKeypointsAdapter` 是否正确配置
- 或者修改代码添加 MANO 模型转换逻辑

### 问题：误差过大

**可能原因**：
- 坐标系不一致
- 相机标定参数差异
- 时间戳对齐不准确

**解决方案**：
- 检查 VR 的相机参数是否准确
- 调整对齐帧率 `--alignment-fps`
- 查看单帧对比图，分析具体差异

## 依赖项

- NumPy
- Matplotlib
- Pandas
- 标准库：json, pickle, pathlib, argparse, logging, dataclasses

## 代码结构

```
VRKeypointsLoader    # VR 数据加载和解析
  ├── _load_json()           # 加载 JSON 文件
  ├── _parse_frames()        # 解析帧数据
  ├── quaternion_to_rotation_matrix()  # 四元数转旋转矩阵
  └── world_to_camera()     # 坐标系转换

HaworKeypointsLoader   # HaWoR 数据加载和解析
  ├── _load_pkl()            # 加载 PKL 文件
  └── _parse_frames()        # 解析帧数据

KeypointComparator     # 对比和可视化
  ├── align_data()           # 时间戳对齐
  ├── interpolate_keypoints() # 插值
  ├── plot_3d_keypoints()   # 多帧 3D 绘制
  ├── plot_single_frame_3d()  # 单帧 3D 绘制
  └── calculate_errors()     # 误差计算
```

## 扩展建议

1. **添加 MANO 模型支持**：自动将 HaWoR 参数转换为 3D 关键点
2. **误差热图**：在 3D 空间中绘制误差分布热图
3. **视频对比**：生成视频形式的对比结果
4. **统计报告**：生成更详细的统计报告（中位数、标准差等）
5. **交互式可视化**：使用 plotly 等库创建交互式 3D 图

## 许可

此脚本为 HaWoR 项目的一部分，遵循项目的开源许可证。
