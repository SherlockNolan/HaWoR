# VR 数据处理工具集

这个目录包含处理和可视化 VR 数据的工具，用于对比 VR 原始关键点和 HaWoR Pipeline 的预测结果。

## 📁 工具列表

| 工具 | 描述 | 状态 |
|------|------|------|
| [compare_keypoints_3d.py](compare_keypoints_3d.py) | VR vs HaWoR 关键点对比和可视化 | ✅ 可用 |
| [visualize_vr_keypoints.py](visualize_vr_keypoints.py) | VR 关键点视频可视化 | ✅ 可用 |
| [convert_vr_data_to_lerobot.py](convert_vr_data_to_lerobot.py) | VR 数据转 LeRobot 格式 | ✅ 可用 |
| [generate_test_data.py](generate_test_data.py) | 生成模拟测试数据 | ✅ 可用 |
| [example_usage.py](example_usage.py) | 使用示例和测试 | ✅ 可用 |

## 🚀 快速开始

### 1. 安装依赖

**Linux/macOS:**
```bash
bash scripts/VR/install_dependencies.sh
```

**Windows:**
```cmd
scripts\VR\install_dependencies.bat
```

### 2. 生成测试数据（可选）

```bash
python scripts/VR/generate_test_data.py
```

这会生成：
- `test_data/test_vr_keypoints.json` - VR 模拟数据
- `test_data/test_hawor_output.pkl` - HaWoR 模拟数据

### 3. 运行对比

```bash
# 使用测试数据
python scripts/VR/compare_keypoints_3d.py \
    --vr-json test_data/test_vr_keypoints.json \
    --hawor-pkl test_data/test_hawor_output.pkl \
    --interactive

# 使用真实数据
python scripts/VR/compare_keypoints_3d.py \
    --vr-json path/to/your_vr_keypoints.json \
    --hawor-pkl path/to/your_hawor_output.pkl \
    --duration 10.0 \
    --interactive
```

## 📖 详细文档

### [compare_keypoints_3d.py](compare_keypoints_3d.py)

**功能：**
- 读取 VR 原始关键点（JSON 格式）
- 读取 HaWoR 预测关键点（PKL 格式）
- 自动时间戳对齐和插值
- 3D 空间中对比两种方法的关键点
- 计算误差统计

**使用示例：**
```bash
# 基础对比
python scripts/VR/compare_keypoints_3d.py \
    --vr-json vr_data.json \
    --hawor-pkl hawor_output.pkl

# 查看特定帧
python scripts/VR/compare_keypoints_3d.py \
    --vr-json vr_data.json \
    --hawor-pkl hawor_output.pkl \
    --frame-idx 50 \
    --interactive
```

**输出：**
- 多帧对比图
- 单帧详细对比图
- 误差统计 CSV
- 误差曲线图

详细文档：[COMPARE_README.md](COMPARE_README.md)

### [visualize_vr_keypoints.py](visualize_vr_keypoints.py)

**功能：**
- 将 VR 3D 关键点投影到视频上
- 在视频中标注关键点索引
- 支持时间戳匹配

**使用示例：**
```bash
# 先查看数据信息
python scripts/VR/visualize_vr_keypoints.py \
    --json vr_keypoints.json \
    --video original_video.mp4 \
    --info-only

# 生成可视化视频
python scripts/VR/visualize_vr_keypoints.py \
    --json vr_keypoints.json \
    --video original_video.mp4 \
    --output annotated_video.mp4 \
    --use-timestamp
```

### [convert_vr_data_to_lerobot.py](convert_vr_data_to_lerobot.py)

**功能：**
- 将 VR 数据转换为 LeRobot 数据集格式
- 自动时间戳对齐和插值
- 支持多相机数据

**使用示例：**
```bash
python scripts/VR/convert_vr_data_to_lerobot.py \
    --raw-dir /path/to/vr/data \
    --repo-id my-dataset/vr-data \
    --task "VR hand manipulation"
```

## 🎨 可视化说明

### 关键点颜色标识

- 🔵 **蓝色圆点**：VR 原始关键点（12 个）
  - 前 6 个点：左手
  - 后 6 个点：右手

- 🔴 **红色三角形**：HaWoR 左手关键点（21 个）
  - 完整的手部关键点
  - 包含手腕、手掌、5 个手指

- 🟢 **绿色三角形**：HaWoR 右手关键点（21 个）
  - 完整的手部关键点

### 误差统计

由于 VR 数据（12 点）和 HaWoR 数据（21 点）数量不同：

- VR 前 6 个点与 HaWoR 左手的前 6 个点对比
- VR 后 6 个点与 HaWoR 右手的前 6 个点对比
- 使用欧氏距离计算误差（米）

## 📊 数据格式

### VR JSON 格式

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
      }
    ],
    "cameraPosition": {"x": 0.02, "y": 1.08, "z": 0.02},
    "cameraRotation": {"x": -0.35, "y": -0.01, "z": 0.006, "w": -0.93},
    "textureSize": {"width": 1280, "height": 960}
  }
]
```

### HaWoR PKL 格式

```python
{
    'pred_keypoints_3d': np.array([
        # 左手 (T, 21, 3)
        # 右手 (T, 21, 3)
    ]),
    'pred_valid': np.array([
        # 左手掩码 (T,)
        # 右手掩码 (T,)
    ]),
    'pred_trans': np.array([(2, T, 3)]),
    'pred_rot': np.array([(2, T, 3)]),
    'pred_hand_pose': np.array([(2, T, 45)]),
    'pred_betas': np.array([(2, T, 10)]),
    'R_c2w': np.array([(T, 3, 3)]),
    't_c2w': np.array([(T, 3)])
}
```

## 🔧 技术特性

### 坐标系转换

- VR 数据：世界坐标系 → 相机坐标系
- HaWoR 数据：直接在相机坐标系
- 统一在相机坐标系下进行对比

### 时间戳对齐

1. 提取两种数据的时间戳范围
2. 计算重叠时间范围
3. 按指定帧率采样
4. 线性插值获取关键点位置

### 插值算法

对每个关键点的 x, y, z 坐标分别进行线性插值：

```python
interpolated_value = np.interp(target_timestamp, timestamps, keypoint_values)
```

## 🐛 故障排除

### 问题：PKL 文件中没有 pred_keypoints_3d

**原因**：HaWoRPipeline 输出中不包含关键点数据

**解决方案**：
1. 确保 HaWoRPipeline 配置正确
2. 使用 `HaworToKeypointsAdapter` 处理输出
3. 或者修改 `HaworKeypointsLoader` 添加 MANO 转换逻辑

### 问题：时间戳不匹配

**原因**：VR 和 HaWoR 数据的时间戳范围没有重叠

**解决方案**：
1. 检查两种数据的时间戳范围
2. 使用 `--duration` 参数限制对齐时长
3. 确认视频对应关系

### 问题：误差过大

**原因**：坐标系不一致或相机标定误差

**解决方案**：
1. 检查 VR 相机参数是否准确
2. 确认 HaWoR 数据在相机坐标系中
3. 查看单帧对比图分析具体差异

## 📝 依赖项

- NumPy
- Matplotlib
- Pandas
- OpenCV (cv2)
- 标准库：json, pickle, pathlib, argparse, logging, dataclasses

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可

此工具集为 HaWoR 项目的一部分，遵循项目的开源许可证。

## 🔗 相关链接

- HaWoR 项目：https://github.com/your-repo/hawor
- LeRobot：https://github.com/huggingface/lerobot

---

**注意**：这些工具主要用于开发和测试，生产环境使用前请充分验证。
