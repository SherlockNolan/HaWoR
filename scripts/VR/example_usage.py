"""
使用示例：对比 VR 和 HaWoR 关键点

这个脚本演示如何使用 compare_keypoints_3d.py 进行关键点对比
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.VR.compare_keypoints_3d import (
    VRKeypointsLoader,
    HaworKeypointsLoader,
    KeypointComparator
)


def simple_comparison_example():
    """简单对比示例"""
    print("=" * 60)
    print("VR vs HaWoR 关键点对比示例")
    print("=" * 60)

    # 1. 加载 VR 数据
    print("\n步骤 1: 加载 VR 数据")
    print("-" * 60)
    vr_json_path = "path/to/vr_keypoints.json"  # 替换为实际路径

    try:
        vr_loader = VRKeypointsLoader(vr_json_path)
        print(f"✓ 成功加载 VR 数据: {len(vr_loader.frames_data)} 帧")
        print(f"  时间范围: {vr_loader.frames_data[0].timestamp:.3f}s - {vr_loader.frames_data[-1].timestamp:.3f}s")
    except FileNotFoundError:
        print(f"✗ 未找到 VR 数据文件: {vr_json_path}")
        print("  请修改 vr_json_path 为实际文件路径")
        return

    # 2. 加载 HaWoR 数据
    print("\n步骤 2: 加载 HaWoR 数据")
    print("-" * 60)
    hawor_pkl_path = "path/to/hawor_output.pkl"  # 替换为实际路径

    try:
        hawor_loader = HaworKeypointsLoader(hawor_pkl_path)
        print(f"✓ 成功加载 HaWoR 数据: {len(hawor_loader.frames_data)} 帧")
        print(f"  时间范围: {hawor_loader.frames_data[0].timestamp:.3f}s - {hawor_loader.frames_data[-1].timestamp:.3f}s")
    except FileNotFoundError:
        print(f"✗ 未找到 HaWoR 数据文件: {hawor_pkl_path}")
        print("  请修改 hawor_pkl_path 为实际文件路径")
        return

    # 3. 创建对比器并对齐数据
    print("\n步骤 3: 对齐数据")
    print("-" * 60)
    comparator = KeypointComparator(vr_loader, hawor_loader)

    try:
        result = comparator.align_data(
            alignment_fps=30.0,
            duration=10.0  # 对齐前 10 秒
        )
        print(f"✓ 数据对齐成功")
        print(f"  对齐时长: {result.aligned_timestamps[-1] - result.aligned_timestamps[0]:.3f}s")
        print(f"  对齐帧数: {len(result.aligned_timestamps)}")
    except ValueError as e:
        print(f"✗ 数据对齐失败: {e}")
        print("  检查 VR 和 HaWoR 数据的时间戳是否有重叠")
        return

    # 4. 可视化对比结果
    print("\n步骤 4: 可视化对比结果")
    print("-" * 60)

    # 创建输出目录
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)

    # 绘制多帧对比
    print("生成多帧对比图...")
    comparator.plot_3d_keypoints(
        result,
        frame_indices=None,  # 自动选择帧
        save_path=str(output_dir / "multi_frame_comparison.png"),
        interactive=False  # 不显示交互式窗口
    )
    print(f"✓ 多帧对比图已保存: {output_dir / 'multi_frame_comparison.png'}")

    # 绘制单帧详细对比
    frame_idx = len(result.aligned_timestamps) // 2  # 中间帧
    print(f"\n生成单帧对比图 (帧 {frame_idx})...")
    comparator.plot_single_frame_3d(
        result,
        frame_idx=frame_idx,
        save_path=str(output_dir / f"frame_{frame_idx}_comparison.png"),
        interactive=False
    )
    print(f"✓ 单帧对比图已保存: {output_dir / f'frame_{frame_idx}_comparison.png'}")

    # 5. 计算误差
    print("\n步骤 5: 计算误差统计")
    print("-" * 60)
    errors = comparator.calculate_errors(result)

    mean_left_error = error_mean(errors['vr_left_error'])
    mean_right_error = error_mean(errors['vr_right_error'])
    mean_total_error = error_mean(errors['mean_error'])

    print(f"  左手平均误差: {mean_left_error:.4f} 米")
    print(f"  右手平均误差: {mean_right_error:.4f} 米")
    print(f"  总体平均误差: {mean_total_error:.4f} 米")

    # 保存误差数据
    import pandas as pd
    df = pd.DataFrame({
        'timestamp': errors['timestamps'],
        'vr_left_error': errors['vr_left_error'],
        'vr_right_error': errors['vr_right_error'],
        'mean_error': errors['mean_error']
    })
    df.to_csv(output_dir / 'errors.csv', index=False)
    print(f"✓ 误差数据已保存: {output_dir / 'errors.csv'}")

    print("\n" + "=" * 60)
    print("对比完成！")
    print(f"所有结果已保存到: {output_dir}")
    print("=" * 60)


def error_mean(errors):
    """计算误差的平均值（忽略 NaN）"""
    import numpy as np
    return np.nanmean(errors)


def quick_test():
    """快速测试：检查数据是否可以加载"""
    print("=" * 60)
    print("快速数据加载测试")
    print("=" * 60)

    # 测试 VR 数据
    print("\n测试 VR 数据加载...")
    vr_json_path = "path/to/vr_keypoints.json"
    try:
        vr_loader = VRKeypointsLoader(vr_json_path)
        print(f"✓ VR 数据加载成功")
        print(f"  帧数: {len(vr_loader.frames_data)}")
        print(f"  每帧关键点数: {len(vr_loader.frames_data[0].keypoints_3d)}")
        print(f"  第一帧时间戳: {vr_loader.frames_data[0].timestamp:.3f}s")
    except Exception as e:
        print(f"✗ VR 数据加载失败: {e}")

    # 测试 HaWoR 数据
    print("\n测试 HaWoR 数据加载...")
    hawor_pkl_path = "path/to/hawor_output.pkl"
    try:
        hawor_loader = HaworKeypointsLoader(hawor_pkl_path)
        print(f"✓ HaWoR 数据加载成功")
        print(f"  帧数: {len(hawor_loader.frames_data)}")

        # 检查是否有有效数据
        valid_frames = [f for f in hawor_loader.frames_data if f.left_hand_3d is not None]
        print(f"  有效帧数: {len(valid_frames)}")

        if valid_frames:
            print(f"  左手关键点数: {len(valid_frames[0].left_hand_3d)}")
            print(f"  右手关键点数: {len(valid_frames[0].right_hand_3d)}")
        else:
            print("  ⚠ 警告: 没有有效的关键点数据")

    except Exception as e:
        print(f"✗ HaWoR 数据加载失败: {e}")


if __name__ == '__main__':
    print("\n选择运行模式:")
    print("1. 完整对比示例")
    print("2. 快速数据测试")
    print("3. 退出")

    choice = input("\n请输入选项 (1/2/3): ").strip()

    if choice == '1':
        simple_comparison_example()
    elif choice == '2':
        quick_test()
    else:
        print("退出程序")
