"""
生成测试数据：模拟 VR 和 HaWoR 的关键点数据

用于在没有实际数据的情况下测试对比脚本的功能
"""

import json
import pickle
import numpy as np
from pathlib import Path


def generate_vr_test_data(num_frames: int = 100, duration: float = 10.0,
                         output_path: str = "test_vr_keypoints.json"):
    """
    生成模拟的 VR 关键点数据

    Args:
        num_frames: 帧数
        duration: 时长（秒）
        output_path: 输出文件路径
    """
    print(f"生成 VR 测试数据: {num_frames} 帧, {duration} 秒")

    vr_data = []

    for i in range(num_frames):
        timestamp = i * duration / num_frames
        frame_index = 42846 + i  # 模拟 VR 的 frameIndex

        # 模拟左右手各 6 个关键点
        # 左手（前 6 个点）：手掌中心 + 5 个指尖
        # 右手（后 6 个点）：手掌中心 + 5 个指尖
        keypoints = []

        # 左手关键点（世界坐标系）
        # 手掌中心
        keypoints.append({
            "position": {"x": -0.15, "y": 0.85, "z": 0.05},
            "confidence": 1.0,
            "hand": "left",
            "fingerName": "palm",
            "index": 0
        })

        # 拇指
        for j in range(1, 3):
            keypoints.append({
                "position": {"x": -0.12 + j*0.03, "y": 0.88 - j*0.02, "z": 0.08 - j*0.01},
                "confidence": 1.0,
                "hand": "left",
                "fingerName": "thumb",
                "index": j
            })

        # 食指、中指等（简化）
        for j in range(3, 6):
            keypoints.append({
                "position": {"x": -0.16 + j*0.02, "y": 0.82 - j*0.01, "z": 0.07 - j*0.015},
                "confidence": 1.0,
                "hand": "left",
                "fingerName": f"finger_{j-2}",
                "index": j
            })

        # 右手关键点（世界坐标系）
        # 手掌中心
        keypoints.append({
            "position": {"x": 0.35, "y": 0.80, "z": 0.05},
            "confidence": 1.0,
            "hand": "right",
            "fingerName": "palm",
            "index": 6
        })

        # 拇指和其他手指
        for j in range(7, 12):
            keypoints.append({
                "position": {"x": 0.32 + (j-7)*0.04, "y": 0.78 - (j-7)*0.02, "z": 0.06 - (j-7)*0.008},
                "confidence": 1.0,
                "hand": "right",
                "fingerName": f"finger_{j-6}",
                "index": j
            })

        # 添加一些时间变化（模拟运动）
        time_factor = np.sin(timestamp * 2) * 0.02  # 周期性运动
        for kp in keypoints:
            kp["position"]["y"] += time_factor

        # 相机参数（模拟）
        camera_position = {"x": 0.02, "y": 1.08, "z": 0.02}
        camera_rotation = {"x": -0.35, "y": -0.01, "z": 0.006, "w": -0.93}

        frame_data = {
            "timestamp": timestamp,
            "frameIndex": frame_index,
            "syncedTimestamp": 603000.0 + timestamp,
            "unityTime": 602.0 + timestamp,
            "keypointCount": 12,
            "keypoints": keypoints,
            "cameraPosition": camera_position,
            "cameraRotation": camera_rotation,
            "textureSize": {"width": 1280, "height": 960}
        }

        vr_data.append(frame_data)

    # 保存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vr_data, f, indent=2)

    print(f"✓ VR 测试数据已保存到: {output_path}")


def generate_hawor_test_data(num_frames: int = 300, fps: float = 30.0,
                            output_path: str = "test_hawor_output.pkl"):
    """
    生成模拟的 HaWoR 关键点数据

    Args:
        num_frames: 帧数
        fps: 帧率
        output_path: 输出文件路径
    """
    print(f"生成 HaWoR 测试数据: {num_frames} 帧, {fps} fps")

    # 模拟 MANO 参数
    # pred_trans: (2, T, 3)
    # pred_rot: (2, T, 3)
    # pred_hand_pose: (2, T, 45)
    # pred_betas: (2, T, 10)

    pred_trans = np.random.randn(2, num_frames, 3) * 0.01  # 小幅平移
    pred_rot = np.random.randn(2, num_frames, 3) * 0.01  # 小幅旋转
    pred_hand_pose = np.random.randn(2, num_frames, 45) * 0.1  # 手部姿态
    pred_betas = np.random.randn(2, num_frames, 10) * 0.05  # 手部形状

    # 模拟 3D 关键点（相机坐标系）
    # 左手 21 个点
    left_hand_keypoints = []
    for t in range(num_frames):
        timestamp = t / fps

        # 手腕和手掌（4 个点）
        left_wrist = np.array([-0.02, 0.03, 0.5])
        left_palm = np.array([
            [-0.02, 0.04, 0.48],
            [-0.01, 0.04, 0.48],
            [0.00, 0.04, 0.48]
        ])

        # 手指（17 个点，每个手指 4 个关节）
        left_fingers = []
        base_x = -0.02
        base_y = 0.04
        base_z = 0.46

        for finger_idx in range(5):  # 5 个手指
            for joint_idx in range(4):  # 4 个关节
                # 拇指位置稍有不同
                if finger_idx == 0:  # 拇指
                    x = base_x + 0.03 + joint_idx * 0.01
                    y = base_y + 0.01 + joint_idx * 0.005
                    z = base_z - joint_idx * 0.015
                else:  # 其他手指
                    x = base_x + (finger_idx - 2.5) * 0.012
                    y = base_y + 0.02 - joint_idx * 0.008
                    z = base_z - joint_idx * 0.02

                # 添加时间变化（模拟运动）
                time_factor = np.sin(timestamp * 2) * 0.01
                left_fingers.append([x + time_factor, y, z])

        # 组合左手关键点
        left_hand = np.vstack([
            left_wrist.reshape(1, 3),
            np.array(left_palm),
            np.array(left_fingers)
        ])  # (21, 3)

        left_hand_keypoints.append(left_hand)

    # 右手 21 个点
    right_hand_keypoints = []
    for t in range(num_frames):
        timestamp = t / fps

        # 手腕和手掌（4 个点）
        right_wrist = np.array([0.02, 0.03, 0.5])
        right_palm = np.array([
            [0.02, 0.04, 0.48],
            [0.01, 0.04, 0.48],
            [0.00, 0.04, 0.48]
        ])

        # 手指（17 个点）
        right_fingers = []
        base_x = 0.02
        base_y = 0.04
        base_z = 0.46

        for finger_idx in range(5):
            for joint_idx in range(4):
                if finger_idx == 0:  # 拇指
                    x = base_x - 0.03 - joint_idx * 0.01
                    y = base_y + 0.01 + joint_idx * 0.005
                    z = base_z - joint_idx * 0.015
                else:
                    x = base_x - (finger_idx - 2.5) * 0.012
                    y = base_y + 0.02 - joint_idx * 0.008
                    z = base_z - joint_idx * 0.02

                # 添加时间变化
                time_factor = np.sin(timestamp * 2) * 0.01
                right_fingers.append([x + time_factor, y, z])

        # 组合右手关键点
        right_hand = np.vstack([
            right_wrist.reshape(1, 3),
            np.array(right_palm),
            np.array(right_fingers)
        ])  # (21, 3)

        right_hand_keypoints.append(right_hand)

    # 转换为 numpy 数组
    pred_keypoints_3d = np.array([
        np.array(left_hand_keypoints),   # (T, 21, 3)
        np.array(right_hand_keypoints)   # (T, 21, 3)
    ])  # (2, T, 21, 3)

    # 有效帧掩码（假设所有帧都有效）
    pred_valid = np.ones((2, num_frames), dtype=bool)

    # 相机外参
    R_c2w = np.eye(3).reshape(1, 3, 3).repeat(num_frames, axis=0)  # (T, 3, 3)
    t_c2w = np.zeros((num_frames, 3))  # (T, 3)

    # 保存 PKL
    hawor_data = {
        'pred_trans': pred_trans,
        'pred_rot': pred_rot,
        'pred_hand_pose': pred_hand_pose,
        'pred_betas': pred_betas,
        'pred_keypoints_3d': pred_keypoints_3d,
        'pred_valid': pred_valid,
        'R_c2w': R_c2w,
        't_c2w': t_c2w
    }

    with open(output_path, 'wb') as f:
        pickle.dump(hawor_data, f)

    print(f"✓ HaWoR 测试数据已保存到: {output_path}")


def main():
    """生成测试数据"""
    print("=" * 60)
    print("生成测试数据：VR 和 HaWoR 关键点")
    print("=" * 60)
    print()

    # 创建测试数据目录
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    # 生成 VR 测试数据
    generate_vr_test_data(
        num_frames=100,
        duration=10.0,
        output_path=str(test_dir / "test_vr_keypoints.json")
    )

    print()

    # 生成 HaWoR 测试数据
    generate_hawor_test_data(
        num_frames=300,
        fps=30.0,
        output_path=str(test_dir / "test_hawor_output.pkl")
    )

    print()
    print("=" * 60)
    print("测试数据生成完成！")
    print(f"测试数据目录: {test_dir.absolute()}")
    print()
    print("使用示例:")
    print(f"python scripts/VR/compare_keypoints_3d.py \\")
    print(f"    --vr-json {test_dir / 'test_vr_keypoints.json'} \\")
    print(f"    --hawor-pkl {test_dir / 'test_hawor_output.pkl'} \\")
    print(f"    --duration 10.0 \\")
    print(f"    --interactive")
    print("=" * 60)


if __name__ == '__main__':
    main()
