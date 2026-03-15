import json
import pickle
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def visualize_comparison(json_path, pkl_path, output_video_path):
    print(f"Loading VR JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        vr_data = json.load(f)

    print(f"Loading Hawor PKL from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        hawor_data = pickle.load(f)
        
    # --- 1. 处理 VR 数据到相机坐标系 ---
    vr_frames = []
    for frame in vr_data:
        keypoints = frame.get("keypoints", [])
        
        cam_pos_data = frame.get("cameraPosition", {})
        cam_pos = np.array([cam_pos_data.get("x", 0), cam_pos_data.get("y", 0), cam_pos_data.get("z", 0)])
        
        cam_rot_data = frame.get("cameraRotation", {})
        cam_quat = [cam_rot_data.get("x", 0), cam_rot_data.get("y", 0), cam_rot_data.get("z", 0), cam_rot_data.get("w", 1)]
        
        rot = R.from_quat(cam_quat)
        rot_inv = rot.inv()
        
        pts_cam = []
        for kp in keypoints:
            pos = kp.get("position", {})
            pt_world = np.array([pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)])
            # 转换到相机坐标系: P_cam = R_inv * (P_world - C_pos)
            pt_cam = rot_inv.apply(pt_world - cam_pos)
            pts_cam.append(pt_cam)
        vr_frames.append(np.array(pts_cam))
        
    vr_frames = np.array(vr_frames)
    
    # --- 2. 处理 Hawor 模型数据 ---
    # 使用 smoothed_result
    # results = hawor_data.get('smoothed_result', hawor_data.get('original_result', []))
    results = hawor_data.get('original_result', [])
    
    hawor_frames = []
    for frame in results:
        hands = frame.get('hands', [])
        frame_pts = []
        for hand in hands:
            # Hawor 的 pred_keypoints_3d 已经是相机坐标系下 (OpenCV 标准: +X右, +Y下, +Z前)
            # 为了与 Unity 的 (+X右, +Y上, +Z前) 坐标系统一，我们将 Hawor 的 Y 坐标取反
            kps_3d = hand['pred_keypoints_3d'].copy()
            kps_3d[:, 1] = -kps_3d[:, 1] 
            frame_pts.append({
                "is_right": hand['is_right'],
                "kps": kps_3d
            })
        hawor_frames.append(frame_pts)

    num_vr_frames = len(vr_frames)
    num_hawor_frames = len(hawor_frames)
    
    print(f"VR frames: {num_vr_frames}, Hawor frames: {num_hawor_frames}")

    # 将两者的帧率进行大致对齐 (以 Hawor 的帧数为基准)
    # 因为可能两个序列录制的起止点略有偏差或者帧率不同
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 计算全局坐标范围
    all_x, all_y, all_z = [], [], []
    for f in vr_frames:
        all_x.extend(f[:, 0]); all_y.extend(f[:, 1]); all_z.extend(f[:, 2])
    for f in hawor_frames:
        for hand in f:
            kps = hand['kps']
            all_x.extend(kps[:, 0]); all_y.extend(kps[:, 1]); all_z.extend(kps[:, 2])
            
    min_x, max_x = min(all_x) - 0.1, max(all_x) + 0.1
    min_y, max_y = min(all_y) - 0.1, max(all_y) + 0.1
    min_z, max_z = min(all_z) - 0.1, max(all_z) + 0.1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    trajectory_length = 20

    print("Generating comparison video...")
    for i in range(min(140, num_hawor_frames)):
        ax.clear()
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])
        ax.set_xlabel('X (Right)')
        ax.set_ylabel('Y (Up)')
        ax.set_zlabel('Z (Forward)')
        ax.set_title(f'Comparison: VR (Dots/Lines) vs Hawor Model (Stars/Dashed) - Frame {i}')

        # 画相机原点
        ax.scatter([0], [0], [0], c='g', marker='*', s=150, label='Camera (Origin)')

        # 1. 绘制 Hawor 帧数据 (21个关键点)
        hawor_frame = hawor_frames[i]
        for hand in hawor_frame:
            kps = hand['kps']
            color = 'purple' if hand['is_right'] == 0 else 'orange'
            label = 'Hawor Left' if hand['is_right'] == 0 else 'Hawor Right'
            
            # 手腕索引是0，指尖通常是 4, 8, 12, 16, 20
            fingertips = [4, 8, 12, 16, 20]
            ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c=color, marker='*', s=40, label=label)
            # 简单连线
            for tip in fingertips:
                ax.plot([kps[0, 0], kps[tip, 0]], 
                        [kps[0, 1], kps[tip, 1]], 
                        [kps[0, 2], kps[tip, 2]], c=color, linestyle='--', linewidth=1.5)

        # 2. 绘制对应的 VR 帧数据 (12个关键点)
        # 将 Hawor 的索引映射到 VR 的索引
        vr_idx = int(i * (num_vr_frames / num_hawor_frames))
        if vr_idx < num_vr_frames:
            vr_kps = vr_frames[vr_idx]
            half_kp = len(vr_kps) // 2
            
            # VR 左手 (红)
            ax.scatter(vr_kps[:half_kp, 0], vr_kps[:half_kp, 1], vr_kps[:half_kp, 2], c='r', marker='o', s=40, label='VR Left')
            for tip in range(1, half_kp):
                ax.plot([vr_kps[0, 0], vr_kps[tip, 0]], 
                        [vr_kps[0, 1], vr_kps[tip, 1]], 
                        [vr_kps[0, 2], vr_kps[tip, 2]], c='r', linewidth=2)
                        
            # VR 右手 (蓝)
            ax.scatter(vr_kps[half_kp:, 0], vr_kps[half_kp:, 1], vr_kps[half_kp:, 2], c='b', marker='^', s=40, label='VR Right')
            for tip in range(half_kp + 1, len(vr_kps)):
                ax.plot([vr_kps[half_kp, 0], vr_kps[tip, 0]], 
                        [vr_kps[half_kp, 1], vr_kps[tip, 1]], 
                        [vr_kps[half_kp, 2], vr_kps[tip, 2]], c='b', linewidth=2)

        # 去重图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # 转换为图像
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if out is None:
            height, width, _ = img.shape
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

        out.write(img)

        if i % 20 == 0:
            print(f"Processed {i}/{num_hawor_frames} frames...")

    if out is not None:
        out.release()
    plt.close(fig)
    print(f"Comparison video saved to {output_video_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "../dataset/example/recording_2026-01-13T12-13-48/recording_2026-01-13T12-13-48_keypoints.json")
    pkl_path = os.path.join(script_dir, "../dataset/example/recording_2026-01-13T12-13-48/recording_2026-01-13T12-13-48_remote_0_hawor.pkl")
    output_video = os.path.join(script_dir, "comparison_trajectory_origin.mp4")
    
    if os.path.exists(json_path) and os.path.exists(pkl_path):
        visualize_comparison(json_path, pkl_path, output_video)
    else:
        print("Error: Missing JSON or PKL files.")
