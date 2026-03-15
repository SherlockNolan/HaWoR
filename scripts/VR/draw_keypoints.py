import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def visualize_keypoints(json_file_path, output_video_path, fps=30):
    # 1. 加载数据
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    print(f"Total frames: {len(data)}")

    # 2. 提取所有帧的坐标以确定坐标轴范围
    all_coords = []
    frames_data = []
    
    for frame in data:
        pts = []
        for kp in frame['keypoints']:
            pos = kp['position']
            pts.append([pos['x'], pos['y'], pos['z']])
        pts = np.array(pts)
        frames_data.append(pts)
        all_coords.append(pts)

    all_coords = np.concatenate(all_coords, axis=0)
    min_bound = all_coords.min(axis=0)
    max_bound = all_coords.max(axis=0)

    # 3. 设置绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 初始化散点图
    scatter = ax.scatter([], [], [], c='blue', s=50)
    
    # 设置坐标轴标签和范围
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand 3D Keypoints Visualization')

    # 为了保持比例，手动设置范围
    ax.set_xlim(min_bound[0], max_bound[0])
    ax.set_ylim(min_bound[1], max_bound[1])
    ax.set_zlim(min_bound[2], max_bound[2])

    # 4. 动画更新函数
    def update(frame_idx):
        pts = frames_data[frame_idx]
        # 更新散点位置
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        # 可选：更新标题显示当前帧/时间
        ax.set_title(f"Frame: {data[frame_idx]['frameIndex']} | Time: {data[frame_idx]['timestamp']:.2f}s")
        return scatter,

    # 5. 创建动画并保存
    ani = FuncAnimation(fig, update, frames=len(frames_data), interval=1000/fps, blit=False)

    print(f"Saving video to {output_video_path}...")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_video_path, writer=writer)
    
    plt.close()
    print("Done!")

if __name__ == "__main__":
    # 你的文件名
    INPUT_JSON = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json"
    OUTPUT_VIDEO = "./results/hand_visualization.mp4"
    
    visualize_keypoints(INPUT_JSON, OUTPUT_VIDEO, fps=25)
    
"""
python scripts/VR/draw_keypoints_with_origin_video.py\
    --json="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json"\
    --video="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0.mp4"\
    --output="./results/hand_visualize_with_origin_video.mp4"
    
    --json="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json"
    --video="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0.mp4"
    --output="./results/hand_visualize_with_origin_video.mp4"
"""