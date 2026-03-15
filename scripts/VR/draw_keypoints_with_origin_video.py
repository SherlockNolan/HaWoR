"""
可视化 VR 数据集中的 3D 关键点

将 JSON 中的 3D 关键点投影到视频帧上，并标注关键点索引
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VRKeypointsVisualizer:
    """VR 关键点可视化器"""

    def __init__(self, json_file_path: str, video_path: str, output_path: str,
                 use_timestamp: bool = False):
        """
        初始化可视化器

        Args:
            json_file_path: JSON 文件路径
            video_path: 原视频路径
            output_path: 输出视频路径
            use_timestamp: 是否使用 timestamp 匹配帧（默认使用 frameIndex）
        """
        self.json_file_path = Path(json_file_path)
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.use_timestamp = use_timestamp

        # 加载 JSON 数据
        self.frames_data = self._load_json()
        logger.info(f"加载了 {len(self.frames_data)} 帧数据")

        

        # 打开视频
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 分析 JSON 数据
        self._analyze_json_data()

        logger.info(f"视频信息: {self.width}x{self.height}, {self.fps}fps, 总帧数: {self.total_frames}")

    def _load_json(self) -> List[Dict]:
        """加载 JSON 文件"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _analyze_json_data(self):
        """分析 JSON 数据的统计信息"""
        if not self.frames_data:
            return

        # 获取 frameIndex 范围
        frame_indices = [frame['frameIndex'] for frame in self.frames_data]
        frame_indices.sort()

        # 获取 timestamp 范围
        timestamps = [frame.get('timestamp', 0) for frame in self.frames_data]
        timestamps.sort()

        logger.info("=" * 50)
        logger.info("JSON 数据分析:")
        logger.info(f"  总帧数: {len(self.frames_data)}")
        logger.info(f"  frameIndex 范围: {frame_indices[0]} - {frame_indices[-1]}")
        logger.info(f"  frameIndex 间隔: {frame_indices[1] - frame_indices[0] if len(frame_indices) > 1 else 0}")
        logger.info(f"  timestamp 范围: {timestamps[0]:.3f}s - {timestamps[-1]:.3f}s")
        logger.info(f"  timestamp 总时长: {timestamps[-1] - timestamps[0]:.3f}s")

        # 检查关键点数量
        keypoint_counts = [frame.get('keypointCount', 0) for frame in self.frames_data]
        unique_counts = list(set(keypoint_counts))
        logger.info(f"  关键点数量: {unique_counts}")

        # 检查纹理尺寸
        texture_sizes = []
        for frame in self.frames_data:
            ts = frame.get('textureSize', {})
            texture_sizes.append((ts.get('width', 0), ts.get('height', 0)))
        unique_sizes = list(set(texture_sizes))
        logger.info(f"  纹理尺寸: {unique_sizes}")

        logger.info("=" * 50)

        # 检查 frameIndex 是否合理
        if frame_indices[0] > self.total_frames * 0.9:
            logger.warning(f"⚠️  JSON 的 frameIndex ({frame_indices[0]}) 远大于视频帧数 ({self.total_frames})")
            logger.warning("   建议: 使用 --use-timestamp 参数按 timestamp 匹配")

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        将四元数转换为旋转矩阵

        Args:
            q: 四元数 [x, y, z, w]

        Returns:
            旋转矩阵 (3, 3)
        """
        x, y, z, w = q

        # 归一化
        q_norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
        x, y, z, w = x/q_norm, y/q_norm, z/q_norm, w/q_norm

        # 转换为旋转矩阵
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])

        return R

    @staticmethod
    def project_3d_to_2d(
        points_3d: np.ndarray,
        camera_pos: np.ndarray,
        camera_rot_quat: np.ndarray,
        img_width: int,
        img_height: int,
        focal_length: float = 1000.0
    ) -> np.ndarray:
        """
        将 3D 点从世界坐标系投影到 2D 图像坐标系

        Args:
            points_3d: 3D 点数组 (N, 3)，世界坐标系
            camera_pos: 相机位置 (3,)
            camera_rot_quat: 相机旋转四元数 (4,) [x, y, z, w]
            img_width: 图像宽度
            img_height: 图像高度
            focal_length: 焦距（像素）

        Returns:
            2D 点数组 (N, 2)，图像坐标系
        """
        # 转换四元数为旋转矩阵
        R = VRKeypointsVisualizer.quaternion_to_rotation_matrix(camera_rot_quat)

        # 世界坐标系到相机坐标系的转换
        # P_camera = R_world_to_camera * (P_world - camera_pos)
        points_camera = points_3d - camera_pos
        points_camera = R @ points_camera.T  # (3, N)

        # 提取深度（Z 坐标）
        z = points_camera[2, :]

        # 透视投影
        # x_image = x * f / z + cx
        # y_image = y * f / z + cy
        cx = img_width / 2
        cy = img_height / 2

        x_proj = points_camera[0, :] * focal_length / z + cx
        y_proj = points_camera[1, :] * focal_length / z + cy

        points_2d = np.stack([x_proj, y_proj], axis=1)  # (N, 2)

        return points_2d

    def _draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints_2d: np.ndarray,
        indices: List[int],
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        在帧上绘制关键点和索引

        Args:
            frame: 视频帧 (H, W, 3)
            keypoints_2d: 2D 关键点 (N, 2)
            indices: 关键点索引列表
            confidence: 置信度数组 (N,)

        Returns:
            绘制后的帧
        """
        # 复制帧以避免修改原帧
        frame_draw = frame.copy()

        for i, (x, y) in enumerate(keypoints_2d):
            # 检查点是否在图像范围内
            if 0 <= x < self.width and 0 <= y < self.height:
                # 根据置信度设置颜色（如果有）
                if confidence is not None and confidence[i] < 0.5:
                    color = (0, 0, 255)  # 红色表示低置信度
                else:
                    color = (0, 255, 0)  # 绿色表示高置信度

                # 绘制关键点
                cv2.circle(frame_draw, (int(x), int(y)), 5, color, -1)

                # 绘制索引标签
                label = str(indices[i])
                cv2.putText(
                    frame_draw,
                    label,
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # 白色文字
                    1,
                    cv2.LINE_AA
                )

        return frame_draw

    def visualize(self):
        """生成可视化视频"""
        # 创建输出视频写入器 - 使用 H.264 编码器以提高兼容性
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 (更兼容)
        # if fourcc == -1:
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 回退到 mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 回退到 mp4v
        out = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not out.isOpened():
            raise ValueError(f"无法创建输出视频: {self.output_path}")

        logger.info("开始生成可视化视频...")
        logger.info(f"匹配模式: {'timestamp' if self.use_timestamp else 'frameIndex'}")

        # 准备帧映射
        if self.use_timestamp:
            # 使用 timestamp 匹配
            frame_map = {frame['timestamp']: frame for frame in self.frames_data}
            logger.info(f"使用 timestamp 匹配，共 {len(frame_map)} 个时间戳")
        else:
            # 使用 frameIndex 匹配
            frame_map = {frame['frameIndex']: frame for frame in self.frames_data}
            logger.info(f"使用 frameIndex 匹配，共 {len(frame_map)} 个帧索引")

        processed_count = 0
        total_to_process = len(self.frames_data)
        matched_count = 0

        for frame_idx in range(self.total_frames):
            # 读取视频帧
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"无法读取第 {frame_idx} 帧")
                break

            # 计算当前帧的 timestamp
            current_timestamp = frame_idx / self.fps if self.fps > 0 else 0

            # 检查当前帧是否有对应的关键点数据
            frame_data = None
            if self.use_timestamp:
                # 使用 timestamp 匹配 - 寻找最接近的 timestamp
                if frame_map:
                    closest_timestamp = min(frame_map.keys(), key=lambda x: abs(x - current_timestamp))
                    if abs(closest_timestamp - current_timestamp) < 0.5:  # 允许 0.5 秒误差
                        frame_data = frame_map[closest_timestamp]
            else:
                # 使用 frameIndex 匹配
                if frame_idx in frame_map:
                    frame_data = frame_map[frame_idx]

            if frame_data is not None:
                matched_count += 1

                # 提取关键点（12个点）
                keypoints_3d = []
                confidences = []
                for kp in frame_data['keypoints']:
                    pos = kp['position']
                    keypoints_3d.append([pos['x'], pos['y'], pos['z']])
                    confidences.append(kp.get('confidence', 1.0))

                keypoints_3d = np.array(keypoints_3d)
                confidences = np.array(confidences)

                # 提取相机参数
                camera_pos = np.array([
                    frame_data['cameraPosition']['x'],
                    frame_data['cameraPosition']['y'],
                    frame_data['cameraPosition']['z']
                ])

                camera_rot_quat = np.array([
                    frame_data['cameraRotation']['x'],
                    frame_data['cameraRotation']['y'],
                    frame_data['cameraRotation']['z'],
                    frame_data['cameraRotation']['w']
                ])

                texture_size = frame_data['textureSize']
                img_width = texture_size['width']
                img_height = texture_size['height']

                # 投影到 2D
                keypoints_2d = self.project_3d_to_2d(
                    points_3d=keypoints_3d,
                    camera_pos=camera_pos,
                    camera_rot_quat=camera_rot_quat,
                    img_width=img_width,
                    img_height=img_height
                )

                # 绘制关键点
                frame = self._draw_keypoints(
                    frame=frame,
                    keypoints_2d=keypoints_2d,
                    indices=list(range(len(keypoints_2d))),
                    confidence=confidences
                )

                processed_count += 1

                # 显示进度
                if processed_count % 10 == 0:
                    logger.info(f"已匹配 {matched_count} 帧，已处理 {processed_count}/{total_to_process} 帧")

            # 写入输出视频
            out.write(frame)

        # 释放资源
        self.cap.release()
        out.release()

        logger.info("=" * 50)
        logger.info(f"可视化视频已保存到: {self.output_path}")
        logger.info(f"视频总帧数: {self.total_frames}")
        logger.info(f"JSON 帧数: {total_to_process}")
        logger.info(f"匹配成功: {matched_count} 帧")
        logger.info(f"处理成功: {processed_count} 帧")
        logger.info("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='可视化 VR 数据集中的 3D 关键点'
    )
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='JSON 文件路径，包含 3D 关键点和相机参数'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='原视频路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_visualization.mp4',
        help='输出视频路径（默认: output_visualization.mp4）'
    )
    parser.add_argument(
        '--focal-length',
        type=float,
        default=1000.0,
        help='相机焦距（像素，默认: 1000.0）'
    )
    parser.add_argument(
        '--use-timestamp',
        action='store_true',
        help='使用 timestamp 匹配视频帧（默认使用 frameIndex）'
    )
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='只显示 JSON 数据分析，不生成视频'
    )

    args = parser.parse_args()

    try:
        visualizer = VRKeypointsVisualizer(
            json_file_path=args.json,
            video_path=args.video,
            output_path=args.output,
            use_timestamp=args.use_timestamp
        )

        if args.info_only:
            logger.info("仅显示数据信息，不生成视频")
            return

        visualizer.visualize()
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
