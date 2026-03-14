"""
对比 VR 原始关键点和 HaWoR 预测关键点

功能：
1. 读取 VR 原始数据（JSON 格式）
2. 读取 HaWoR Pipeline 输出数据（PKL 格式）
3. 进行时间戳对齐和插值
4. 在 3D 空间中对比两种方法的关键点
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from dataclasses import dataclass, asdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VRFrameData:
    """VR 单帧数据"""
    timestamp: float
    frame_index: int
    keypoints_3d: np.ndarray  # (12, 3) - 世界坐标系
    camera_position: np.ndarray  # (3,)
    camera_rotation: np.ndarray  # (4,) - 四元数 [x, y, z, w]
    confidences: np.ndarray  # (12,)


@dataclass
class HaworFrameData:
    """HaWoR 单帧数据"""
    timestamp: float
    frame_index: int
    left_hand_3d: Optional[np.ndarray]  # (21, 3) - 相机坐标系
    right_hand_3d: Optional[np.ndarray]  # (21, 3) - 相机坐标系
    left_hand_2d: Optional[np.ndarray]  # (21, 2)
    right_hand_2d: Optional[np.ndarray]  # (21, 2)
    left_valid: bool
    right_valid: bool


@dataclass
class ComparisonResult:
    """对比结果"""
    vr_data: List[VRFrameData]
    hawor_data: List[HaworFrameData]
    aligned_timestamps: List[float]
    vr_interpolated: List[np.ndarray]  # 对齐后的 VR 关键点
    hawor_interpolated: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]  # 对齐后的 HaWoR 关键点


class VRKeypointsLoader:
    """VR 关键点加载器"""

    def __init__(self, json_file_path: str):
        """
        初始化 VR 关键点加载器

        Args:
            json_file_path: JSON 文件路径
        """
        self.json_file_path = Path(json_file_path)
        self.data = self._load_json()
        self.frames_data = self._parse_frames()

    def _load_json(self) -> List[Dict]:
        """加载 JSON 文件"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _parse_frames(self) -> List[VRFrameData]:
        """解析帧数据"""
        frames = []

        for frame_dict in self.data:
            # 提取关键点
            keypoints_3d = []
            confidences = []
            for kp in frame_dict['keypoints']:
                pos = kp['position']
                keypoints_3d.append([pos['x'], pos['y'], pos['z']])
                confidences.append(kp.get('confidence', 1.0))

            keypoints_3d = np.array(keypoints_3d)
            confidences = np.array(confidences)

            # 提取相机参数
            camera_position = np.array([
                frame_dict['cameraPosition']['x'],
                frame_dict['cameraPosition']['y'],
                frame_dict['cameraPosition']['z']
            ])

            camera_rotation = np.array([
                frame_dict['cameraRotation']['x'],
                frame_dict['cameraRotation']['y'],
                frame_dict['cameraRotation']['z'],
                frame_dict['cameraRotation']['w']
            ])

            frame_data = VRFrameData(
                timestamp=frame_dict['timestamp'],
                frame_index=frame_dict['frameIndex'],
                keypoints_3d=keypoints_3d,
                camera_position=camera_position,
                camera_rotation=camera_rotation,
                confidences=confidences
            )

            frames.append(frame_data)

        # 按时间戳排序
        frames.sort(key=lambda x: x.timestamp)

        logger.info(f"加载了 {len(frames)} 帧 VR 数据")
        logger.info(f"时间戳范围: {frames[0].timestamp:.3f}s - {frames[-1].timestamp:.3f}s")

        return frames

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        x, y, z, w = q
        q_norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
        x, y, z, w = x/q_norm, y/q_norm, z/q_norm, w/q_norm

        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])

        return R

    def world_to_camera(self, points_world: np.ndarray,
                     camera_position: np.ndarray,
                     camera_rotation: np.ndarray) -> np.ndarray:
        """
        将世界坐标系点转换到相机坐标系

        Args:
            points_world: 世界坐标系点 (N, 3)
            camera_position: 相机位置 (3,)
            camera_rotation: 相机旋转四元数 (4,)

        Returns:
            相机坐标系点 (N, 3)
        """
        R = self.quaternion_to_rotation_matrix(camera_rotation)
        points_camera = points_world - camera_position
        points_camera = R @ points_camera.T
        return points_camera.T


class HaworKeypointsLoader:
    """HaWoR 关键点加载器"""

    def __init__(self, pkl_file_path: str, fps: float = 30.0):
        """
        初始化 HaWoR 关键点加载器

        Args:
            pkl_file_path: PKL 文件路径
            fps: 视频帧率（用于计算时间戳）
        """
        self.pkl_file_path = Path(pkl_file_path)
        self.fps = fps
        self.data = self._load_pkl()
        self.frames_data = self._parse_frames()

    def _load_pkl(self) -> Dict:
        """加载 PKL 文件"""
        with open(self.pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _parse_frames(self) -> List[HaworFrameData]:
        """解析帧数据"""
        frames = []

        # HaWoR 输出格式：pred_trans, pred_rot, pred_hand_pose, pred_betas, R_c2w, t_c2w
        # 转换为关键点格式
        num_frames = self.data['pred_trans'].shape[1]  # (2, T, 3) -> T

        # 这里需要使用 MANO 模型将参数转换为 3D 关键点
        # 为了简化，我们假设数据已经包含了关键点
        # 如果没有，需要调用 MANO 模型

        logger.info(f"加载了 {num_frames} 帧 HaWoR 数据")

        # 检查数据中是否已经有关键点
        if 'pred_keypoints_3d' in self.data:
            # 数据中已经包含关键点
            keypoints_3d_left = self.data['pred_keypoints_3d'][0]  # (T, 21, 3)
            keypoints_3d_right = self.data['pred_keypoints_3d'][1]  # (T, 21, 3)

            # 获取有效帧掩码（如果有）
            pred_valid = self.data.get('pred_valid', None)
            if pred_valid is not None:
                left_valid = pred_valid[0]  # (T,)
                right_valid = pred_valid[1]  # (T,)
            else:
                left_valid = np.ones(num_frames, dtype=bool)
                right_valid = np.ones(num_frames, dtype=bool)

            for frame_idx in range(num_frames):
                frame_data = HaworFrameData(
                    timestamp=frame_idx / self.fps,
                    frame_index=frame_idx,
                    left_hand_3d=keypoints_3d_left[frame_idx] if left_valid[frame_idx] else None,
                    right_hand_3d=keypoints_3d_right[frame_idx] if right_valid[frame_idx] else None,
                    left_hand_2d=None,
                    right_hand_2d=None,
                    left_valid=left_valid[frame_idx],
                    right_valid=right_valid[frame_idx]
                )
                frames.append(frame_data)

        else:
            logger.warning("PKL 文件中没有 pred_keypoints_3d，需要使用 MANO 模型转换")
            logger.warning("请确保 HaWoRPipeline 输出包含关键点数据")

            # 这里可以添加 MANO 转换代码
            # 但需要导入 MANO 模型，比较复杂
            # 暂时创建空数据
            for frame_idx in range(num_frames):
                frame_data = HaworFrameData(
                    timestamp=frame_idx / self.fps,
                    frame_index=frame_idx,
                    left_hand_3d=None,
                    right_hand_3d=None,
                    left_hand_2d=None,
                    right_hand_2d=None,
                    left_valid=False,
                    right_valid=False
                )
                frames.append(frame_data)

        return frames


class KeypointComparator:
    """关键点对比器"""

    def __init__(self, vr_loader: VRKeypointsLoader, hawor_loader: HaworKeypointsLoader):
        """
        初始化对比器

        Args:
            vr_loader: VR 数据加载器
            hawor_loader: HaWoR 数据加载器
        """
        self.vr_loader = vr_loader
        self.hawor_loader = hawor_loader

    @staticmethod
    def interpolate_keypoints(timestamps: List[float], keypoints: List[np.ndarray],
                          target_timestamp: float) -> Optional[np.ndarray]:
        """
        插值获取目标时间点的关键点

        Args:
            timestamps: 时间戳列表
            keypoints: 关键点列表
            target_timestamp: 目标时间戳

        Returns:
            插值后的关键点，如果超出范围则返回 None
        """
        timestamps = np.array(timestamps)
        keypoints = np.array(keypoints)

        # 检查是否超出范围
        if target_timestamp < timestamps[0] or target_timestamp > timestamps[-1]:
            return None

        # 对每个维度进行插值
        interpolated = np.zeros_like(keypoints[0])
        for i in range(keypoints.shape[1]):
            interpolated[:, i] = np.interp(
                [target_timestamp],
                timestamps,
                keypoints[:, :, i].T
            )[0]

        return interpolated

    def align_data(self, alignment_fps: float = 30.0,
                  duration: Optional[float] = None) -> ComparisonResult:
        """
        对齐两种数据的时间戳

        Args:
            alignment_fps: 对齐后的采样帧率
            duration: 对齐时长（秒），如果为 None 则使用 VR 数据的时长

        Returns:
            对齐结果
        """
        # 获取时间戳范围
        vr_start = self.vr_loader.frames_data[0].timestamp
        vr_end = self.vr_loader.frames_data[-1].timestamp
        hawor_start = self.hawor_loader.frames_data[0].timestamp
        hawor_end = self.hawor_loader.frames_data[-1].timestamp

        # 确定对齐时长
        if duration is None:
            duration = vr_end - vr_start

        # 确定对齐的时间范围（取重叠部分）
        start_time = max(vr_start, hawor_start)
        end_time = min(vr_start + duration, hawor_end)

        if start_time >= end_time:
            raise ValueError("VR 和 HaWoR 数据没有时间重叠")

        logger.info(f"对齐时间范围: {start_time:.3f}s - {end_time:.3f}s")
        logger.info(f"对齐时长: {end_time - start_time:.3f}s")

        # 生成对齐的时间戳
        num_frames = int((end_time - start_time) * alignment_fps)
        aligned_timestamps = [start_time + i / alignment_fps for i in range(num_frames)]

        # 提取 VR 时间戳和关键点
        vr_timestamps = [frame.timestamp for frame in self.vr_loader.frames_data]
        vr_keypoints = [frame.keypoints_3d for frame in self.vr_loader.frames_data]

        # 提取 HaWoR 时间戳和关键点
        hawor_timestamps = [frame.timestamp for frame in self.hawor_loader.frames_data]
        hawor_left_keypoints = [frame.left_hand_3d for frame in self.hawor_loader.frames_data]
        hawor_right_keypoints = [frame.right_hand_3d for frame in self.hawor_loader.frames_data]

        # 对齐数据
        vr_interpolated = []
        hawor_interpolated = []

        for ts in aligned_timestamps:
            # 插值 VR 关键点
            vr_kp = self.interpolate_keypoints(vr_timestamps, vr_keypoints, ts)
            vr_interpolated.append(vr_kp)

            # 插值 HaWoR 关键点（左右手）
            hawor_left = self.interpolate_keypoints(hawor_timestamps, hawor_left_keypoints, ts)
            hawor_right = self.interpolate_keypoints(hawor_timestamps, hawor_right_keypoints, ts)
            hawor_interpolated.append((hawor_left, hawor_right))

        logger.info(f"对齐完成，共 {len(aligned_timestamps)} 帧")

        return ComparisonResult(
            vr_data=self.vr_loader.frames_data,
            hawor_data=self.hawor_loader.frames_data,
            aligned_timestamps=aligned_timestamps,
            vr_interpolated=vr_interpolated,
            hawor_interpolated=hawor_interpolated
        )

    @staticmethod
    def plot_3d_keypoints(result: ComparisonResult, frame_indices: Optional[List[int]] = None,
                          save_path: Optional[str] = None, interactive: bool = True):
        """
        3D 绘制对比关键点

        Args:
            result: 对齐结果
            frame_indices: 要绘制的帧索引列表，如果为 None 则绘制所有帧（每 5 帧绘制一次）
            save_path: 保存路径，如果为 None 则不保存
            interactive: 是否显示交互式图表
        """
        if frame_indices is None:
            # 每隔 5 帧绘制一次
            frame_indices = list(range(0, len(result.aligned_timestamps), 5))

        fig = plt.figure(figsize=(15, 10))

        # 选择几个帧进行绘制
        num_plots = min(len(frame_indices), 6)
        for plot_idx, frame_idx in enumerate(frame_indices[:num_plots]):
            ax = fig.add_subplot(2, 3, plot_idx + 1, projection='3d')

            ts = result.aligned_timestamps[frame_idx]
            vr_kp = result.vr_interpolated[frame_idx]
            hawor_left, hawor_right = result.hawor_interpolated[frame_idx]

            if vr_kp is not None:
                # 绘制 VR 关键点（蓝色）
                ax.scatter(vr_kp[:, 0], vr_kp[:, 1], vr_kp[:, 2],
                          c='blue', s=50, marker='o', label='VR (12 points)', alpha=0.7)

                # 绘制 VR 关键点索引
                for i, (x, y, z) in enumerate(vr_kp):
                    ax.text(x, y, z, str(i), fontsize=8, color='blue')

            if hawor_left is not None:
                # 绘制 HaWoR 左手关键点（红色）
                ax.scatter(hawor_left[:, 0], hawor_left[:, 1], hawor_left[:, 2],
                          c='red', s=30, marker='^', label='HaWoR Left (21 points)', alpha=0.7)

            if hawor_right is not None:
                # 绘制 HaWoR 右手关键点（绿色）
                ax.scatter(hawor_right[:, 0], hawor_right[:, 1], hawor_right[:, 2],
                          c='green', s=30, marker='^', label='HaWoR Right (21 points)', alpha=0.7)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame_idx} (t={ts:.2f}s)')
            ax.legend()

            # 设置相等的坐标轴比例
            all_points = []
            if vr_kp is not None:
                all_points.append(vr_kp)
            if hawor_left is not None:
                all_points.append(hawor_left)
            if hawor_right is not None:
                all_points.append(hawor_right)

            if all_points:
                all_points = np.vstack(all_points)
                max_range = np.max(np.abs(all_points))
                ax.set_xlim([-max_range, max_range])
                ax.set_ylim([-max_range, max_range])
                ax.set_zlim([-max_range, max_range])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D 对比图已保存到: {save_path}")

        if interactive:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_single_frame_3d(result: ComparisonResult, frame_idx: int,
                           save_path: Optional[str] = None, interactive: bool = True):
        """
        3D 绘制单个帧的对比关键点（更大更详细）

        Args:
            result: 对齐结果
            frame_idx: 帧索引
            save_path: 保存路径，如果为 None 则不保存
            interactive: 是否显示交互式图表
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ts = result.aligned_timestamps[frame_idx]
        vr_kp = result.vr_interpolated[frame_idx]
        hawor_left, hawor_right = result.hawor_interpolated[frame_idx]

        if vr_kp is not None:
            # 绘制 VR 关键点（蓝色）
            ax.scatter(vr_kp[:, 0], vr_kp[:, 1], vr_kp[:, 2],
                      c='blue', s=100, marker='o', label='VR (12 points)', alpha=0.8)

            # 绘制 VR 关键点索引
            for i, (x, y, z) in enumerate(vr_kp):
                ax.text(x, y, z, f'VR-{i}', fontsize=10, color='blue', weight='bold')

        if hawor_left is not None:
            # 绘制 HaWoR 左手关键点（红色）
            ax.scatter(hawor_left[:, 0], hawor_left[:, 1], hawor_left[:, 2],
                      c='red', s=50, marker='^', label='HaWoR Left (21 points)', alpha=0.8)

            # 绘制 HaWoR 左手关键点索引
            for i, (x, y, z) in enumerate(hawor_left):
                ax.text(x, y, z, f'L-{i}', fontsize=8, color='red')

        if hawor_right is not None:
            # 绘制 HaWoR 右手关键点（绿色）
            ax.scatter(hawor_right[:, 0], hawor_right[:, 1], hawor_right[:, 2],
                      c='green', s=50, marker='^', label='HaWoR Right (21 points)', alpha=0.8)

            # 绘制 HaWoR 右手关键点索引
            for i, (x, y, z) in enumerate(hawor_right):
                ax.text(x, y, z, f'R-{i}', fontsize=8, color='green')

        ax.set_xlabel('X (相机坐标系)', fontsize=12)
        ax.set_ylabel('Y (相机坐标系)', fontsize=12)
        ax.set_zlabel('Z (相机坐标系)', fontsize=12)
        ax.set_title(f'Frame {frame_idx} (t={ts:.2f}s) - 关键点对比', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 设置相等的坐标轴比例
        all_points = []
        if vr_kp is not None:
            all_points.append(vr_kp)
        if hawor_left is not None:
            all_points.append(hawor_left)
        if hawor_right is not None:
            all_points.append(hawor_right)

        if all_points:
            all_points = np.vstack(all_points)
            max_range = np.max(np.abs(all_points)) * 1.1  # 留一点边距
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"单帧 3D 对比图已保存到: {save_path}")

        if interactive:
            plt.show()
        else:
            plt.close()

    def calculate_errors(self, result: ComparisonResult) -> Dict[str, List[float]]:
        """
        计算两种方法之间的误差

        注意：由于 VR 数据（12点）和 HaWoR 数据（21点）数量不同，
        这里只计算对应点的平均误差

        Args:
            result: 对齐结果

        Returns:
            误差字典，包含每帧的误差统计
        """
        errors = {
            'timestamps': [],
            'vr_left_error': [],  # VR 左手与 HaWoR 左手的误差
            'vr_right_error': [],  # VR 右手与 HaWoR 右手的误差
            'mean_error': []
        }

        for ts, vr_kp, (hawor_left, hawor_right) in zip(
            result.aligned_timestamps,
            result.vr_interpolated,
            result.hawor_interpolated
        ):
            errors['timestamps'].append(ts)

            # 计算 VR 前6个点（左手）与 HaWoR 左手的误差
            if vr_kp is not None and hawor_left is not None:
                vr_left = vr_kp[:6]  # VR 前6个点
                # 找到 HaWoR 左手中最接近的点（简化处理）
                hawor_left_subset = hawor_left[:6]
                error = np.mean(np.linalg.norm(vr_left - hawor_left_subset, axis=1))
                errors['vr_left_error'].append(error)
            else:
                errors['vr_left_error'].append(np.nan)

            # 计算 VR 后6个点（右手）与 HaWoR 右手的误差
            if vr_kp is not None and hawor_right is not None:
                vr_right = vr_kp[6:]  # VR 后6个点
                hawor_right_subset = hawor_right[:6]
                error = np.mean(np.linalg.norm(vr_right - hawor_right_subset, axis=1))
                errors['vr_right_error'].append(error)
            else:
                errors['vr_right_error'].append(np.nan)

            # 计算总平均误差
            if not np.isnan(errors['vr_left_error'][-1]) and not np.isnan(errors['vr_right_error'][-1]):
                errors['mean_error'].append((errors['vr_left_error'][-1] + errors['vr_right_error'][-1]) / 2)
            else:
                errors['mean_error'].append(np.nan)

        return errors


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='对比 VR 原始关键点和 HaWoR 预测关键点'
    )
    parser.add_argument(
        '--vr-json',
        type=str,
        required=True,
        help='VR 关键点 JSON 文件路径'
    )
    parser.add_argument(
        '--hawor-pkl',
        type=str,
        required=True,
        help='HaWoR Pipeline 输出 PKL 文件路径'
    )
    parser.add_argument(
        '--alignment-fps',
        type=float,
        default=30.0,
        help='对齐后的采样帧率（默认: 30.0）'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='对齐时长（秒），如果为 None 则使用 VR 数据的时长'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='comparison_output',
        help='输出目录（默认: comparison_output）'
    )
    parser.add_argument(
        '--frame-idx',
        type=int,
        default=None,
        help='绘制指定帧的单帧对比图，如果为 None 则绘制多帧'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='显示交互式图表'
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    logger.info("加载 VR 数据...")
    vr_loader = VRKeypointsLoader(args.vr_json)

    logger.info("加载 HaWoR 数据...")
    hawor_loader = HaworKeypointsLoader(args.hawor_pkl)

    # 创建对比器
    comparator = KeypointComparator(vr_loader, hawor_loader)

    # 对齐数据
    logger.info("对齐数据...")
    result = comparator.align_data(
        alignment_fps=args.alignment_fps,
        duration=args.duration
    )

    # 绘制对比图
    if args.frame_idx is not None:
        # 绘制单帧
        output_path = output_dir / f'frame_{args.frame_idx}_comparison.png'
        comparator.plot_single_frame_3d(
            result,
            frame_idx=args.frame_idx,
            save_path=str(output_path),
            interactive=args.interactive
        )
    else:
        # 绘制多帧
        output_path = output_dir / 'multi_frame_comparison.png'
        comparator.plot_3d_keypoints(
            result,
            save_path=str(output_path),
            interactive=args.interactive
        )

    # 计算误差
    logger.info("计算误差统计...")
    errors = comparator.calculate_errors(result)

    # 保存误差统计
    errors_df = {
        'timestamp': errors['timestamps'],
        'vr_left_error': errors['vr_left_error'],
        'vr_right_error': errors['vr_right_error'],
        'mean_error': errors['mean_error']
    }

    import pandas as pd
    df = pd.DataFrame(errors_df)
    errors_path = output_dir / 'errors.csv'
    df.to_csv(errors_path, index=False)
    logger.info(f"误差统计已保存到: {errors_path}")

    # 绘制误差曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(errors['timestamps'], errors['vr_left_error'], label='左手误差', color='red')
    ax1.plot(errors['timestamps'], errors['vr_right_error'], label='右手误差', color='green')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('误差 (米)')
    ax1.set_title('VR vs HaWoR 关键点误差')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(errors['timestamps'], errors['mean_error'], label='平均误差', color='blue')
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('误差 (米)')
    ax2.set_title('平均误差曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    error_plot_path = output_dir / 'error_curves.png'
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"误差曲线图已保存到: {error_plot_path}")

    if args.interactive:
        plt.show()
    else:
        plt.close()

    # 打印统计信息
    mean_left_error = np.nanmean(errors['vr_left_error'])
    mean_right_error = np.nanmean(errors['vr_right_error'])
    mean_total_error = np.nanmean(errors['mean_error'])

    logger.info("=" * 50)
    logger.info("误差统计摘要:")
    logger.info(f"  左手平均误差: {mean_left_error:.4f} 米")
    logger.info(f"  右手平均误差: {mean_right_error:.4f} 米")
    logger.info(f"  总体平均误差: {mean_total_error:.4f} 米")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
