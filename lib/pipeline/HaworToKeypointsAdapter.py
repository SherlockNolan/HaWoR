"""
HaWoR结果到关键点格式的适配器

将HaWoRPipeline的reconstruct结果转换为相机坐标系下的3D keypoints、2D keypoints
以及MANO模型表示的手部数据，按照每一帧的方式存储。
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """视频元数据"""
    video_path: str
    width: int
    height: int
    frame_count: int
    fps: Optional[float] = None


class HaworToKeypointsAdapter:
    """
    将HaWoR pipeline的输出转换为关键点格式

    主要功能：
    1. 从MANO参数计算3D关键点
    2. 将世界坐标系转换到相机坐标系
    3. 投影3D关键点到2D图像平面
    4. 按帧组织数据
    """

    def __init__(self, video_metadata: VideoMetadata, use_smoothed: bool = True):
        """
        初始化适配器

        Args:
            video_metadata: 视频元数据（包含宽高信息用于投影）
            use_smoothed: 是否使用平滑后的结果
        """
        self.metadata = video_metadata
        self.use_smoothed = use_smoothed

        # 导入必要的工具函数
        from hawor.utils.process import run_mano, run_mano_left
        from hawor.utils.geometry import perspective_projection

        self.run_mano = run_mano
        self.run_mano_left = run_mano_left
        self.perspective_projection = perspective_projection

    def convert(self, hawor_result: Dict) -> List[Dict]:
        """
        将HaWoR pipeline结果转换为按帧存储的关键点格式

        Args:
            hawor_result: HaWoRPipeline.reconstruct()的返回结果

        Returns:
            List[Dict]: 每个元素是一帧的数据，包含左右手的关键点和相机位姿
        """
        # 选择使用平滑或原始结果
        if self.use_smoothed and hawor_result.get('smoothed_result'):
            result = hawor_result['smoothed_result']
        else:
            result = hawor_result

        # 提取基本参数
        pred_trans = result['pred_trans']       # (2, T, 3)
        pred_rot = result['pred_rot']           # (2, T, 3)
        pred_hand_pose = result['pred_hand_pose']  # (2, T, 45)
        pred_betas = result['pred_betas']       # (2, T, 10)
        pred_valid = result.get('pred_valid')   # (2, T) - 可选

        R_c2w = result['R_c2w']                # (T, 3, 3)
        t_c2w = result['t_c2w']                # (T, 3)
        R_w2c = result['R_w2c']                # (T, 3, 3)
        t_w2c = result['t_w2c']                # (T, 3)

        img_focal = result['img_focal']         # float

        # 获取帧数
        num_frames = pred_trans.shape[1]

        # 为每一帧准备数据
        frames_data = []

        for frame_idx in range(num_frames):
            frame_dict = {
                'frame_idx': frame_idx,
                'hands': [],
                'camera_pose': self._extract_camera_pose(
                    R_w2c[frame_idx], t_w2c[frame_idx],
                    R_c2w[frame_idx], t_c2w[frame_idx]
                )
            }

            # 处理左手 (index=0)
            left_hand = self._process_single_hand(
                hand_idx=0,
                frame_idx=frame_idx,
                is_right=False,
                pred_trans=pred_trans,
                pred_rot=pred_rot,
                pred_hand_pose=pred_hand_pose,
                pred_betas=pred_betas,
                pred_valid=pred_valid,
                R_w2c=R_w2c[frame_idx:frame_idx+1],
                img_focal=img_focal,
                width=self.metadata.width,
                height=self.metadata.height
            )
            if left_hand is not None:
                frame_dict['hands'].append(left_hand)

            # 处理右手 (index=1)
            right_hand = self._process_single_hand(
                hand_idx=1,
                frame_idx=frame_idx,
                is_right=True,
                pred_trans=pred_trans,
                pred_rot=pred_rot,
                pred_hand_pose=pred_hand_pose,
                pred_betas=pred_betas,
                pred_valid=pred_valid,
                R_w2c=R_w2c[frame_idx:frame_idx+1],
                img_focal=img_focal,
                width=self.metadata.width,
                height=self.metadata.height
            )
            if right_hand is not None:
                frame_dict['hands'].append(right_hand)

            frames_data.append(frame_dict)

        return frames_data

    def _extract_camera_pose(self, R_w2c: torch.Tensor, t_w2c: torch.Tensor,
                           R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> Dict:
        """
        提取相机位姿信息

        Args:
            R_w2c: 世界到相机的旋转矩阵 (3, 3)
            t_w2c: 世界到相机的平移 (3,)
            R_c2w: 相机到世界的旋转矩阵 (3, 3)
            t_c2w: 相机到世界的平移 (3,)

        Returns:
            Dict: 包含相机位姿的字典
        """
        return {
            'R_w2c': R_w2c.cpu().numpy(),
            't_w2c': t_w2c.cpu().numpy(),
            'R_c2w': R_c2w.cpu().numpy(),
            't_c2w': t_c2w.cpu().numpy(),
        }

    def _process_single_hand(self, hand_idx: int, frame_idx: int, is_right: bool,
                           pred_trans: torch.Tensor, pred_rot: torch.Tensor,
                           pred_hand_pose: torch.Tensor, pred_betas: torch.Tensor,
                           pred_valid: Optional[torch.Tensor],
                           R_w2c: torch.Tensor, img_focal: float,
                           width: int, height: int) -> Optional[Dict]:
        """
        处理单只手的关键点生成

        Args:
            hand_idx: 手的索引 (0=左手, 1=右手)
            frame_idx: 帧索引
            is_right: 是否为右手
            pred_trans: 平移参数 (2, T, 3)
            pred_rot: 旋转参数 (2, T, 3)
            pred_hand_pose: 手部姿态 (2, T, 45)
            pred_betas: 形状参数 (2, T, 10)
            pred_valid: 有效帧掩码 (2, T) - 可选
            R_w2c: 世界到相机的旋转 (1, 3, 3)
            img_focal: 焦距
            width: 图像宽度
            height: 图像高度

        Returns:
            Optional[Dict]: 手部数据字典，如果手部无效则返回None
        """
        # 检查手部是否有效
        if pred_valid is not None:
            if pred_valid[hand_idx, frame_idx] == 0:
                return None

        # 提取当前手和帧的参数
        # 注意：MANO模型需要 (B, T, ...) 格式，这里我们构建 (1, 1, ...)
        trans = pred_trans[hand_idx:hand_idx+1, frame_idx:frame_idx+1]  # (1, 1, 3)
        rot = pred_rot[hand_idx:hand_idx+1, frame_idx:frame_idx+1]     # (1, 1, 3)
        hand_pose = pred_hand_pose[hand_idx:hand_idx+1, frame_idx:frame_idx+1]  # (1, 1, 45)
        betas = pred_betas[hand_idx:hand_idx+1, frame_idx:frame_idx+1]  # (1, 1, 10)

        # 将手部姿态从 (1, 1, 45) 转换为 (1, 1, 15, 3)
        hand_pose = hand_pose.view(1, 1, 15, 3)

        # 运行MANO模型获取3D关键点和顶点
        if is_right:
            mano_output = self.run_mano(trans, rot, hand_pose, betas=betas)
        else:
            mano_output = self.run_mano_left(trans, rot, hand_pose, betas=betas)

        # 提取3D关键点 - 在相机坐标系中
        # mano_output['joints'] 的形状为 (1, 1, 21, 3) - 21个关节点
        keypoints_3d_camera = mano_output['joints'][0, 0].cpu().numpy()  # (21, 3)

        # 提取顶点 - 在相机坐标系中
        vertices_3d_camera = mano_output['vertices'][0, 0].cpu().numpy()  # (778, 3)

        # 将3D关键点转换为PyTorch tensor用于投影
        keypoints_3d_tensor = mano_output['joints'][0:1, 0:1].view(1, 21, 3)  # (1, 21, 3)

        # 投影到2D
        # 构建相机参数
        translation = torch.zeros(1, 3).to(keypoints_3d_tensor.device)  # 相机坐标系下不需要平移
        focal_length = torch.tensor([[img_focal, img_focal]]).to(keypoints_3d_tensor.device)  # (1, 2)
        camera_center = torch.tensor([[width/2, height/2]]).to(keypoints_3d_tensor.device)  # (1, 2)
        rotation = torch.eye(3).unsqueeze(0).to(keypoints_3d_tensor.device)  # (1, 3, 3)

        # 计算投影
        keypoints_2d = self.perspective_projection(
            points=keypoints_3d_tensor,
            translation=translation,
            focal_length=focal_length,
            camera_center=camera_center,
            rotation=rotation
        )

        keypoints_2d = keypoints_2d[0].cpu().numpy()  # (21, 2)

        # 构建手部数据字典
        hand_dict = {
            "is_right": int(is_right),
            "pred_keypoints_3d": keypoints_3d_camera,
            "pred_vertices_3d": vertices_3d_camera,
            "pred_keypoints_2d": keypoints_2d,
            "pred_cam_t_full": translation.cpu().numpy(),
            "scaled_focal_length": img_focal,
            "mano_params": {
                "trans": trans[0, 0].cpu().numpy(),
                "rot": rot[0, 0].cpu().numpy(),
                "hand_pose": hand_pose[0, 0].cpu().numpy(),
                "betas": betas[0, 0].cpu().numpy(),
            }
        }

        return hand_dict

    @staticmethod
    def get_video_metadata(video_path: str) -> VideoMetadata:
        """
        获取视频元数据

        Args:
            video_path: 视频文件路径

        Returns:
            VideoMetadata: 视频元数据对象
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        return VideoMetadata(
            video_path=video_path,
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps
        )

    @staticmethod
    def format_for_vla(frames_data: List[Dict]) -> Dict:
        """
        格式化为VLA模型预训练所需的格式

        Args:
            frames_data: convert()方法的输出

        Returns:
            Dict: VLA模型所需的数据格式
        """
        vla_data = {
            'metadata': {
                'num_frames': len(frames_data),
                'hands_per_frame': [len(frame['hands']) for frame in frames_data],
            },
            'frames': []
        }

        for frame_data in frames_data:
            frame_entry = {
                'frame_idx': frame_data['frame_idx'],
                'camera_pose': frame_data['camera_pose'],
                'hands': frame_data['hands']
            }
            vla_data['frames'].append(frame_entry)

        return vla_data


def convert_hawor_to_keypoints(hawor_result: Dict, video_path: str,
                              use_smoothed: bool = True) -> List[Dict]:
    """
    便捷函数：将HaWoR结果转换为关键点格式

    Args:
        hawor_result: HaWoRPipeline.reconstruct()的返回结果
        video_path: 原始视频路径（用于获取尺寸信息）
        use_smoothed: 是否使用平滑后的结果

    Returns:
        List[Dict]: 按帧存储的关键点数据
    """
    metadata = HaworToKeypointsAdapter.get_video_metadata(video_path)
    adapter = HaworToKeypointsAdapter(metadata, use_smoothed=use_smoothed)
    return adapter.convert(hawor_result)


def convert_and_format_for_vla(hawor_result: Dict, video_path: str,
                               use_smoothed: bool = True) -> Dict:
    """
    便捷函数：将HaWoR结果转换为VLA模型格式

    Args:
        hawor_result: HaWoRPipeline.reconstruct()的返回结果
        video_path: 原始视频路径
        use_smoothed: 是否使用平滑后的结果

    Returns:
        Dict: VLA模型格式化后的数据
    """
    frames_data = convert_hawor_to_keypoints(hawor_result, video_path, use_smoothed)
    return HaworToKeypointsAdapter.format_for_vla(frames_data)
