"""
HaWoR Pipeline with Keypoints Extraction

在 HaWoRPipeline 的基础上直接提取 3D/2D keypoints，从 MANO 模型的中间结果直接获取，
避免重复调用 MANO 模型。

Author: Claude
Date: 2025-03
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# 导入 HaWoRPipeline 的基础组件
from lib.pipeline.HaWoRPipeline import (
    HaWoRPipeline,
    HaWoRConfig,
    LazyVideoFrames,
    _FACES_NEW,
    _R_X,
    smooth_hand_predictions,
    smooth_camera_trajectory,
    _mad_detect_jitter,
    _interp_translation,
    _interp_rotation_aa,
    _gaussian_smooth_1d,
)
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.geometry import perspective_projection


@dataclass
class VideoMetadata:
    """视频元数据"""
    video_path: str
    width: int
    height: int
    frame_count: int
    fps: Optional[float] = None


# ---------------------------------------------------------------------------
# 构建 hand dicts 时同时提取 keypoints
# ---------------------------------------------------------------------------

def _build_faces():
    """构建右手 / 左手的 face 数组。"""
    faces_base = get_mano_faces()
    faces_right = np.concatenate([faces_base, _FACES_NEW], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_right, faces_left


def _build_hand_dicts_with_keypoints(pred_trans, pred_rot, pred_hand_pose, pred_betas,
                                      vis_start, vis_end, faces_right, faces_left):
    """
    用 MANO 模型前向推理，得到双手的顶点和关键点字典。

    返回 (right_dict, left_dict)，其中每个 dict 包含：
        - 'vertices': (1, T, N, 3) Tensor
        - 'joints': (1, T, 21, 3) Tensor  - 21个关节点
        - 'faces': np.ndarray
    """
    hand2idx = {"right": 1, "left": 0}

    # 右手
    hi = hand2idx["right"]
    pred_glob_r = run_mano(
        pred_trans[hi:hi + 1, vis_start:vis_end],
        pred_rot[hi:hi + 1, vis_start:vis_end],
        pred_hand_pose[hi:hi + 1, vis_start:vis_end],
        betas=pred_betas[hi:hi + 1, vis_start:vis_end],
    )
    right_dict = {
        "vertices": pred_glob_r["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "joints": pred_glob_r["joints"][0].unsqueeze(0),      # (1, T, 21, 3)
        "faces": faces_right,
    }

    # 左手
    hi = hand2idx["left"]
    pred_glob_l = run_mano_left(
        pred_trans[hi:hi + 1, vis_start:vis_end],
        pred_rot[hi:hi + 1, vis_start:vis_end],
        pred_hand_pose[hi:hi + 1, vis_start:vis_end],
        betas=pred_betas[hi:hi + 1, vis_start:vis_end],
    )
    left_dict = {
        "vertices": pred_glob_l["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "joints": pred_glob_l["joints"][0].unsqueeze(0),      # (1, T, 21, 3)
        "faces": faces_left,
    }

    return right_dict, left_dict


def _apply_coord_transform_with_keypoints(right_dict, left_dict,
                                           R_c2w_sla_all, t_c2w_sla_all):
    """
    将双手顶点、关键点和相机位姿统一变换到渲染坐标系（与 demo.py 保持一致）。

    返回 (right_dict, left_dict, R_w2c_sla_all, t_w2c_sla_all,
           R_c2w_sla_all, t_c2w_sla_all)
    """
    R_x = _R_X

    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    # 变换顶点
    left_dict["vertices"] = torch.einsum(
        "ij,btnj->btni", R_x, left_dict["vertices"].cpu()
    )
    right_dict["vertices"] = torch.einsum(
        "ij,btnj->btni", R_x, right_dict["vertices"].cpu()
    )

    # 变换关键点
    if "joints" in left_dict:
        left_dict["joints"] = torch.einsum(
            "ij,btnj->btni", R_x, left_dict["joints"].cpu()
        )
    if "joints" in right_dict:
        right_dict["joints"] = torch.einsum(
            "ij,btnj->btni", R_x, right_dict["joints"].cpu()
        )

    return (right_dict, left_dict,
            R_w2c_sla_all, t_w2c_sla_all,
            R_c2w_sla_all, t_c2w_sla_all)


class HaworPipelineKeypoints(HaWoRPipeline):
    """
    带有 Keypoints 提取功能的 HaWoR Pipeline。

    继承自 HaWoRPipeline，重写 reconstruct 方法，在 MANO 模型推理过程中
    直接获取 joints（关键点），避免重复调用 MANO 模型。

    返回的结果中直接包含按帧组织的 keypoints 数据。
    """

    def __init__(self, cfg: HaWoRConfig | None = None):
        """
        初始化 Pipeline。

        Args:
            cfg: HaWoR 配置对象，如果为 None 则使用默认配置
        """
        super().__init__(cfg)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "HaworPipelineKeypoints":
        """
        便捷工厂方法，允许用关键字参数直接构造。

        示例:
            pipeline = HaworPipelineKeypoints.from_kwargs(verbose=True, smooth_hands=True)
        """
        return cls(HaWoRConfig(**kwargs))

    def _project_keypoints_to_2d(
        self,
        keypoints_3d: torch.Tensor,
        img_focal: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        将 3D 关键点投影到 2D 图像平面。

        Args:
            keypoints_3d: 3D 关键点 (1, 21, 3) 或 (21, 3)，在相机坐标系中
            img_focal: 焦距
            width: 图像宽度
            height: 图像高度

        Returns:
            np.ndarray: 2D 关键点 (21, 2)
        """
        if keypoints_3d.dim() == 2:
            keypoints_3d = keypoints_3d.unsqueeze(0)

        device = keypoints_3d.device
        translation = torch.zeros(1, 3).to(device)
        focal_length = torch.tensor([[img_focal, img_focal]]).to(device)
        camera_center = torch.tensor([[width/2, height/2]]).to(device)
        rotation = torch.eye(3).unsqueeze(0).to(device)

        keypoints_2d = perspective_projection(
            points=keypoints_3d,
            translation=translation,
            focal_length=focal_length,
            camera_center=camera_center,
            rotation=rotation
        )

        return keypoints_2d[0].cpu().numpy()

    def _compute_keypoints_from_mano_output(
        self,
        right_dict: Dict,
        left_dict: Dict,
        R_w2c: torch.Tensor,
        t_w2c: torch.Tensor,
        pred_valid: torch.Tensor,
        img_focal: float,
        width: int,
        height: int,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        从 MANO 输出中直接提取 keypoints 数据。

        Args:
            right_dict: 右手字典，包含 'vertices', 'joints', 'faces'
            left_dict: 左手字典，包含 'vertices', 'joints', 'faces'
            R_w2c: 世界到相机的旋转 (T, 3, 3)
            t_w2c: 世界到相机的平移 (T, 3)
            pred_valid: 有效帧掩码 (2, T)
            img_focal: 焦距
            width: 图像宽度
            height: 图像高度
            verbose: 是否打印详细信息

        Returns:
            List[Dict]: 每帧的 keypoints 数据列表
        """
        from tqdm import tqdm

        # 获取帧数
        T = right_dict['joints'].shape[1]  # (1, T, 21, 3)
        frames_data = []

        if verbose:
            print(f"[Keypoints] Extracting keypoints for {T} frames from MANO output...")

        # 提取关键点和顶点
        right_joints = right_dict['joints'][0]    # (T, 21, 3)
        right_vertices = right_dict['vertices'][0]  # (T, 778, 3)
        left_joints = left_dict['joints'][0]       # (T, 21, 3)
        left_vertices = left_dict['vertices'][0]   # (T, 778, 3)

        for frame_idx in tqdm(range(T), desc="Extracting keypoints", disable=not verbose):
            frame_dict = {
                'frame_idx': frame_idx,
                'hands': [],
                'camera_pose': {
                    'R_w2c': R_w2c[frame_idx].cpu().numpy(),
                    't_w2c': t_w2c[frame_idx].cpu().numpy(),
                }
            }

            # 处理左手 (index=0)
            if pred_valid[0, frame_idx] > 0:
                # 世界坐标系的关键点和顶点
                joints_world = left_joints[frame_idx]  # (21, 3)
                vertices_world = left_vertices[frame_idx]  # (778, 3)

                # 转换到相机坐标系
                R = R_w2c[frame_idx].cpu().numpy()
                t = t_w2c[frame_idx].cpu().numpy()
                joints_camera = (R @ joints_world.T).T + t
                vertices_camera = (R @ vertices_world.T).T + t

                # 投影到 2D
                joints_for_proj = torch.from_numpy(joints_camera).float()
                if torch.cuda.is_available():
                    joints_for_proj = joints_for_proj.cuda()
                keypoints_2d = self._project_keypoints_to_2d(
                    joints_for_proj.unsqueeze(0), img_focal, width, height
                )

                left_hand_dict = {
                    "is_right": 0,
                    "pred_keypoints_3d": joints_camera,           # 相机坐标系
                    "pred_keypoints_3d_world": joints_world.cpu().numpy() if torch.is_tensor(joints_world) else joints_world,  # 世界坐标系
                    "pred_vertices_3d": vertices_camera,          # 相机坐标系
                    "pred_vertices_3d_world": vertices_world.cpu().numpy() if torch.is_tensor(vertices_world) else vertices_world,  # 世界坐标系
                    "pred_keypoints_2d": keypoints_2d,
                    "scaled_focal_length": img_focal,
                }
                frame_dict['hands'].append(left_hand_dict)

            # 处理右手 (index=1)
            if pred_valid[1, frame_idx] > 0:
                # 世界坐标系的关键点和顶点
                joints_world = right_joints[frame_idx]  # (21, 3)
                vertices_world = right_vertices[frame_idx]  # (778, 3)

                # 转换到相机坐标系
                R = R_w2c[frame_idx].cpu().numpy()
                t = t_w2c[frame_idx].cpu().numpy()
                joints_camera = (R @ joints_world.T).T + t
                vertices_camera = (R @ vertices_world.T).T + t

                # 投影到 2D
                joints_for_proj = torch.from_numpy(joints_camera).float()
                if torch.cuda.is_available():
                    joints_for_proj = joints_for_proj.cuda()
                keypoints_2d = self._project_keypoints_to_2d(
                    joints_for_proj.unsqueeze(0), img_focal, width, height
                )

                right_hand_dict = {
                    "is_right": 1,
                    "pred_keypoints_3d": joints_camera,           # 相机坐标系
                    "pred_keypoints_3d_world": joints_world.cpu().numpy() if torch.is_tensor(joints_world) else joints_world,  # 世界坐标系
                    "pred_vertices_3d": vertices_camera,          # 相机坐标系
                    "pred_vertices_3d_world": vertices_world.cpu().numpy() if torch.is_tensor(vertices_world) else vertices_world,  # 世界坐标系
                    "pred_keypoints_2d": keypoints_2d,
                    "scaled_focal_length": img_focal,
                }
                frame_dict['hands'].append(right_hand_dict)

            frames_data.append(frame_dict)

        return frames_data

    def reconstruct(
        self,
        video_path: str,
        output_dir: str = "./results",
        start_idx: int = 0,
        end_idx: int | None = -1,
        image_focal: float | None = None,
        rendering: bool = False,
        vis_mode: str = "world",
        use_progress_bar: bool = False,
        compute_keypoints: bool = True,
        use_smoothed_keypoints: bool = True,
    ) -> dict:
        """
        对单个视频执行完整重建 pipeline，并直接从 MANO 输出中提取 keypoints。

        Args:
            video_path : str
                输入视频路径。
            output_dir : str
                渲染视频的输出目录。只有 rendering 开启时有效。
            start_idx : int
                起始帧索引。
            end_idx : int | None
                结束帧索引，-1 表示到视频末尾。
            image_focal : float | None
                焦距，如果为 None 则使用默认值 600。
            rendering : bool
                是否渲染并合成 mp4 视频。
            vis_mode : str
                渲染视角：'world' 或 'cam'。
            use_progress_bar : bool
                是否显示进度条。
            compute_keypoints : bool
                是否计算 keypoints 数据。
            use_smoothed_keypoints : bool
                是否使用平滑后的结果计算 keypoints。

        Returns:
            result : dict
                包含以下键：
                - 所有 HaWoRPipeline.reconstruct() 返回的键
                - 'keypoints' : List[Dict] - 按帧组织的 keypoints 数据
                - 'keypoints_metadata' : Dict - keypoints 相关的元数据
        """
        # Setup overall progress bar across stages
        smoothing_enabled = bool(self.smooth_hands or self.smooth_camera)
        num_stages = 4 + (1 if smoothing_enabled else 0) + (1 if compute_keypoints else 0)
        self.progress_percentage = 0.0
        overall_pb = None
        if use_progress_bar:
            overall_pb = tqdm(total=num_stages, desc="Overall Progress", unit="stage")

        # ── Step 1: 检测 & 追踪 ─────────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 1/4 — Detect & Track")
        import os
        file = video_path
        os.makedirs(output_dir, exist_ok=True)
        if self.verbose:
            print(f'Running detect_track on {file} ...')

        ##### Extract Frames #####
        images_BGR = self._extract_frames(video_path, start_idx, end_idx)

        ##### Detection + Track #####
        if self.verbose:
            print('Detect and Track ...')
        boxes, tracks = self._detect_track(images_BGR, thresh=0.2)
        if overall_pb is not None:
            self.progress_percentage += 1/num_stages
            overall_pb.update(1)

        # ── Step 2: HaWoR 运动估计 ──────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 2/4 — Motion Estimation")
        if image_focal is None:
            image_focal = 600
            print(f'No focal length provided, use default {image_focal}')
        frame_chunks_all, model_masks, pred_hand_json = self._hawor_motion_estimation(
            images_BGR, image_focal, tracks
        )
        if overall_pb is not None:
            self.progress_percentage += 1/num_stages
            overall_pb.update(1)

        # ── Step 3: SLAM ─────────────────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 3/4 — SLAM")
        pred_cam = self._hawor_slam(images_BGR, model_masks, image_focal)
        if overall_pb is not None:
            self.progress_percentage += 1/num_stages
            overall_pb.update(1)

        from lib.eval_utils.custom_utils import quaternion_to_matrix

        def _load_slam_cam(pred_cam):
            pred_traj = pred_cam['traj']
            t_c2w_sla = torch.tensor(pred_traj[:, :3]) * pred_cam['scale']
            pred_camq = torch.tensor(pred_traj[:, 3:])
            R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]])
            R_w2c_sla = R_c2w_sla.transpose(-1, -2)
            R_w2c_sla = R_w2c_sla.float()
            t_c2w_sla = t_c2w_sla.float()
            t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
            return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla

        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = \
            _load_slam_cam(pred_cam)
        slam_cam = (R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all)

        # ── Step 4: Infiller ─────────────────────────────────────────────
        print("[HaWoR] Step 4/4 — Infiller")
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
            self._hawor_infiller(images_BGR, frame_chunks_all, slam_cam, pred_hand_json)

        if overall_pb is not None:
            self.progress_percentage += 1/num_stages
            overall_pb.update(1)

        # ── Step 5: 抖动检测与平滑 ────────────────────────────────────────
        pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth = (None, None, None)
        R_c2w_sla_all_smooth, t_c2w_sla_all_smooth = (None, None)

        if self.smooth_hands:
            if self.verbose:
                print("[HaWoR] Step 5a — Smoothing hand predictions")
            pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth = smooth_hand_predictions(
                pred_trans, pred_rot, pred_hand_pose, pred_valid,
                jitter_thresh_k=self.jitter_thresh_k_hands,
                smooth_sigma_trans=self.smooth_sigma_trans,
                smooth_sigma_rot=self.smooth_sigma_rot,
                smooth_sigma_pose=self.smooth_sigma_pose,
                verbose=self.verbose,
            )

        if self.smooth_camera:
            if self.verbose:
                print("[HaWoR] Step 5b — Smoothing camera trajectory")
            R_c2w_sla_all_smooth, t_c2w_sla_all_smooth = smooth_camera_trajectory(
                R_c2w_sla_all, t_c2w_sla_all,
                jitter_thresh_k=self.jitter_thresh_k_cam,
                smooth_sigma=self.smooth_sigma_cam,
                verbose=self.verbose,
            )
            R_w2c_sla_all_smooth = R_c2w_sla_all_smooth.transpose(-1, -2)
            t_w2c_sla_all_smooth = -torch.einsum("bij,bj->bi", R_w2c_sla_all_smooth, t_c2w_sla_all_smooth)
        else:
            R_c2w_sla_all_smooth, t_c2w_sla_all_smooth = R_c2w_sla_all.clone(), t_c2w_sla_all.clone()
            R_w2c_sla_all_smooth, t_w2c_sla_all_smooth = R_w2c_sla_all.clone(), t_w2c_sla_all.clone()

        if overall_pb is not None:
            if smoothing_enabled:
                self.progress_percentage += 1/num_stages
                overall_pb.update(1)

        # ── 构建双手网格字典（包含 keypoints）─────────────────────────────
        faces_right, faces_left = _build_faces()
        vis_start = 0
        vis_end = pred_trans.shape[1]

        # 原始结果的 hand dicts
        right_dict, left_dict = _build_hand_dicts_with_keypoints(
            pred_trans, pred_rot, pred_hand_pose, pred_betas,
            vis_start, vis_end, faces_right, faces_left
        )

        # 平滑结果的 hand dicts
        right_dict_smooth, left_dict_smooth = (None, None)
        if self.smooth_hands:
            right_dict_smooth, left_dict_smooth = _build_hand_dicts_with_keypoints(
                pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth, pred_betas,
                vis_start, vis_end, faces_right, faces_left
            )

        # ── 坐标系变换 ───────────────────────────────────────────────────
        (right_dict, left_dict,
         R_w2c_sla_all, t_w2c_sla_all,
         R_c2w_sla_all, t_c2w_sla_all) = _apply_coord_transform_with_keypoints(
            right_dict, left_dict, R_c2w_sla_all, t_c2w_sla_all
        )

        if self.smooth_hands or self.smooth_camera:
            if not self.smooth_hands:
                right_dict_smooth, left_dict_smooth = right_dict.copy(), left_dict.copy()
            (right_dict_smooth, left_dict_smooth,
             R_w2c_sla_all_smooth, t_w2c_sla_all_smooth,
             R_c2w_sla_all_smooth, t_c2w_sla_all_smooth) = _apply_coord_transform_with_keypoints(
                right_dict_smooth, left_dict_smooth, R_c2w_sla_all_smooth, t_c2w_sla_all_smooth
            )

        # ── 提取 keypoints ───────────────────────────────────────────────────
        frames_data = None
        keypoints_metadata = None

        if compute_keypoints:
            # 获取视频尺寸
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # 选择使用原始结果还是平滑结果
            if use_smoothed_keypoints and (self.smooth_hands or self.smooth_camera):
                source_right_dict = right_dict_smooth
                source_left_dict = left_dict_smooth
                source_R_w2c = R_w2c_sla_all_smooth
                source_t_w2c = t_w2c_sla_all_smooth
                if self.verbose:
                    print("[Keypoints] Using smoothed results for keypoints extraction")
            else:
                source_right_dict = right_dict
                source_left_dict = left_dict
                source_R_w2c = R_w2c_sla_all
                source_t_w2c = t_w2c_sla_all
                if self.verbose:
                    print("[Keypoints] Using original results for keypoints extraction")

            frames_data = self._compute_keypoints_from_mano_output(
                right_dict=source_right_dict,
                left_dict=source_left_dict,
                R_w2c=source_R_w2c,
                t_w2c=source_t_w2c,
                pred_valid=pred_valid,
                img_focal=image_focal,
                width=width,
                height=height,
                verbose=self.verbose,
            )

            keypoints_metadata = {
                'num_frames': len(frames_data),
                'hands_per_frame': [len(frame['hands']) for frame in frames_data],
                'image_width': width,
                'image_height': height,
                'focal_length': image_focal,
                'use_smoothed': use_smoothed_keypoints and (self.smooth_hands or self.smooth_camera),
            }

            if overall_pb is not None:
                self.progress_percentage += 1/num_stages
                overall_pb.update(1)

        if overall_pb is not None:
            overall_pb.close()

        # ── 整理返回结果 ─────────────────────────────────────────────────
        result = dict(
            pred_trans=pred_trans,
            pred_rot=pred_rot,
            pred_hand_pose=pred_hand_pose,
            pred_betas=pred_betas,
            pred_valid=pred_valid,
            right_dict=right_dict,
            left_dict=left_dict,
            R_c2w=R_c2w_sla_all,
            t_c2w=t_c2w_sla_all,
            R_w2c=R_w2c_sla_all,
            t_w2c=t_w2c_sla_all,
            img_focal=image_focal,
            rendered_video=None,
            seq_folder=None,

            smooth_hand_enabled=self.smooth_hands,
            smooth_camera_enabled=self.smooth_camera,

            # 平滑结果
            smoothed_result=dict(
                pred_trans=pred_trans_smooth,
                pred_rot=pred_rot_smooth,
                pred_hand_pose=pred_hand_pose_smooth,
                pred_betas=pred_betas,
                pred_valid=pred_valid,
                right_dict=right_dict_smooth,
                left_dict=left_dict_smooth,
                R_c2w=R_c2w_sla_all_smooth,
                t_c2w=t_c2w_sla_all_smooth,
                R_w2c=R_w2c_sla_all_smooth,
                t_w2c=t_w2c_sla_all_smooth,
                img_focal=image_focal,
            ),

            # keypoints 数据
            keypoints=frames_data,
            keypoints_metadata=keypoints_metadata,
        )

        # ── 可选：渲染 mp4 ───────────────────────────────────────────────
        if rendering:
            if self.verbose:
                print("[WARNING] Rendering is enabled which may slow down processing.")
            # 调用父类的渲染逻辑
            from natsort import natsorted
            from glob import glob
            import subprocess

            root = os.path.dirname(file)
            seq = os.path.basename(file).split('.')[0]
            seq_folder = f'{root}/{seq}'
            img_folder = f'{seq_folder}/extracted_images'
            os.makedirs(seq_folder, exist_ok=True)
            os.makedirs(img_folder, exist_ok=True)

            def extract_frames(video_path, output_folder):
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                command = [
                    'ffmpeg', '-i', video_path, '-vf', 'fps=30',
                    '-start_number', '0',
                    os.path.join(output_folder, '%04d.jpg')
                ]
                subprocess.run(command, check=True)

            imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
            if len(imgfiles) == 0:
                _ = extract_frames(file, img_folder)
            imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))

            result_render = result['smoothed_result'] if self.smooth_hands or self.smooth_camera else result
            rendered_video = self._render(
                result=result_render,
                imgfiles=imgfiles,
                vis_start=vis_start,
                vis_end=vis_end,
                output_dir=output_dir,
                vis_mode=vis_mode,
                video_path=video_path,
            )
            result["seq_folder"] = seq_folder
            result["rendered_video"] = rendered_video

        return result

    @staticmethod
    def format_for_vla(frames_data: List[Dict]) -> Dict:
        """
        格式化为 VLA 模型预训练所需的格式。

        Args:
            frames_data: reconstruct() 返回的 keypoints 数据

        Returns:
            Dict: VLA 模型所需的数据格式
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


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def process_video_to_keypoints(
    video_path: str,
    output_dir: str = "./results",
    use_smoothed: bool = True,
    **pipeline_kwargs
) -> Dict:
    """
    便捷函数：处理视频并返回 keypoints 数据。

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        use_smoothed: 是否使用平滑后的结果
        **pipeline_kwargs: 传递给 HaworPipelineKeypoints 的额外参数

    Returns:
        Dict: 包含 keypoints 数据的结果字典

    示例:
        result = process_video_to_keypoints("input.mp4", verbose=True)
        keypoints = result['keypoints']
        for frame_data in keypoints:
            for hand in frame_data['hands']:
                print(f"3D keypoints: {hand['pred_keypoints_3d'].shape}")
    """
    pipeline = HaworPipelineKeypoints.from_kwargs(**pipeline_kwargs)
    result = pipeline.reconstruct(
        video_path=video_path,
        output_dir=output_dir,
        compute_keypoints=True,
        use_smoothed_keypoints=use_smoothed,
    )
    return result


# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    使用示例：

    python -c "
    from lib.pipeline.HaworPipelineKeypoints import process_video_to_keypoints

    result = process_video_to_keypoints('path/to/video.mp4', verbose=True)
    print(f'Processed {len(result[\"keypoints\"])} frames')

    for frame in result['keypoints'][:3]:  # 打印前3帧
        print(f'Frame {frame[\"frame_idx\"]}: {len(frame[\"hands\"])} hands')
        for hand in frame['hands']:
            print(f'  is_right={hand[\"is_right\"]}')
            print(f'  3D keypoints shape: {hand[\"pred_keypoints_3d\"].shape}')
            print(f'  2D keypoints shape: {hand[\"pred_keypoints_2d\"].shape}')
    "
    """
    pass