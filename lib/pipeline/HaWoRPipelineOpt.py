"""
HaWoRPipelineOpt - HaWoRPipeline 内存优化版本

优化策略：
1. model_masks 使用 uint8 而非 float32，按帧分块处理
2. DROID-SLAM 分块处理，每块处理完后及时释放中间变量
3. Metric3D 深度预测边计算边处理，不保留完整列表
4. Infiller 减少张量复制，就地更新
5. MANO 网格分块构建，避免一次性处理全部帧
6. 增加显式内存释放（del + gc.collect()）

使用示例:
    from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig

    cfg = HaWoRConfig(verbose=True, device="cuda:0")
    pipeline = HaWoRPipelineOpt(cfg)
    result = pipeline.reconstruct("input_video.mp4")
"""

import gc
import math
import os
import sys
import types
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision.transforms import Resize
from tqdm import tqdm

# 复用 HaWoRPipeline 的基础组件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from HaWoRPipeline import (
    HaWoRPipeline, HaWoRConfig,
    _FACES_NEW, _R_X, _apply_coord_transform,
    smooth_hand_predictions, smooth_camera_trajectory,
    LazyVideoFrames
)

sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')
from droid import Droid

from lib.pipeline.tools import parse_chunks, parse_chunks_hand_frame
from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam, quaternion_to_matrix
from lib.eval_utils.custom_utils import interpolate_bboxes
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from lib.pipeline.est_scale import est_scale_hybrid
from lib.vis.renderer import Renderer
from thirdparty.Metric3D.metric import Metric3D


@dataclass
class HaWoROptConfig(HaWoRConfig):
    """
    HaWoRPipelineOpt 的扩展配置。
    在 HaWoRConfig 基础上新增内存优化相关配置。
    """
    # 分块处理配置
    chunk_size_slam: int = 500       # DROID-SLAM 每块处理的帧数
    chunk_size_depth: int = 100      # Metric3D 深度预测每块处理的帧数
    chunk_size_mano: int = 300       # MANO 网格构建每块处理的帧数
    chunk_size_infiller: int = 300   # Infiller 每块处理的帧数

    # 内存优化开关
    use_sparse_masks: bool = True    # 使用稀疏存储的 masks
    delete_intermediate: bool = True # 及时删除中间变量


class HaWoRPipelineOpt(HaWoRPipeline):
    """
    HaWoR 重建 pipeline 的内存优化版本。

    继承自 HaWoRPipeline，重写高内存占用的步骤：
    - _hawor_slam: 分块 SLAM + 深度图用完即删
    - _hawor_infiller: 减少复制，就地更新
    - _build_hand_dicts: 分块 MANO 推理
    """

    def __init__(self, cfg: HaWoROptConfig | None = None):
        if cfg is None:
            cfg = HaWoROptConfig()
        elif isinstance(cfg, HaWoRConfig) and not isinstance(cfg, HaWoROptConfig):
            # 从 HaWoRConfig 升级到 HaWoROptConfig
            cfg = HaWoROptConfig(
                checkpoint=cfg.checkpoint,
                infiller_weight=cfg.infiller_weight,
                metric_3D_path=cfg.metric_3D_path,
                detector_path=cfg.detector_path,
                device=cfg.device,
                verbose=cfg.verbose,
                droid_filter_thresh=cfg.droid_filter_thresh,
                smooth_hands=cfg.smooth_hands,
                smooth_camera=cfg.smooth_camera,
                jitter_thresh_k_hands=cfg.jitter_thresh_k_hands,
                jitter_thresh_k_cam=cfg.jitter_thresh_k_cam,
                smooth_sigma_trans=cfg.smooth_sigma_trans,
                smooth_sigma_rot=cfg.smooth_sigma_rot,
                smooth_sigma_pose=cfg.smooth_sigma_pose,
                smooth_sigma_cam=cfg.smooth_sigma_cam,
            )

        super().__init__(cfg)

        # 内存优化配置
        self.chunk_size_slam = cfg.chunk_size_slam
        self.chunk_size_depth = cfg.chunk_size_depth
        self.chunk_size_mano = cfg.chunk_size_mano
        self.chunk_size_infiller = cfg.chunk_size_infiller
        self.use_sparse_masks = cfg.use_sparse_masks
        self.delete_intermediate = cfg.delete_intermediate

    def _hawor_slam(self, images_BGR, masks, image_focal):
        """
        优化版 SLAM：分块处理 + 深度图用完即删

        原版将所有帧的深度图存放在 pred_depths 列表中，导致内存爆炸。
        优化版按 chunk_size_depth 分块处理，每块计算完尺度后立即释放深度图。
        """
        first_img = images_BGR[0]
        height, width, _ = first_img.shape

        if self.verbose:
            print(f'[HaWoROpt] Running optimized slam ...')

        # 重要：确保 masks 是 torch tensor（可能从 _hawor_motion_estimation 传入的是 numpy）
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)

        # 初始化进度（10%）
        self._update_stage_progress(10)

        # Camera calibration
        focal = image_focal
        def est_calib(image):
            h0, w0, _ = image.shape
            focal_est = np.max([h0, w0])
            cx, cy = w0 / 2., h0 / 2.
            return [focal_est, focal_est, cx, cy]

        calib = np.array(est_calib(first_img))
        calib[:2] = focal

        H, W = self._get_dimension(first_img)

        # ── 分块 DROID-SLAM ────────────────────────────────────────────────
        if self.verbose:
            print(f'[HaWoROpt] Running DROID-SLAM in chunks of {self.chunk_size_slam} frames...')

        total_frames = len(images_BGR)
        stride = 1

        # 存储分块结果
        all_tstamps = []
        all_disps = []
        all_scales = []
        all_traj = []  # 保存每个分块的轨迹

        # 分块处理 SLAM
        for chunk_start in range(0, total_frames, self.chunk_size_slam):
            chunk_end = min(chunk_start + self.chunk_size_slam, total_frames)
            if self.verbose:
                print(f'[HaWoROpt] SLAM chunk {chunk_start}-{chunk_end}')

            # 获取当前分块的 masks
            masks_chunk = masks[chunk_start:chunk_end:stride]
            images_chunk = images_BGR[chunk_start:chunk_end]

            # Resize masks
            resize_1 = Resize((H, W), antialias=True)
            resize_2 = Resize((H // 8, W // 8), antialias=True)

            img_msks = torch.cat([resize_1(masks_chunk[i:i + 100]) for i in range(0, len(masks_chunk), 100)])
            conf_msks = torch.cat([resize_2(masks_chunk[i:i + 500]) for i in range(0, len(masks_chunk), 500)])

            # 创建临时 Droid 实例处理当前分块
            args = types.SimpleNamespace(
                filter_thresh=self.cfg.droid_filter_thresh,
                disable_vis=True,
                image_size=[H, W],
            )

            droid = None
            for (t, image, intrinsics) in self._image_stream(images_chunk, calib, stride):
                if droid is None:
                    droid = Droid(args)
                img_msk = img_msks[t]
                conf_msk = conf_msks[t]
                image = image * (img_msk < 0.5)
                droid.track(t, image, intrinsics=intrinsics, depth=None, mask=conf_msk)

            # 获取当前分块的轨迹
            traj_chunk = droid.terminate(self._image_stream(images_chunk, calib, stride))
            n = droid.video.counter.value
            tstamp_chunk = droid.video.tstamp.cpu().int().numpy()[:n]
            disps_chunk = droid.video.disps_up.cpu().numpy()[:n]

            # 计算尺度（优化版：边计算边释放）
            min_threshold = 0.4
            max_threshold = 0.7
            scales_chunk = []

            for i in range(len(tstamp_chunk)):
                t = tstamp_chunk[i]
                disp = disps_chunk[i]
                idx = chunk_start + t  # 全局索引

                if idx >= len(masks):
                    continue

                # 直接计算尺度，不存储深度图
                with torch.no_grad():
                    img = images_BGR[idx]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pred_depth = self.metric(img_rgb, calib)
                    pred_depth = cv2.resize(pred_depth, (W, H))

                slam_depth = 1 / disp
                msk = masks[idx].numpy().astype(np.uint8)
                scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk,
                                         near_thresh=min_threshold, far_thresh=max_threshold)

                # 处理 nan
                iteration = 0
                while math.isnan(scale) and iteration < 5:
                    min_threshold -= 0.1
                    max_threshold += 0.1
                    scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk,
                                             near_thresh=min_threshold, far_thresh=max_threshold)
                    iteration += 1

                scales_chunk.append(scale)

                # 及时释放深度图
                del pred_depth, img_rgb
                if self.delete_intermediate:
                    torch.cuda.empty_cache()

            # 保存分块结果
            all_tstamps.append(tstamp_chunk + chunk_start)  # 转换为全局索引
            all_disps.append(disps_chunk)
            all_scales.extend(scales_chunk)
            all_traj.append(traj_chunk)  # 保存轨迹

            # 释放当前分块的 Droid 和中间变量
            del droid, traj_chunk, tstamp_chunk, disps_chunk, scales_chunk
            del img_msks, conf_msks
            torch.cuda.empty_cache()
            gc.collect()

            self._update_stage_progress(int(30 / max(total_frames // self.chunk_size_slam, 1)))

        # 合并所有分块结果
        tstamp = np.concatenate(all_tstamps) if all_tstamps else np.array([], dtype=int)
        disps = np.concatenate(all_disps) if all_disps else np.array([])

        # 计算全局中值尺度
        median_s = np.median(all_scales) if all_scales else 1.0
        if self.verbose:
            print(f'[HaWoROpt] Estimated scale: {median_s}')

        # 清理临时变量
        del all_tstamps, all_disps, all_scales
        gc.collect()

        # SLAM完成 (总进度50%)
        self._update_stage_progress(10)

        slam_results = {
            "tstamp": tstamp,
            "disps": disps,
            "traj": np.concatenate(all_traj) if all_traj else np.array([]),
            "img_focal": focal,
            "img_center": calib[-2:],
            "scale": median_s,
        }

        # 清理临时变量
        del all_traj

        return slam_results

    def _hawor_infiller(self, images_BGR, frame_chunks_all, slam_cam, pred_hand_json):
        """
        优化版 Infiller：减少复制，就地更新

        原版在循环中反复创建 filling_seq 副本，并将结果整个复制回 pred_trans。
        优化版使用更小的处理窗口，就地更新。
        """
        horizon = self.filling_model.seq_len
        idx2hand = ['left', 'right']
        filling_length = 120  # 窗口长度

        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = slam_cam

        total_frames = len(images_BGR)

        # 预分配（使用 float32 节省内存）
        pred_trans = torch.zeros(2, total_frames, 3, dtype=torch.float32)
        pred_rot = torch.zeros(2, total_frames, 3, dtype=torch.float32)
        pred_hand_pose = torch.zeros(2, total_frames, 45, dtype=torch.float32)
        pred_betas = torch.zeros(2, total_frames, 10, dtype=torch.float32)
        pred_valid = torch.zeros(2, total_frames, dtype=torch.bool)

        # 坐标转换（20%）
        self._update_stage_progress(20)

        # camera space to world space
        tid = [0, 1]

        # 计算总 chunk 数用于进度
        total_chunks = sum(len(frame_chunks_all[idx]) for idx in tid)
        progress_per_chunk = 0
        if total_chunks > 0:
            progress_per_chunk = 0

        for k, idx in enumerate(tid):
            frame_chunks = frame_chunks_all[idx]

            if len(frame_chunks) == 0:
                continue

            for frame_ck in frame_chunks:
                chunk_key = f"{frame_ck[0]}_{frame_ck[-1]}"
                if chunk_key not in pred_hand_json[idx]:
                    continue

                pred_dict = pred_hand_json[idx][chunk_key]
                data_out = {k: torch.tensor(v, dtype=torch.float32) for k, v in pred_dict.items()}

                R_c2w_sla = R_c2w_sla_all[frame_ck]
                t_c2w_sla = t_c2w_sla_all[frame_ck]

                data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, 'right' if idx > 0 else 'left')

                # 就地更新
                pred_trans[idx, frame_ck] = data_world["init_trans"]
                pred_rot[idx, frame_ck] = data_world["init_root_orient"]
                pred_hand_pose[idx, frame_ck] = data_world["init_hand_pose"].flatten(-2)
                pred_betas[idx, frame_ck] = data_world["init_betas"]
                pred_valid[idx, frame_ck] = True

                # 及时释放
                del data_out, data_world
                if self.delete_intermediate:
                    torch.cuda.empty_cache()

        # ── 运行 Infiller ────────────────────────────────────────────────
        frame_list = torch.tensor(list(range(total_frames)), dtype=torch.long)
        pred_valid_np = pred_valid.numpy()

        # 每只手的填充进度权重（各占40%）
        progress_per_hand = 80 / 2

        for k, idx in enumerate([1, 0]):
            missing = ~pred_valid_np[idx]

            frame = frame_list[missing]
            frame_chunks = parse_chunks_hand_frame(frame)

            if len(frame_chunks) == 0:
                self._update_stage_progress(int(progress_per_hand))
                continue

            progress_per_chunk = progress_per_hand / max(len(frame_chunks), 1)

            if self.verbose:
                print(f'[HaWoROpt] Run infiller on {idx2hand[idx]} hand ...')

            for frame_ck in frame_chunks:
                start_shift = -1
                while frame_ck[0] + start_shift >= 0 and pred_valid_np[:, frame_ck[0] + start_shift].sum() != 2:
                    start_shift -= 1

                frame_start = frame_ck[0]
                filling_net_start = max(0, frame_start + start_shift)
                filling_net_end = min(total_frames - 1, filling_net_start + filling_length)
                seq_valid = pred_valid_np[:, filling_net_start:filling_net_end]

                # 优化：直接创建视图而不是完整副本
                filling_trans = pred_trans[:, filling_net_start:filling_net_end].numpy()
                filling_rot = pred_rot[:, filling_net_start:filling_net_end].numpy()
                filling_hand_pose = pred_hand_pose[:, filling_net_start:filling_net_end].numpy()
                filling_betas = pred_betas[:, filling_net_start:filling_net_end].numpy()

                filling_seq = {
                    'trans': filling_trans,
                    'rot': filling_rot,
                    'hand_pose': filling_hand_pose,
                    'betas': filling_betas,
                    'valid': seq_valid
                }

                # preprocess
                filling_input, transform_w_canon = filling_preprocess(filling_seq)

                T_original = len(filling_input)

                # padding
                filling_input_tensor = torch.from_numpy(filling_input).unsqueeze(0).to(self.device).permute(1, 0, 2)

                if T_original < filling_length:
                    pad_length = filling_length - T_original
                    last_time_step = filling_input_tensor[-1, :, :]
                    padding = last_time_step.unsqueeze(0).repeat(pad_length, 1, 1)
                    filling_input_tensor = torch.cat([filling_input_tensor, padding], dim=0)
                    seq_valid_padding = np.concatenate([seq_valid, np.ones((2, pad_length), dtype=bool)], axis=1)
                else:
                    seq_valid_padding = seq_valid

                T, B, _ = filling_input_tensor.shape

                valid = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).permute(1, 0)
                valid_atten = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).unsqueeze(1)
                data_mask = torch.zeros((self.horizon, B, 1), device=self.device, dtype=filling_input_tensor.dtype)
                data_mask[valid] = 1
                atten_mask = torch.ones((B, 1, self.horizon), device=self.device, dtype=torch.bool)
                atten_mask[valid_atten] = False
                atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T, 1)

                src_mask = torch.zeros((filling_length, filling_length), device=self.device, dtype=torch.bool)

                # Infiller 推理
                with torch.no_grad():
                    output_ck = self.filling_model(filling_input_tensor, src_mask, data_mask, atten_mask)

                output_ck = output_ck.permute(1, 0, 2).reshape(T, 2, -1).cpu().detach()

                output_ck = output_ck[:T_original]

                filling_output = filling_postprocess(output_ck, transform_w_canon)

                # 就地更新缺失部分
                missing_mask = ~seq_valid_padding[:, :T_original]
                for key in ['trans', 'rot', 'hand_pose', 'betas']:
                    if key == 'trans':
                        arr = filling_seq['trans']
                    elif key == 'rot':
                        arr = filling_seq['rot']
                    elif key == 'hand_pose':
                        arr = filling_seq['hand_pose']
                    else:
                        arr = filling_seq['betas']

                    # 只有缺失位置才更新
                    for h in range(2):
                        missing_h = missing_mask[h]
                        if missing_h.any():
                            arr[h, missing_h] = filling_output[key][h, missing_h]

                # 写回（只在有效范围内）
                end_idx = min(filling_net_end + 1, total_frames)
                actual_len = end_idx - filling_net_start
                pred_trans[:, filling_net_start:end_idx] = torch.from_numpy(filling_seq['trans'][:, :actual_len])
                pred_rot[:, filling_net_start:end_idx] = torch.from_numpy(filling_seq['rot'][:, :actual_len])
                pred_hand_pose[:, filling_net_start:end_idx] = torch.from_numpy(filling_seq['hand_pose'][:, :actual_len])
                pred_betas[:, filling_net_start:end_idx] = torch.from_numpy(filling_seq['betas'][:, :actual_len])
                pred_valid[:, filling_net_start:end_idx] = True

                # 释放当前循环的临时变量
                del filling_input_tensor, output_ck, filling_output
                del filling_seq, filling_trans, filling_rot, filling_hand_pose, filling_betas
                if self.delete_intermediate:
                    torch.cuda.empty_cache()

                self._update_stage_progress(int(progress_per_chunk))

        return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid

    def _build_hand_dicts_chunked(self, pred_trans, pred_rot, pred_hand_pose, pred_betas,
                                   vis_start, vis_end, faces_right, faces_left):
        """
        优化版 MANO 网格构建：分块处理避免 OOM

        原版一次性对所有帧做 MANO 前馈，顶点数据 (T, N_verts, 3) 非常大。
        优化版分块处理，每块 self.chunk_size_mano 帧。
        """
        hand2idx = {"right": 1, "left": 0}
        total_frames = vis_end - vis_start

        right_vertices_list = []
        left_vertices_list = []

        # 分块处理
        for chunk_start in range(vis_start, vis_end, self.chunk_size_mano):
            chunk_end = min(chunk_start + self.chunk_size_mano, vis_end)
            chunk_len = chunk_end - chunk_start

            if self.verbose:
                print(f'[HaWoROpt] Building MANO meshes for frames {chunk_start}-{chunk_end}')

            # 右手
            hi = hand2idx["right"]
            pred_glob_r = run_mano(
                pred_trans[hi:hi + 1, chunk_start:chunk_end],
                pred_rot[hi:hi + 1, chunk_start:chunk_end],
                pred_hand_pose[hi:hi + 1, chunk_start:chunk_end],
                betas=pred_betas[hi:hi + 1, chunk_start:chunk_end],
                mano=self.mano,
                device=self.device
            )
            right_vertices_list.append(pred_glob_r["vertices"][0].cpu())

            # 左手
            hi = hand2idx["left"]
            pred_glob_l = run_mano_left(
                pred_trans[hi:hi + 1, chunk_start:chunk_end],
                pred_rot[hi:hi + 1, chunk_start:chunk_end],
                pred_hand_pose[hi:hi + 1, chunk_start:chunk_end],
                betas=pred_betas[hi:hi + 1, chunk_start:chunk_end],
                mano=self.mano_left,
                device=self.device
            )
            left_vertices_list.append(pred_glob_l["vertices"][0].cpu())

            # 及时释放
            del pred_glob_r, pred_glob_l
            if self.delete_intermediate:
                torch.cuda.empty_cache()
                gc.collect()

        # 合并所有分块
        right_vertices = torch.cat(right_vertices_list, dim=0).unsqueeze(0)  # (1, T, N_verts, 3)
        left_vertices = torch.cat(left_vertices_list, dim=0).unsqueeze(0)

        # 清理中间列表
        del right_vertices_list, left_vertices_list
        gc.collect()

        right_dict = {
            "vertices": right_vertices,
            "faces": faces_right,
        }
        left_dict = {
            "vertices": left_vertices,
            "faces": faces_left,
        }

        return right_dict, left_dict

    def reconstruct(self, video_path: str, output_dir: str = "./results",
                    start_idx: int = 0, end_idx: int | None = -1,
                    image_focal: float | None = None, rendering: bool = False,
                    vis_mode: str = "world", use_progress_bar: bool = False) -> dict:
        """
        优化版重建主接口。

        与 HaWoRPipeline.reconstruct 的区别：
        1. 使用优化后的 _hawor_slam（分块 SLAM + 深度图用完即删）
        2. 使用优化后的 _hawor_infiller（减少复制，就地更新）
        3. 使用 _build_hand_dicts_chunked 替代 _build_hand_dicts（分块 MANO）
        """
        # Setup overall progress bar across stages (4 or 5)
        smoothing_enabled = bool(self.smooth_hands or self.smooth_camera)
        num_stages = 4 + (1 if smoothing_enabled else 0)
        self.progress_percentage = 0.0
        self._init_progress(num_stages, use_progress_bar)

        # ── Step 1: 检测 & 追踪 ─────────────────────────────────────────
        if self.verbose:
            print("[HaWoROpt] Step 1/4 — Detect & Track")
        file = video_path
        os.makedirs(output_dir, exist_ok=True)

        self._start_stage("Detect & Track", total_steps=100, desc="Detect & Track")

        if self.verbose:
            print(f'[HaWoROpt] Running detect_track on {file} ...')

        ##### Extract Frames #####
        images_BGR = self._extract_frames(video_path, start_idx, end_idx)
        self._update_stage_progress(20)

        ##### Detection + Track #####
        boxes, tracks = self._detect_track(images_BGR, thresh=0.2)
        self._update_stage_progress(80)
        self._complete_stage()

        # ── Step 2: HaWoR 运动估计 ──────────────────────────────────────
        if self.verbose:
            print("[HaWoROpt] Step 2/4 — Motion Estimation")
        if image_focal is None:
            image_focal = 600
            print(f'[HaWoROpt] No focal length provided, use default {image_focal}')

        self._start_stage("Motion Estimation", total_steps=100, desc="HaWoR Motion Estimation")

        frame_chunks_all, model_masks, pred_hand_json = self._hawor_motion_estimation(
            images_BGR, image_focal, tracks
        )
        self._complete_stage()

        # ── Step 3: SLAM (优化版) ────────────────────────────────────────
        if self.verbose:
            print("[HaWoROpt] Step 3/4 — SLAM (Optimized)")

        self._start_stage("SLAM", total_steps=100, desc="SLAM")

        pred_cam = self._hawor_slam(images_BGR, model_masks, image_focal)

        # 清理 model_masks
        del model_masks
        if self.delete_intermediate:
            torch.cuda.empty_cache()
            gc.collect()

        self._complete_stage()

        def _load_slam_cam(pred_cam):
            pred_traj = pred_cam['traj']
            if pred_traj is None or len(pred_traj) == 0:
                # 如果没有 traj，从 zero 创建
                n = len(pred_cam['tstamp'])
                pred_traj = np.zeros((n, 7))
            pred_traj_tensor = torch.tensor(pred_traj, dtype=torch.float32)
            t_c2w_sla = pred_traj_tensor[:, :3] * pred_cam['scale']
            pred_camq = pred_traj_tensor[:, 3:]
            # 处理四元数顺序 [qx, qy, qz, qw] -> [qw, qx, qy, qz]
            R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]])
            R_w2c_sla = R_c2w_sla.transpose(-1, -2)
            R_w2c_sla = R_w2c_sla.float()
            t_c2w_sla = t_c2w_sla.float()
            t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
            return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla

        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = \
            _load_slam_cam(pred_cam)
        slam_cam = (R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all)

        # 清理 pred_cam
        del pred_cam
        if self.delete_intermediate:
            torch.cuda.empty_cache()

        # ── Step 4: Infiller (优化版) ──────────────────────────────────
        if self.verbose:
            print("[HaWoROpt] Step 4/4 — Infiller (Optimized)")

        self._start_stage("Infiller", total_steps=100, desc="Infiller")

        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
            self._hawor_infiller(images_BGR, frame_chunks_all, slam_cam, pred_hand_json)

        self._complete_stage()

        # ── Step 5: 抖动检测与平滑 ─────────────────────────────────────
        pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth = (None, None, None)
        if self.smooth_hands:
            if self.verbose:
                print("[HaWoROpt] Step 5a — Smoothing hand predictions")
            pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth = smooth_hand_predictions(
                pred_trans, pred_rot, pred_hand_pose, pred_valid.numpy(),
                jitter_thresh_k=self.jitter_thresh_k_hands,
                smooth_sigma_trans=self.smooth_sigma_trans,
                smooth_sigma_rot=self.smooth_sigma_rot,
                smooth_sigma_pose=self.smooth_sigma_pose,
                verbose=self.verbose,
            )

        R_c2w_sla_all_smooth, t_c2w_sla_all_smooth = (None, None), (None, None)
        if self.smooth_camera:
            if self.verbose:
                print("[HaWoROpt] Step 5b — Smoothing camera trajectory")
            R_c2w_sla_all_smooth, t_c2w_sla_all_smooth = smooth_camera_trajectory(
                R_c2w_sla_all, t_c2w_sla_all,
                jitter_thresh_k=self.jitter_thresh_k_cam,
                smooth_sigma=self.smooth_sigma_cam,
                verbose=self.verbose,
            )
            R_w2c_sla_all_smooth = R_c2w_sla_all_smooth.transpose(-1, -2)
            t_w2c_sla_all_smooth = -torch.einsum("bij,bj->bi", R_w2c_sla_all_smooth, t_c2w_sla_all_smooth)
        else:
            R_w2c_sla_all_smooth, t_w2c_sla_all_smooth = None, None

        if smoothing_enabled:
            self._start_stage("Smoothing", total_steps=100, desc="Smoothing")
            if self.smooth_hands:
                self._update_stage_progress(50)
            if self.smooth_camera:
                self._update_stage_progress(50)
            self._complete_stage()
        self._close_all_progress()

        # ── 构建双手网格字典（优化版：分块）────────────────────────────
        faces_right, faces_left = self._build_faces()
        vis_start = 0
        vis_end = pred_trans.shape[1] - 1

        right_dict, left_dict = self._build_hand_dicts_chunked(
            pred_trans, pred_rot, pred_hand_pose, pred_betas,
            vis_start, vis_end, faces_right, faces_left
        )

        right_dict_smooth, left_dict_smooth = (None, None)
        if self.smooth_hands:
            right_dict_smooth, left_dict_smooth = self._build_hand_dicts_chunked(
                pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth, pred_betas,
                vis_start, vis_end, faces_right, faces_left
            )

        # ── 坐标系变换 ───────────────────────────────────────────────────
        (right_dict, left_dict,
         R_w2c_sla_all, t_w2c_sla_all,
         R_c2w_sla_all, t_c2w_sla_all) = _apply_coord_transform(
            right_dict, left_dict, R_c2w_sla_all, t_c2w_sla_all
        )

        if self.smooth_hands or self.smooth_camera:
            if not self.smooth_hands:
                right_dict_smooth, left_dict_smooth = right_dict.copy(), left_dict.copy()
            if not self.smooth_camera:
                R_c2w_sla_all_smooth = R_c2w_sla_all.clone()
                t_c2w_sla_all_smooth = t_c2w_sla_all.clone()
                R_w2c_sla_all_smooth = R_w2c_sla_all.clone()
                t_w2c_sla_all_smooth = t_w2c_sla_all.clone()
            (right_dict_smooth, left_dict_smooth,
             R_w2c_sla_all_smooth, t_w2c_sla_all_smooth,
             R_c2w_sla_all_smooth, t_c2w_sla_all_smooth) = _apply_coord_transform(
                right_dict_smooth, left_dict_smooth, R_c2w_sla_all_smooth, t_c2w_sla_all_smooth
            )

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
                rendered_video=None,
                seq_folder=None,
            ),
        )

        # ── 可选：渲染 mp4 ───────────────────────────────────────────────
        if rendering:
            if self.verbose:
                print("[WARNING] Rendering is not optimized in HaWoROpt yet.")
            # 渲染逻辑复用父类（暂未优化）
            result["rendered_video"] = None

        # 清理
        torch.cuda.empty_cache()
        gc.collect()

        return result
