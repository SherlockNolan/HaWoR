import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sys
import gc
from infiller.lib.model.network import TransformerModel

sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')
from droid import Droid
import cv2
import cv2
import joblib
import numpy as np
import numpy as np
import torch
from torchvision.transforms import Resize
from natsort import natsorted
from natsort import natsorted
from tqdm import tqdm
from typing import Optional
import subprocess
if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics.models import YOLO
import numpy as np
import joblib
from scripts.scripts_test_video.hawor_video import hawor_infiller_plain, hawor_infiller
from lib.pipeline.tools import parse_chunks
from lib.models.hawor import HAWOR
from lib.models.mano_wrapper import MANO
from lib.eval_utils.custom_utils import load_slam_cam, quaternion_to_matrix
from lib.eval_utils.custom_utils import interpolate_bboxes
from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam
from lib.eval_utils.custom_utils import interpolate_bboxes
from lib.pipeline.masked_droid_slam import *
from lib.pipeline.est_scale import *
from lib.vis.renderer import Renderer
from lib.pipeline.tools import parse_chunks, parse_chunks_hand_frame
from lib.models.hawor import HAWOR
from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam
from lib.eval_utils.custom_utils import interpolate_bboxes
from lib.eval_utils.filling_utils import filling_postprocess, filling_preprocess
from lib.vis.renderer import Renderer
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from hawor.utils.process import block_print, enable_print
from infiller.lib.model.network import TransformerModel
from thirdparty.Metric3D.metric import Metric3D
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
from HaWoRPipeline import (
    HaWoRPipeline, HaWoRConfig,
    _FACES_NEW, _R_X, _apply_coord_transform,
    smooth_hand_predictions, smooth_camera_trajectory,
    LazyVideoFrames
)

# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class HaWoRPipelineOpt:
    """
    HaWoR 重建 pipeline 的核心类。

    推荐用法
    --------
    >>> cfg = HaWoRConfig(verbose=True, smooth_hands=True)
    >>> pipeline = HaWoRPipeline(cfg)
    >>> result = pipeline.reconstruct("input.mp4")

    也可以直接用关键字参数（内部会自动构造 HaWoRConfig）：
    >>> pipeline = HaWoRPipeline.from_kwargs(verbose=True, smooth_hands=False)
    """

    def __init__(self, cfg: HaWoRConfig | None = None):
        if cfg is None:
            cfg = HaWoRConfig()
        self.cfg = cfg

        # ── 路径 & 运行配置 ────────────────────────────────────────────
        self.checkpoint = cfg.checkpoint
        self.infiller_weight = cfg.infiller_weight
        self.verbose = cfg.verbose
        self.device = cfg.device
        

        # ── 加载模型 ───────────────────────────────────────────────────
        print(f"[INIT] HaWoRPipeline Using device: {self.device}")
        self.model, self.model_cfg = self._load_hawor(self.checkpoint, self.device)
        self.hand_detect_model = YOLO(cfg.detector_path)
        self.hand_detect_model = self.hand_detect_model.to(self.device)
        self.metric = Metric3D(cfg.metric_3D_path, device=self.device)
        
        MANO_cfg = {
            'DATA_DIR': '_DATA/data/',
            'MODEL_PATH': '_DATA/data/mano',
            'GENDER': 'neutral',
            'NUM_HAND_JOINTS': 15,
            'CREATE_BODY_POSE': False
        }
        mano_cfg = {k.lower(): v for k,v in MANO_cfg.items()}
        self.mano = MANO(**mano_cfg).to(self.device)
        self.mano.eval()
        
        MANO_cfg_left = {
            'DATA_DIR': '_DATA/data_left/',
            'MODEL_PATH': '_DATA/data_left/mano_left',
            'GENDER': 'neutral',
            'NUM_HAND_JOINTS': 15,
            'CREATE_BODY_POSE': False,
            'is_rhand': False
        }
        mano_cfg_left = {k.lower(): v for k,v in MANO_cfg_left.items()}
        self.mano_left = MANO(**mano_cfg_left).to(self.device)
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        self.mano_left.shapedirs[:, 0, :] *= -1
        self.mano_left.eval()

        # ── DROID-SLAM 参数 ───────────────────────────────────────────
        import types
        self.args_droid = types.SimpleNamespace(
            # 基础参数
            imagedir=None,
            calib=None,
            t0=0,
            stride=1,
            weights="weights/external/droid.pth",
            buffer=512,
            image_size=None, # 每个视频单独动态创建
            disable_vis=True,
            stereo=False,
            upsample=True,
            # DROID-SLAM 核心参数
            beta=0.3,
            filter_thresh=cfg.droid_filter_thresh,
            warmup=8,
            keyframe_thresh=4.0,
            frontend_thresh=16.0,
            frontend_window=25,
            frontend_radius=2,
            frontend_nms=1,
            backend_thresh=22.0,
            backend_radius=2,
            backend_nms=3,
            reconstruction_path=None,
        )

        # ── 抖动平滑配置（直接引用 cfg，便于运行时修改） ───────────────
        self.smooth_hands = cfg.smooth_hands
        self.smooth_camera = cfg.smooth_camera
        self.jitter_thresh_k_hands = cfg.jitter_thresh_k_hands
        self.jitter_thresh_k_cam = cfg.jitter_thresh_k_cam
        self.smooth_sigma_trans = cfg.smooth_sigma_trans
        self.smooth_sigma_rot = cfg.smooth_sigma_rot
        self.smooth_sigma_pose = cfg.smooth_sigma_pose
        self.smooth_sigma_cam = cfg.smooth_sigma_cam

        # ── Infiller 模型 ──────────────────────────────────────────────
        self.filling_model = self._load_infiller_model()
        
        # ── 进度管理 ──────────────────────────────────────────────
        self.progress_percentage = 0.0  # 0-1的实数表示总处理进度
        self._overall_pb = None  # 外层进度条对象
        self._stage_pb = None    # 当前阶段的子进度条对象
        self._current_stage_base = 0.0  # 当前阶段的起始进度（0-1）
        self._current_stage_weight = 0.0  # 当前阶段占总进度的权重

    @classmethod
    def from_kwargs(cls, **kwargs) -> "HaWoRPipeline":
        """
        便捷工厂方法，允许用关键字参数直接构造，无需手动创建 HaWoRConfig。

        示例
        ----
        >>> pipeline = HaWoRPipeline.from_kwargs(verbose=True, smooth_hands=False)
        """
        return cls(HaWoRConfig(**kwargs))

    def _load_infiller_model(self):
        if self.verbose:
            print(f"[INIT] Loading infiller model from {self.infiller_weight}...")
        weight_path = self.infiller_weight
        ckpt = torch.load(weight_path, map_location="cpu")
        pos_dim = 3
        shape_dim = 10
        num_joints = 15
        rot_dim = (num_joints + 1) * 6  # rot6d
        repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
        nhead = 8  # repr_dim = 154
        self.horizon = 120  # 用于infiller模型的参数
        filling_model = TransformerModel(seq_len=self.horizon, input_dim=repr_dim, d_model=384, nhead=nhead, d_hid=2048,
                                         nlayers=8, dropout=0.05, out_dim=repr_dim, masked_attention_stage=True)
        filling_model.to(self.device)
        filling_model.load_state_dict(ckpt['transformer_encoder_state_dict'])
        filling_model = filling_model.to(self.device)
        filling_model.eval()
        return filling_model

    def _cleanup_temp_files(self):
        """
        清理临时文件（如 memmap 文件）。
        在 reconstruct 方法结束时调用，释放磁盘空间。
        """
        # 清理 memmap 临时文件
        if hasattr(self, '_mask_tmpfile_path') and self._mask_tmpfile_path is not None:
            try:
                if os.path.exists(self._mask_tmpfile_path):
                    os.remove(self._mask_tmpfile_path)
                    if self.verbose:
                        print(f'[CLEANUP] 已删除临时文件: {self._mask_tmpfile_path}')
            except Exception as e:
                if self.verbose:
                    print(f'[CLEANUP] 删除临时文件失败: {e}')
            finally:
                self._mask_tmpfile_path = None

    # ────────────────────────────────────────────────────────────────────────
    # 进度管理辅助方法
    # ────────────────────────────────────────────────────────────────────────

    def _init_progress(self, num_stages: int, use_progress_bar: bool = True):
        """
        初始化进度管理。

        Parameters
        ----------
        num_stages : int
            总的阶段数。
        use_progress_bar : bool
            是否显示进度条。
        """
        if use_progress_bar:
            from tqdm import tqdm
            self._overall_pb = tqdm(total=num_stages, desc="Overall Progress", unit="stage")
        else:
            self._overall_pb = None
        self._current_stage = 0
        self._num_stages = num_stages

    def _start_stage(self, stage_name: str, total_steps: int = 100, desc: Optional[str] = None):
        """
        开始一个新阶段，创建子进度条。

        Parameters
        ----------
        stage_name : str
            阶段名称（用于显示）。
        total_steps : int
            该阶段的子步骤总数。
        desc : str
            子进度条的描述。
        """
        self._current_stage += 1
        self._current_stage_base = (self._current_stage - 1) / self._num_stages
        self._current_stage_weight = 1.0 / self._num_stages
        self._stage_step = 0
        self._stage_total_steps = total_steps

        if self._overall_pb is not None:
            if desc is None:
                desc = f"Stage {self._current_stage}/{self._num_stages}: {stage_name}"
            self._stage_pb = tqdm(total=total_steps, desc=desc, unit="step",
                                leave=False, position=1, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        else:
            self._stage_pb = None

    def _update_stage_progress(self, steps: int = 1):
        """
        更新当前阶段的进度。

        Parameters
        ----------
        steps : int
            完成的子步骤数。
        """
        if self._stage_pb is not None:
            self._stage_pb.update(steps)
        self._stage_step += steps

        # 更新总进度
        if self._stage_total_steps > 0:
            stage_progress = self._stage_step / self._stage_total_steps
            self.progress_percentage = self._current_stage_base + stage_progress * self._current_stage_weight

    def _complete_stage(self):
        """完成当前阶段，关闭子进度条并更新总进度条。"""
        if self._stage_pb is not None:
            # 确保子进度条填满
            remaining = self._stage_total_steps - self._stage_step
            if remaining > 0:
                self._stage_pb.update(remaining)
            self._stage_pb.close()
            self._stage_pb = None

        if self._overall_pb is not None:
            self._overall_pb.update(1)

        # 更新总进度到阶段完成状态
        self.progress_percentage = self._current_stage_base + self._current_stage_weight

    def _close_all_progress(self):
        """关闭所有进度条。"""
        if self._stage_pb is not None:
            self._stage_pb.close()
            self._stage_pb = None
        if self._overall_pb is not None:
            self._overall_pb.close()
            self._overall_pb = None

    # # ------------------------------------------------------------------
    # # 内部辅助：把 args namespace 透传给各子模块
    # # ------------------------------------------------------------------
    # def _make_args(self, video_path: str):
    #     """构造一个兼容子模块签名的简单 namespace 对象。"""
    #     import types
    #     args = types.SimpleNamespace(
    #         video_path     = video_path,
    #         input_type     = "file",
    #     )
    #     return args

    def _detect_track(self, images_BGR, thresh=0.5):

        hand_detect_model = self.hand_detect_model

        # Run
        boxes = []
        tracks = {}
        for t, image_BGR in enumerate(tqdm(images_BGR)):
            img_cv2 = image_BGR

            ### --- Detection ---
            with torch.no_grad():
                with autocast():
                    results = hand_detect_model.track(img_cv2, conf=thresh, persist=True, verbose=False)

                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    handedness = results[0].boxes.cls.cpu().numpy()
                    if not results[0].boxes.id is None:
                        track_id = results[0].boxes.id.cpu().numpy()
                    else:
                        track_id = [-1] * len(boxes)

                    boxes = np.hstack([boxes, confs[:, None]])
                    find_right = False
                    find_left = False
                    for idx, box in enumerate(boxes):
                        if track_id[idx] == -1:
                            if handedness[[idx]] > 0:
                                id = int(10000)
                            else:
                                id = int(5000)
                        else:
                            id = track_id[idx]
                        subj = dict()
                        subj['frame'] = t
                        subj['det'] = True
                        subj['det_box'] = boxes[[idx]]
                        subj['det_handedness'] = handedness[[idx]]

                        if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]] == 0):
                            if id in tracks:
                                tracks[id].append(subj)
                            else:
                                tracks[id] = [subj]

                            if handedness[[idx]] > 0:
                                find_right = True
                            elif handedness[[idx]] == 0:
                                find_left = True
        tracks = np.array(tracks, dtype=object)
        boxes = np.array(boxes, dtype=object)

        return boxes, tracks

    def _load_hawor(self, checkpoint_path, device):
        from pathlib import Path
        from hawor.configs import get_config
        model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
        model_cfg = get_config(model_cfg, update_cachedir=True)

        # Override some config values, to crop bbox correctly
        if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
            model_cfg.defrost()
            assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()

        model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, map_location="cpu") # fix: 避免并行化初始化的时候开销过大
        model = model.to(device)
        return model, model_cfg

    def _hawor_motion_estimation(self, images_BGR, image_focal, tracks_np):
        """

        Returns:
            frame_chunks_all, model_masks, pred_hand_json.
            pred_hand_json 是原来硬盘写入的json文件，主要分为idx=0,1，区分表示左右手
        """
        model = self.model
        model.eval()

        # file = video_path
        # video_root = os.path.dirname(file)
        # video = os.path.basename(file).split('.')[0]
        # img_folder = f"{video_root}/{video}/extracted_images"
        # imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))

        # tracks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', allow_pickle=True).item()
        tracks = tracks_np.item()

        tid = np.array([tr for tr in tracks])

        # if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy'):
        #     print("skip hawor motion estimation")
        #     frame_chunks_all = joblib.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
        #     return frame_chunks_all, image_focal

        if self.verbose:
            print(f'Running hawor ...')

        # 统计总chunk数用于进度更新
        total_hands = len(tid)

        left_trk = []
        right_trk = []
        for k, idx in enumerate(tid):
            trk = tracks[idx]

            valid = np.array([t['det'] for t in trk])
            is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]

            if is_right.sum() / len(is_right) < 0.5:
                left_trk.extend(trk)
            else:
                right_trk.extend(trk)
        left_trk = sorted(left_trk, key=lambda x: x['frame'])
        right_trk = sorted(right_trk, key=lambda x: x['frame'])
        final_tracks = {
            0: left_trk,
            1: right_trk
        }
        tid = [0, 1]  # 0表示左手， 1表示右手， 区分轨迹的左右手

        img = images_BGR[0]
        img_center = [img.shape[1] / 2, img.shape[0] / 2]  # w/2, h/2
        H, W = img.shape[:2]
        
        # 使用 np.memmap 替代 np.zeros，避免大数组驻留内存
        # 10000帧1080p: ~20GB 内存 → 磁盘存储，内存仅缓存访问页
        # 临时文件存储在 ./tmp/ 目录下，使用唯一前缀
        import uuid
        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        unique_prefix = f"mask_{uuid.uuid4().hex[:8]}"
        tmp_file_path = os.path.join(tmp_dir, f"{unique_prefix}.dat")
        self._mask_tmpfile_path = tmp_file_path  # 保存路径用于清理
        model_masks = np.memmap(
            tmp_file_path, 
            dtype=np.uint8, 
            mode='w+', 
            shape=(len(images_BGR), H, W)
        )
        model_masks[:] = 0  # 初始化
        if self.verbose:
            print(f'[Motion Estimation] model_masks 使用 memmap: {model_masks.shape}, 临时文件: {tmp_file_path}')

        bin_size = 128
        max_faces_per_bin = 20000
        renderer = Renderer(img.shape[1], img.shape[0], image_focal, self.device,
                            bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)

        self._update_stage_progress(10)  # 初始化完成

        # get faces
        faces = get_mano_faces()
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces_right = np.concatenate([faces, faces_new], axis=0)
        faces_left = faces_right[:, [0, 2, 1]]

        frame_chunks_all = defaultdict(list)
        pred_hand_json = {}

        # 计算每只手的进度权重
        progress_per_hand = 90 / max(total_hands, 1)  # 剩余90%分配给各个手

        for hand_idx, idx in enumerate(tid):
            print(f"tracklet {idx}:")
            pred_hand_json[idx] = {}
            trk = final_tracks[idx]

            # interp bboxes
            valid = np.array([t['det'] for t in trk])
            if valid.sum() < 2:
                self._update_stage_progress(int(progress_per_hand))  # 即使跳过也更新进度
                continue
            boxes = np.concatenate([t['det_box'] for t in trk])
            non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
            first_non_zero = non_zero_indices[0]
            last_non_zero = non_zero_indices[-1]
            boxes[first_non_zero:last_non_zero + 1] = interpolate_bboxes(boxes[first_non_zero:last_non_zero + 1])
            valid[first_non_zero:last_non_zero + 1] = True

            boxes = boxes[first_non_zero:last_non_zero + 1]
            is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
            frame = np.array([t['frame'] for t in trk])[valid]

            if is_right.sum() / len(is_right) < 0.5:
                is_right = np.zeros((len(boxes), 1))
            else:
                is_right = np.ones((len(boxes), 1))

            frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
            frame_chunks_all[idx] = frame_chunks

            if len(frame_chunks) == 0:
                self._update_stage_progress(int(progress_per_hand))
                continue

            # 计算当前手每个chunk的进度权重
            progress_per_chunk = progress_per_hand / max(len(frame_chunks), 1)

            for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
                if self.verbose:
                    print(f"inference from frame {frame_ck[0]} to {frame_ck[-1]}")
                img_ck = images_BGR[frame_ck]  # BGR格式的！
                if is_right[0] > 0:
                    do_flip = False
                else:
                    do_flip = True

                with torch.no_grad():
                    results = model.inference(img_ck, boxes_ck, img_focal=image_focal, img_center=img_center,
                                          device=self.device, do_flip=do_flip)

                data_out = {
                    "init_root_orient": results["pred_rotmat"][None, :, 0],  # (B, T, 3, 3)
                    "init_hand_pose": results["pred_rotmat"][None, :, 1:],  # (B, T, 15, 3, 3)
                    "init_trans": results["pred_trans"][None, :, 0],  # (B, T, 3)
                    "init_betas": results["pred_shape"][None, :]  # (B, T, 10)
                }

                # flip left hand
                init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
                init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
                if do_flip:
                    init_root[..., 1] *= -1
                    init_root[..., 2] *= -1
                    init_hand_pose[..., 1] *= -1
                    init_hand_pose[..., 2] *= -1
                data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
                data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

                # save camera-space results
                pred_dict = {
                    k: v.tolist() for k, v in data_out.items()
                }
                # 3. 使用片段的起始和结束帧作为 Key，避免覆盖
                chunk_key = f"{frame_ck[0]}_{frame_ck[-1]}"
                pred_hand_json[idx][chunk_key] = pred_dict
                # pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
                # if not os.path.exists(os.path.join(seq_folder, 'cam_space', str(idx))):
                #     os.makedirs(os.path.join(seq_folder, 'cam_space', str(idx)))
                # with open(pred_path, "w") as f:
                #     json.dump(pred_dict, f, indent=1)

                # get hand mask
                data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
                data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
                if do_flip:  # left
                    outputs = run_mano_left(data_out["init_trans"], data_out["init_root_orient"],
                                            data_out["init_hand_pose"], betas=data_out["init_betas"])
                else:  # right
                    outputs = run_mano(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"],
                                       betas=data_out["init_betas"])

                vertices = outputs["vertices"][0].cpu()  # (T, N, 3)
                for img_i, _ in enumerate(img_ck):
                    if do_flip:
                        faces = torch.from_numpy(faces_left).to(self.device)
                    else:
                        faces = torch.from_numpy(faces_right).to(self.device)
                    cam_R = torch.eye(3).unsqueeze(0).to(self.device)
                    cam_T = torch.zeros(1, 3).to(self.device)
                    cameras, lights = renderer.create_camera_from_cv(cam_R, cam_T)
                    verts_color = torch.tensor([0, 0, 255, 255]) / 255
                    vertices_i = vertices[[img_i]]
                    rend, mask = renderer.render_multiple(vertices_i.unsqueeze(0).to(self.device), faces,
                                                          verts_color.unsqueeze(0).to(self.device), cameras, lights)

                    model_masks[frame_ck[img_i]] += mask

                self._update_stage_progress(int(progress_per_chunk))  # 每处理一个chunk更新进度

        # 原地转换为 0/1（bool 结果），避免创建新数组
        # 使用 np.greater 原地操作，保持 memmap 特性
        np.greater(model_masks, 0, out=model_masks)
        model_masks = model_masks.astype(np.uint8)  # 这会创建新数组，但数据量已很小
        if isinstance(model_masks, np.memmap):
            model_masks.flush()
        if self.verbose:
            print(f'[Motion Estimation] model_masks 生成完成')
        # np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', model_masks)
        # joblib.dump(frame_chunks_all, f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
        return frame_chunks_all, model_masks, pred_hand_json

    def _extract_frames(self, video_path: str | Path, start_idx: int = 0, end_idx: int | None = -1, frame_step=1):
        """
        从给定视频路径构建懒加载帧序列，不一次性将所有帧读入内存。
        返回 LazyVideoFrames 对象，支持 len()、随机索引、切片和迭代。

        Args:
            video_path : 输入 mp4 路径
            start_idx  : 起始帧（包含）
            end_idx    : 终止帧（不含）；-1 或 None 表示到末尾
            frame_step : 采样步长（1 = 每帧）
        """
        lazy = LazyVideoFrames(
            video_path=str(video_path),
            frame_step=frame_step,
            start_idx=start_idx,
            end_idx=end_idx if (end_idx and end_idx != -1) else None,
        )
        if len(lazy) == 0:
            print("No frames collected, exiting")
            return lazy
        if self.verbose:
            print(f"LazyVideoFrames ready: {lazy}")
        return lazy

    def _image_stream(self, images_BGR, calib, stride, max_frame=None):
        """ Image generator for DROID """
        fx, fy, cx, cy = calib[:4]

        K = np.eye(3)
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy

        image_list = images_BGR
        image_list = image_list[::stride]

        if max_frame is not None:
            image_list = image_list[:max_frame]

        for t, image_BGR in enumerate(image_list):
            if len(calib) > 4:
                image_BGR = cv2.undistort(image_BGR, K, calib[4:])

            h0, w0, _ = image_BGR.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image_BGR, (w1, h1))
            image = image[:h1 - h1 % 8, :w1 - w1 % 8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield t, image[None], intrinsics

    @staticmethod
    def _get_dimension(image):
        """
        Get proper image dimension for DROID
        DROID-SLAM 需要将图像尺寸除以 8，所以输入必须是 8 的倍数。
        """
        # if isinstance(imagedir, list):
        #     imgfiles = imagedir
        # else:
        #     imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
        # image = cv2.imread(imgfiles[0])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        H, W, _ = image.shape
        return H, W

    def _run_slam(self, images_BGR, masks, calib, depth=None, stride=1,
                  filter_thresh=2.4, disable_vis=True):
        """
        流式处理 Masked DROID-SLAM，核心改进：
        - mask 的 resize 改为逐帧惰性计算，不再 pre-compute 全量 tensor
        - images_BGR 应为 LazyVideoFrames，支持惰性读取
        - masks 可以是 numpy memmap，按需读取单帧转换为 torch tensor
        """
        depth = None
        droid = None
        self.args_droid.filter_thresh = filter_thresh
        self.args_droid.disable_vis = disable_vis

        # 仅取 stride 后的索引，用于惰性获取对应帧的 mask
        mask_indices = list(range(0, len(masks), stride)) if stride > 1 else list(range(len(masks)))
        n_strided = len(mask_indices)

        # 流式构建 resize 变换器（只存变换器，不预计算任何 tensor）
        H, W = self._get_dimension(images_BGR[0])
        resize_1 = Resize((H, W), antialias=True)
        resize_2 = Resize((H // 8, W // 8), antialias=True)

        # 追踪当前 mask 列表位置（惰性 resize，需要手动管理索引映射）
        mask_idx = 0  # masks 原始索引
        
        total_frames = len(images_BGR)
        required_buffer = total_frames + 64
        if required_buffer > self.args_droid.buffer:
            self.args_droid.buffer = required_buffer
            if self.verbose:
                print(f"[SLAM] buffer 自动扩容: {required_buffer} (视频帧数: {total_frames})")

        for (t, image, intrinsics) in tqdm(self._image_stream(images_BGR, calib, stride),
                                            total=n_strided, desc="SLAM tracking"):
            if droid is None:
                self.args_droid.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(self.args_droid)

            # ── 惰性 resize：只对当前帧的 mask 做 resize，不分配全量数组 ──
            # masks 可能是 numpy memmap，按需读取单帧并转换为 torch tensor
            raw_idx = mask_indices[mask_idx]
            mask_frame = masks[raw_idx:raw_idx + 1]  # numpy array (1, H, W)
            # 仅对当前帧转换为 torch tensor，避免全量转换
            mask_frame_tensor = torch.from_numpy(mask_frame).float()  # (1, H, W)
            img_msk = resize_1(mask_frame_tensor)[0]   # (H, W)
            conf_msk = resize_2(mask_frame_tensor)[0]   # (H//8, W//8)
            mask_idx += 1

            image = image * (img_msk < 0.5)

            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)

        # terminate 需要重新遍历帧流进行 pose 插值填充（非关键帧）
        traj = droid.terminate(self._image_stream(images_BGR, calib, stride))
        return droid, traj

    def _hawor_slam(self, images_BGR, masks, image_focal):
        # File and folders
        # file = args.video_path
        # video_root = os.path.dirname(file)
        # video = os.path.basename(file).split('.')[0]
        # seq_folder = os.path.join(video_root, video)
        # os.makedirs(seq_folder, exist_ok=True)
        # video_folder = os.path.join(video_root, video)
        #
        # img_folder = f'{video_folder}/extracted_images'
        # imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))

        # first_img = cv2.imread(imgfiles[0])
        first_img = images_BGR[0]
        height, width, _ = first_img.shape

        if self.verbose:
            print(f'Running slam ...')

        # 初始化进度（10%）
        self._update_stage_progress(10)

        ##### Run SLAM #####
        # Use Masking
        # masks 已经是 memmap 或 numpy array，不需要转换为 torch tensor
        # 直接使用 numpy 格式进行流式处理，避免内存复制
        if isinstance(masks, np.memmap):
            if self.verbose:
                print(f'[SLAM] 使用 memmap masks: {masks.shape}, dtype: {masks.dtype}')
        else:
            if self.verbose:
                print(f'[SLAM] 使用 numpy masks: {masks.shape}, dtype: {masks.dtype}')
        print(f'Masks shape: {masks.shape}')

        # Camera calibration (intrinsics) for SLAM
        focal = image_focal

        def est_calib(image):
            """
            estimate calibration 估计相机内参
            """
            h0, w0, _ = image.shape
            focal = np.max([h0, w0])
            cx, cy = w0 / 2., h0 / 2.
            calib = [focal, focal, cx, cy]
            return calib

        calib = np.array(est_calib(first_img))  # [focal, focal, cx, cy]
        center = calib[2:]
        calib[:2] = focal

        # Droid-slam with masking (30%)
        self._update_stage_progress(30)
        droid, traj = self._run_slam(images_BGR, masks=masks, calib=calib)
        n = droid.video.counter.value
        tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
        disps = droid.video.disps_up.cpu().numpy()[:n]
        if self.verbose:
            print('DBA errors:', droid.backend.errors)

        del droid  # 一条视频单独使用一个droid实例！
        torch.cuda.empty_cache()

        # SLAM完成 (总进度50%)
        self._update_stage_progress(10)

        # Estimate scale
        # block_print() # 临时禁用打印输出，避免 Metric3D 模型加载或推理时产生过多日志干扰
        # 加载metric3D改到模型公用
        # enable_print()

        min_threshold = 0.4
        max_threshold = 0.7

        if self.verbose:
            print('Predicting Metric Depth ...')

        H, W = self._get_dimension(first_img)
        n = len(tstamp)
        # 深度预测(35%) + 尺度估计(5%) 合并为流式循环，用完即丢 depth buffer
        total_depth_and_scale = n
        progress_per_step = 40 / max(total_depth_and_scale, 1)

        scales_ = []
        for i, t in enumerate(tstamp):
            img = images_BGR[t]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_depth = self.metric(img_rgb, calib)
            pred_depth = cv2.resize(pred_depth, (W, H))
            slam_depth = 1 / disps[i]

            # Estimate scene scale — pred_depth 用完即释放，不驻留全程列表
            # masks 可能是 numpy memmap，直接访问即可，不需要 .numpy()
            msk = masks[t]
            if hasattr(msk, 'astype'):
                msk = msk.astype(np.uint8)
            else:
                msk = np.array(msk, dtype=np.uint8)
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=int(min_threshold),
                                     far_thresh=int(max_threshold))
            while math.isnan(scale):
                min_threshold -= 0.1
                max_threshold += 0.1
                scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=int(min_threshold),
                                         far_thresh=int(max_threshold))
            scales_.append(scale)

            # pred_depth 离开作用域后由 Python GC 释放，不再占用内存
            del pred_depth, slam_depth
            self._update_stage_progress(int(progress_per_step))

        median_s = np.median(scales_)
        if self.verbose:
            print(f"estimated scale: {median_s}")

        # Save results
        # os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
        # save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
        # np.savez(save_path,
        #          tstamp=tstamp, disps=disps, traj=traj,
        #          img_focal=focal, img_center=calib[-2:],
        #          scale=median_s)
        slam_results = {
            "tstamp": tstamp,
            "disps": disps,
            "traj": traj,
            "img_focal": focal,
            "img_center": calib[-2:],
            "scale": median_s,
        }
        return slam_results

    def _hawor_infiller(self, images_BGR, frame_chunks_all, slam_cam, pred_hand_json):
        # load infiller
        # file = args.video_path
        # video_root = os.path.dirname(file)
        # video = os.path.basename(file).split('.')[0]
        # seq_folder = os.path.join(video_root, video)
        # img_folder = f"{video_root}/{video}/extracted_images"

        # Previous steps
        # imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))

        horizon = self.filling_model.seq_len

        idx2hand = ['left', 'right']
        filling_length = 120

        # fpath = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = slam_cam

        pred_trans = torch.zeros(2, len(images_BGR), 3)
        pred_rot = torch.zeros(2, len(images_BGR), 3)
        pred_hand_pose = torch.zeros(2, len(images_BGR), 45)
        pred_betas = torch.zeros(2, len(images_BGR), 10)
        pred_valid = torch.zeros((2, pred_betas.size(1)))

        # 坐标转换（20%）
        self._update_stage_progress(20)

        # camera space to world space
        tid = [0, 1]
        total_chunks = 0
        for idx in tid:
            frame_chunks = frame_chunks_all[idx]
            total_chunks += len(frame_chunks)

        progress_per_chunk = 0
        if total_chunks > 0:
            progress_per_chunk = 0  # 坐标转换已经在上面的_update_stage_progress(20)中处理

        for k, idx in enumerate(tid):
            frame_chunks = frame_chunks_all[idx]

            if len(frame_chunks) == 0:
                continue

            for frame_ck in frame_chunks:
                # print(f"from frame {frame_ck[0]} to {frame_ck[-1]}")
                # pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
                # with open(pred_path, "r") as f:
                #     pred_dict = json.load(f)
                chunk_key = f"{frame_ck[0]}_{frame_ck[-1]}"
                pred_dict = pred_hand_json[idx][chunk_key]
                data_out = {
                    k: torch.tensor(v) for k, v in pred_dict.items()
                }

                R_c2w_sla = R_c2w_sla_all[frame_ck]
                t_c2w_sla = t_c2w_sla_all[frame_ck]

                data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, 'right' if idx > 0 else 'left')

                pred_trans[[idx], frame_ck] = data_world["init_trans"]
                pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
                pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
                pred_betas[[idx], frame_ck] = data_world["init_betas"]
                pred_valid[[idx], frame_ck] = 1

        # runing fillingnet for this video
        frame_list = torch.tensor(list(range(pred_trans.size(1))))
        pred_valid = (pred_valid > 0).numpy()

        # 计算每只手的填充进度权重（各占40%）
        progress_per_hand = 80 / 2  # 总共80%，两只手平分

        for k, idx in enumerate([1, 0]):
            missing = ~pred_valid[idx]

            frame = frame_list[missing]
            frame_chunks = parse_chunks_hand_frame(frame)  # 这边进行帧分块处理

            if len(frame_chunks) == 0:
                self._update_stage_progress(int(progress_per_hand))  # 即使没有缺失帧也更新进度
                continue

            progress_per_chunk = progress_per_hand / max(len(frame_chunks), 1)

            if self.verbose:
                print(f"run infiller on {idx2hand[idx]} hand ...")
            for frame_ck in frame_chunks:
                start_shift = -1
                while frame_ck[0] + start_shift >= 0 and pred_valid[:, frame_ck[0] + start_shift].sum() != 2:
                    start_shift -= 1  # Shift to find the previous valid frame as start
                if self.verbose:
                    print(
                        f"run infiller on frame {frame_ck[0] + start_shift} to frame {min(len(images_BGR) - 1, frame_ck[0] + start_shift + filling_length)}")

                frame_start = frame_ck[0]
                filling_net_start = max(0, frame_start + start_shift)
                filling_net_end = min(len(images_BGR) - 1, filling_net_start + filling_length)
                seq_valid = pred_valid[:, filling_net_start:filling_net_end]
                filling_seq = {}
                filling_seq['trans'] = pred_trans[:, filling_net_start:filling_net_end].numpy()
                filling_seq['rot'] = pred_rot[:, filling_net_start:filling_net_end].numpy()
                filling_seq['hand_pose'] = pred_hand_pose[:, filling_net_start:filling_net_end].numpy()
                filling_seq['betas'] = pred_betas[:, filling_net_start:filling_net_end].numpy()
                filling_seq['valid'] = seq_valid
                # preprocess (convert to canonical + slerp)
                filling_input, transform_w_canon = filling_preprocess(filling_seq)
                src_mask = torch.zeros((filling_length, filling_length), device=self.device).type(torch.bool)
                src_mask = src_mask.to(self.device)
                filling_input = torch.from_numpy(filling_input).unsqueeze(0).to(self.device).permute(1, 0,
                                                                                                     2)  # (seq_len, B, in_dim)
                T_original = len(filling_input)
                filling_length = 120
                if T_original < filling_length:
                    pad_length = filling_length - T_original
                    last_time_step = filling_input[-1, :, :]
                    padding = last_time_step.unsqueeze(0).repeat(pad_length, 1, 1)
                    filling_input = torch.cat([filling_input, padding], dim=0)
                    seq_valid_padding = np.ones((2, filling_length - T_original))
                    seq_valid_padding = np.concatenate([seq_valid, seq_valid_padding], axis=1)
                else:
                    seq_valid_padding = seq_valid

                T, B, _ = filling_input.shape

                valid = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).permute(1, 0)  # (T,B)
                valid_atten = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).unsqueeze(1)  # (B,1,T)
                data_mask = torch.zeros((self.horizon, B, 1), device=self.device, dtype=filling_input.dtype)
                data_mask[valid] = 1
                atten_mask = torch.ones((B, 1, self.horizon),
                                        device=self.device, dtype=torch.bool)
                atten_mask[valid_atten] = False
                atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T, 1)  # (B,1,T,T)

                output_ck = self.filling_model(filling_input, src_mask, data_mask, atten_mask)

                output_ck = output_ck.permute(1, 0, 2).reshape(T, 2, -1).cpu().detach()  # two hands

                output_ck = output_ck[:T_original]

                filling_output = filling_postprocess(output_ck, transform_w_canon)

                # repalce the missing prediciton with infiller output
                filling_seq['trans'][~seq_valid] = filling_output['trans'][~seq_valid]
                filling_seq['rot'][~seq_valid] = filling_output['rot'][~seq_valid]
                filling_seq['hand_pose'][~seq_valid] = filling_output['hand_pose'][~seq_valid]
                filling_seq['betas'][~seq_valid] = filling_output['betas'][~seq_valid]

                pred_trans[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['trans'][:])
                pred_rot[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['rot'][:])
                pred_hand_pose[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['hand_pose'][:])
                pred_betas[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['betas'][:])
                pred_valid[:, filling_net_start:filling_net_end] = 1

                self._update_stage_progress(int(progress_per_chunk))  # 每处理一个chunk更新进度
        # save_path = os.path.join(seq_folder, "world_space_res.pth")
        # joblib.dump([pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid], save_path)
        return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid

    def _build_faces(self,):
        """构建右手 / 左手的 face 数组。"""
        faces_base = self.mano.faces
        faces_right = np.concatenate([faces_base, _FACES_NEW], axis=0)
        faces_left = faces_right[:, [0, 2, 1]]
        return faces_right, faces_left
    
    def _build_hand_dicts(self, pred_trans, pred_rot, pred_hand_pose, pred_betas,
                      vis_start, vis_end, faces_right, faces_left):
        """
        用 MANO 模型前向推理，得到双手的顶点字典。

        返回 (right_dict, left_dict)，其中每个 dict 包含：
            - 'vertices': (1, T, N, 3) Tensor
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
            mano=self.mano,
            device=self.device
        )
        right_dict = {
            "vertices": pred_glob_r["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
            "faces": faces_right,
        }

        # 左手
        hi = hand2idx["left"]
        pred_glob_l = run_mano_left(
            pred_trans[hi:hi + 1, vis_start:vis_end],
            pred_rot[hi:hi + 1, vis_start:vis_end],
            pred_hand_pose[hi:hi + 1, vis_start:vis_end],
            betas=pred_betas[hi:hi + 1, vis_start:vis_end],
            mano=self.mano_left,
            device=self.device
        )
        left_dict = {
            "vertices": pred_glob_l["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
            "faces": faces_left,
        }

        return right_dict, left_dict
    # ------------------------------------------------------------------
    # 重建主接口
    # ------------------------------------------------------------------
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
        ) -> dict:
        """
        对单个视频执行完整重建 pipeline。

        Args:
            video_path : str
                输入视频路径。
            output_dir : str | None
                渲染视频的输出目录。若为 None，则默认保存在与视频同名的子目录中。只有rendering开启的时候有效
            rendering : bool
                是否渲染并合成 mp4 视频。
            vis_mode : str
                渲染视角：'world' 或 'cam'。
        Returns:
            result : dict
                包含以下键：
                - 'pred_trans'    : Tensor (2, T, 3)  — 双手平移
                - 'pred_rot'      : Tensor (2, T, 3)  — 双手根方向（轴角）
                - 'pred_hand_pose': Tensor (2, T, 45) — 双手姿态（轴角展开）
                - 'pred_betas'    : Tensor (2, T, 10) — 形状参数
                - 'pred_valid'    : Tensor (2, T)     — 有效帧掩码
                - 'right_dict'    : dict              — 右手顶点 & 面片（变换后）
                - 'left_dict'     : dict              — 左手顶点 & 面片（变换后）
                - 'R_c2w'         : Tensor (T, 3, 3)  — 相机→世界旋转
                - 't_c2w'         : Tensor (T, 3)     — 相机→世界平移
                - 'R_w2c'         : Tensor (T, 3, 3)  — 世界→相机旋转
                - 't_w2c'         : Tensor (T, 3)     — 世界→相机平移
                - 'image_focal'     : float             — 使用的焦距
                - 'rendered_video': str | None        — 渲染视频路径（仅 rendering=True 时有值）
        """

        # Setup overall progress bar across stages (4 or 5)
        smoothing_enabled = bool(self.smooth_hands or self.smooth_camera)
        num_stages = 4 + (1 if smoothing_enabled else 0)
        self.progress_percentage = 0.0
        self._init_progress(num_stages, use_progress_bar)

        # ── Step 1: 检测 & 追踪 ─────────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 1/4 — Detect & Track")
        file = video_path
        os.makedirs(output_dir, exist_ok=True)

        # Step 1 的子步骤：提取帧(20%) + 检测追踪(80%)
        self._start_stage("Detect & Track", total_steps=100, desc="Detect & Track")

        if self.verbose:
            print(f'Running detect_track on {file} ...')

        ##### Extract Frames #####
        images_BGR = self._extract_frames(video_path, start_idx, end_idx)
        self._update_stage_progress(20)  # 提取帧完成

        ##### Detection + Track #####
        if self.verbose:
            print('Detect and Track ...')
        boxes, tracks = self._detect_track(images_BGR, thresh=0.2)
        self._update_stage_progress(80)  # 检测追踪完成
        self._complete_stage()

        # ── Step 2: HaWoR 运动估计 ──────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 2/4 — Motion Estimation")
        if image_focal is None:
            image_focal = 600
            print(f'No focal length provided, use default {image_focal}')

        # Step 2 的子步骤：初始化(10%) + 左手推理(45%) + 右手推理(45%)
        self._start_stage("Motion Estimation", total_steps=100, desc="HaWoR Motion Estimation")

        frame_chunks_all, model_masks, pred_hand_json = self._hawor_motion_estimation(
            images_BGR, image_focal, tracks
        )
        self._complete_stage()

        # ── Step 3: SLAM ─────────────────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 3/4 — SLAM")

        # Step 3 的子步骤：SLAM运行(40%) + 深度预测(40%) + 尺度估计(20%)
        self._start_stage("SLAM", total_steps=100, desc="SLAM")

        pred_cam = self._hawor_slam(images_BGR, model_masks, image_focal)
        self._complete_stage()

        def _load_slam_cam(pred_cam):
            pred_traj = pred_cam['traj']
            t_c2w_sla = torch.tensor(pred_traj[:, :3]) * pred_cam['scale']
            pred_camq = torch.tensor(pred_traj[:, 3:])
            R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]])
            R_w2c_sla = R_c2w_sla.transpose(-1, -2)
            # 将 R 和 t 都转换为 float32 精度
            R_w2c_sla = R_w2c_sla.float()
            t_c2w_sla = t_c2w_sla.float()
            t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
            return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla

        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = \
            _load_slam_cam(pred_cam)
        slam_cam = (R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all)

        # ── Step 4: Infiller ─────────────────────────────────────────────
        print("[HaWoR] Step 4/4 — Infiller")

        # Step 4 的子步骤：坐标转换(20%) + 左手填充(40%) + 右手填充(40%)
        self._start_stage("Infiller", total_steps=100, desc="Infiller")

        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
            self._hawor_infiller(images_BGR, frame_chunks_all, slam_cam, pred_hand_json)
        # pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        #     hawor_infiller_plain(args, start_idx, end_idx, frame_chunks_all)

        self._complete_stage()

        # ── Step 5: 抖动检测与平滑 ────────────────────────────────────────
        is_smooth_failed = False
        try:
            pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth = (None, None, None)
            if self.smooth_hands:
                if self.verbose:
                    print("[HaWoR] Step 5a — Smoothing hand predictions (MAD jitter detection + Gaussian)")
                pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth = smooth_hand_predictions(
                    pred_trans, pred_rot, pred_hand_pose, pred_valid,
                    jitter_thresh_k=self.jitter_thresh_k_hands,
                    smooth_sigma_trans=self.smooth_sigma_trans,
                    smooth_sigma_rot=self.smooth_sigma_rot,
                    smooth_sigma_pose=self.smooth_sigma_pose,
                    verbose=self.verbose,
                )
 
            R_c2w_sla_all_smooth, t_c2w_sla_all_smooth, R_w2c_sla_all_smooth, t_w2c_sla_all_smooth = (None, None, None, None)
            if self.smooth_camera:
                if self.verbose:
                    print("[HaWoR] Step 5b — Smoothing camera trajectory (MAD jitter detection + Gaussian)")
                R_c2w_sla_all_smooth, t_c2w_sla_all_smooth = smooth_camera_trajectory(
                    R_c2w_sla_all, t_c2w_sla_all,
                    jitter_thresh_k=self.jitter_thresh_k_cam,
                    smooth_sigma=self.smooth_sigma_cam,
                    verbose=self.verbose,
                )
                R_w2c_sla_all_smooth = R_c2w_sla_all_smooth.transpose(-1, -2)
                t_w2c_sla_all_smooth = -torch.einsum("bij,bj->bi", R_w2c_sla_all_smooth, t_c2w_sla_all_smooth)
        except Exception as e:
            if self.verbose:
                print(f"[HaWoR] Warning: smooth_hand_predictions failed: {e}")
            pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth = (None, None, None)
            R_c2w_sla_all_smooth, t_c2w_sla_all_smooth, R_w2c_sla_all_smooth, t_w2c_sla_all_smooth = (None, None, None, None)
            is_smooth_failed = True
        
        # Update overall progress after smoothing stage if enabled
        if smoothing_enabled:
            self._start_stage("Smoothing", total_steps=100, desc="Smoothing")
            if self.smooth_hands:
                self._update_stage_progress(50)  # 手部平滑占一半
            if self.smooth_camera:
                self._update_stage_progress(50)  # 相机平滑占一半
            self._complete_stage()
        self._close_all_progress()

        # ── 构建双手网格字典 ─────────────────────────────────────────────
        faces_right, faces_left = self._build_faces()
        vis_start = 0
        vis_end = pred_trans.shape[1] - 1

        right_dict, left_dict = self._build_hand_dicts(
            pred_trans, pred_rot, pred_hand_pose, pred_betas,
            vis_start, vis_end, faces_right, faces_left
        )
        right_dict_smooth, left_dict_smooth = (None, None)
        if self.smooth_hands and not is_smooth_failed:
            right_dict_smooth, left_dict_smooth = self._build_hand_dicts(
                pred_trans_smooth, pred_rot_smooth, pred_hand_pose_smooth, pred_betas,
                vis_start, vis_end, faces_right, faces_left
            )
            

        # ── 坐标系变换 ───────────────────────────────────────────────────
        (right_dict, left_dict,
         R_w2c_sla_all, t_w2c_sla_all,
         R_c2w_sla_all, t_c2w_sla_all) = _apply_coord_transform(
            right_dict, left_dict, R_c2w_sla_all, t_c2w_sla_all
        )
         
        if self.smooth_hands or self.smooth_camera and not is_smooth_failed:
            if not self.smooth_hands:
                right_dict_smooth, left_dict_smooth = right_dict.copy(), left_dict.copy()
            if not self.smooth_camera:
                R_c2w_sla_all_smooth, t_c2w_sla_all_smooth = R_c2w_sla_all.clone(), t_c2w_sla_all.clone()
                R_w2c_sla_all_smooth, t_w2c_sla_all_smooth = R_w2c_sla_all.clone(), t_w2c_sla_all.clone()
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
            
            smooth_hand_enabled=self.smooth_hands if not is_smooth_failed else False,
            smooth_camera_enabled=self.smooth_camera if not is_smooth_failed else False,
            
            # 使用了smooth的结果
            smoothed_result = dict(
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
            ) if not is_smooth_failed else None,
        )

        # ── 可选：渲染 mp4 ───────────────────────────────────────────────
        if rendering:
            if self.verbose:
                print("[WARNING] You are trying to render the video which calls the original old API and it will generate temp frame images to seq_folder which degrades the performance. It's recommended to use rendering in testing single video only.")
            # collect image files 仅用于接口的统一。渲染接口需要把每一帧拆成images，太多处了，不想改了。为了接口统一，暂时生成images
            file = video_path
            root = os.path.dirname(file)
            seq = os.path.basename(file).split('.')[0]

            seq_folder = f'{root}/{seq}'
            img_folder = f'{seq_folder}/extracted_images'
            os.makedirs(seq_folder, exist_ok=True)
            os.makedirs(img_folder, exist_ok=True)
            print(f'Running detect_track on {file} ...')

            ##### Extract Frames #####
            def extract_frames(video_path, output_folder):
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                command = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vf', 'fps=30',
                    '-start_number', '0',
                    os.path.join(output_folder, '%04d.jpg')
                ]

                subprocess.run(command, check=True)
            imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
            if len(imgfiles) > 0:
                print("Skip extracting frames")
            else:
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
            result["seq_folder"]=seq_folder
            result["rendered_video"] = rendered_video
            if rendered_video:
                print(f"[HaWoR] Rendered video saved to: {rendered_video}")
        
        # 清理临时文件（memmap）
        self._cleanup_temp_files()
        
        # 在处理循环的末尾添加
        torch.cuda.empty_cache()
        gc.collect()
        return result

    # ------------------------------------------------------------------
    # 渲染（可选）
    # ------------------------------------------------------------------
    def _render(
            self,
            result: dict,
            imgfiles: list,
            vis_start: int,
            vis_end: int,
            output_dir: str,
            vis_mode: str,
            video_path: str = "",
    ) -> str | None:
        """
        调用公共渲染函数 render_hand_results，返回生成的 mp4 路径（失败则返回 None）。
        """
        from lib.vis.run_vis2 import render_hand_results

        video_stem = os.path.splitext(os.path.basename(video_path))[0] if video_path else "output"
        image_names = list(imgfiles[vis_start:vis_end])

        print(f"[HaWoR] Rendering frames {vis_start} → {vis_end}  (mode={vis_mode})")

        return render_hand_results(
            left_dict=result["left_dict"],
            right_dict=result["right_dict"],
            image_names=image_names,
            img_focal=result["img_focal"],
            output_dir=output_dir,
            vis_start=vis_start,
            vis_end=vis_end,
            vis_mode=vis_mode,
            R_c2w=result["R_c2w"],
            t_c2w=result["t_c2w"],
            R_w2c=result["R_w2c"],
            t_w2c=result["t_w2c"],
            video_stem=video_stem,
        )

