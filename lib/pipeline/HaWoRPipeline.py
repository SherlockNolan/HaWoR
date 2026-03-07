import math
import os
import sys
from collections import defaultdict
from pathlib import Path
import sys

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
from ultralytics import YOLO
import numpy as np
import joblib
from scripts.scripts_test_video.hawor_video import hawor_infiller_plain, hawor_infiller
from lib.pipeline.tools import parse_chunks
from lib.models.hawor import HAWOR
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



# ---------------------------------------------------------------------------
# 面片常量（与 demo.py 保持一致）
# ---------------------------------------------------------------------------
_FACES_NEW = np.array([
    [92, 38, 234],
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
    [78, 108, 79],
])

# 绕 X 轴旋转 180° 的矩阵（坐标系对齐用）
_R_X = torch.tensor([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=torch.float32)


def _build_faces():
    """构建右手 / 左手的 face 数组。"""
    faces_base = get_mano_faces()
    faces_right = np.concatenate([faces_base, _FACES_NEW], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_right, faces_left


def _build_hand_dicts(pred_trans, pred_rot, pred_hand_pose, pred_betas,
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
        pred_trans[hi:hi+1, vis_start:vis_end],
        pred_rot[hi:hi+1, vis_start:vis_end],
        pred_hand_pose[hi:hi+1, vis_start:vis_end],
        betas=pred_betas[hi:hi+1, vis_start:vis_end],
    )
    right_dict = {
        "vertices": pred_glob_r["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "faces": faces_right,
    }

    # 左手
    hi = hand2idx["left"]
    pred_glob_l = run_mano_left(
        pred_trans[hi:hi+1, vis_start:vis_end],
        pred_rot[hi:hi+1, vis_start:vis_end],
        pred_hand_pose[hi:hi+1, vis_start:vis_end],
        betas=pred_betas[hi:hi+1, vis_start:vis_end],
    )
    left_dict = {
        "vertices": pred_glob_l["vertices"][0].unsqueeze(0),  # (1, T, N, 3)
        "faces": faces_left,
    }

    return right_dict, left_dict


def _apply_coord_transform(right_dict, left_dict,
                           R_c2w_sla_all, t_c2w_sla_all):
    """
    将双手顶点和相机位姿统一变换到渲染坐标系（与 demo.py 保持一致）。

    返回 (right_dict, left_dict, R_w2c_sla_all, t_w2c_sla_all,
           R_c2w_sla_all, t_c2w_sla_all)
    """
    R_x = _R_X

    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    left_dict["vertices"] = torch.einsum(
        "ij,btnj->btni", R_x, left_dict["vertices"].cpu()
    )
    right_dict["vertices"] = torch.einsum(
        "ij,btnj->btni", R_x, right_dict["vertices"].cpu()
    )

    return (right_dict, left_dict,
            R_w2c_sla_all, t_w2c_sla_all,
            R_c2w_sla_all, t_c2w_sla_all)




# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class HaWoRPipeline:
    """
    HaWoR 重建 pipeline 的核心类。

    参数
    ----
    checkpoint : str
        HaWoR 模型权重路径。
    infiller_weight : str
        Infiller 模型权重路径。
    img_focal : float | None
        相机焦距，若为 None 则自动估计。
    """

    DEFAULT_CHECKPOINT = "./weights/hawor/checkpoints/hawor.ckpt"
    DEFAULT_INFILLER   = "./weights/hawor/checkpoints/infiller.pt"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        infiller_weight: str = DEFAULT_INFILLER,
        verbose: bool = False,
        metric_3D_path: str = 'thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth',
        droid_filter_thresh: float = 2.4, # 原HaWoR实现都是使用这个默认值
    ):
        self.checkpoint      = checkpoint
        self.infiller_weight = infiller_weight
        self.verbose         = verbose
        self.model, self.model_cfg = self._load_hawor(self.checkpoint)
        self.hand_detect_model = YOLO('./weights/external/detector.pt')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"[INIT] HaWoRPipeline Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.metric_3D_path = metric_3D_path
        self.metric = Metric3D(self.metric_3D_path)
        import types
        self.args_droid = types.SimpleNamespace(
            filter_thresh = droid_filter_thresh,
            disable_vis = True, # 禁止可视化
            image_size = None, # 后面每个视频单独动态创建
        )
        self.filling_model = self._load_infiller_model()

    def _load_infiller_model(self):
        if self.verbose:
            print(f"[INIT] Loading infiller model from {self.infiller_weight}...")
        weight_path = self.infiller_weight
        ckpt = torch.load(weight_path, map_location=self.device)
        pos_dim = 3
        shape_dim = 10
        num_joints = 15
        rot_dim = (num_joints + 1) * 6  # rot6d
        repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
        nhead = 8  # repr_dim = 154
        self.horizon = 120 # 用于infiller模型的参数
        filling_model = TransformerModel(seq_len=self.horizon, input_dim=repr_dim, d_model=384, nhead=nhead, d_hid=2048,
                                         nlayers=8, dropout=0.05, out_dim=repr_dim, masked_attention_stage=True)
        filling_model.to(self.device)
        filling_model.load_state_dict(ckpt['transformer_encoder_state_dict'])
        filling_model.eval()
        return filling_model

        
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
    
    def _load_hawor(self, checkpoint_path):
        from pathlib import Path
        from hawor.configs import get_config
        model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
        model_cfg = get_config(model_cfg, update_cachedir=True)

        # Override some config values, to crop bbox correctly
        if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
            model_cfg.defrost()
            assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
            model_cfg.MODEL.BBOX_SHAPE = [192,256]
            model_cfg.freeze()

        model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
        return model, model_cfg
    
    def _hawor_motion_estimation(self, images_BGR, image_focal, tracks):
        """

        Returns:
            frame_chunks_all, model_masks, pred_hand_dict.
            pred_hand_dict 是原来硬盘写入的json文件，主要分为idx=0,1，区分表示左右手
        """
        model = self.model
        model.eval()
        
        # file = video_path
        # video_root = os.path.dirname(file)
        # video = os.path.basename(file).split('.')[0]
        # img_folder = f"{video_root}/{video}/extracted_images"
        # imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))

        # tracks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', allow_pickle=True).item()


        tid = np.array([tr for tr in tracks])

        # if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy'):
        #     print("skip hawor motion estimation")
        #     frame_chunks_all = joblib.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
        #     return frame_chunks_all, image_focal

        if self.verbose:
            print(f'Running hawor ...')

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
        tid = [0, 1] # 0表示左手， 1表示右手， 区分轨迹的左右手

        img = images_BGR[0]
        img_center = [img.shape[1] / 2, img.shape[0] / 2]# w/2, h/2  
        H, W = img.shape[:2]
        model_masks = np.zeros((len(images_BGR), H, W))

        bin_size = 128
        max_faces_per_bin = 20000
        renderer = Renderer(img.shape[1], img.shape[0], image_focal, self.device,
                        bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)
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
        faces_left = faces_right[:,[0,2,1]]

        frame_chunks_all = defaultdict(list)
        pred_hand_json = []
        for idx in tid:
            print(f"tracklet {idx}:")
            trk = final_tracks[idx]

            # interp bboxes
            valid = np.array([t['det'] for t in trk])
            if valid.sum() < 2:
                continue
            boxes = np.concatenate([t['det_box'] for t in trk])
            non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
            first_non_zero = non_zero_indices[0]
            last_non_zero = non_zero_indices[-1]
            boxes[first_non_zero:last_non_zero+1] = interpolate_bboxes(boxes[first_non_zero:last_non_zero+1])
            valid[first_non_zero:last_non_zero+1] = True


            boxes = boxes[first_non_zero:last_non_zero+1]
            is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
            frame = np.array([t['frame'] for t in trk])[valid]
            
            if is_right.sum() / len(is_right) < 0.5:
                is_right = np.zeros((len(boxes), 1))
            else:
                is_right = np.ones((len(boxes), 1))

            frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
            frame_chunks_all[idx] = frame_chunks

            if len(frame_chunks) == 0:
                continue

            for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
                print(f"inference from frame {frame_ck[0]} to {frame_ck[-1]}")
                img_ck = images_BGR[frame_ck] # BGR格式的！
                if is_right[0] > 0:
                    do_flip = False
                else:
                    do_flip = True
                    
                results = model.inference(img_ck, boxes_ck, img_focal=image_focal, img_center=img_center, do_flip=do_flip)

                data_out = {
                    "init_root_orient": results["pred_rotmat"][None, :, 0], # (B, T, 3, 3)
                    "init_hand_pose": results["pred_rotmat"][None, :, 1:], # (B, T, 15, 3, 3)
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
                pred_dict={
                    k:v.tolist() for k, v in data_out.items()
                }
                pred_hand_json[idx] = pred_dict
                # pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
                # if not os.path.exists(os.path.join(seq_folder, 'cam_space', str(idx))):
                #     os.makedirs(os.path.join(seq_folder, 'cam_space', str(idx)))
                # with open(pred_path, "w") as f:
                #     json.dump(pred_dict, f, indent=1)


                # get hand mask
                data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
                data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
                if do_flip: # left
                    outputs = run_mano_left(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"])
                else: # right
                    outputs = run_mano(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"])
                
                vertices = outputs["vertices"][0].cpu()  # (T, N, 3)
                for img_i, _ in enumerate(img_ck):
                    if do_flip:
                        faces = torch.from_numpy(faces_left).cuda()
                    else:
                        faces = torch.from_numpy(faces_right).cuda()
                    cam_R = torch.eye(3).unsqueeze(0).cuda()
                    cam_T = torch.zeros(1, 3).cuda()
                    cameras, lights = renderer.create_camera_from_cv(cam_R, cam_T)
                    verts_color = torch.tensor([0, 0, 255, 255]) / 255
                    vertices_i = vertices[[img_i]]
                    rend, mask = renderer.render_multiple(vertices_i.unsqueeze(0).cuda(), faces, verts_color.unsqueeze(0).cuda(), cameras, lights)
                    
                    model_masks[frame_ck[img_i]] += mask
                    
        model_masks = model_masks > 0 # bool
        # np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', model_masks)
        # joblib.dump(frame_chunks_all, f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
        return frame_chunks_all, model_masks, pred_hand_json

    def _extract_frames(self, video_path: str | Path, start_idx:int = 0, end_idx:int = -1, frame_step = 1):
        """
        从给定视频路径提取视频帧，返回images: list (BGR)

        Args:
            video_path: 输入 mp4 路径
            frame_step: 采样步长（1 = 每帧）
            max_frames: 最大处理帧数（None 表示全部）
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        images_BGR = []
        frame_indices = []
        frame_idx = 0
        collected = 0
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            total_frames = 0
        if not end_idx or end_idx==-1:
            end_idx = total_frames

        if self.verbose:
            print(f"Reading frames from video {video_path} (total frames ~{total_frames})")

        # Collect frames according to frame_step and max_frames
        # Collect frames according to frame_step and max_frames, show progress
        with tqdm(total=total_frames if total_frames > 0 else None,
                  desc=f"Reading {video_path}", unit='frame') as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pbar.update(1)

                if frame_idx % frame_step == 0:
                    # # Convert BGR -> RGB # 不要自动执行这一步！后面的YOLO模型会有问题
                    # img_rgb = frame[:, :, ::-1]
                    # images.append(img_rgb)
                    images_BGR.append(frame)
                    frame_indices.append(frame_idx)
                    collected += 1

                    if end_idx is not None and collected >= end_idx:
                        break

                frame_idx += 1

        cap.release()

        if len(images_BGR) == 0:
            print("No frames collected, exiting")
            return

        if self.verbose:
            print(f"Collected {len(images_BGR)} frames, running batch reconstruction via recon.recon(images)")

        return images_BGR

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

    def _run_slam(self, images_BGR, masks, calib, depth=None, stride=1,
                 filter_thresh=2.4, disable_vis=True):
        """ Maksed DROID-SLAM """
        depth = None
        droid = None
        args.filter_thresh = filter_thresh
        args.disable_vis = disable_vis
        masks = masks[::stride]

        """ Resize masks for masked droid """
        H, W = images_BGR[0].shape[:2]
        resize_1 = Resize((H, W), antialias=True)
        resize_2 = Resize((H // 8, W // 8), antialias=True)

        img_msks = []
        for i in range(0, len(masks), 500):
            m = resize_1(masks[i:i + 500])
            img_msks.append(m)
        img_msks = torch.cat(img_msks)

        conf_msks = []
        for i in range(0, len(masks), 500):
            m = resize_2(masks[i:i + 500])
            conf_msks.append(m)
        conf_msks = torch.cat(conf_msks)


        for (t, image, intrinsics) in tqdm(self._image_stream(images_BGR, calib, stride)):
            if droid is None:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(args)

            img_msk = img_msks[t]
            conf_msk = conf_msks[t]
            image = image * (img_msk < 0.5)
            # cv2.imwrite('debug.png', image[0].permute(1, 2, 0).numpy())

            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)

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

        ##### Run SLAM #####
        # Use Masking
        # masks = np.load(f'{video_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', allow_pickle=True)
        masks = torch.from_numpy(masks)
        print(masks.shape)

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

        # Droid-slam with masking
        droid, traj = run_slam(images_BGR, masks=masks, calib=calib)
        n = droid.video.counter.value
        tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
        disps = droid.video.disps_up.cpu().numpy()[:n]
        if self.verbose:
            print('DBA errors:', droid.backend.errors)

        del droid # 一条视频单独使用一个droid实例！
        torch.cuda.empty_cache()

        # Estimate scale
        # block_print() # 临时禁用打印输出，避免 Metric3D 模型加载或推理时产生过多日志干扰
        # 加载metric3D改到模型公用
        # enable_print()

        min_threshold = 0.4
        max_threshold = 0.7

        if self.verbose:
            print('Predicting Metric Depth ...')
        pred_depths = []

        def get_dimention(image):
            """
            Get proper image dimension for DROID
            DROID-SLAM 需要将图像尺寸除以 8，所以输入必须是 8 的倍数。
            """
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1 - h1 % 8, :w1 - w1 % 8]
            H, W, _ = image.shape
            return H, W

        H, W = get_dimention(first_img)
        for t in tqdm(tstamp):
            img = images_BGR[t]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_depth = self.metric(img_rgb, calib)
            pred_depth = cv2.resize(pred_depth, (W, H))
            pred_depths.append(pred_depth)

        ##### Estimate Metric Scale #####
        print('Estimating Metric Scale ...')
        scales_ = []
        n = len(tstamp)  # for each keyframe
        for i in tqdm(range(n)):
            t = tstamp[i]
            disp = disps[i]
            pred_depth = pred_depths[i]
            slam_depth = 1 / disp

            # Estimate scene scale
            msk = masks[t].numpy().astype(np.uint8)
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold,
                                     far_thresh=max_threshold)
            while math.isnan(scale):
                min_threshold -= 0.1
                max_threshold += 0.1
                scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold,
                                         far_thresh=max_threshold)
            scales_.append(scale)

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

        # camera space to world space
        tid = [0, 1]
        for k, idx in enumerate(tid):
            frame_chunks = frame_chunks_all[idx]

            if len(frame_chunks) == 0:
                continue

            for frame_ck in frame_chunks:
                # print(f"from frame {frame_ck[0]} to {frame_ck[-1]}")
                # pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
                # with open(pred_path, "r") as f:
                #     pred_dict = json.load(f)
                pred_dict = pred_hand_json[idx]
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
        for k, idx in enumerate([1, 0]):
            missing = ~pred_valid[idx]

            frame = frame_list[missing]
            frame_chunks = parse_chunks_hand_frame(frame) # 这边进行帧分块处理

            if self.verbose:
                print(f"run infiller on {idx2hand[idx]} hand ...")
            for frame_ck in tqdm(frame_chunks):
                start_shift = -1
                while frame_ck[0] + start_shift >= 0 and pred_valid[:, frame_ck[0] + start_shift].sum() != 2:
                    start_shift -= 1  # Shift to find the previous valid frame as start
                if self.verbose:
                    print(f"run infiller on frame {frame_ck[0] + start_shift} to frame {min(len(images_BGR) - 1, frame_ck[0] + start_shift + filling_length)}")

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
        # save_path = os.path.join(seq_folder, "world_space_res.pth")
        # joblib.dump([pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid], save_path)
        return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid

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
    ) -> dict:
        """
        对单个视频执行完整重建 pipeline。

        Args:
            video_path : str
                输入视频路径。
            output_dir : str | None
                输出目录。若为 None，则默认保存在与视频同名的子目录中。
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

        # ── Step 1: 检测 & 追踪 ─────────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 1/4 — Detect & Track")
        file = video_path
        os.makedirs(output_dir, exist_ok=True)
        if self.verbose:
            print(f'Running detect_track on {file} ...')

        ##### Extract Frames #####
        images_BGR = self._extract_frames(video_path, start_idx, end_idx)
        if not end_idx or end_idx==-1 or end_idx > len(images_BGR):
            end_idx = len(images_BGR)

        ##### Detection + Track #####
        if self.verbose:
            print('Detect and Track ...')
        boxes, tracks = self._detect_track(images_BGR, thresh=0.2)


        # ── Step 2: HaWoR 运动估计 ──────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 2/4 — Motion Estimation")
        if image_focal is None:
            image_focal = 600
            print(f'No focal length provided, use default {image_focal}')
        frame_chunks_all, model_masks, pred_hand_json = self._hawor_motion_estimation(
            images_BGR, image_focal, tracks
        )

        # ── Step 3: SLAM ─────────────────────────────────────────────────
        if self.verbose:
            print("[HaWoR] Step 3/4 — SLAM")
        pred_cam = self._hawor_slam(images_BGR, model_masks, image_focal)
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
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
            self._hawor_infiller(images_BGR, frame_chunks_all, slam_cam, pred_hand_json)
        # pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        #     hawor_infiller_plain(args, start_idx, end_idx, frame_chunks_all)

        # ── 构建双手网格字典 ─────────────────────────────────────────────
        faces_right, faces_left = _build_faces()
        vis_start = 0
        vis_end   = pred_trans.shape[1] - 1

        right_dict, left_dict = _build_hand_dicts(
            pred_trans, pred_rot, pred_hand_pose, pred_betas,
            vis_start, vis_end, faces_right, faces_left
        )

        # ── 坐标系变换 ───────────────────────────────────────────────────
        (right_dict, left_dict,
         R_w2c_sla_all, t_w2c_sla_all,
         R_c2w_sla_all, t_c2w_sla_all) = _apply_coord_transform(
            right_dict, left_dict, R_c2w_sla_all, t_c2w_sla_all
        )

        # ── 整理返回结果 ─────────────────────────────────────────────────
        result = dict(
            pred_trans     = pred_trans,
            pred_rot       = pred_rot,
            pred_hand_pose = pred_hand_pose,
            pred_betas     = pred_betas,
            pred_valid     = pred_valid,
            right_dict     = right_dict,
            left_dict      = left_dict,
            R_c2w          = R_c2w_sla_all,
            t_c2w          = t_c2w_sla_all,
            R_w2c          = R_w2c_sla_all,
            t_w2c          = t_w2c_sla_all,
            img_focal      = image_focal,
            rendered_video = None,
        )

        # ── 可选：渲染 mp4 ───────────────────────────────────────────────
        if rendering:
            rendered_video = self._render(
                result      = result,
                vis_start   = vis_start,
                vis_end     = vis_end,
                output_dir  = output_dir,
                vis_mode    = vis_mode,
                video_path  = video_path,
            )
            result["rendered_video"] = rendered_video
            if rendered_video:
                print(f"[HaWoR] Rendered video saved to: {rendered_video}")

        return result

    # ------------------------------------------------------------------
    # 渲染（可选）
    # ------------------------------------------------------------------
    def _render(
        self,
        result: dict,
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
        image_names = list(result["imgfiles"][vis_start:vis_end])

        print(f"[HaWoR] Rendering frames {vis_start} → {vis_end}  (mode={vis_mode})")

        return render_hand_results(
            left_dict    = result["left_dict"],
            right_dict   = result["right_dict"],
            image_names  = image_names,
            img_focal    = result["img_focal"],
            output_dir   = output_dir,
            vis_start    = vis_start,
            vis_end      = vis_end,
            vis_mode     = vis_mode,
            R_c2w        = result["R_c2w"],
            t_c2w        = result["t_c2w"],
            R_w2c        = result["R_w2c"],
            t_w2c        = result["t_w2c"],
            video_stem   = video_stem,
        )

