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

# ---------------------------------------------------------------------------
# 懒加载视频帧序列（按需从磁盘读帧，避免一次性全部载入内存）
# ---------------------------------------------------------------------------

class LazyVideoFrames:
    """
    按需从视频文件中读取帧的懒加载序列。

    支持的访问方式（与 numpy ndarray 接口兼容）：
      - len(v)                     → 总帧数
      - v[i]                       → 读取第 i 帧 (int/np.integer)
      - v[array_like]              → 按帧号列表批量读取，返回 np.ndarray (N,H,W,3)
      - v[start:stop:step]         → 切片，返回新的 LazyVideoFrames 子视图
      - iter(v)                    → 顺序逐帧迭代
      - v.shape                    → (N, H, W, 3) 虚拟形状

    所有帧均以 BGR uint8 格式返回，与 cv2.imread / cv2.VideoCapture 一致。
    """

    def __init__(
        self,
        video_path: str,
        indices: "list[int] | None" = None,
        frame_step: int = 1,
        start_idx: int = 0,
        end_idx: "int | None" = None,
    ):
        """
        Parameters
        ----------
        video_path : str
            视频文件路径。
        indices : list[int] | None
            若给定，则直接以此作为帧索引列表（忽略 start/end/step）。
        frame_step : int
            采样步长。
        start_idx : int
            起始帧（包含）。
        end_idx : int | None
            终止帧（不含）；None 表示到末尾。
        """
        self.video_path = str(video_path)

        # 先探测视频信息
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self._total_video_frames = total
        self._frame_wh = (w, h)  # (width, height)

        if indices is not None:
            self._indices = list(indices)
        else:
            _end = end_idx if (end_idx is not None and end_idx != -1) else total
            self._indices = list(range(start_idx, _end, frame_step))

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------

    @property
    def shape(self):
        """虚拟形状 (N, H, W, 3)，与 numpy array 接口一致。"""
        h, w = self._frame_wh[1], self._frame_wh[0]
        return (len(self._indices), h, w, 3)

    def __len__(self):
        return len(self._indices)

    # ------------------------------------------------------------------
    # 索引 / 切片
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        # ── 整数索引 ─────────────────────────────────────────────────────
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += len(self._indices)
            if not (0 <= idx < len(self._indices)):
                raise IndexError(f"index {key} out of range for LazyVideoFrames with {len(self._indices)} frames")
            return self._read_single(self._indices[idx])

        # ── 切片 ──────────────────────────────────────────────────────────
        if isinstance(key, slice):
            sub_indices = self._indices[key]
            return LazyVideoFrames(self.video_path, indices=sub_indices)

        # ── numpy array / list / tuple（随机多帧访问）───────────────────
        if isinstance(key, (np.ndarray, list, tuple)):
            key_arr = np.asarray(key)
            if key_arr.ndim == 0:
                return self._read_single(self._indices[int(key_arr)])
            # 将相对索引映射到全局帧号
            global_indices = [self._indices[int(i)] for i in key_arr]
            return self._read_batch(global_indices)

        raise TypeError(f"Unsupported index type: {type(key)}")

    # ------------------------------------------------------------------
    # 迭代
    # ------------------------------------------------------------------

    def __iter__(self):
        """
        顺序逐帧迭代，真正的流式处理——每次只在内存中保留一帧。

        策略：
        - 若 _indices 是单调递增（无乱序），则顺序读取，不重复 seek，效率最高。
        - 若 _indices 有间隔（如 stride>1），则每次 seek 到目标帧后读取。
        - 每帧 yield 完毕即可被 GC 回收，不会累积内存。
        """
        if not self._indices:
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for iteration: {self.video_path}")

        try:
            current_video_pos = -1  # 记录当前视频指针位置（解码后）
            for global_fid in self._indices:
                # 若帧号不连续，需要 seek
                if global_fid != current_video_pos + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, global_fid)
                ret, frame = cap.read()
                current_video_pos = global_fid
                if not ret:
                    frame = np.zeros((self._frame_wh[1], self._frame_wh[0], 3), dtype=np.uint8)
                yield frame
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # 内部读帧工具
    # ------------------------------------------------------------------

    def _read_single(self, global_frame_idx: int) -> np.ndarray:
        """打开视频，seek 到指定帧并读取。"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            h, w = self._frame_wh[1], self._frame_wh[0]
            return np.zeros((h, w, 3), dtype=np.uint8)
        return frame

    def _read_batch(self, global_indices: "list[int]") -> np.ndarray:
        """
        顺序 seek 读取多帧，返回 (N, H, W, 3) uint8 ndarray。
        对于已排好序的帧号会使用连续读取优化。
        """
        if not global_indices:
            h, w = self._frame_wh[1], self._frame_wh[0]
            return np.empty((0, h, w, 3), dtype=np.uint8)

        cap = cv2.VideoCapture(self.video_path)
        frames = []
        prev = -2
        for fid in global_indices:
            if fid != prev + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                h, w = self._frame_wh[1], self._frame_wh[0]
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            frames.append(frame)
            prev = fid
        cap.release()
        return np.stack(frames)

    def __repr__(self):
        return (
            f"LazyVideoFrames(video='{self.video_path}', "
            f"n_frames={len(self._indices)}, "
            f"wh={self._frame_wh})"
        )


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
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
], dtype=torch.float32)



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
# 抖动检测与平滑（Jitter Detection & Smoothing）
# ---------------------------------------------------------------------------

def _mad_detect_jitter(signal: np.ndarray, window: int = 5, thresh_k: float = 3.5) -> np.ndarray:
    """
    用滑动窗口 MAD（中位数绝对偏差）检测抖动帧。

    原理
    ----
    对信号的逐帧速度（一阶差分的 L2 范数）计算全局 MAD 统计量：
        MAD = median(|v_i - median(v)|)
    若某帧速度满足 |v_i - median(v)| > thresh_k * MAD / 0.6745 则视为抖动。
    0.6745 使 MAD 成为高斯分布标准差的一致估计量。

    Parameters
    ----------
    signal : np.ndarray, shape (T, D)
        待检测的时序信号（D 维，T 帧）。
    window : int
        局部平滑窗口大小（用于预去噪，奇数）。
    thresh_k : float
        MAD 倍数阈值，越小越敏感；建议 3.0 ~ 5.0。

    Returns
    -------
    jitter_mask : np.ndarray, shape (T,), dtype=bool
        True 表示该帧被判定为抖动帧（需要被修复）。
    """
    T = signal.shape[0]
    if T < 3:
        return np.zeros(T, dtype=bool)

    # 一阶差分速度（帧间变化幅度）
    vel = np.linalg.norm(np.diff(signal, axis=0), axis=-1)   # (T-1,)

    # 全局 MAD 统计
    median_v = np.median(vel)
    mad = np.median(np.abs(vel - median_v))
    sigma_hat = mad / 0.6745 + 1e-8   # 鲁棒标准差估计

    # 离群判定
    outlier = np.abs(vel - median_v) > thresh_k * sigma_hat  # (T-1,)

    # 若第 t 帧的离开速度或到达速度均为离群 → 该帧本身是抖动帧
    jitter_mask = np.zeros(T, dtype=bool)
    # vel[t-1] 是帧 t-1→t 的速度，vel[t] 是帧 t→t+1 的速度
    for t in range(1, T - 1):
        if outlier[t - 1] and outlier[t]:   # 到达和离开都异常 → 孤立跳变帧
            jitter_mask[t] = True
    # 边界：第 0 帧或最后一帧若速度异常，也标记
    if T >= 2 and outlier[0]:
        jitter_mask[0] = True
    if T >= 2 and outlier[-1]:
        jitter_mask[-1] = True

    return jitter_mask


def _interp_translation(trans: np.ndarray, jitter_mask: np.ndarray) -> np.ndarray:
    """
    对平移量中的抖动帧用三次样条插值修复。

    Parameters
    ----------
    trans : np.ndarray, shape (T, 3)
    jitter_mask : np.ndarray, shape (T,), bool

    Returns
    -------
    trans_fixed : np.ndarray, shape (T, 3)
    """
    T = trans.shape[0]
    t_all = np.arange(T)
    good = ~jitter_mask

    if good.sum() < 2:
        return trans.copy()

    trans_fixed = trans.copy()
    t_good = t_all[good]
    v_good = trans[good]   # (N_good, 3)

    if good.sum() >= 4:
        cs = CubicSpline(t_good, v_good, extrapolate=True)
    else:
        # 点太少，退化为线性插值
        from scipy.interpolate import interp1d
        cs = interp1d(t_good, v_good, axis=0, kind='linear', fill_value='extrapolate')

    bad_t = t_all[jitter_mask]
    if len(bad_t) > 0:
        trans_fixed[bad_t] = cs(bad_t)

    return trans_fixed


def _interp_rotation_aa(rot_aa: np.ndarray, jitter_mask: np.ndarray) -> np.ndarray:
    """
    对轴角旋转中的抖动帧用 SLERP 修复。

    Parameters
    ----------
    rot_aa : np.ndarray, shape (T, 3)  轴角表示
    jitter_mask : np.ndarray, shape (T,), bool

    Returns
    -------
    rot_fixed : np.ndarray, shape (T, 3)
    """
    T = rot_aa.shape[0]
    t_all = np.arange(T, dtype=float)
    good = ~jitter_mask

    if good.sum() < 2:
        return rot_aa.copy()

    rot_fixed = rot_aa.copy()
    t_good = t_all[good]
    rots_good = ScipyRotation.from_rotvec(rot_aa[good])
    slerp_fn = Slerp(t_good, rots_good)

    bad_t = t_all[jitter_mask]
    if len(bad_t) > 0:
        # 超出已知帧范围的 bad_t 需要 clamp，避免外推失败
        bad_t_clamped = np.clip(bad_t, t_good[0], t_good[-1])
        rot_fixed[jitter_mask] = slerp_fn(bad_t_clamped).as_rotvec()

    return rot_fixed


def _gaussian_smooth_1d(signal: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    对时序信号每一维施加高斯核平滑（边界用镜像填充）。

    Parameters
    ----------
    signal : np.ndarray, shape (T, D)
    sigma  : float  高斯核标准差（帧数单位），越大越平滑

    Returns
    -------
    smoothed : np.ndarray, shape (T, D)
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(signal, sigma=sigma, axis=0, mode='mirror')


def smooth_hand_predictions(
    pred_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_hand_pose: torch.Tensor,
    pred_valid: np.ndarray,
    *,
    jitter_thresh_k: float = 3.5,
    smooth_sigma_trans: float = 1.2,
    smooth_sigma_rot: float = 0.8,
    smooth_sigma_pose: float = 0.6,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对 HaWoR 输出的双手平移 / 根旋转 / 手指姿态做抖动检测与平滑。

    流程
    ----
    1. **MAD 抖动检测**：在有效帧范围内，基于帧间速度的 MAD 统计量识别跳变帧。
    2. **插值修复**：平移用三次样条插值，旋转用 SLERP，替换掉跳变帧的原始值。
    3. **高斯平滑**：对平移、根旋转、手指姿态分别施加各自 sigma 的高斯核平滑，
       消除高频颤抖。

    Parameters
    ----------
    pred_trans     : Tensor (2, T, 3)   平移（世界坐标）
    pred_rot       : Tensor (2, T, 3)   根方向（轴角）
    pred_hand_pose : Tensor (2, T, 45)  手指姿态（轴角展开）
    pred_valid     : np.ndarray (2, T)  有效帧掩码（bool）
    jitter_thresh_k : float  MAD 检测阈值倍数（越小越敏感，建议 3~5）
    smooth_sigma_trans : float  平移高斯平滑 sigma（帧数单位）
    smooth_sigma_rot   : float  根旋转高斯平滑 sigma
    smooth_sigma_pose  : float  手指姿态高斯平滑 sigma
    verbose : bool

    Returns
    -------
    pred_trans_s, pred_rot_s, pred_hand_pose_s : 同形状平滑后的 Tensor
    """
    n_hands, T, _ = pred_trans.shape
    trans_np    = pred_trans.numpy().copy()       # (2, T, 3)
    rot_np      = pred_rot.numpy().copy()         # (2, T, 3)
    pose_np     = pred_hand_pose.numpy().copy()   # (2, T, 45)

    hand_names = ['left', 'right']

    for h in range(n_hands):
        valid_mask = pred_valid[h].astype(bool)   # (T,)
        valid_idx  = np.where(valid_mask)[0]

        if len(valid_idx) < 4:
            if verbose:
                print(f"[Smooth] {hand_names[h]} hand: too few valid frames ({len(valid_idx)}), skip.")
            continue

        # ── 1. MAD 抖动检测（仅在有效帧上做） ────────────────────────
        trans_valid  = trans_np[h][valid_idx]   # (N_valid, 3)
        rot_valid    = rot_np[h][valid_idx]     # (N_valid, 3)

        jitter_trans = _mad_detect_jitter(trans_valid, thresh_k=jitter_thresh_k)
        jitter_rot   = _mad_detect_jitter(rot_valid,   thresh_k=jitter_thresh_k)
        jitter_union = jitter_trans | jitter_rot        # 任一跳变则修复

        n_jitter = jitter_union.sum()
        if verbose:
            print(f"[Smooth] {hand_names[h]} hand: detected {n_jitter}/{len(valid_idx)} jitter frames "
                  f"(thresh_k={jitter_thresh_k})")

        # ── 2. 插值修复跳变帧 ────────────────────────────────────────
        if n_jitter > 0:
            trans_fixed = _interp_translation(trans_valid, jitter_union)
            rot_fixed   = _interp_rotation_aa(rot_valid,   jitter_union)
            trans_np[h][valid_idx] = trans_fixed
            rot_np[h][valid_idx]   = rot_fixed

            # 手指姿态：每 3 维为一个关节的轴角，逐关节 SLERP 修复
            for j in range(15):
                pose_j = pose_np[h][valid_idx, j * 3:(j + 1) * 3]   # (N_valid, 3)
                pose_np[h][valid_idx, j * 3:(j + 1) * 3] = _interp_rotation_aa(pose_j, jitter_union)

        # ── 3. 高斯平滑（对有效帧段整体平滑） ───────────────────────
        # 平移
        trans_np[h][valid_idx] = _gaussian_smooth_1d(
            trans_np[h][valid_idx], sigma=smooth_sigma_trans
        )
        # 根旋转（在向量空间近似平滑，适合小角度旋转）
        rot_np[h][valid_idx] = _gaussian_smooth_1d(
            rot_np[h][valid_idx], sigma=smooth_sigma_rot
        )
        # 手指姿态
        pose_np[h][valid_idx] = _gaussian_smooth_1d(
            pose_np[h][valid_idx], sigma=smooth_sigma_pose
        )

    pred_trans_s    = torch.from_numpy(trans_np)
    pred_rot_s      = torch.from_numpy(rot_np)
    pred_hand_pose_s = torch.from_numpy(pose_np)

    return pred_trans_s, pred_rot_s, pred_hand_pose_s


def smooth_camera_trajectory(
    R_c2w: torch.Tensor,
    t_c2w: torch.Tensor,
    *,
    jitter_thresh_k: float = 4.0,
    smooth_sigma: float = 1.0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 SLAM 估计的相机轨迹做抖动检测与平滑。

    Parameters
    ----------
    R_c2w : Tensor (T, 3, 3)  相机→世界旋转
    t_c2w : Tensor (T, 3)     相机→世界平移
    jitter_thresh_k : float   MAD 阈值倍数
    smooth_sigma : float      高斯平滑 sigma
    verbose : bool

    Returns
    -------
    R_c2w_s, t_c2w_s : 平滑后的旋转与平移
    """
    T = t_c2w.shape[0]
    t_np = t_c2w.numpy().copy()   # (T, 3)

    # 平移抖动检测+插值
    jitter_t = _mad_detect_jitter(t_np, thresh_k=jitter_thresh_k)
    n_jitter = jitter_t.sum()
    if verbose:
        print(f"[Smooth] Camera trajectory: detected {n_jitter}/{T} jitter frames")

    if n_jitter > 0:
        t_np = _interp_translation(t_np, jitter_t)

    # 平移高斯平滑
    t_np = _gaussian_smooth_1d(t_np, sigma=smooth_sigma)

    # 旋转：转为四元数 → SLERP 插值 → 高斯平滑轴角
    R_np   = R_c2w.numpy()                                          # (T, 3, 3)
    rot_aa = ScipyRotation.from_matrix(R_np).as_rotvec()           # (T, 3)

    if n_jitter > 0:
        rot_aa = _interp_rotation_aa(rot_aa, jitter_t)

    rot_aa = _gaussian_smooth_1d(rot_aa, sigma=smooth_sigma)
    R_np_s = ScipyRotation.from_rotvec(rot_aa).as_matrix()         # (T, 3, 3)

    R_c2w_s = torch.from_numpy(R_np_s).float()
    t_c2w_s = torch.from_numpy(t_np).float()

    return R_c2w_s, t_c2w_s


# ---------------------------------------------------------------------------
# 配置 dataclass
# ---------------------------------------------------------------------------

@dataclass
class HaWoRConfig:
    """
    HaWoRPipeline 的统一配置类。

    字段分组
    --------
    路径配置
        checkpoint        : HaWoR 模型权重路径。
        infiller_weight   : Infiller 模型权重路径。
        metric_3D_path    : Metric3D 权重路径。
        detector_path     : 手部检测器（YOLO）权重路径。
        device            : 运行设备（"cuda"、"cpu" 或具体 GPU）。

    运行配置
        verbose           : 是否打印详细日志。
        droid_filter_thresh : DROID-SLAM 的 filter_thresh 参数。

    抖动平滑配置
        smooth_hands      : 是否对手部预测结果做抖动平滑。
        smooth_camera     : 是否对相机轨迹做抖动平滑。
        jitter_thresh_k_hands : 手部 MAD 抖动检测阈值倍数（越小越敏感，建议 3~5）。
        jitter_thresh_k_cam   : 相机 MAD 抖动检测阈值倍数。
        smooth_sigma_trans    : 手部平移高斯平滑 sigma（帧数单位）。
        smooth_sigma_rot      : 手部根旋转高斯平滑 sigma。
        smooth_sigma_pose     : 手指姿态高斯平滑 sigma。
        smooth_sigma_cam      : 相机轨迹高斯平滑 sigma。
    """
    # ── 路径配置 ─────────────────────────────────────────────────────
    checkpoint: str = "./weights/hawor/checkpoints/hawor.ckpt"
    infiller_weight: str = "./weights/hawor/checkpoints/infiller.pt"
    metric_3D_path: str = "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"
    detector_path: str = "./weights/external/detector.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # "cuda" 或 "cpu" 或者自定义哪个具体的GPU

    # ── 运行配置 ─────────────────────────────────────────────────────
    verbose: bool = False
    droid_filter_thresh: float = 2.4   # 原 HaWoR 实现的默认值

    # ── 抖动平滑配置 ─────────────────────────────────────────────────
    smooth_hands: bool = True
    smooth_camera: bool = True
    jitter_thresh_k_hands: float = 3.5
    jitter_thresh_k_cam: float = 4.0
    smooth_sigma_trans: float = 1.2
    smooth_sigma_rot: float = 0.8
    smooth_sigma_pose: float = 0.6
    smooth_sigma_cam: float = 1.0


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class HaWoRPipeline:
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
        model_masks = np.zeros((len(images_BGR), H, W))

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

        model_masks = model_masks > 0  # bool
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
        """ Maksed DROID-SLAM """
        depth = None
        droid = None
        self.args_droid.filter_thresh = filter_thresh
        self.args_droid.disable_vis = disable_vis
        masks = masks[::stride]

        """ Resize masks for masked droid """
        H, W = self._get_dimension(images_BGR[0])
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
                self.args_droid.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(self.args_droid)

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

        # 初始化进度（10%）
        self._update_stage_progress(10)

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
        pred_depths = []

        H, W = self._get_dimension(first_img)
        total_keyframes = len(tstamp)
        progress_per_depth = 35 / max(total_keyframes, 1)  # 深度预测占35%

        for t in tstamp:
            img = images_BGR[t]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_depth = self.metric(img_rgb, calib)
            pred_depth = cv2.resize(pred_depth, (W, H))
            pred_depths.append(pred_depth)
            self._update_stage_progress(int(progress_per_depth))  # 每预测一帧更新进度

        ##### Estimate Metric Scale #####
        print('Estimating Metric Scale ...')
        scales_ = []
        n = len(tstamp)  # for each keyframe
        progress_per_scale = 5 / max(n, 1)  # 尺度估计占5%

        for i in range(n):
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
            self._update_stage_progress(int(progress_per_scale))  # 每估计一个尺度更新进度

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
        if self.smooth_hands:
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
         
        if self.smooth_hands or self.smooth_camera:
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
            
            smooth_hand_enabled=self.smooth_hands,
            smooth_camera_enabled=self.smooth_camera,
            
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
            ),
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

