"""
对比 VR JSON 与 HaWoR(Adapter 输出 PKL) 的 3D 关键点。

说明：
1) 这里的 `--hawor-pkl` 输入不是原始 HaWoRPipeline MANO 参数，而是
   `reconstruct_egocentric.py` 中经过 `convert_hawor_to_keypoints(...)` 后保存的结构：

    {
        "original_result": List[frame_dict],
        "smoothed_result": List[frame_dict]
    }

2) 每个 frame_dict 的关键字段：
    {
        "frame_idx": int,
        "hands": [
            {
                "is_right": 0/1,
                "pred_keypoints_3d": (21, 3),
                ...
            },
            ...
        ],
        "camera_pose": {...}
    }

3) VR JSON 采用与 `convert_vr_data_to_lerobot.py` 一致的时间处理方式：
   使用 `timestamp / 1000.0` 作为秒时间轴，并插值到 HaWoR 帧时刻。

Linux 示例：
python scripts/VR/compare_vr_data_and_hawor.py \
    --vr-json "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json" \
    --hawor-pkl "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0_hawor.pkl" \
    --stream smoothed_result \
    --mode video \
    --output results/compare.mp4
    
full arguments:
    --vr-json "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json" 
    --hawor-pkl "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0_hawor.pkl" 
    --stream smoothed_result 
    --mode video 
    --output results/compare.mp4
"""

import argparse
import glob
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def _safe_np(array_like, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(array_like, dtype=dtype)
    return arr


class VRDataLoader:
    """读取并处理 VR JSON。"""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.raw_data = self._load_json()
        self.processed_data: Optional[Dict] = None

    def _load_json(self) -> List[Dict]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"VR JSON 无有效帧: {self.json_path}")
        return data

    def process(self) -> Dict:
        timestamps_s: List[float] = []
        keypoints: List[np.ndarray] = []
        confidences: List[np.ndarray] = []

        for item in self.raw_data:
            # 与 convert_vr_data_to_lerobot.py 保持一致：毫秒 -> 秒
            t_rel_s = float(item["timestamp"]) / 1000.0
            timestamps_s.append(t_rel_s)

            kpts = []
            confs = []
            for kp in item["keypoints"]:
                pos = kp["position"]
                kpts.append([float(pos["x"]), float(pos["y"]), float(pos["z"])])
                confs.append(float(kp.get("confidence", 1.0)))

            kpt_arr = _safe_np(kpts)
            if kpt_arr.ndim != 2 or kpt_arr.shape[1] != 3:
                raise ValueError(f"VR keypoints 形状异常: {kpt_arr.shape}")

            keypoints.append(kpt_arr)
            confidences.append(_safe_np(confs))

        # 排序保证单调递增时间，避免插值报错
        timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
        order = np.argsort(timestamps_s)

        self.processed_data = {
            "timestamps": timestamps_s[order],
            "keypoints": np.asarray(keypoints, dtype=np.float32)[order],
            "confidences": np.asarray(confidences, dtype=np.float32)[order],
        }
        return self.processed_data

    def interpolate_to_times(self, target_times: np.ndarray) -> Dict:
        if self.processed_data is None:
            self.process()

        processed = self.processed_data
        if processed is None:
            raise RuntimeError("VR 数据初始化失败")

        src_times = processed["timestamps"]
        src_kpts = processed["keypoints"]  # (T, K, 3)
        src_confs = processed["confidences"]  # (T, K)

        if len(src_times) < 2:
            # 单帧退化情况：直接复制
            interp_kpts = np.repeat(src_kpts[:1], len(target_times), axis=0)
            interp_confs = np.repeat(src_confs[:1], len(target_times), axis=0)
            return {
                "timestamps": target_times,
                "keypoints": interp_kpts,
                "confidences": interp_confs,
            }

        num_kpts = src_kpts.shape[1]
        interp_kpts = np.zeros((len(target_times), num_kpts, 3), dtype=np.float32)
        interp_confs = np.zeros((len(target_times), num_kpts), dtype=np.float32)

        for k in range(num_kpts):
            for axis in range(3):
                fn = interp1d(
                    src_times,
                    src_kpts[:, k, axis],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(src_kpts[0, k, axis], src_kpts[-1, k, axis]),
                    assume_sorted=True,
                )
                interp_kpts[:, k, axis] = fn(target_times).astype(np.float32)

            fn_conf = interp1d(
                src_times,
                src_confs[:, k],
                kind="linear",
                bounds_error=False,
                fill_value=(src_confs[0, k], src_confs[-1, k]),
                assume_sorted=True,
            )
            interp_confs[:, k] = fn_conf(target_times).astype(np.float32)

        return {
            "timestamps": target_times,
            "keypoints": interp_kpts,
            "confidences": interp_confs,
        }


class HaWoRResultLoader:
    """读取 `convert_hawor_to_keypoints` 产出的 PKL。"""

    def __init__(self, pkl_path: str, stream: str = "smoothed_result"):
        self.pkl_path = Path(pkl_path)
        self.stream = stream
        self.raw = self._load_pkl()

    def _load_pkl(self) -> Dict:
        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("HaWoR PKL 不是 dict")
        return data

    def extract_keypoints(self) -> Dict:
        if self.stream not in self.raw:
            if "original_result" in self.raw:
                frames = self.raw["original_result"]
            else:
                raise KeyError(
                    f"PKL 中未找到 '{self.stream}'，可用 keys={list(self.raw.keys())}"
                )
        else:
            frames = self.raw[self.stream]

        if not isinstance(frames, list) or len(frames) == 0:
            raise ValueError(f"{self.stream} 为空或格式错误")

        num_frames = len(frames)
        left_list: List[np.ndarray] = []
        right_list: List[np.ndarray] = []

        for frame in frames:
            hands = frame.get("hands", [])
            left_kpts = None
            right_kpts = None
            for hand in hands:
                kpts = hand.get("pred_keypoints_3d")
                if kpts is None:
                    continue
                kpts = _safe_np(kpts)
                if kpts.ndim != 2 or kpts.shape[1] != 3:
                    continue
                if int(hand.get("is_right", 0)) == 1:
                    right_kpts = kpts
                else:
                    left_kpts = kpts

            if left_kpts is None:
                left_kpts = np.full((21, 3), np.nan, dtype=np.float32)
            if right_kpts is None:
                right_kpts = np.full((21, 3), np.nan, dtype=np.float32)

            left_list.append(left_kpts.astype(np.float32))
            right_list.append(right_kpts.astype(np.float32))

        # 这里按 reconstruct 输出默认 25fps 映射时间轴（与 VR 秒时间轴做插值对齐）
        fps = 25.0
        timestamps_s = np.arange(num_frames, dtype=np.float64) / fps

        return {
            "timestamps": timestamps_s,
            "keypoints_left": np.stack(left_list, axis=0),   # (T, 21, 3)
            "keypoints_right": np.stack(right_list, axis=0), # (T, 21, 3)
            "num_frames": num_frames,
            "stream": self.stream,
        }


class KeypointVisualizer:
    """VR 与 HaWoR 3D 对比绘制。"""

    VR_LEFT_COLOR = (0.20, 0.55, 1.00)
    VR_RIGHT_COLOR = (0.10, 0.35, 0.90)
    HAWOR_LEFT_COLOR = (1.00, 0.50, 0.20)
    HAWOR_RIGHT_COLOR = (0.90, 0.25, 0.10)

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize

    @staticmethod
    def _finite_rows(points: np.ndarray) -> np.ndarray:
        if points is None or points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        mask = np.isfinite(points).all(axis=1)
        return points[mask]

    def _set_equal_axes(self, ax, *point_sets: np.ndarray) -> None:
        merged = []
        for pts in point_sets:
            finite = self._finite_rows(pts)
            if len(finite) > 0:
                merged.append(finite)

        if len(merged) == 0:
            return

        all_pts = np.concatenate(merged, axis=0)
        mins = np.nanmin(all_pts, axis=0)
        maxs = np.nanmax(all_pts, axis=0)
        center = (mins + maxs) * 0.5
        radius = max(np.max(maxs - mins) * 0.5, 1e-4)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def plot_frame(
        self,
        vr_keypoints: np.ndarray,
        hawor_left: np.ndarray,
        hawor_right: np.ndarray,
        frame_idx: int,
        stream_name: str,
        output_image_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # VR: 12 点结构（前6左手，后6右手）
        vr_left = None
        vr_right = None
        if vr_keypoints is not None and len(vr_keypoints) >= 12:
            vr_left = vr_keypoints[:6]
            vr_right = vr_keypoints[6:12]
            if len(vr_left) > 0:
                ax.scatter(
                    vr_left[:, 0], vr_left[:, 1], vr_left[:, 2],
                    c=[self.VR_LEFT_COLOR], s=38, marker="o", label="VR Left"
                )
            if len(vr_right) > 0:
                ax.scatter(
                    vr_right[:, 0], vr_right[:, 1], vr_right[:, 2],
                    c=[self.VR_RIGHT_COLOR], s=38, marker="o", label="VR Right"
                )

        # HaWoR: 21 点（Adapter 输出）
        if hawor_left is not None:
            l = self._finite_rows(hawor_left)
            if len(l) > 0:
                ax.scatter(
                    l[:, 0], l[:, 1], l[:, 2],
                    c=[self.HAWOR_LEFT_COLOR], s=32, marker="^", label="HaWoR Left"
                )

        if hawor_right is not None:
            r = self._finite_rows(hawor_right)
            if len(r) > 0:
                ax.scatter(
                    r[:, 0], r[:, 1], r[:, 2],
                    c=[self.HAWOR_RIGHT_COLOR], s=32, marker="^", label="HaWoR Right"
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"VR vs HaWoR ({stream_name}) - frame {frame_idx}")
        ax.legend(loc="upper left")

        self._set_equal_axes(ax, vr_left, vr_right, hawor_left, hawor_right)

        if output_image_path is not None:
            plt.savefig(output_image_path, dpi=120)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def create_video(
        self,
        vr_aligned: Dict,
        hawor_data: Dict,
        output_path: str,
        fps: int = 25,
    ) -> None:
        num_frames = min(
            len(vr_aligned["keypoints"]),
            len(hawor_data["keypoints_left"]),
            len(hawor_data["keypoints_right"]),
        )
        output_path = str(output_path)

        temp_dir = Path(output_path).parent / "_tmp_compare_frames"
        # if temp_dir.exists():
        #     shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        stream_name = hawor_data.get("stream", "smoothed_result")
        for i in range(num_frames):
            image_path = str(temp_dir / f"frame_{i:06d}.png")
            self.plot_frame(
                vr_keypoints=vr_aligned["keypoints"][i],
                hawor_left=hawor_data["keypoints_left"][i],
                hawor_right=hawor_data["keypoints_right"][i],
                frame_idx=i,
                stream_name=stream_name,
                output_image_path=image_path,
                show=False,
            )
            if i % 100 == 0:
                print(f"Processed frame {i}/{num_frames}")

        self._images_to_video(str(temp_dir), output_path, fps)
        shutil.rmtree(temp_dir)
        print(f"Video saved to: {output_path}")

    @staticmethod
    def _images_to_video(image_dir: str, output_path: str, fps: int) -> None:
        images = sorted(glob.glob(f"{image_dir}/frame_*.png"))
        if not images:
            raise RuntimeError("未找到中间图片帧，无法生成视频")

        first = cv2.imread(images[0])
        if first is None:
            raise RuntimeError(f"无法读取图片: {images[0]}")

        height, width = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)

        writer.release()


def build_result_dict_for_frame(vr_aligned: Dict, hawor_data: Dict, frame_idx: int) -> Dict:
    """输出单帧对比数据，便于外部二次可视化/分析。"""
    return {
        "frame_idx": frame_idx,
        "vr": {
            "timestamp": float(vr_aligned["timestamps"][frame_idx]),
            "keypoints": vr_aligned["keypoints"][frame_idx],
            "confidences": vr_aligned["confidences"][frame_idx],
        },
        "hawor": {
            "timestamp": float(hawor_data["timestamps"][frame_idx]),
            "stream": hawor_data.get("stream", "smoothed_result"),
            "keypoints_left": hawor_data["keypoints_left"][frame_idx],
            "keypoints_right": hawor_data["keypoints_right"][frame_idx],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VR JSON and HaWoR keypoints PKL")
    parser.add_argument("--vr-json", type=str, required=True, help="VR keypoints json path")
    parser.add_argument("--hawor-pkl", type=str, required=True, help="hawor pkl path")
    parser.add_argument(
        "--stream",
        type=str,
        default="smoothed_result",
        choices=["smoothed_result", "original_result"],
        help="Use which result stream from pkl",
    )
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "video"], help="visualization mode"
    )
    parser.add_argument("--frame", type=int, default=0, help="frame index for single mode")
    parser.add_argument("--output", type=str, default=None, help="output path for image/video")
    parser.add_argument("--fps", type=int, default=25, help="fps when mode=video")
    args = parser.parse_args()

    print("Loading VR json...")
    vr_loader = VRDataLoader(args.vr_json)
    _ = vr_loader.process()

    print("Loading HaWoR pkl...")
    hawor_loader = HaWoRResultLoader(args.hawor_pkl, stream=args.stream)
    hawor_data = hawor_loader.extract_keypoints()

    # 用 HaWoR 帧时间轴对齐 VR
    vr_aligned = vr_loader.interpolate_to_times(hawor_data["timestamps"])

    vis = KeypointVisualizer()

    if args.mode == "single":
        frame_idx = int(np.clip(args.frame, 0, hawor_data["num_frames"] - 1))
        result_dict = build_result_dict_for_frame(vr_aligned, hawor_data, frame_idx)
        print(f"Visualizing frame {frame_idx}, stream={args.stream}")
        vis.plot_frame(
            vr_keypoints=result_dict["vr"]["keypoints"],
            hawor_left=result_dict["hawor"]["keypoints_left"],
            hawor_right=result_dict["hawor"]["keypoints_right"],
            frame_idx=frame_idx,
            stream_name=args.stream,
            output_image_path=args.output,
            show=True,
        )
    else:
        if args.output is None:
            args.output = str(Path(args.vr_json).with_name(f"compare_{args.stream}.mp4"))
        print(f"Creating video: {args.output}")
        vis.create_video(vr_aligned, hawor_data, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
