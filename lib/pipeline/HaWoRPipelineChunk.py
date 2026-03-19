"""
HaWoRPipelineChunk - 大视频分段处理版本

此模块提供 HaWoRPipeline 的分段处理版本，用于降低大视频处理时的内存峰值占用。

设计思路：
1. reconstruct 方法在开始时对视频进行分块
2. 对每个分块调用 _reconstruct 方法处理（复用原始逻辑）
3. 合并各分块的结果

使用示例:
    from lib.pipeline.HaWoRPipelineChunk import HaWoRPipelineChunk, HaWoRConfig

    cfg = HaWoRConfig(verbose=True, device="cuda:0")
    pipeline = HaWoRPipelineChunk(cfg)

    # 分段处理，每1000帧一块，重叠120帧
    result = pipeline.reconstruct(
        "large_video.mp4",
        chunk_size=1000,
        overlap_frames=120
    )
"""

import gc
import glob
import math
import os
import subprocess
from typing import Optional

import cv2
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

from lib.pipeline.HaWoRPipeline import HaWoRPipeline, HaWoRConfig


class HaWoRPipelineChunk(HaWoRPipeline):
    """
    HaWoRPipeline 的大视频分段处理版本。

    通过在入口处分块处理、调用内部 _reconstruct 方法、最后合并结果，
    实现大视频的低内存处理。
    """

    def __init__(self, cfg: HaWoRConfig | None = None):
        super().__init__(cfg)
        # 分块处理配置
        self._chunk_results = []  # 存储各分块结果
        self._original_total_frames = 0  # 原始视频总帧数

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
        chunk_size: int = 1000,
        overlap_frames: int = 120,
    ) -> dict:
        """
        对单个视频执行分段重建 pipeline。

        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            start_idx: 起始帧索引
            end_idx: 结束帧索引（不含）
            image_focal: 相机焦距
            rendering: 是否渲染视频
            vis_mode: 渲染视角
            use_progress_bar: 是否显示进度条
            chunk_size: 每块的帧数，默认 1000
            overlap_frames: 块之间的重叠帧数，默认 120

        Returns:
            result: 包含重建结果的字典
        """
        # ── 预处理：获取视频总帧数并计算分块 ─────────────────────────────
        first_frame = self._extract_frames(video_path, 0, 1)
        if len(first_frame) == 0:
            raise RuntimeError(f"无法读取视频: {video_path}")

        # 获取完整视频信息
        full_video = self._extract_frames(video_path, 0, -1)
        self._original_total_frames = len(full_video)

        if self.verbose:
            print(f"[HaWoRChunk] 视频总帧数: {self._original_total_frames}")
            print(f"[HaWoRChunk] 分块配置: chunk_size={chunk_size}, overlap={overlap_frames}")

        # 如果视频帧数小于等于 chunk_size，直接使用原始逻辑不分块
        if self._original_total_frames <= chunk_size:
            if self.verbose:
                print(f"[HaWoRChunk] 视频帧数较小，使用标准模式处理")
            return super().reconstruct(
                video_path, output_dir, start_idx, end_idx,
                image_focal, rendering, vis_mode, use_progress_bar
            )

        # ── 计算分块边界 ───────────────────────────────────────────────────
        chunk_ranges = self._compute_chunk_ranges(
            self._original_total_frames, chunk_size, overlap_frames, start_idx, end_idx
        )

        if self.verbose:
            print(f"[HaWoRChunk] 将处理 {len(chunk_ranges)} 个分块:")
            for i, (s, e) in enumerate(chunk_ranges):
                print(f"  Chunk {i+1}: 帧 {s}-{e} (共 {e-s} 帧)")

        # ── 设置进度条（如果启用）───────────────────────────────────────────
        self.progress_percentage = 0.0
        if use_progress_bar:
            self._overall_pbar = tqdm(
                total=len(chunk_ranges),
                desc="Processing chunks",
                unit="chunk",
                position=0
            )
        else:
            self._overall_pbar = None

        # ── 处理每个分块 ───────────────────────────────────────────────────
        self._chunk_results = []

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            if self.verbose:
                print(f"\n[HaWoRChunk] ========== 处理分块 {chunk_idx+1}/{len(chunk_ranges)} ==========")
                print(f"[HaWoRChunk] 帧范围: {chunk_start} - {chunk_end}")

            # 创建分块专用的输出目录（可选，用于调试）
            chunk_output_dir = os.path.join(output_dir, f".chunk_{chunk_idx}")

            try:
                # 调用内部 _reconstruct 方法处理当前分块
                chunk_result = self._reconstruct(
                    video_path=video_path,
                    output_dir=chunk_output_dir,
                    start_idx=chunk_start,
                    end_idx=chunk_end,
                    image_focal=image_focal,
                    rendering=False,  # 分块处理时不渲染
                    vis_mode=vis_mode,
                    use_progress_bar=False  # 使用总进度条
                )
                self._chunk_results.append((chunk_start, chunk_end, chunk_result))

                # 更新总进度
                progress = (chunk_idx + 1) / len(chunk_ranges)
                self.progress_percentage = progress
                if self._overall_pbar:
                    self._overall_pbar.update(1)
                    self._overall_pbar.set_postfix_str(f"{progress*100:.1f}%")

                if self.verbose:
                    print(f"[HaWoRChunk] 分块 {chunk_idx+1} 完成")

            except Exception as e:
                import traceback
                print(f"[HaWoRChunk] 分块 {chunk_idx+1} 处理失败: {e}")
                print(traceback.format_exc())
                raise

        # ── 关闭总进度条 ───────────────────────────────────────────────────
        if self._overall_pbar:
            self._overall_pbar.close()
            self._overall_pbar = None

        # ── 合并分块结果 ───────────────────────────────────────────────────
        if self.verbose:
            print(f"\n[HaWoRChunk] ========== 合并 {len(self._chunk_results)} 个分块结果 ==========")

        merged_result = self._merge_chunk_results(
            self._chunk_results, chunk_ranges, overlap_frames, smoothed=False
        )
        
        merged_result["smoothed_result"] = self._merge_chunk_results(
            self._chunk_results, chunk_ranges, overlap_frames, smoothed=True
        )

        # ── 清理分块临时数据 ───────────────────────────────────────────────
        self._chunk_results = []
        torch.cuda.empty_cache()
        gc.collect()

        # ── 可选：渲染合并后的结果 ───────────────────────────────────────────
        if rendering:
            if self.verbose:
                print(f"\n[HaWoRChunk] ========== 渲染视频 ==========")
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
            imgfiles = natsorted(glob.glob(f'{img_folder}/*.jpg'))
            if len(imgfiles) > 0:
                print("Skip extracting frames")
            else:
                _ = extract_frames(file, img_folder)
            imgfiles = natsorted(glob.glob(f'{img_folder}/*.jpg'))
            result_render = merged_result['smoothed_result'] if self.smooth_hands or self.smooth_camera else merged_result
            rendered_video = self._render(
                result=result_render,
                imgfiles=imgfiles,
                vis_start=0,
                vis_end=len(imgfiles) - 1,
                output_dir=output_dir,
                vis_mode=vis_mode,
                video_path=video_path,
            )
            merged_result["seq_folder"]=seq_folder
            merged_result["rendered_video"] = rendered_video

            merged_result["rendered_video"] = rendered_video

            if self.verbose and rendered_video:
                print(f"[HaWoRChunk] 渲染视频已保存: {rendered_video}")

        return merged_result

    def _compute_chunk_ranges(self, total_frames: int, chunk_size: int,
                              overlap_frames: int, start_idx: int,
                              end_idx: int | None) -> list[tuple[int, int]]:
        """计算分块边界"""
        # 处理 end_idx
        if end_idx is None or end_idx == -1:
            end_idx = total_frames
        else:
            end_idx = min(end_idx, total_frames)

        # 调整起始位置
        actual_start = max(0, start_idx)
        actual_end = min(total_frames, end_idx)

        chunk_ranges = []
        current = actual_start

        while current < actual_end:
            chunk_end = min(current + chunk_size, actual_end)
            chunk_ranges.append((current, chunk_end))
            if chunk_end >= actual_end:
                break
            current = chunk_end - overlap_frames

        return chunk_ranges

    def _merge_chunk_results(
        self,
        chunk_results: list[tuple[int, int, dict]],
        chunk_ranges: list[tuple[int, int]],
        overlap_frames: int,
        smoothed: bool = False
    ) -> dict:
        """
        合并多个分块的结果。

        策略：
        1. 对于非重叠区域，直接使用该分块的结果
        2. 对于重叠区域，使用后一个分块的结果（因为其有更多上下文）
        """
        if not chunk_results:
            raise RuntimeError("没有可合并的分块结果")

        # 获取第一块的结果作为模板
        _, _, first_result = chunk_results[0]

        # 初始化合并后的张量
        total_frames = self._original_total_frames
        merged = {
            "pred_trans": torch.zeros(2, total_frames, 3),
            "pred_rot": torch.zeros(2, total_frames, 3),
            "pred_hand_pose": torch.zeros(2, total_frames, 45),
            "pred_betas": torch.zeros(2, total_frames, 10),
            "pred_valid": torch.zeros(2, total_frames),
        }

        # 记录相机位姿（需要特殊处理，因为有全局坐标）
        R_c2w_list = []
        t_c2w_list = []
        R_w2c_list = []
        t_w2c_list = []
        frame_ranges = []

        # 处理每个分块
        for chunk_idx, (chunk_start, chunk_end, chunk_result) in enumerate(chunk_results):
            if smoothed:
                chunk_result = chunk_result.get("smoothed_result", chunk_result)
            chunk_length = chunk_end - chunk_start

            # 计算要复制的范围（处理重叠）
            if chunk_idx == 0:
                # 第一块：全部复制
                copy_start = chunk_start
                copy_end = chunk_end
                local_start = 0
                local_end = chunk_length
            elif chunk_idx == len(chunk_results) - 1:
                # 最后一块：从上一个块的结束位置开始
                prev_end = chunk_ranges[chunk_idx - 1][1]
                copy_start = prev_end - overlap_frames
                copy_end = chunk_end
                local_start = copy_start - chunk_start
                local_end = chunk_length
            else:
                # 中间块：只取非重叠部分
                prev_end = chunk_ranges[chunk_idx - 1][1]
                copy_start = prev_end - overlap_frames
                copy_end = chunk_end
                local_start = copy_start - chunk_start
                local_end = chunk_length

            # 复制手部数据（确保类型一致）
            merged["pred_trans"][:, copy_start:copy_end] = \
                torch.as_tensor(chunk_result["pred_trans"][:, local_start:local_end], dtype=merged["pred_trans"].dtype)
            merged["pred_rot"][:, copy_start:copy_end] = \
                torch.as_tensor(chunk_result["pred_rot"][:, local_start:local_end], dtype=merged["pred_rot"].dtype)
            merged["pred_hand_pose"][:, copy_start:copy_end] = \
                torch.as_tensor(chunk_result["pred_hand_pose"][:, local_start:local_end], dtype=merged["pred_hand_pose"].dtype)
            merged["pred_betas"][:, copy_start:copy_end] = \
                torch.as_tensor(chunk_result["pred_betas"][:, local_start:local_end], dtype=merged["pred_betas"].dtype)

            # pred_valid 可能是 numpy 或 torch，需要统一处理
            valid_data = chunk_result["pred_valid"][:, local_start:local_end]
            if isinstance(valid_data, np.ndarray):
                valid_data = torch.from_numpy(valid_data)
            merged["pred_valid"][:, copy_start:copy_end] = valid_data.to(merged["pred_valid"].dtype)

            # 保存相机位姿
            if "R_c2w" in chunk_result:
                R_c2w_list.append((copy_start, copy_end, chunk_result["R_c2w"]))
            if "t_c2w" in chunk_result:
                t_c2w_list.append((copy_start, copy_end, chunk_result["t_c2w"]))
            if "R_w2c" in chunk_result:
                R_w2c_list.append((copy_start, copy_end, chunk_result["R_w2c"]))
            if "t_w2c" in chunk_result:
                t_w2c_list.append((copy_start, copy_end, chunk_result["t_w2c"]))

        # 合并相机位姿（确保类型一致）
        if R_c2w_list:
            R_c2w_merged = torch.zeros(total_frames, 3, 3)
            for s, e, data in R_c2w_list:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                R_c2w_merged[s:e] = data
            merged["R_c2w"] = R_c2w_merged

        if t_c2w_list:
            t_c2w_merged = torch.zeros(total_frames, 3)
            for s, e, data in t_c2w_list:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                t_c2w_merged[s:e] = data
            merged["t_c2w"] = t_c2w_merged

        if R_w2c_list:
            R_w2c_merged = torch.zeros(total_frames, 3, 3)
            for s, e, data in R_w2c_list:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                R_w2c_merged[s:e] = data
            merged["R_w2c"] = R_w2c_merged

        if t_w2c_list:
            t_w2c_merged = torch.zeros(total_frames, 3)
            for s, e, data in t_w2c_list:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                t_w2c_merged[s:e] = data
            merged["t_w2c"] = t_w2c_merged

        # 构建完整的手部网格字典（分段构建以降低内存）
        if self.verbose:
            print(f"[HaWoRChunk] 构建手部网格字典...")

        from lib.pipeline.HaWoRPipeline import _FACES_NEW, _R_X, _apply_coord_transform
        from hawor.utils.process import run_mano, run_mano_left

        faces_base = self.mano.faces
        faces_right = np.concatenate([faces_base, _FACES_NEW], axis=0)
        faces_left = faces_right[:, [0, 2, 1]]

        # 分段构建手部网格
        right_vertices_list = []
        left_vertices_list = []
        mesh_chunk_size = 500  # 手部网格的分块大小

        for mesh_start in range(0, total_frames, mesh_chunk_size):
            mesh_end = min(mesh_start + mesh_chunk_size, total_frames)

            # 右手
            pred_glob_r = run_mano(
                merged["pred_trans"][1:2, mesh_start:mesh_end],
                merged["pred_rot"][1:2, mesh_start:mesh_end],
                merged["pred_hand_pose"][1:2, mesh_start:mesh_end],
                betas=merged["pred_betas"][1:2, mesh_start:mesh_end],
                mano=self.mano,
                device=self.device
            )
            right_vertices_list.append(pred_glob_r["vertices"][0].cpu())

            # 左手
            pred_glob_l = run_mano_left(
                merged["pred_trans"][0:1, mesh_start:mesh_end],
                merged["pred_rot"][0:1, mesh_start:mesh_end],
                merged["pred_hand_pose"][0:1, mesh_start:mesh_end],
                betas=merged["pred_betas"][0:1, mesh_start:mesh_end],
                mano=self.mano_left,
                device=self.device
            )
            left_vertices_list.append(pred_glob_l["vertices"][0].cpu())

            # 及时清理
            torch.cuda.empty_cache()
            gc.collect()

        # 合并手部网格
        right_vertices = torch.cat(right_vertices_list, dim=0).unsqueeze(0)
        left_vertices = torch.cat(left_vertices_list, dim=0).unsqueeze(0)

        merged["right_dict"] = {"vertices": right_vertices, "faces": faces_right}
        merged["left_dict"] = {"vertices": left_vertices, "faces": faces_left}

        # 坐标系变换
        (merged["right_dict"], merged["left_dict"],
         merged["R_w2c"], merged["t_w2c"],
         merged["R_c2w"], merged["t_c2w"]) = _apply_coord_transform(
            merged["right_dict"], merged["left_dict"],
            merged["R_c2w"], merged["t_c2w"]
        )

        # 添加其他元数据（从第一个分块结果获取）
        first_result = chunk_results[0][2]
        merged["img_focal"] = first_result.get("img_focal", 600)
        merged["rendered_video"] = None
        merged["seq_folder"] = None
        merged["smooth_hand_enabled"] = first_result.get("smooth_hand_enabled", False)
        merged["smooth_camera_enabled"] = first_result.get("smooth_camera_enabled", False)

        return merged

    def _render_merged_result(
        self,
        video_path: str,
        result: dict,
        output_dir: str,
        vis_mode: str
    ) -> str | None:
        """
        渲染合并后的结果。

        需要提取所有帧到临时目录，然后调用渲染函数。
        """
        # 提取视频帧
        file = video_path
        root = os.path.dirname(file)
        seq = os.path.basename(file).split('.')[0]

        seq_folder = os.path.join(output_dir, seq)
        img_folder = os.path.join(seq_folder, 'extracted_images')
        os.makedirs(seq_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)

        # 使用 ffmpeg 提取帧
        self._extract_frames_to_folder(video_path, img_folder)

        # 获取图像文件列表
        imgfiles = natsorted(glob.glob(os.path.join(img_folder, '*.jpg')))

        if len(imgfiles) == 0:
            print(f"[HaWoRChunk] 警告: 未能提取图像帧")
            return None

        if self.verbose:
            print(f"[HaWoRChunk] 提取了 {len(imgfiles)} 帧，开始渲染...")

        # 调用渲染函数
        try:
            rendered_video = self._render(
                result=result,
                imgfiles=imgfiles,
                vis_start=0,
                vis_end=len(imgfiles),
                output_dir=output_dir,
                vis_mode=vis_mode,
                video_path=video_path,
            )
            return rendered_video
        except Exception as e:
            import traceback
            print(f"[HaWoRChunk] 渲染失败: {e}")
            print(traceback.format_exc())
            return None

    def _extract_frames_to_folder(self, video_path: str, output_folder: str):
        """使用 ffmpeg 提取视频帧到文件夹"""
        import glob

        # 检查是否已经提取过
        existing_files = glob.glob(os.path.join(output_folder, '*.jpg'))
        if len(existing_files) > 0:
            if self.verbose:
                print(f"[HaWoRChunk] 跳过帧提取（已存在 {len(existing_files)} 个文件）")
            return

        # 使用 ffmpeg 提取帧
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=30',
            '-start_number', '0',
            '-q:v', '2',  # 高质量
            os.path.join(output_folder, '%04d.jpg')
        ]

        if self.verbose:
            print(f"[HaWoRChunk] 提取视频帧: {' '.join(command)}")

        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"[HaWoRChunk] ffmpeg 执行失败: {e}")
            if e.stderr:
                print(f"[HaWoRChunk] stderr: {e.stderr.decode()}")

    def _reconstruct(
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
        内部重建方法，处理指定范围的帧。

        直接复用父类的 reconstruct 逻辑，通过传递 start_idx 和 end_idx 参数来处理指定范围。

        Returns:
            result: 包含重建结果的字典
        """
        # 复用父类的 reconstruct 逻辑
        # 通过传递 start_idx 和 end_idx 参数来处理指定范围
        return super().reconstruct(
            video_path=video_path,
            output_dir=output_dir,
            start_idx=start_idx,
            end_idx=end_idx,
            image_focal=image_focal,
            rendering=rendering,
            vis_mode=vis_mode,
            use_progress_bar=use_progress_bar,
        )

    def _extract_frames(self, video_path: str, start_idx: int = 0,
                       end_idx: int | None = -1, frame_step: int = 1):
        """
        提取视频帧（支持索引切片）。

        如果是分块模式且已经在初始化时获取了完整视频，
        则直接切片返回，避免重复读取。
        """
        # 如果已经缓存了完整视频，直接切片
        if hasattr(self, '_cached_full_video') and self._cached_full_video is not None:
            full_video = self._cached_full_video
            if end_idx is None or end_idx == -1:
                end_idx = len(full_video)
            result = full_video[start_idx:end_idx:frame_step]
            return result

        # 否则调用父类方法
        return super()._extract_frames(video_path, start_idx, end_idx, frame_step)
