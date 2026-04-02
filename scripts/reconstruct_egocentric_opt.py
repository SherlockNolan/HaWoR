import os
import glob
import pickle
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
import threading
import queue as queue_module
import torch
import time
import os
import contextlib
# --- HaWoR Imports ---
# 将当前脚本的父目录（即根目录）加入路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# NOTE: HaWoR pipeline and adapter import are intentionally delayed
# to avoid initializing CUDA at module import time in child processes.
# They will be imported inside worker/main branches after setting
# `CUDA_VISIBLE_DEVICES` appropriately.

"""
python scripts/reconstruct_egocentric.py --video-path=\
"/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0.mp4" \
--output="results/" \
--start=0 --end=1 --no-interleave --num-workers=1 --save-origin


python scripts/reconstruct_egocentric.py \
    --output="results/" \
    --num-workers=4 --test

# 2
python scripts/reconstruct_egocentric_opt.py \
    --start=0 --end=100000 --frame-end-idx=2000 \
    --num-workers=72

# 3
python scripts/reconstruct_egocentric_opt.py \
    --start=100000 --end=200000 --frame-end-idx=2000 \
    --num-workers=72

python scripts/reconstruct_egocentric_opt.py \
    --start=100000 --end=200000 --frame-end-idx=2000 \
    --num-workers=32 --inverse

24 
100不行


python scripts/reconstruct_egocentric.py \
    --start=100000 --end=100007 \
    --num-workers=2
    
python scripts/reconstruct_egocentric.py \
    --start=100007 --end=100015 \
    --num-workers=2
"""


#--- Default Configuration ---
DEFAULT_DATASET_ROOT = "/inspire/dataset/egocentric-10k/v20251211"
DEFAULT_OUTPUT_ROOT = (
    "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/egocentric-10k-hawor-2000frames"
)
DEFAULT_DATASET_ROOT_TEST = "./test_video"
DEFAULT_OUTPUT_ROOT_TEST = (
    "./test_results"
)

# 用于传递额外参数的 dict，替代全局变量


def find_all_mp4_files(root_dir: str) -> list[str]:
    """Scans a directory recursively for all .mp4 files."""
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory '{root_dir}' not found.")
        return []
    pattern = os.path.join(root_dir, "**", "*.mp4")
    mp4_files = glob.glob(pattern, recursive=True)
    mp4_files.sort()
    print(f"Found {len(mp4_files)} MP4 files in '{root_dir}'.")
    return mp4_files


def reorder_videos_by_factory_worker(all_vids: list[str], root_dir: str) -> list[str]:
    """Reorder videos so that we round-robin across factories and their workers.

    Expected folder layout (relative to `root_dir`):
      factory_XXX/workers/worker_YYY/<videos...>

    Behavior: for round r=0.., pick the r-th video from each worker in sorted(factory)->sorted(worker) order.
    Videos that don't match the structure are appended at the end in sorted order.
    """
    # Build mapping: factory -> worker -> [videos]
    mapping: dict[str, dict[str, list[str]]] = {}
    others: list[str] = []

    for v in all_vids:
        try:
            rel = os.path.relpath(v, root_dir)
        except Exception:
            rel = os.path.basename(v)
        parts = rel.split(os.sep)
        if len(parts) >= 3 and parts[1] == "workers":
            factory = parts[0]
            worker = parts[2]
            mapping.setdefault(factory, {}).setdefault(worker, []).append(v)
        else:
            # not following expected structure
            others.append(v)

    # Sort videos inside each worker for deterministic order
    for factory in mapping:
        for worker in mapping[factory]:
            mapping[factory][worker].sort()

    # Determine max number of shards per worker
    max_shards = 0
    for factory in mapping:
        for worker in mapping[factory]:
            max_shards = max(max_shards, len(mapping[factory][worker]))

    ordered: list[str] = []
    factories_sorted = sorted(mapping.keys())
    for r in range(max_shards):
        for factory in factories_sorted:
            workers_sorted = sorted(mapping[factory].keys())
            for worker in workers_sorted:
                vids = mapping[factory][worker]
                if r < len(vids):
                    ordered.append(vids[r])

    # Append any remaining videos that didn't match structure or were missed
    others.sort()
    ordered.extend(others)
    # As a final safety, append any videos from original list that were not included yet
    included = set(ordered)
    for v in all_vids:
        if v not in included:
            ordered.append(v)

    return ordered


# 进程级全局 pipeline 缓存（每个子进程独立持有）
_process_pipe = None
_process_device = None
_process_dtype = None
# Placeholders for lazy imports (set inside worker or main when needed)
HaWoRPipeline = None
HaWoRConfig = None
convert_hawor_to_keypoints = None


@contextlib.contextmanager
def _suppress_all_output(enabled=True):
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)

def _worker_process_main(
    worker_id: int, device_str: str, dtype, task_queue, result_queue, init_queue
):
    """自定义进程工作函数：每个进程绑定指定 GPU，循环从任务队列取任务执行。

    相比 ProcessPoolExecutor 的 initializer 只能统一参数，这里可以为每个
    进程精确分配不同 GPU，实现真正的多 GPU 并行推理。

    Args:
        worker_id:    进程编号（用于日志）
        device_str:   分配给此进程的 GPU，如 "cuda:0"
        dtype:        模型推理精度
        task_queue:   任务队列，每项为 (video_path, video_idx, progress_queue, extra_args) 或 None（哨兵值）
        result_queue: 结果队列，每项为 (video_idx, status, message)
        init_queue:   初始化完成后上报 worker_id，主进程用于显示初始化进度
    """
    # ── 关键修复：在任何 CUDA 操作之前设置 CUDA_VISIBLE_DEVICES ──────────
    # DROID-SLAM 内部硬编码使用 "cuda:0"（droid.py、depth_video.py、
    # motion_filter.py 均如此）。在多 GPU 场景下，若进程的 CUDA context
    # 已初始化在 cuda:1/2/... 上，自定义 CUDA 算子（如 CorrSampler）会按
    # 该 GPU 架构编译/加载；之后 DROID-SLAM 强行访问 cuda:0（不同架构），
    # 就会触发 "no kernel image is available for execution on the device"。
    # 解决方案：限制本进程只能看到分配的物理 GPU，使其在进程内以 cuda:0
    # 身份出现，DROID-SLAM 的硬编码 cuda:0 便自动指向正确的物理 GPU。
    gpu_idx = device_str.split(":")[-1] if ":" in device_str else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    device_str = "cuda:0"  # 限制可见性后，进程内 cuda:0 即为分配的物理 GPU
    # ─────────────────────────────────────────────────────────────────────

    global _process_pipe
    pid = os.getpid()
    device = torch.device(device_str)
    print(f"[Worker {worker_id} / PID {pid}] 初始化 pipeline 在 物理GPU {gpu_idx} (进程内为 {device})", flush=True)
    
    # 延迟导入 HaWoR，以保证在设置好 CUDA_VISIBLE_DEVICES 后再触发 CUDA 初始化
    try:
        from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig
        from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints as _convert_fn
        # 将 adapter 函数注入到模块全局，这样其他 worker 内部函数可以使用它
        globals()["convert_hawor_to_keypoints"] = _convert_fn
    except Exception:
        # 如果导入失败，仍然让错误冒出来以便主进程获知
        raise

    # 使用 HaWoRPipeline
    cfg = HaWoRConfig(verbose=False, device=device_str)
    _process_pipe = HaWoRPipelineOpt(cfg)
    
    # 通知主进程：本 Worker 初始化完成
    init_queue.put((worker_id, device_str))
    print(f"[Worker {worker_id} / PID {pid}] 就绪，等待任务...", flush=True)

    while True:
        task = task_queue.get()
        if task is None:  # 哨兵值，退出
            break
        video_path, video_idx, progress_queue, extra_args = task
        try:
            msg = process_video_worker_proc(
                video_path,
                video_idx,
                progress_queue,
                worker_id,
                extra_args=extra_args,
            )
            result_queue.put((video_idx, "ok", msg))
        except Exception as e:
            # 捕获所有异常，记录后继续处理下一个任务，进程不退出
            import traceback

            err_msg = f"{os.path.basename(video_path)} -> {traceback.format_exc()}"
            result_queue.put((video_idx, "error", err_msg))


def process_video_worker_proc(
    video_path,
    video_idx,
    progress_queue,
    worker_id=0,
    extra_args=None,
):
    """Worker function for use in ProcessPoolExecutor.

    每个子进程在进程初始化时已经创建好自己的 pipeline（_process_pipe），
    此处直接使用，不再重复创建，彻底绕开 GIL。
    """
    global _process_pipe
    if extra_args is None:
        extra_args = {}
    dataset_root = extra_args.get("dataset_root", "")
    output_root = extra_args.get("output_root", "")
    save_origin = extra_args.get("save_origin", False)
    frame_start_idx = extra_args.get("frame_start_idx", 0)
    frame_end_idx = extra_args.get("frame_end_idx", -1)
    pkl_path = video_path.replace(".mp4", "_hawor.pkl")
    pkl_path = pkl_path.replace(dataset_root, output_root)
    pkl_dir = os.path.dirname(pkl_path)
    if os.path.exists(pkl_path):
        if progress_queue is not None:
            progress_queue.put(("done", video_idx, 0, worker_id))
        return f"Skipped (already exists): {os.path.basename(video_path)}"
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir, exist_ok=True)

    success = False
    try:
        if _process_pipe is None:
            raise RuntimeError("Worker pipeline 未初始化。")

        # 启动进度监听线程（针对 HaWoR 的百分比进度）
        stop_monitoring = threading.Event()
        def monitor_progress():
            last_p = -1.0
            while not stop_monitoring.is_set():
                p = getattr(_process_pipe, "progress_percentage", 0.0)
                if p != last_p:
                    if progress_queue is not None:
                        progress_queue.put(("progress", video_idx, p, worker_id))
                    last_p = p
                time.sleep(0.1)
        
        mon_thread = threading.Thread(target=monitor_progress, daemon=True)
        mon_thread.start()

        if progress_queue is not None:
             progress_queue.put(("init", video_idx, 1.0, worker_id, os.path.basename(video_path)))

        
        with _suppress_all_output(enabled=True):
                result_dict_origin = _process_pipe.reconstruct(video_path, output_dir=pkl_dir, image_focal=1031, start_idx=frame_start_idx, end_idx=frame_end_idx) # 屏蔽原有输出
                result_dict = dict()
                # Ensure adapter is imported in this process
                if globals().get("convert_hawor_to_keypoints") is None:
                    from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints as _convert_fn
                    globals()["convert_hawor_to_keypoints"] = _convert_fn
                _convert = globals()["convert_hawor_to_keypoints"]
                result_dict["original_result"] = _convert(result_dict_origin, video_path, use_smoothed=False)
                result_dict["smoothed_result"] = _convert(result_dict_origin, video_path, use_smoothed=True) if result_dict_origin["smoothed_result"] is not None else None

        # result_dict = _process_pipe.reconstruct(video_path, output_dir=pkl_dir) # 有完整输出信息的
        
        stop_monitoring.set()
        mon_thread.join()

        if result_dict:
            with open(pkl_path, "wb") as f:
                pickle.dump(result_dict, f)
            if save_origin:
                with open(pkl_path.replace(".pkl", "_origin_dict.pkl"), "wb") as f:
                    pickle.dump(result_dict_origin, f)
            success = True
            return f"Success: {os.path.basename(video_path)}"
        else:
            success = True
            return f"Warning (no data): {os.path.basename(video_path)}"
    except Exception as e:
        import traceback

        err_detail = traceback.format_exc()
        if progress_queue is not None:
            progress_queue.put(
                ("error_video", video_idx, f"{os.path.basename(video_path)} -> {err_detail}", worker_id)
            )
        # 不再 raise，直接返回错误字符串，让 _worker_process_main 写入 result_queue
        print(err_detail)
        return f"Error: {os.path.basename(video_path)} ->\n{err_detail}"
    finally:
        if success and progress_queue is not None:
            progress_queue.put(("done", video_idx, 1.0, worker_id))


def process_video_worker(video_path, pipe, extra_args=None):
    """
    A worker function that handles processing and saving for a single video.
    Designed to be called from a ProcessPoolExecutor.
    """
    if extra_args is None:
        extra_args = {}
    dataset_root = extra_args.get("dataset_root", "")
    output_root = extra_args.get("output_root", "")
    save_origin = extra_args.get("save_origin", False)
    frame_start_idx = extra_args.get("frame_start_idx", 0)
    frame_end_idx = extra_args.get("frame_end_idx", -1)
    pkl_path = video_path.replace(".mp4", "_hawor.pkl")
    pkl_path = pkl_path.replace(dataset_root, output_root)
    pkl_dir = os.path.dirname(pkl_path)
    if os.path.exists(pkl_path):
        return f"Skipped (already exists): {os.path.basename(video_path)}"
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir, exist_ok=True)
    try:
        with _suppress_all_output(enabled=True):
            result_dict_origin = pipe.reconstruct(video_path, output_dir=pkl_dir, image_focal=1031, start_idx=frame_start_idx, end_idx=frame_end_idx)
            result_dict = dict()
            # Ensure adapter available in this process
            if globals().get("convert_hawor_to_keypoints") is None:
                from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints as _convert_fn
                globals()["convert_hawor_to_keypoints"] = _convert_fn
            _convert = globals()["convert_hawor_to_keypoints"]
            result_dict["original_result"] = _convert(result_dict_origin, video_path, use_smoothed=False)
            result_dict["smoothed_result"] = _convert(result_dict_origin, video_path, use_smoothed=True) if result_dict["smoothed_result"] is not None else None

        if result_dict:
            with open(pkl_path, "wb") as f:
                pickle.dump(result_dict, f)
            if save_origin:
                with open(pkl_path.replace(".pkl", "_origin_dict.pkl"), "wb") as f:
                    pickle.dump(result_dict_origin, f)
            return f"Success: {os.path.basename(video_path)}"
        else:
            return f"Warning (no data): {os.path.basename(video_path)}"
    except Exception as e:
        import traceback
        import contextlib
        return f"Error: {os.path.basename(video_path)} -> {e}\n{traceback.format_exc()}"


def _get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def check_single_pkl(pkl_path, verbose=True):
    if not os.path.exists(pkl_path):
        if verbose:
            print(f"Error: File not found - {pkl_path}")
        return False
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        if verbose:
            print(f"Error loading {pkl_path}: {str(e)}")
        # delete crushed files

        try:
            os.remove(pkl_path)
            print(f"文件 '{pkl_path}' 删除成功。")
        except FileNotFoundError:
            print(f"文件 '{pkl_path}' 不存在，无法删除。")
        except OSError as e:
            print(f"删除文件时出错：{e}")

        return False
    if not isinstance(data, dict):
        if verbose:
            print(f"Invalid structure: {pkl_path} is not a dict")
        return False
    # HaWoR result keys check
    required_keys = ['original_result', 'smoothed_result']
    for k in required_keys:
        if k not in data:
            if verbose:
                print(f"Missing key {k} in {pkl_path}")
            return False
    return True


def check_pkl_files(video_list, verbose=True, extra_args=None):
    # ... (code is identical)
    results = {}
    if extra_args is None:
        extra_args = {}
    dataset_root = extra_args.get("dataset_root", "")
    output_root = extra_args.get("output_root", "")
    for video_path in tqdm(video_list, desc="Checking pkl files", unit="video"):
        pkl_path = video_path.replace(".mp4", "_hawor.pkl")
        pkl_path = pkl_path.replace(dataset_root, output_root)
        valid = check_single_pkl(pkl_path, verbose=verbose)
        results[pkl_path] = valid
    return results


def print_summary(results):
    # ... (code is identical)
    total = len(results)
    if total == 0:
        print("\nSummary: No files to check.")
        return
    valid_count = sum(1 for v in results.values() if v)
    invalid_files = [k for k, v in results.items() if not v]
    print(f"\nSummary:")
    print(f"Total files checked: {total}")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {len(invalid_files)}")

def process_multi_workers(args, dtype, selected_vid_list, extra_args=None):
    if extra_args is None:
        extra_args = {}
    # 检测所有可用 GPU，按进程数轮询分配
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_devices = [f"cuda:{i}" for i in range(num_gpus)]
        print(f"检测到 {num_gpus} 张 GPU: {gpu_devices}")
    else:
        gpu_devices = ["cpu"]
        print("未检测到 GPU，使用 CPU。")

    # 为每个进程分配一张 GPU（轮询）
    num_workers = args.num_workers
    device_assignments = [
        gpu_devices[i % len(gpu_devices)] for i in range(num_workers)
    ]
    print(f"进程数: {num_workers}，进程-GPU 分配: {device_assignments}")

    # 使用 multiprocessing.Manager 创建跨进程共享队列
    mp_ctx = multiprocessing.get_context("spawn")
    mp_manager = multiprocessing.Manager()
    progress_queue = mp_manager.Queue()  # 子进程 -> 主进程的进度上报
    task_queue = mp_manager.Queue()  # 主进程 -> 子进程的任务分发
    result_queue = mp_manager.Queue()  # 子进程 -> 主进程的结果上报

    # 启动自定义进程池：每个进程绑定特定 GPU
    init_queue = mp_manager.Queue()  # 子进程 -> 主进程的初始化完成通知
    workers = []
    for wid, dev_str in enumerate(device_assignments):
        p = mp_ctx.Process(
            target=_worker_process_main,
            args=(wid, dev_str, dtype, task_queue, result_queue, init_queue),
            daemon=True,
            name=f"Worker-{wid}",
        )
        p.start()
        workers.append(p)

    # 等待所有 Worker 初始化完成（含进度条显示）
    print(f"等待 {num_workers} 个 Worker 完成 GPU/Pipeline 初始化...", flush=True)
    with tqdm(
        total=num_workers,
        desc="Worker Init",
        unit="worker",
        position=0,
        dynamic_ncols=True
    ) as init_pbar:
        ready_count = 0
        while ready_count < num_workers:
            try:
                wid, dev = init_queue.get() 
                init_pbar.set_postfix_str(f"Worker-{wid} on {dev} ready")
                init_pbar.update(1)
                ready_count += 1
            except Exception:
                # 检查是否有进程已死
                dead = [p for p in workers if not p.is_alive() and p.exitcode not in (0, None)]
                if dead:
                    raise RuntimeError(
                        f"Worker 初始化期间进程意外退出: "
                        f"{[f'{p.name}(exitcode={p.exitcode})' for p in dead]}"
                    )
    print("所有 Worker 初始化完成，开始投递任务。", flush=True)

    # 向任务队列投递所有视频任务
    for idx, video_path in enumerate(selected_vid_list):
        task_queue.put((video_path, idx, progress_queue, extra_args))
    # 投递哨兵值，每个进程收到后退出
    for _ in workers:
        task_queue.put(None)

    # 每个 worker 一个进度条，按进程 id 固定位置
    # worker_frame_counts[wid] 保存该 worker 当前视频的总帧数
    worker_frame_counts = [None] * num_workers

    # 总进度条（position=0）+ 每个 worker 的进度条（position=1..num_workers）
    total_pbar = tqdm(
        total=len(selected_vid_list),
        desc="All videos",
        position=0,
        unit="video",
        dynamic_ncols=True,
    )
    worker_pbars = [
        tqdm(
            total=100,
            desc=f"Worker-{i} [idle]",
            position=i + 1,
            leave=True,
            unit="frame",
            dynamic_ncols=True,
        )
        for i in range(num_workers)
    ]

    stop_event = threading.Event()
    # 已完成的视频数
    finished = [False] * len(selected_vid_list)

    def _worker_progress_handler(wid, vidx, payload, msg_type, worker_pbars, worker_frame_counts, video_name=None):
        bar = worker_pbars[wid]
        if msg_type == "init":
            # 对于 HaWoR，payload 约定为 1.0 (表示 100%)
            bar.total = 100
            bar.n = 0
            desc_name = video_name[:28] if video_name is not None else "[idle]"
            bar.set_description(f"W{wid} {desc_name}")
            bar.refresh()
        elif msg_type == "progress":
            # payload 是 0-1 的浮点数
            new_n = int(payload * 100)
            if new_n > bar.n:
                bar.n = new_n
                bar.refresh()
        elif msg_type == "done":
            bar.n = 100
            bar.refresh()

    def progress_listener():
        """主进程监听线程：同时消费 progress_queue（帧进度）和 result_queue（完成通知）。"""
        while not stop_event.is_set():
            # --- 消费进度 ---
            while True:
                try:
                    item = progress_queue.get_nowait()
                except Exception:
                    break
                msg_type = item[0]
                vidx = item[1]
                payload = item[2]
                wid = item[3] if len(item) > 3 else 0
                
                video_name = item[4] if len(item) > 4 else None
                _worker_progress_handler(wid, vidx, payload, msg_type, worker_pbars, worker_frame_counts, video_name)
                
                if msg_type == "error_video":
                    tqdm.write(f"[ERROR] {payload}")

            # --- 消费结果（视频完成）---
            while True:
                try:
                    vidx, status, msg = result_queue.get_nowait()
                except Exception:
                    break
                if not finished[vidx]:
                    finished[vidx] = True
                    total_pbar.update(1)

                if status == "ok":
                    tqdm.write(msg)
                else:
                    # 单个视频错误，只记录，不停止整体流程
                    tqdm.write(f"[ERROR] {msg}")

            if all(finished):
                break

            stop_event.wait(timeout=0.1)  # 短暂休眠，避免忙等

    listener_thread = threading.Thread(target=progress_listener, daemon=True)
    listener_thread.start()

    # 等待监听线程退出（即所有视频处理完毕）
    while listener_thread.is_alive():
        # 检查工作进程是否意外退出（exitcode != 0 且不是正常退出）
        dead = [
            p
            for p in workers
            if not p.is_alive() and p.exitcode not in (0, None)
        ]
        if dead:
            tqdm.write(
                f"[WARN] 以下工作进程意外退出（将继续等待其他进程完成）: "
                f"{[f'{p.name}(exitcode={p.exitcode})' for p in dead]}"
            )
            # 意外死亡的进程负责的任务不会再有结果写入 result_queue，
            # 将所有尚未完成的任务标记为 error，避免监听线程永久阻塞
            for vidx in range(len(selected_vid_list)):
                if not finished[vidx]:
                    result_queue.put(
                        (vidx, "error", f"video[{vidx}] 所在工作进程意外崩溃")
                    )
            for p in dead:
                workers.remove(p)
        listener_thread.join(timeout=1.0)

    stop_event.set()
    listener_thread.join(timeout=5)

    # 关闭所有进度条
    for bar in worker_pbars:
        if bar is None:
            continue
        bar.set_description(f"{bar.desc.split(' ')[0]} [done]")
        bar.refresh()
        bar.close()
    total_pbar.n = len(selected_vid_list)
    total_pbar.refresh()
    total_pbar.close()

    # 等待所有子进程正常退出
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    mp_manager.shutdown()

    tqdm.write("\n--- 所有视频处理完毕 ---")

def main():
    parser = argparse.ArgumentParser(
        description="Process videos to extract hand keypoints using MediaPipe."
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Specific Video. 单视频处理逻辑，和整个数据集的优先遍历处理逻辑不同。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Starting index for video processing."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending index for video processing (exclusive).",
    )
    parser.add_argument(
        "--frame-start-idx", type=int, default=0, help="视频处理帧数开始"
    )
    parser.add_argument(
        "--frame-end-idx", type=int, default=-1, help="视频处理帧数结束"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check existing pkl files without processing videos.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Using test video.",
    )
    parser.add_argument(
        "--save-origin",
        action="store_true",
        help="同时保存一份原始的hawor的结果dict.",
    )
    parser.add_argument(
        "--no-interleave",
        action="store_false",
        dest="interleave",
        help="Disable interleaving by factory/worker. By default interleaving is enabled. 取消轮询式优先遍历每个不同种类的视频",
    )
    # New argument for controlling parallelism
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of CPU processes to use for parallel processing. Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Reverse the final video list. Useful for running multiple processes head-to-tail to cover the full dataset.",
    )

    args = parser.parse_args()
    
    if args.test:
        args.dataroot = DEFAULT_DATASET_ROOT_TEST
        args.output = DEFAULT_OUTPUT_ROOT_TEST
    

    extra_args = {
        "dataset_root": args.dataroot,
        "output_root": args.output,
        "save_origin": args.save_origin,
        "frame_start_idx": args.frame_start_idx,
        "frame_end_idx": args.frame_end_idx
    }
    

    if args.video_path:
        selected_vid_list = [args.video_path]
    else:
        all_vid_list = find_all_mp4_files(args.dataroot)
        if args.interleave:
            print("Applying factory/worker interleaving to video list...")
            all_vid_list = reorder_videos_by_factory_worker(all_vid_list, args.dataroot)
        if not all_vid_list:
            print("No MP4 files found. Exiting.")
            exit()
        print(f"Total Videos: {len(all_vid_list)}")

        if args.end is None:
            args.end = len(all_vid_list)
        selected_vid_list = all_vid_list[
            args.start : args.end
        ]  # todo: 重新处理排序逻辑，使得每个factory的worker的视频首先被遍历一次
        if args.inverse:
            selected_vid_list = selected_vid_list[::-1]
            print("Video list reversed (--inverse enabled).")
        print(f"Process: {len(selected_vid_list)}")
        print("Selected Videos:")
        print(selected_vid_list[:10])

        if not selected_vid_list:
            print("No videos selected based on start/end indices. Exiting.")
            exit()
        print(
            f"Selected {len(selected_vid_list)} videos for processing (indices from {args.start} to {args.end})."
        )

    if args.check:
        print("Running in check-only mode.")
        results = check_pkl_files(selected_vid_list, verbose=True, extra_args=extra_args)
        print_summary(results)
        exit()

    # Avoid any torch.cuda.* checks here to prevent CUDA initialization
    # in the parent process before child processes set their CUDA_VISIBLE_DEVICES.
    dtype = torch.float16
    print("Running in detection mode.")
    # If num_workers <= 1, keep sequential processing to preserve original behaviour.

    if args.num_workers <= 1:
        # Sequential mode: import HaWoR here (after deciding not to spawn workers)
        from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig
        from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints
        cfg = HaWoRConfig(verbose=False)
        pipe = HaWoRPipelineOpt(cfg)
        for video_path in tqdm(
            selected_vid_list, desc="Processing videos", unit="video"
        ):
            print(process_video_worker(video_path, pipe, extra_args=extra_args))
    else:
        # 需要在多进程相关函数中传递 extra_args
        process_multi_workers(args, dtype, selected_vid_list, extra_args=extra_args)

    print("\n--- Detection complete. Now checking generated PKL files ---")
    results = check_pkl_files(selected_vid_list, verbose=False, extra_args=extra_args)
    print_summary(results)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Windows 上使用 spawn 启动子进程时，必须把主逻辑放在此保护块内，
    # 否则子进程 import 模块时会再次执行顶层代码，导致递归启动。
    multiprocessing.freeze_support()
    main()