"""
纯 CPU 运行版本的 HaWoR 重建脚本。
基于 reconstruct_egocentric_opt.py 修改，强制使用 CPU 运行。

使用示例:
    python scripts/reconstruct_egocentric_cpu.py \
        --start=0 --end=100000 --frame-end-idx=2000 \
        --num-workers=8
"""

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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig
from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints as _convert_fn

# 全局 pipeline（每个子进程独立持有）
_process_pipe = None
_process_device = None

"""
# test
python scripts/reconstruct_egocentric_cpu.py \
    --start=0 --end=2 --frame-end-idx=2000 \
    --num-workers=1 \
    --tmp-dir="/inspire/qb-ilm/project/robot-reasoning/xuyue-p-xuyue/zy/tmp" \
    --test
"""

DEFAULT_DATASET_ROOT = "/inspire/dataset/egocentric-100k/v20251211"
DEFAULT_OUTPUT_ROOT = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/egocentric-100k-hawor-2000frames"
DEFAULT_DATASET_ROOT_TEST = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/DATASET/Self"
DEFAULT_OUTPUT_ROOT_TEST = "./test_results"
DEFAULT_TMP_DIR = None


def find_all_mp4_files(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory '{root_dir}' not found.")
        return []
    pattern = os.path.join(root_dir, "**", "*.mp4")
    mp4_files = glob.glob(pattern, recursive=True)
    mp4_files.sort()
    print(f"Found {len(mp4_files)} MP4 files in '{root_dir}'.")
    return mp4_files


def reorder_videos_by_factory_worker(all_vids: list[str], root_dir: str) -> list[str]:
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
            others.append(v)

    for factory in mapping:
        for worker in mapping[factory]:
            mapping[factory][worker].sort()

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

    others.sort()
    ordered.extend(others)
    included = set(ordered)
    for v in all_vids:
        if v not in included:
            ordered.append(v)
    return ordered


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
    worker_id: int, device_str: str, task_queue, result_queue, init_queue, extra_args=None,
):
    """子进程工作函数：强制使用 CPU。"""
    # 强制使用 CPU，忽略 GPU
    device_str = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用所有 CUDA
    pid = os.getpid()
    device = torch.device(device_str)
    print(f"[Worker {worker_id} / PID {pid}] 初始化 pipeline 使用 {device}", flush=True)

    try:
        from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig
        from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints as _convert_fn
        globals()["convert_hawor_to_keypoints"] = _convert_fn
    except Exception:
        raise

    _tmp_dir = (extra_args or {}).get("tmp_dir")
    cfg = HaWoRConfig(verbose=False, device=device_str, tmp_dir=_tmp_dir)
    global _process_pipe
    _process_pipe = HaWoRPipelineOpt(cfg)

    init_queue.put((worker_id, device_str))
    print(f"[Worker {worker_id} / PID {pid}] 就绪，等待任务...", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            break
        video_path, video_idx, progress_queue, extra_args = task
        try:
            msg = process_video_worker_proc(
                video_path, video_idx, progress_queue, worker_id, extra_args=extra_args,
            )
            result_queue.put((video_idx, "ok", msg))
        except Exception as e:
            import traceback
            err_msg = f"{os.path.basename(video_path)} -> {traceback.format_exc()}"
            result_queue.put((video_idx, "error", err_msg))


def process_video_worker_proc(
    video_path, video_idx, progress_queue, worker_id=0, extra_args=None,
):
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
            result_dict_origin = _process_pipe.reconstruct(
                video_path, output_dir=pkl_dir, image_focal=1031,
                start_idx=frame_start_idx, end_idx=frame_end_idx
            )
            result_dict = dict()
            _convert = globals().get("convert_hawor_to_keypoints")
            if _convert is None:
                from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints as _convert_fn
                globals()["convert_hawor_to_keypoints"] = _convert_fn
                _convert = globals()["convert_hawor_to_keypoints"]
            result_dict["original_result"] = _convert(result_dict_origin, video_path, use_smoothed=False)
            result_dict["smoothed_result"] = _convert(result_dict_origin, video_path, use_smoothed=True) if result_dict_origin["smoothed_result"] is not None else None

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
            progress_queue.put(("error_video", video_idx, f"{os.path.basename(video_path)} -> {err_detail}", worker_id))
        print(err_detail)
        return f"Error: {os.path.basename(video_path)} ->\n{err_detail}"
    finally:
        if success and progress_queue is not None:
            progress_queue.put(("done", video_idx, 1.0, worker_id))


def process_video_worker(video_path, pipe, extra_args=None):
    """单进程模式下的 worker 函数。"""
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
            result_dict_origin = pipe.reconstruct(
                video_path, output_dir=pkl_dir, image_focal=1031,
                start_idx=frame_start_idx, end_idx=frame_end_idx
            )
            result_dict = dict()
            _convert = globals().get("convert_hawor_to_keypoints")
            if _convert is None:
                from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints as _convert_fn
                globals()["convert_hawor_to_keypoints"] = _convert_fn
                _convert = globals()["convert_hawor_to_keypoints"]
            result_dict["original_result"] = _convert(result_dict_origin, video_path, use_smoothed=False)
            result_dict["smoothed_result"] = _convert(result_dict_origin, video_path, use_smoothed=True) if result_dict_origin["smoothed_result"] is not None else None

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
    required_keys = ['original_result', 'smoothed_result']
    for k in required_keys:
        if k not in data:
            if verbose:
                print(f"Missing key {k} in {pkl_path}")
            return False
    return True


def check_pkl_files(video_list, verbose=True, extra_args=None):
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


def process_multi_workers(args, selected_vid_list, extra_args=None):
    if extra_args is None:
        extra_args = {}
    num_workers = args.num_workers
    device_assignments = ["cpu"] * num_workers
    print(f"进程数: {num_workers}，全部使用 CPU")

    mp_ctx = multiprocessing.get_context("spawn")
    mp_manager = multiprocessing.Manager()
    progress_queue = mp_manager.Queue()
    task_queue = mp_manager.Queue()
    result_queue = mp_manager.Queue()
    init_queue = mp_manager.Queue()
    workers = []

    for wid, dev_str in enumerate(device_assignments):
        p = mp_ctx.Process(
            target=_worker_process_main,
            args=(wid, dev_str, task_queue, result_queue, init_queue, extra_args),
            daemon=True,
            name=f"Worker-{wid}",
        )
        p.start()
        workers.append(p)

    print(f"等待 {num_workers} 个 Worker 完成初始化...", flush=True)
    with tqdm(
        total=num_workers, desc="Worker Init", unit="worker", position=0, dynamic_ncols=True
    ) as init_pbar:
        ready_count = 0
        while ready_count < num_workers:
            try:
                wid, dev = init_queue.get()
                init_pbar.set_postfix_str(f"Worker-{wid} on {dev} ready")
                init_pbar.update(1)
                ready_count += 1
            except Exception:
                dead = [p for p in workers if not p.is_alive() and p.exitcode not in (0, None)]
                if dead:
                    raise RuntimeError(
                        f"Worker 初始化期间进程意外退出: "
                        f"{[f'{p.name}(exitcode={p.exitcode})' for p in dead]}"
                    )

    print("所有 Worker 初始化完成，开始投递任务。", flush=True)

    for idx, video_path in enumerate(selected_vid_list):
        task_queue.put((video_path, idx, progress_queue, extra_args))
    for _ in workers:
        task_queue.put(None)

    worker_frame_counts = [None] * num_workers
    total_pbar = tqdm(
        total=len(selected_vid_list), desc="All videos", position=0, unit="video", dynamic_ncols=True,
    )
    worker_pbars = [
        tqdm(total=100, desc=f"Worker-{i} [idle]", position=i + 1, leave=True, unit="frame", dynamic_ncols=True)
        for i in range(num_workers)
    ]

    stop_event = threading.Event()
    finished = [False] * len(selected_vid_list)

    def _worker_progress_handler(wid, vidx, payload, msg_type, worker_pbars, worker_frame_counts, video_name=None):
        bar = worker_pbars[wid]
        if msg_type == "init":
            bar.total = 100
            bar.n = 0
            desc_name = video_name[:28] if video_name is not None else "[idle]"
            bar.set_description(f"W{wid} {desc_name}")
            bar.refresh()
        elif msg_type == "progress":
            new_n = int(payload * 100)
            if new_n > bar.n:
                bar.n = new_n
                bar.refresh()
        elif msg_type == "done":
            bar.n = 100
            bar.refresh()

    def progress_listener():
        while not stop_event.is_set():
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
                    tqdm.write(f"[ERROR] {msg}")

            if all(finished):
                break
            stop_event.wait(timeout=0.1)

    listener_thread = threading.Thread(target=progress_listener, daemon=True)
    listener_thread.start()

    while listener_thread.is_alive():
        dead = [p for p in workers if not p.is_alive() and p.exitcode not in (0, None)]
        if dead:
            tqdm.write(f"[WARN] 工作进程意外退出: {[f'{p.name}(exitcode={p.exitcode})' for p in dead]}")
            for vidx in range(len(selected_vid_list)):
                if not finished[vidx]:
                    result_queue.put((vidx, "error", f"video[{vidx}] 所在工作进程意外崩溃"))
            for p in dead:
                workers.remove(p)
        listener_thread.join(timeout=1.0)

    stop_event.set()
    listener_thread.join(timeout=5)

    for bar in worker_pbars:
        if bar is None:
            continue
        bar.set_description(f"{bar.desc.split(' ')[0]} [done]")
        bar.refresh()
        bar.close()
    total_pbar.n = len(selected_vid_list)
    total_pbar.refresh()
    total_pbar.close()

    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    mp_manager.shutdown()
    tqdm.write("\n--- 所有视频处理完毕 ---")


def main():
    parser = argparse.ArgumentParser(description="Process videos using HaWoR (CPU mode).")
    parser.add_argument("--dataroot", type=str, default=DEFAULT_DATASET_ROOT, help="Root directory of the dataset.")
    parser.add_argument("--video-path", type=str, default=None, help="Specific video path.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root directory.")
    parser.add_argument("--start", type=int, default=0, help="Starting index for video processing.")
    parser.add_argument("--end", type=int, default=None, help="Ending index for video processing (exclusive).")
    parser.add_argument("--frame-start-idx", type=int, default=0, help="视频处理帧数开始")
    parser.add_argument("--frame-end-idx", type=int, default=-1, help="视频处理帧数结束")
    parser.add_argument("--check", action="store_true", help="Only check existing pkl files without processing videos.")
    parser.add_argument("--test", action="store_true", help="Using test video.")
    parser.add_argument("--save-origin", action="store_true", help="同时保存一份原始的hawor的结果dict.")
    parser.add_argument("--no-interleave", action="store_false", dest="interleave", help="Disable interleaving.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--inverse", action="store_true", help="Reverse the final video list.")
    parser.add_argument("--tmp-dir", type=str, default=DEFAULT_TMP_DIR, help="临时文件目录")

    args = parser.parse_args()

    if args.test:
        args.dataroot = DEFAULT_DATASET_ROOT_TEST
        args.output = DEFAULT_OUTPUT_ROOT_TEST

    extra_args = {
        "dataset_root": args.dataroot,
        "output_root": args.output,
        "save_origin": args.save_origin,
        "frame_start_idx": args.frame_start_idx,
        "frame_end_idx": args.frame_end_idx,
        "tmp_dir": args.tmp_dir,
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
        selected_vid_list = all_vid_list[args.start:args.end]
        if args.inverse:
            selected_vid_list = selected_vid_list[::-1]
            print("Video list reversed (--inverse enabled).")
        print(f"Process: {len(selected_vid_list)}")
        print("Selected Videos:")
        print(selected_vid_list[:10])

        if not selected_vid_list:
            print("No videos selected based on start/end indices. Exiting.")
            exit()

    if args.check:
        print("Running in check-only mode.")
        results = check_pkl_files(selected_vid_list, verbose=True, extra_args=extra_args)
        print_summary(results)
        exit()

    print("Running in CPU detection mode.")

    if args.num_workers <= 1:
        from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt, HaWoRConfig
        from lib.pipeline.HaworToKeypointsAdapter import convert_hawor_to_keypoints
        cfg = HaWoRConfig(verbose=False, device="cpu", tmp_dir=extra_args.get("tmp_dir"))
        pipe = HaWoRPipelineOpt(cfg)
        for video_path in tqdm(selected_vid_list, desc="Processing videos", unit="video"):
            print(process_video_worker(video_path, pipe, extra_args=extra_args))
    else:
        process_multi_workers(args, selected_vid_list, extra_args=extra_args)

    print("\n--- Detection complete. Now checking generated PKL files ---")
    results = check_pkl_files(selected_vid_list, verbose=False, extra_args=extra_args)
    print_summary(results)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
