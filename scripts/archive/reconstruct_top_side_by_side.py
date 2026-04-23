import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline.HaWoRPipeline import HaWoRConfig
from lib.pipeline.HaWoRPipelineOpt import HaWoRPipelineOpt


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_side_by_side(original_video: Path, rendered_video: Path, output_video: Path) -> None:
    ensure_parent(output_video)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(original_video),
        "-i",
        str(rendered_video),
        "-filter_complex",
        "[0:v]fps=30,scale=-2:1080,setsar=1[left];"
        "[1:v]fps=30,scale=-2:1080,setsar=1[right];"
        "[left][right]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(output_video),
    ]
    run_ffmpeg(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct one egocentric video and create a side-by-side comparison video.")
    parser.add_argument("--video-path", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--tmp-dir", default=None, help="Temporary directory for memmap and caches.")
    parser.add_argument("--image-focal", type=float, default=1031.0, help="Camera focal length in pixels.")
    parser.add_argument("--start-idx", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end-idx", type=int, default=-1, help="End frame index, -1 means all frames.")
    parser.add_argument("--vis-mode", default="world", choices=["world", "cam"], help="3D render viewpoint.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("MGLW_WINDOW", "moderngl_window.context.headless.Window")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = HaWoRConfig(
        verbose=True,
        tmp_dir=args.tmp_dir,
    )
    pipeline = HaWoRPipelineOpt(cfg)

    result = pipeline.reconstruct(
        video_path=args.video_path,
        output_dir=str(output_dir),
        rendering=True,
        vis_mode=args.vis_mode,
        image_focal=args.image_focal,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        use_progress_bar=True,
    )

    rendered_video = result.get("rendered_video")
    if not rendered_video:
        raise RuntimeError("3D rendered video was not generated.")

    comparison_video = output_dir / "comparison_side_by_side.mp4"
    make_side_by_side(Path(args.video_path), Path(rendered_video), comparison_video)

    print("rendered_video:", rendered_video)
    print("comparison_video:", comparison_video)


if __name__ == "__main__":
    main()
