主要运行reconstruct_egocentric_opy.py。里面调用HaWoRPipelineOpt.py（优化过内存，放一些tmp到硬盘中）->HaWoRPipelineAdapter转换成每一帧的数据->Smooth。具体查看文件里面的操作。

注意需要修改数据集目录、输出目录（不同帧数长度的输出目录要区分。不同帧数长度需要通过--end_frame_idx=来指定）、tmp_dir（不要导致硬盘爆了）

建议开启大规模训练之前，先进行单个视频的测试：

```bash
export MGLW_WINDOW=moderngl_window.context.headless.Window
export PYOPENGL_PLATFORM=egl
xvfb-run -a python scripts/reconstruct_opt.py --video_path example/video_0.mp4 --output_dir ./results --rendering --vis_mode cam
```

**目前CPU不可用！！！**需要重新从源代码编译CPU版本的pytorch3D！