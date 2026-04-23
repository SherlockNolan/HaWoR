使用的python环境和命令（防止lietorch找不到等错误）

目前CPU不可用！！！需要重新从源代码编译CPU版本的pytorch3D！

```bash
/root/miniconda3/envs/hawor/bin/python tests/test_single_video.py --video-path=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/HaWoR/example/video_0.mp4 --device=cpu 2>&1 | tail -80
```