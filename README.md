<div align="center">

# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

[Jinglei Zhang]()`<sup>`1`</sup>` &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)`<sup>`2`</sup>` &emsp; [Chao Ma](https://scholar.google.com/citations?user=syoPhv8AAAAJ&hl=en)`<sup>`1`</sup>` &emsp; [Rolandos Alexandros Potamias](https://rolpotamias.github.io)`<sup>`2`</sup>` &emsp;

`<sup>`1`</sup>`Shanghai Jiao Tong University, China
`<sup>`2`</sup>`Imperial College London, UK `<br>`

`<font color="blue"><strong>`CVPR 2025 Highlight✨`</strong></font>`

`<a href='https://arxiv.org/abs/2501.02973'><img src='https://img.shields.io/badge/Arxiv-2501.02973-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>``</a>`
`<a href='https://arxiv.org/pdf/2501.02973'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>``</a>`
`<a href='https://hawor-project.github.io/'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'>``</a>`
`<a href='https://github.com/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'>``</a>`
`<a href='https://huggingface.co/spaces/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'>``</a>`

</div>

This is the official implementation of **[HaWoR](https://hawor-project.github.io/)**, a hand reconstruction model in the world coordinates:

![teaser](assets/teaser.png)

## Installation

### Installing Through UV

```bash
uv sync --no-build-isolation
```

在服务器上运行的时候请指定：`~/.config/uv/uv.toml`

```toml
# uv.toml in qizhi
preview = true

link-mode = "symlink"

cache-dir = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/cache/uv"

[[index]]

url = "https://pypi.tuna.tsinghua.edu.cn/simple" # 清华源

# url = " https://mirrors.aliyun.com/pypi/simple/"

default = true

```

测试：

```bash
uv run tests/test_single_video.py --video-path=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/HaWoR/example/video_0.mp4
```
或者`.venv/bin/python tests/test_single_video.py --video-path=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/HaWoR/example/video_0.mp4`

### Installation (CU12+)


```
git clone --recursive https://github.com/ThunderVVV/HaWoR.git
cd HaWoR
```

The code has been tested with PyTorch 1.13 and CUDA 11.7. Higher torch and cuda versions should be also compatible. It is suggested to use an anaconda environment to install the the required dependencies:

```bash
conda create --name hawor python=3.10
conda activate hawor

# Install requirements
pip install -r requirements.txt
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0

pip install ninja
# 安装适配 CUDA 12.4 的版本 (以 2.4.0 为例)
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

**使用** `pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable#egg=pytorch3d`这个指令来安装 `pytorch3d`！同时我已经更新到cu124版本的torch！

chumpy同理 `pip install --no-build-isolation git+https://github.com/mattloper/chumpy`

mmcv单独处理，之前的版本太老了 `pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html`

还需要

```bash
sudo apt upgrade
sudo apt install ffmpeg
```



### Install masked DROID-SLAM

```bash
cd thirdparty/DROID-SLAM
python setup.py install
```

Download DROID-SLAM official weights [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing), put it under `./weights/external/`.


**检查建议** 
在重新编译 DROID-SLAM 之前，请务必确认安装成功且版本匹配：
1. **检查 PyTorch 认领的 CUDA：**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
   输出应该是 `12.4`。
2. **检查编译器 (NVCC) 版本：**
   ```bash
   nvcc --version
   ```
   输出也应该是 `12.4` 左右。

只有这两个数字“对齐”了，接下来的 `python setup.py install` 或 `pip install -e .` 才能顺利通过。


验证Droid-SLAM安装了支持H200的版本，请在Droid-SLAM目录下面运行：

```bash
# 验证 droid_backends
SO_DROID=$(find . -name "droid_backends*.so" | head -n 1)
echo "Checking $SO_DROID"
cuobjdump "$SO_DROID" | grep sm_90

# 验证 lietorch_extras (这才是真正的 CorrSampler 所在)
SO_LIE=$(find . -name "lietorch_extras*.so" | head -n 1)
echo "Checking $SO_LIE"
cuobjdump "$SO_LIE" | grep sm_90
```

关于lietorch无法识别的问题，请在lietorch目录下面运行
```bash
python setup.py develop
```
因为lietorch还在更新。随着版本更新会有一些变化



### Install Metric3D

Download Metric3D official weights [metric_depth_vit_large_800k.pth](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link), put it under `thirdparty/Metric3D/weights`.

### Download the model weights

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./weights/hawor/
```

国内使用
```bash
wget https://hf-mirror.com/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/external/
wget https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
wget https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./weights/hawor/checkpoints/
wget https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./weights/hawor/
```


It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de).
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and put the hand model to the `_DATA/data/mano/MANO_RIGHT.pkl` and `_DATA/data_left/mano_left/MANO_LEFT.pkl`.

Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).



## Demo

For visualizaiton in world view, run with:

```bash
python demo.py --video_path ./example/video_0.mp4  --vis_mode world
```

For visualizaiton in camera view, run with:

```bash
python demo.py --video_path ./example/video_0.mp4 --vis_mode cam
```

## Training

The training code will be released soon.

## Acknowledgements

Parts of the code are taken or adapted from the following repos:

- [HaMeR](https://github.com/geopavlakos/hamer/)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [SLAHMR](https://github.com/vye16/slahmr)
- [TRAM](https://github.com/yufu-wang/tram)
- [CMIB](https://github.com/jihoonerd/Conditional-Motion-In-Betweening)

## License

HaWoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.

## Citing

If you find HaWoR useful for your research, please consider citing our paper:

```bibtex
@article{zhang2025hawor,
      title={HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos},
      author={Zhang, Jinglei and Deng, Jiankang and Ma, Chao and Potamias, Rolandos Alexandros},
      journal={arXiv preprint arXiv:2501.02973},
      year={2025}
    }
```
