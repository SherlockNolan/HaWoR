# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HaWoR (World-Space Hand Motion Reconstruction from Egocentric Videos) is a CVPR 2025 Highlight paper implementation that reconstructs 3D hand motion from egocentric videos. The system combines multiple computer vision models:

- **DROID-SLAM**: Camera pose estimation and SLAM
- **Metric3D**: Depth estimation
- **HaWoR**: Hand motion reconstruction model
- **Infiller**: Transformer model for temporal consistency
- **MANO**: Parametric hand model for mesh generation

## 项目说明（写在前面）

目前主要开发HaWoRPipeline这个类，其它的都是论文里面的结果，只进行诸如进度条之类的修改，但是算法主体不改变。

意图：使用当前的HaWoRPipeline对数据集进行处理，生成3dKeypoints用于后面的VLA大模型的预训练。之前主要编辑的文件为：

- `scripts/`文件夹下面的脚本
- `lib/pipeline`下面的 `HaWoRPipeline.py`等。

编写代码时提供必要的注释，遵守良好的设计模式原则。

思考推荐使用英文。注释和最后的总结回答推荐使用中文。

编辑单个文件的时候不要写README.md，基本用法通过注释的方式写在编辑的代码文件的底端。不需要考虑环境问题，如有需要安装的python包则一并写在注释中。

## Installation

**note:** 在当前的工作环境中，并没有同步虚拟环境，只是纯coding代码。最后手动上传到linux服务器上进行运行。服务器是linux，注释中给出的python脚本示例运行等命令均需要linux系统的语法
