#!/bin/bash
# 强制环境变量
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/usr/local/cuda-12.4
CONDA_LIB="/root/miniconda3/envs/hawor/lib/python3.10/site-packages"

echo "1. 正在清理所有残留的 .egg 黑盒..."
rm -rf $CONDA_LIB/*.egg
rm -rf $CONDA_LIB/droid_backends*
rm -rf $CONDA_LIB/lietorch*

echo "2. 强制编译并安装到 site-packages..."
# 我们不再用 install -e，直接强行 build 并拷贝
cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/HaWoR/thirdparty/DROID-SLAM/
rm -rf build/
python setup.py build

echo "3. 物理覆盖 .so 文件..."
# 找到刚编出来的 sm_90 文件并直接拍到 site-packages 里
cp build/lib.linux-x86_64-3.10/*.so $CONDA_LIB/

echo "4. 建立软链接兼容性..."
cd $CONDA_LIB
ln -sf droid_backends.cpython-310-x86_64-linux-gnu.so droid_backend.so
ln -sf droid_backends.cpython-310-x86_64-linux-gnu.so droid_backends.so

echo "5. 修复 lietorch 源码导入..."
# 必须让 Python 看到 lietorch 的文件夹
rm -rf lietorch
cp -r /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zy/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch .

echo "✅ 修复完成！请运行测试。"