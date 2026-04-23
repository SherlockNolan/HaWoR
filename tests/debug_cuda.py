import torch
import os
import ctypes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')
# 1. 打印所有相关模块的物理路径
import droid_backends
import lietorch_extras
print(f"DEBUG: droid_backends from {droid_backends.__file__}")
print(f"DEBUG: lietorch_extras from {lietorch_extras.__file__}")

# 2. 模拟报错位置的最小 CUDA 调用
from lietorch import SE3
try:
    print("测试 lietorch CUDA 算子...")
    # 尝试创建一个恒等变换并转到 GPU，触发 lietorch_backends.so
    groups = SE3.Identity(1, device="cuda")
    print("✅ lietorch 基础算子 OK")
except Exception as e:
    print(f"❌ lietorch 算子失败: {e}")

try:
    print("测试 DROID-SLAM Correlation 算子 (CorrSampler)...")
    # 强制调用引发报错的 CorrSampler.apply
    # 这会直接通过 lietorch_extras.so 或 droid_backends.so 访问 Kernel
    from droid_slam.geom.corr import CorrSampler
    fmap = torch.randn(1, 1, 8, 8, 32).cuda()
    coords = torch.randn(1, 1, 8, 8, 2).cuda()
    out = CorrSampler.apply(fmap, coords, 3)
    print("✅ CorrSampler 算子 OK")
except Exception as e:
    print(f"❌ CorrSampler 算子失败: {e}")