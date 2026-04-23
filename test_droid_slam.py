import torch
import numpy as np

def test_corr_sampler():
    print("--- 1. 测试 CorrSampler (Correlation Kernel) ---")
    try:
        from droid_slam.geom.corr import CorrSampler
        
        # 模拟 DROID-SLAM 内部张量 (Batch, Pairs, Height, Width, Dim)
        fmap = torch.randn(1, 1, 48, 64, 128).cuda()
        coords = torch.randn(1, 1, 48, 64, 2).cuda()
        
        # 核心算子调用
        out = CorrSampler.apply(fmap, coords, 3) # radius=3
        
        print(f"✅ CorrSampler 运行成功! 输出形状: {out.shape}")
    except Exception as e:
        print(f"❌ CorrSampler 失败: \n{e}")

def test_backend_ba():
    print("\n--- 2. 测试 Backend Factor Graph (BA Kernel) ---")
    try:
        # 这里模拟 DROID-SLAM 后端核心算子的最小加载逻辑
        import droid_backends
        print("✅ droid_backends 模块加载成功")
        
        # 尝试调用后端的一个基础函数
        # 注意：这里仅检查链接库是否能正常寻址
        if hasattr(droid_backends, 'corr_kernel'):
            print("✅ 找到 backend CUDA 符号: corr_kernel")
        else:
            print("⚠️ 未能在 backend 中找到预期的 symbols")
            
    except ImportError as e:
        print(f"❌ 加载 droid_backends 失败 (通常是.so文件没找着): {e}")
    except Exception as e:
        print(f"❌ 后端测试异常: {e}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 未检测到 GPU")
    else:
        print(f"检测到设备: {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability()})")
        test_corr_sampler()
        test_backend_ba()