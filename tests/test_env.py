import torch
import droid_backends
import lietorch_extras
import os

print(f"--- 核心检查 ---")
print(f"Torch Version: {torch.__version__}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")

print(f"\n--- 模块路径检查 ---")
print(f"droid_backends file: {os.path.abspath(droid_backends.__file__)}")
try:
    print(f"lietorch_extras file: {os.path.abspath(lietorch_extras.__file__)}")
except:
    print("lietorch_extras 未能直接导入")

print(f"\n--- 物理一致性检查 ---")
# 检查加载的文件是否就是你刚才 cuobjdump 的那个