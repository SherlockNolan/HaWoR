import json

# 打开并读取 JSON 文件
file_name = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json"
with open(file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看读取内容
print(data)
print(type(data))  # 通常是 dict 或 list