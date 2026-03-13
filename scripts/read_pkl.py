import pickle

# 以二进制只读模式 ('rb') 打开文件
pkl_file_path = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0_hawor.pkl"
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

print(data)