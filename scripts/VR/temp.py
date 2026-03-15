"""
Compare VR data (from JSON) and HaWoR reconstruction results (from PKL).

This script loads both VR keypoint data from JSON files and HaWoR reconstruction
results from PKL files, then visualizes them side-by-side for comparison.

Example usage:
    python scripts/VR/compare_vr_data_and_hawor.py \\
        --vr-json /path/to/vr_data.json \\
        --hawor-pkl /path/to/hawor_results.pkl \\
        --output /path/to/output_video.mp4

Dependencies (install via pip if needed):
    matplotlib, numpy, opencv-python, scipy
    
python scripts/VR/compare_vr_data_and_hawor.py \
    --vr-json="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json" \
    --hawor-pkl="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0_hawor_origin_dict.pkl" \
    --output="./results/compare_output_video.mp4"
    
python scripts/VR/compare_vr_data_and_hawor.py \
    --vr-json="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_keypoints.json" --hawor-pkl="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/DATASET/Rhos_VR_EgoHands/align_blocks/recording_2026-01-13T12-13-48_remote_0_hawor_origin_dict.pkl" --output="./results/compare_output_video.mp4"
"""