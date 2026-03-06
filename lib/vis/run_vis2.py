import os
import cv2
import numpy as np
import torch
import trimesh

import lib.vis.viewer as viewer_utils
from lib.vis.wham_tools.tools import checkerboard_geometry

def camera_marker_geometry(radius, height):
    vertices = np.array(
        [
            [-radius, -radius, 0],
            [radius, -radius, 0],
            [radius, radius, 0],
            [-radius, radius, 0],
            [0, 0, - height],
        ]
    )


    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4],]
    )

    face_colors = np.array(
        [
            [0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    return vertices, faces, face_colors


def run_vis2_on_video(res_dict, res_dict2, output_pth, focal_length, image_names, R_c2w=None, t_c2w=None, interactive=False):
    
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            # "color": "pace-green",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1
    
    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            # "color": "pace-blue",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1
    
    v, f, vc, fc = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
    v[:, 2] -= 2 # z plane
    gound_meshes = {
        "v3d": v,
        "f3d": f,
        "vc": vc,
        "name": "ground",
        "fc": fc,
        "color": -1,
    }
    vis_dict["ground"] = gound_meshes

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_c2w[:num_frames]
    Rt[:, :3, 3] = t_c2w[:num_frames]

    verts, faces, face_colors = camera_marker_geometry(0.05, 0.1)
    verts = np.einsum("tij,nj->tni", Rt[:, :3, :3], verts) + Rt[:, None, :3, 3]
    camera_meshes = {
        "v3d": verts,
        "f3d": faces,
        "vc": None,
        "name": "camera",
        "fc": face_colors,
        "color": -1,
    }
    vis_dict["camera"] = camera_meshes

    side_source = torch.tensor([0.463, -0.478, 2.456])
    side_target = torch.tensor([0.026, -0.481, -3.184])
    up = torch.tensor([1.0, 0.0, 0.0])
    view_camera = lookat_matrix(side_source, side_target, up)
    viewer_Rt = np.tile(view_camera[:3, :4], (num_frames, 1, 1))

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    vis_h, vis_w = (height, width)
    K = np.array(
        [
            [1000, 0, vis_w / 2],
            [0, 1000, vis_h / 2],
            [0, 0, 1]
        ]
    )
    
    data = viewer_utils.ViewerData(viewer_Rt, K, vis_w, vis_h)
    batch = (meshes, data)

    if interactive:
        viewer = viewer_utils.ARCTICViewer(interactive=True, size=(vis_w, vis_h))
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    else:
        viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'])
        if os.path.exists(os.path.join(output_pth, 'aitviewer', "video_0.mp4")):
            os.remove(os.path.join(output_pth, 'aitviewer', "video_0.mp4"))
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
        return os.path.join(output_pth, 'aitviewer', "video_0.mp4")

def run_vis2_on_video_cam(res_dict, res_dict2, output_pth, focal_length, image_names, R_w2c=None, t_w2c=None):
    
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            # "color": "pace-green",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1
    
    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            # "color": "pace-blue",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_w2c[:num_frames]
    Rt[:, :3, 3] = t_w2c[:num_frames]

    cols, rows = (width, height)
    K = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ]
    )
    vis_h = height
    vis_w = width

    data = viewer_utils.ViewerData(Rt, K, cols, rows, imgnames=image_names)
    batch = (meshes, data)

    viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h),  render_types=['video']) # 输出视频
    viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))

def lookat_matrix(source_pos, target_pos, up):
    """
    IMPORTANT: USES RIGHT UP BACK XYZ CONVENTION
    :param source_pos (*, 3)
    :param target_pos (*, 3)
    :param up (3,)
    """
    *dims, _ = source_pos.shape
    up = up.reshape(*(1,) * len(dims), 3)
    up = up / torch.linalg.norm(up, dim=-1, keepdim=True)
    back = normalize(target_pos - source_pos)
    right = normalize(torch.linalg.cross(up, back))
    up = normalize(torch.linalg.cross(back, right))
    R = torch.stack([right, up, back], dim=-1)
    return make_4x4_pose(R, source_pos)

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def normalize(x):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)

def save_mesh_to_obj(vertices, faces, file_path):
    # 创建一个 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 导出为 .obj 文件
    mesh.export(file_path)
    print(f"Mesh saved to {file_path}")


# ---------------------------------------------------------------------------
# 公共渲染入口（供 reconstruct.py / reconstruct_vitra.py 等共同调用）
# ---------------------------------------------------------------------------

def render_hand_results(
    left_dict: dict,
    right_dict: dict,
    image_names: list,
    img_focal: float,
    output_dir: str,
    vis_start: int,
    vis_end: int,
    vis_mode: str = "world",
    R_c2w=None,
    t_c2w=None,
    R_w2c=None,
    t_w2c=None,
    video_stem: str = "output",
) -> str | None:
    """
    公共渲染函数：根据 vis_mode 调用对应的渲染流程，返回最终 mp4 路径。

    参数
    ----
    left_dict : dict
        左手网格字典，包含 'vertices' (1, T, N, 3) 和 'faces'。
    right_dict : dict
        右手网格字典，包含 'vertices' (1, T, N, 3) 和 'faces'。
    image_names : list[str]
        渲染用的图像路径列表（长度 = vis_end - vis_start）。
    img_focal : float
        相机焦距（像素）。
    output_dir : str
        输出目录。
    vis_start : int
        起始帧索引。
    vis_end : int
        结束帧索引（不含）。
    vis_mode : str
        渲染视角：'world' 或 'cam'。
    R_c2w : Tensor (T, 3, 3) | None
        相机→世界旋转（world 模式必填）。
    t_c2w : Tensor (T, 3) | None
        相机→世界平移（world 模式必填）。
    R_w2c : Tensor (T, 3, 3) | None
        世界→相机旋转（cam 模式必填）。
    t_w2c : Tensor (T, 3) | None
        世界→相机平移（cam 模式必填）。
    video_stem : str
        输出文件名（不含扩展名）。

    返回
    ----
    str | None : 生成的 mp4 路径，失败时返回 None。
    """
    import shutil

    vis_output_pth = os.path.join(output_dir, f"vis_{vis_start}_{vis_end}")
    os.makedirs(vis_output_pth, exist_ok=True)
    final_mp4 = os.path.join(output_dir, f"{video_stem}.mp4")

    try:
        if vis_mode == "world":
            assert R_c2w is not None and t_c2w is not None, \
                "render_hand_results: world 模式需提供 R_c2w 和 t_c2w"
            raw_path = run_vis2_on_video(
                left_dict, right_dict,
                vis_output_pth, img_focal, image_names,
                R_c2w=R_c2w[vis_start:vis_end],
                t_c2w=t_c2w[vis_start:vis_end],
            )
        else:  # cam
            assert R_w2c is not None and t_w2c is not None, \
                "render_hand_results: cam 模式需提供 R_w2c 和 t_w2c"
            run_vis2_on_video_cam(
                left_dict, right_dict,
                vis_output_pth, img_focal, image_names,
                R_w2c=R_w2c[vis_start:vis_end],
                t_w2c=t_w2c[vis_start:vis_end],
            )
            raw_path = os.path.join(vis_output_pth, "aitviewer", "video_0.mp4")

        if raw_path and os.path.exists(raw_path):
            shutil.move(raw_path, final_mp4)
            return final_mp4
        return None
    except Exception as e:
        print(f"[render_hand_results] Rendering failed: {e}")
        return None

