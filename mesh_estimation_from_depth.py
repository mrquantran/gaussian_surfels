import torch
import numpy as np
import os
from argparse import ArgumentParser
import sys
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo, SceneInfo, readColmapSceneInfo, readNerfSyntheticInfo, readIDRSceneInfo
from utils.camera_utils import cameraList_from_camInfos
from torchvision.transforms.functional import to_pil_image
from utils.mesh import plot_and_save_mesh_data, render_and_save_mesh, visualize_sparse_depth

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def project_points_to_depth(points_3d, camera):
    """Project 3D points to depth using the camera's intrinsic and extrinsic parameters."""
    R = camera.R.detach().numpy()
    T = camera.T.detach().numpy()
    points_cam = (R @ points_3d.T + T[:, None]).T
    K = camera.get_intrinsic().detach().numpy()
    u = K[0, 0] * points_cam[:, 0] / points_cam[:, 2] + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / points_cam[:, 2] + K[1, 2]
    mask = (u >= 0) & (u < camera.image_width) & (v >= 0) & (v < camera.image_height) & (points_cam[:, 2] > 0)
    return points_cam[:, 2][mask], u[mask], v[mask]

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "IDR": readIDRSceneInfo
}

def depth_to_mesh(camera):
    useMask = True
    mask_gt = camera.get_gtMask(useMask)
    background = torch.rand((3), dtype=torch.float32, device=device)
    background = background.to("cpu")
    gt_image = camera.get_gtImage(background, useMask)
    gt_image = to_pil_image(gt_image.cpu()).convert("RGB")
    if camera.mono is None:
        raise ValueError("Camera does not have monocular depth and normal maps.")

    # Extract depth (single channel) and normal maps
    mono = camera.mono * mask_gt
    monoN = mono[:3].cpu().numpy()
    monoD = mono[3].cpu().numpy()  # Single-channel depth

    print(f"Original monoD.shape: {monoD.shape}")
    print(f"Original monoN.shape: {camera.mono[:3].shape}")
    print(f"Original gt_image.shape: {np.array(gt_image).shape}")

    K = camera.get_intrinsic().cpu().numpy()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h, w = gt_image.size
    print(f"Image size: {h} x {w}")

    # Create pixel coordinates for downsampled points in original image space
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)

    # Unproject to 3D (camera coordinates)
    z = monoD.flatten()
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)

    # Transform to world coordinates
    R = camera.R.detach().cpu().numpy()
    T = camera.T.detach().cpu().numpy()
    points_world = (np.linalg.inv(R) @ (points_cam.T - T[:, None])).T

    # Transform normals
    normals_cam = monoN.reshape(3, -1).T
    normals_world = (R @ normals_cam.T).T

    # Create triangle faces
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            faces.append([idx, idx + 1, idx + w])
            faces.append([idx + 1, idx + w + 1, idx + w])
    faces = np.array(faces, dtype=np.int32)

    # Get colors
    image = np.array(gt_image)
    colors = image.reshape(-1, 3)

    print(f"Points world shape: {points_world.shape}")
    print(f"Normals world shape: {normals_world.shape}")
    print(f"Faces shape: {faces.shape}")
    print(f"Colors shape: {colors.shape}")

    assert points_world.shape[0] == colors.shape[0] == normals_world.shape[0]
    return {
        'vertices': points_world,
        'faces': faces,
        'colors': colors,
        'normals': normals_world
    }

def extract_mesh_from_global_depth(output_dir):
    background = torch.rand((3), dtype=torch.float32, device="cuda")
    os.makedirs(output_dir, exist_ok=True)
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, False)  # Assuming COLMAP
    train_cameras = {1.0: cameraList_from_camInfos(scene_info.train_cameras, 1.0, scene_info.nerf_normalization["radius"], 0.0, args)}
    camera_id = 15

    first_camera = train_cameras[1.0][camera_id]  # Get camera view 15
    point3d = scene_info.point_cloud.points
    sparse_depth, u, v = project_points_to_depth(point3d, first_camera)

    # Visualize sparse depth
    sparse_depth_filename = os.path.join(output_dir, "sparse_depth.png")
    visualize_sparse_depth(sparse_depth, u, v, first_camera.image_width, first_camera.image_height, sparse_depth_filename)

    # Load and scale depth
    mono = first_camera.mono
    monoN = mono[:3].cpu().numpy()
    monoD = mono[3].cpu().numpy()  # Single-channel depth
    if len(sparse_depth) > 0:
        m_X = np.median(sparse_depth)
        p_X = np.percentile(sparse_depth, 10)
        m_D_r = np.median(monoD)
        p_D_r = np.percentile(monoD, 10)
        s = (m_X - p_X) / (m_D_r - p_D_r) if abs(m_D_r - p_D_r) > 1e-6 else 1.0
        o = m_X - s * m_D_r if abs(m_D_r - p_D_r) > 1e-6 else 0.0
        D_g = s * monoD + o
        print(f"Scaled depth map with scale={s:.4f}, offset={o:.4f}")
    else:
        D_g = monoD

    # Update camera with global depth
    first_camera.mono = torch.cat([torch.tensor(monoN), torch.tensor(D_g)[None]], dim=0)
    mesh = depth_to_mesh(first_camera)

    # Save the mesh to a PLY file
    render_and_save_mesh(mesh, output_dir)

    # Plot the estimation normal, ground truth, and estimation depth maps
    plot_and_save_mesh_data(first_camera, background, mono, output_dir)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--source-path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--data-device", type=str, default="cuda")
    parser.add_argument("--images", type=str, default="image")
    args = parser.parse_args(sys.argv[1:])
    extract_mesh_from_global_depth(args.source_path)