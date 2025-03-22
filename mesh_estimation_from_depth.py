import torch
import numpy as np
import os

from arguments import ModelParams
from argparse import ArgumentParser, Namespace
import sys

from scene import Scene
from scene.cameras import Camera

from scene.dataset_readers import CameraInfo, SceneInfo, readColmapSceneInfo, readIDRSceneInfo, readNerfSyntheticInfo
from gaussian_surfels_old.utils.camera_utils import cameraList_from_camInfos

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def project_points_to_depth(points_3d, camera):
    # Transform points to camera coordinates
    R = camera.R.numpy()  # World to camera rotation
    T = camera.T.numpy()  # World to camera translation
    points_cam = (R @ points_3d.T + T[:, None]).T  # (N, 3)

    # Project to image plane
    K = camera.get_intrinsic().numpy()
    u = K[0, 0] * points_cam[:, 0] / points_cam[:, 2] + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / points_cam[:, 2] + K[1, 2]

    # Filter points within image bounds and in front of camera
    mask = (u >= 0) & (u < camera.image_width) & (v >= 0) & (v < camera.image_height) & (points_cam[:, 2] > 0)
    depths = points_cam[:, 2][mask]

    return depths

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "IDR": readIDRSceneInfo
}

def depth_to_mesh(camera, downsample_factor=1):
    """Mesh with 'vertices' (N, 3), 'faces' (F, 3), 'colors' (N, 3), 'normals' (N, 3)."""
    if camera.mono is None:
        raise ValueError("Camera does not have monocular depth and normal maps.")

    # Extract depth and normals (in camera coordinates)
    monoD = camera.mono[3].cpu().numpy()  # Depth map
    monoN = camera.mono[:3].cpu().numpy()  # Normals (x, y, z)

    # Get camera intrinsics
    K = camera.get_intrinsic().cpu().numpy()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Original dimensions
    h, w = monoD.shape

    # Downsample
    if downsample_factor > 1:
        monoD = monoD[::downsample_factor, ::downsample_factor]
        monoN = monoN[:, ::downsample_factor, ::downsample_factor]
        h, w = monoD.shape
        cx /= downsample_factor
        cy /= downsample_factor
        fx /= downsample_factor
        fy /= downsample_factor

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)

    # Unproject to 3D (camera coordinates)
    z = monoD.flatten()
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)  # (N, 3)

    # Transform to world coordinates
    R = camera.R.cpu().numpy()  # World to camera rotation
    T = camera.T.cpu().numpy()  # World to camera translation
    # p_world = R^T * (p_cam - T)
    points_world = (np.linalg.inv(R) @ (points_cam.T - T[:, None])).T

    # Transform normals to world coordinates
    normals_cam = monoN.reshape(3, -1).T  # (N, 3)
    normals_world = (R @ normals_cam.T).T  # R because R is world-to-camera

    # Create triangle faces from grid
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            # Two triangles per quad
            faces.append([idx, idx + 1, idx + w])          # Bottom-left triangle
            faces.append([idx + 1, idx + w + 1, idx + w])  # Top-right triangle
    faces = np.array(faces, dtype=np.int32)

    # Get colors from the image
    image = np.array(camera.image) / 255.0  # Convert to [0, 1]
    if downsample_factor > 1:
        image = image[::downsample_factor, ::downsample_factor, :]
    colors = image.reshape(-1, 3)  # (N, 3)

    # Ensure shape consistency
    assert points_world.shape[0] == colors.shape[0] == normals_world.shape[0], "Mismatch in vertex count"

    return {
        'vertices': points_world,
        'faces': faces,
        'colors': colors,
        'normals': normals_world
    }


def extract_mesh_from_global_depth(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    args_eval = False
    scene_info: SceneInfo = None
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        print("Found sparse directory, assuming COLMAP data format!")
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args_eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming NeRF data format!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args_eval)
    elif os.path.exists(os.path.join(args.source_path, "cameras.npz")):
        print("Found camera.npz file, assuming IDR data format!")
        scene_info = sceneLoadTypeCallbacks["IDR"](args.source_path, args_eval)
    else:
        assert False, "Could not recognize scene type!"

    resolution_scales = [1.0]
    train_cameras = {}
    cameras_extent = scene_info.nerf_normalization["radius"]
    camera_lr: float = 0.0
    background = torch.rand((3), dtype=torch.float32, device="cuda")

    for resolution_scale in resolution_scales:
        train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, cameras_extent, camera_lr, args)

    viewpoint_stack = train_cameras[1.0]

    point3d = scene_info.point_cloud.points
    first_camera: Camera = viewpoint_stack[0]
    gt_image = first_camera.get_gtImage(background, False)
    sparse_depth = project_points_to_depth(point3d, first_camera)

    # Load relative depth map
    relative_depth_map = np.load(os.path.join(args.source_path, "relative_map.npy"))

    if len(sparse_depth) > 0:
        # Compute statistics for sparse depths
        m_X = np.median(sparse_depth)
        p_X = np.percentile(sparse_depth, 10)

        # Compute statistics for relative depth map
        # get depth channel
        D_r = relative_depth_map[:, :, 0]
        m_D_r = np.median(D_r)
        p_D_r = np.percentile(D_r, 10)

        # Calculate scale and offset
        if abs(m_D_r - p_D_r) > 1e-6:  # Avoid division by zero
            s = (m_X - p_X) / (m_D_r - p_D_r)
            o = m_X - s * m_D_r
        else:
            s = 1.0
            o = 0.0

        # Apply transformation
        D_g = s * D_r + o
        global_depth_map = torch.tensor(D_g, dtype=torch.float32).to(device)
        print(f"Scaled depth map with scale={s:.4f}, offset={o:.4f}")

    # Proceed with mesh extraction using the global depth map
    mesh = depth_to_mesh(first_camera, downsample_factor=4)

    # Visualizae the mesh
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh['vertices'])
    pcd.colors = o3d.utility.Vector3dVector(mesh['colors'])
    pcd.normals = o3d.utility.Vector3dVector(mesh['normals'])
    o3d.visualization.draw_geometries([pcd])

# Command: python mesh_estimation_from_depth.py --source_path gaussian_surfels/example/scan114
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source data")
    parser.add_argument("--resolution", type=int, default=1, help="Resolution of the images")
    parser.add_argument("--data-device", type=str, default="cuda", help="Device to use for computation")
    gaussians = None  # No GaussianModel needed for precomputation


    args = parser.parse_args(sys.argv[1:])
    model = args.model
