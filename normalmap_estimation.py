from scene.dataset_readers import readColmapSceneInfo, readIDRSceneInfo, readNerfSyntheticInfo
import torch
import numpy as np
import os
from scene import Scene
from arguments import ModelParams
from argparse import ArgumentParser, Namespace
import sys

from utils.camera_utils import cameraList_from_camInfos

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "IDR": readIDRSceneInfo
}

def precompute_stable_normals(predictor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    args_eval = False
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

    resolution_scales=[1.0]
    train_cameras = {}

    cameras_extent = scene_info.nerf_normalization["radius"]
    camera_lr: float=0.0
    background = torch.rand((3), dtype=torch.float32, device="cuda")

    for resolution_scale in resolution_scales:
        train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, cameras_extent, camera_lr, args)

    viewpoint_stack = train_cameras[1.0]

    for cam in viewpoint_stack:  # Use full resolution
        from torchvision.transforms.functional import to_pil_image, to_tensor
        gt_image = cam.get_gtImage(background, False)
        gt_image = to_pil_image(gt_image.cpu())
        with torch.no_grad():
            normal_map = predictor(gt_image)  # [3, H, W], values in [-1, 1]
        normal_map_tensor = to_tensor(normal_map)
        np.save(os.path.join(output_dir, f"{cam.image_name}_normal.npy"), normal_map_tensor.cpu().numpy())

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source data")
    parser.add_argument("--resolution", type=int, default=1, help="Resolution of the images")
    parser.add_argument("--data-device", type=str, default="cuda", help="Device to use for computation")
    parser.add_argument("--model", type=str, default="Stable-X/StableNormal", help="Model to use for computation")
    gaussians = None  # No GaussianModel needed for precomputation

    args = parser.parse_args(sys.argv[1:])
    model = args.model
    if model == "Stable-X/StableNormal":
        predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    elif model == "alexsax/omnidata_models":
        predictor = torch.hub.load('alexsax/omnidata_models', 'surface_normal_dpt_hybrid_384')

    precompute_stable_normals(predictor, os.path.join(args.source_path, "stable_normal"))
    print("StableNormal maps precomputed and saved.")