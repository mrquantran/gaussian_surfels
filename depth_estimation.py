from scene.dataset_readers import readColmapSceneInfo, readIDRSceneInfo, readNerfSyntheticInfo
import torch
import numpy as np
import os
from scene import Scene
from arguments import ModelParams
from argparse import ArgumentParser, Namespace
import sys
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch.nn.functional as F
from utils.camera_utils import cameraList_from_camInfos

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "IDR": readIDRSceneInfo
}

def load_marigold_model(device, checkpoint="prs-eth/marigold-v1-0", half_precision=False):
    """
    Load the Marigold depth estimation model.

    Args:
        device: The device to load the model on
        checkpoint: The checkpoint to load
        half_precision: Whether to use half precision

    Returns:
        pipeline: The loaded inference pipeline
    """
    from diffusers import DiffusionPipeline

    torch_dtype = torch.float16 if half_precision else torch.float32
    pipeline = DiffusionPipeline.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        variant="fp16" if half_precision else None
    ).to(device)
    return pipeline

def precompute_depth_maps(pipeline, output_dir, device, args):
    os.makedirs(output_dir, exist_ok=True)
    args_eval = False

    # Determine dataset type and load scene info
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
        raise ValueError("Could not recognize scene type!")

    resolution_scales = [1.0]
    train_cameras = {}
    cameras_extent = scene_info.nerf_normalization["radius"]
    camera_lr = 0.0
    background = torch.rand((3), dtype=torch.float32, device=device)

    for resolution_scale in resolution_scales:
        train_cameras[resolution_scale] = cameraList_from_camInfos(
            scene_info.train_cameras, resolution_scale, cameras_extent, camera_lr, args
        )

    viewpoint_stack = train_cameras[1.0]
    print(f"Processing {len(viewpoint_stack)} images...")

    for i, cam in enumerate(viewpoint_stack):
        try:
            # Get ground truth image
            gt_image = cam.get_gtImage(background, False)
            gt_image_pil = to_pil_image(gt_image.cpu()).convert("RGB")  # Ensure RGB
            original_width, original_height = gt_image_pil.size  # (1600, 1200)

            # Run Marigold inference with additional parameters
            depth_output = pipeline(
                gt_image_pil,
                num_inference_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size
            )

            depth_map = depth_output.prediction
            visualization = pipeline.image_processor.export_depth_to_16bit_png(depth_map)

            # Save depth map visualization
            depth_visualization = to_tensor(visualization[0]).cpu().numpy()

            # Save depth map
            depth_filename = os.path.join(output_dir, f"{cam.image_name}_depth.npy")
            np.save(depth_filename, depth_visualization)

            if i % 10 == 0:
                print(f"Processed {i+1}/{len(viewpoint_stack)} images")
                print(f"Original image size: {original_width}x{original_height}, "
                      f"Depth map size: {depth_visualization.shape[-2]}x{depth_visualization.shape[-1]}")
        except Exception as e:
            print(f"Error processing image {cam.image_name}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Depth estimation precomputation script")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source data")
    parser.add_argument("--resolution", type=int, default=1, help="Resolution of the images")
    parser.add_argument("--data-device", type=str, default="cuda", help="Device to use for computation")
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-depth-lcm-v1-0",
                        help="Marigold checkpoint to use")
    parser.add_argument("--processing_res", type=int, default=768,
                        help="Processing resolution for Marigold")
    parser.add_argument("--ensemble_size", type=int, default=10,
                        help="Number of ensemble inferences")
    parser.add_argument("--denoise_steps", type=int, default=4,
                        help="Number of denoising steps")
    parser.add_argument("--half_precision", action="store_true",
                        help="Use half precision for faster inference")
    parser.add_argument("--white_background", action="store_true",
                        help="Use white background (for NeRF synthetic)")
    parser.add_argument("--images", type=str, default="image",
                        help="Image folder name (for COLMAP)")

    args = parser.parse_args(sys.argv[1:])
    device = torch.device(args.data_device)

    # Load Marigold model
    pipeline = load_marigold_model(device, args.checkpoint, args.half_precision)

    output_dir = os.path.join(args.source_path, "depth_maps")
    precompute_depth_maps(pipeline, output_dir, device, args)
    print("Depth maps precomputed and saved to", output_dir)