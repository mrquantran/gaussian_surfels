import open3d as o3d
import os
from pyvirtualdisplay import Display
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import matplotlib.image as mpimg

def render_and_save_mesh(mesh, output_dir):
    # Create Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh['vertices'])
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh['faces'])
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh['colors'])
    mesh_o3d.compute_vertex_normals()

    # Step 1: Save the mesh to a PLY file
    ply_filename = os.path.join(output_dir, "mesh.ply")
    o3d.io.write_triangle_mesh(ply_filename, mesh_o3d)
    print(f"Mesh saved to {ply_filename}")

    # Step 2: Render the mesh to an image using headless rendering
    display = Display(visible=0, size=(800, 600))
    display.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    vis.add_geometry(mesh_o3d)

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(1.0)

    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer(False)
    image_filename = os.path.join(output_dir, "rendered_mesh.png")
    o3d.io.write_image(image_filename, image)
    print(f"Rendered image saved to {image_filename}")

    vis.destroy_window()
    display.stop()

    return ply_filename, image_filename

# Plot the normal, ground truth, and depth maps from dataset
def plot_and_save_mesh_data(first_camera, background, mono, output_dir):
    """
    Plots and saves the ground truth image, normal map, and depth map.

    Parameters:
        first_camera: An object with get_gtImage() method to retrieve the ground truth image.
        background: Background parameter for get_gtImage().
        mono: Tensor containing normal and depth data.
        output_dir: Directory to save the output plot image.
    """
    gt_image = first_camera.get_gtImage(background, False)
    gt_image_pil = to_pil_image(gt_image.cpu()).convert("RGB")  # Ensure RGB
    monoN = mono[:3].cpu().numpy()
    monoD = mono[3:].cpu().numpy()
    if len(monoN.shape) > 2:
        monoN = monoN.transpose(1, 2, 0)
    if len(monoD.shape) > 2:
        monoD = monoD.transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(gt_image_pil)
    axes[0].axis("off")
    axes[0].set_title("Ground Truth Image")
    axes[1].imshow(monoN)
    axes[1].axis("off")
    axes[1].set_title("Normal Map")
    axes[2].imshow(monoD)
    axes[2].axis("off")
    axes[2].set_title("Depth Map")
    plt.tight_layout()

    # Save figure as an image file in output_dir and display inline for Kaggle
    plot_path = os.path.join(output_dir, "mesh_plots.png")
    fig.savefig(plot_path)
    from IPython.display import Image, display
    display(Image(plot_path))
    plt.close(fig)
    print("Mesh extraction and rendering completed.")

def visualize_sparse_depth(sparse_depth, u, v, image_width, image_height, sparse_depth_filename):
    """Visualize sparse depth on a 2D grid and save the figure."""
    sparse_depth_map = np.zeros((image_height, image_width))
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    sparse_depth_map[v, u] = sparse_depth
    plt.figure(figsize=(10, 8))
    plt.imshow(sparse_depth_map, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Sparse Depth Visualization')
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')
    plt.savefig(sparse_depth_filename, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Sparse depth visualization saved as 'sparse_depth.png'")