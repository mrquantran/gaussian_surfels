import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

id_ = "000015"

# Load the normal map (assumed shape is [3, H, W])
# normal_map = np.load(f"./example/scan114/normals_maps/{id_}_normal.npy")
normal_map = np.load(f"./example/scan114/normal/{id_}_normal.npy")
print(normal_map.shape)
# If the data has 3 channels first, transpose it to H x W x C for display
if normal_map.shape[0] == 3:
    normal_map = np.transpose(normal_map, (1, 2, 0))

# Load the depth map (assumed to be a .npy file)
# check shape of depth_map

depth_map = np.load(f"./example/scan114/depth/{id_}_depth.npy")
print(depth_map.shape)
depth_map = depth_map.squeeze()
if depth_map.ndim == 3 and depth_map.shape[0] == 3:
    depth_map = np.transpose(depth_map, (1, 2, 0))
# Load the corresponding image
file_path = f"./example/scan114/image/{id_}.png"
image = Image.open(file_path)
image = np.array(image)  # Convert to NumPy array for plotting

# Create a side-by-side plot with 3 images
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

mask = f"./example/scan114/mask/{id_[-3:]}.png"
mask = Image.open(mask)
mask = np.array(mask)

if True:
    # Apply the mask to the normal map
    normal_map[mask == 0] = 0

    # Apply the mask to the depth map
    depth_map[mask == 0] = 0

    # Apply the mask to the image
    image[mask == 0] = 0

axes[0].imshow(image)
axes[0].axis("off")
axes[0].set_title("Original Image")

axes[1].imshow(normal_map)
axes[1].axis("off")
axes[1].set_title("Normal Map")

axes[2].imshow(depth_map)
axes[2].axis("off")
axes[2].set_title("Depth Map")

plt.tight_layout()
plt.show()