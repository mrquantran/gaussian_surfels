import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

id_ = "000000"

# Load the normal map (assumed shape is [3, H, W])
normal_map = np.load(f"./example/scan114/normal/{id_}_normal.npy")
# If the data has 3 channels first, transpose it to H x W x C for display
if normal_map.shape[0] == 3:
    normal_map = np.transpose(normal_map, (1, 2, 0))
# Convert values from [-1, 1] to [0, 1]
normal_map = (normal_map + 1) / 2.0
normal_map = np.clip(normal_map, 0, 1)

# Load the depth map (assumed to be a .npy file)
# check shape of depth_map

depth_map = np.load(f"./example/scan114/depth/{id_}_depth.npy")
print(depth_map.shape)
depth_map = depth_map.squeeze()

# Load the corresponding image
file_path = f"./example/scan114/image/{id_}.png"
image = Image.open(file_path)
image = np.array(image)  # Convert to NumPy array for plotting

# Create a side-by-side plot with 3 images
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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