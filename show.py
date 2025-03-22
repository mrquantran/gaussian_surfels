import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the normal map (assumed shape is [3, H, W])
normal_map = np.load("./example/000000_normal.npy")

# If the data has 3 channels first, transpose it to H x W x C for display
if normal_map.shape[0] == 3:
    normal_map = np.transpose(normal_map, (1, 2, 0))

# Convert values from [-1, 1] to [0, 1]
normal_map = (normal_map + 1) / 2.0
normal_map = np.clip(normal_map, 0, 1)

# Load the corresponding image
image = Image.open("./example/000000.png")
image = np.array(image)  # Convert to NumPy array for plotting

# Create a side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].axis("off")
axes[0].set_title("Original Image")

axes[1].imshow(normal_map)
axes[1].axis("off")
axes[1].set_title("Normal Map")

plt.tight_layout()
plt.show()