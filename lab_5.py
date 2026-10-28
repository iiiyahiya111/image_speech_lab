"""
py -m pip install opencv-python numpy matplotlib
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 1. Load the ORIGINAL image, not the previous segmentation output
image_path = "character.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"Could not load {image_path}. Check the filename and path."
    )

print("Image dimensions:", image.shape)

# Display original image
plt.figure(figsize=(12, 4))
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# 2. Convert to binary
_, binary = cv2.threshold(
    image,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Optional: connect tiny gaps within characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, kernel
)

# Display binary image with conventional black text on white
plt.figure(figsize=(12, 4))
plt.imshow(255 - binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")
plt.show()

# 3. Find connected components
num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )
)

characters = []

# 4. Extract valid character components
MIN_AREA = 500

for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    # Remove small noise
    if area < MIN_AREA:
        continue

    # Extract character from the binary image
    character = binary[y:y+h, x:x+w]

    # Convert to black text on a white background
    character = 255 - character

    # Add white padding for better visualization
    character = cv2.copyMakeBorder(
        character,
        8, 8, 8, 8,
        cv2.BORDER_CONSTANT,
        value=255
    )

    characters.append((x, character))

# 5. Sort characters from left to right
characters.sort(key=lambda item: item[0])

print("Number of segmented characters:", len(characters))

# 6. Display segmented characters
if characters:

    n_cols = min(5, len(characters))
    n_rows = math.ceil(len(characters) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 2.8 * n_rows)
    )

    axes = np.atleast_1d(axes).ravel()

    for i, (_, character) in enumerate(characters):
        axes[i].imshow(character, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Character {i + 1}")
        axes[i].axis("off")

    # Hide unused subplot positions
    for ax in axes[len(characters):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("No characters detected. Check the input image.")
