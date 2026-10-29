
"""
Install: py -m pip install opencv-python numpy matplotlib
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img, title):
    plt.figure(figsize=(12, 4))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

image_path = "character1.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Cannot load {image_path}. Check the path.")

print("Image dimensions:", image.shape)
show(image, "Original Image")

_, binary = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
show(255 - binary, "Binary Image")

num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

MIN_AREA = 500
characters = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area < MIN_AREA:
        continue

    char = 255 - binary[y:y+h, x:x+w]
    char = cv2.copyMakeBorder(
        char, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255
    )
    characters.append((x, char))

characters.sort(key=lambda item: item[0])
print("Number of segmented characters:", len(characters))

if characters:
    cols = min(5, len(characters))
    rows = (len(characters) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.2 * cols, 2.8 * rows)
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (_, char) in enumerate(characters):
        axes[i].imshow(char, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Character {i + 1}")
        axes[i].axis("off")

    for ax in axes[len(characters):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No characters detected. Check the input image.")
