import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE)

# Convert image to binary
_, binary = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Invert binary image
binary = 255 - binary

# Horizontal Line Detection
horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (20, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)

# Vertical Line Detection
vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, 20)
)

vertical_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel
)

# Display Results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(horizontal_lines, cmap="gray")
plt.title("Horizontal Lines")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(vertical_lines, cmap="gray")
plt.title("Vertical Lines")
plt.axis("off")

plt.tight_layout()
plt.show()