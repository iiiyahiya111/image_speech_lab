"""
py -m pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the original image in COLOR
img = cv2.imread("cat.png", cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("Could not load cat.png. Check the file path.")

# OpenCV uses BGR; convert to RGB for Matplotlib display
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Image enhancement using histogram equalization
# Apply equalization to brightness, not each color channel separately
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
enhanced_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

# 3. Smoothing filters
mean_bgr = cv2.blur(img, (3, 3))
gaussian_bgr = cv2.GaussianBlur(img, (3, 3), 0)
median_bgr = cv2.medianBlur(img, 3)

mean = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2RGB)
gaussian = cv2.cvtColor(gaussian_bgr, cv2.COLOR_BGR2RGB)
median = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2RGB)

# 4. Sharpening filter
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharp_bgr = cv2.filter2D(img, -1, kernel)
sharp = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB)

# 5. Compare all results
titles = [
    "Original Image",
    "Enhanced Image",
    "Mean Filter",
    "Gaussian Filter",
    "Median Filter",
    "Sharpened Image"
]

images = [original, enhanced, mean, gaussian, median, sharp]

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9),
    constrained_layout=True
)

for ax, title, image in zip(axes.flat, titles, images):
    ax.imshow(image)
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis("off")

plt.show()
