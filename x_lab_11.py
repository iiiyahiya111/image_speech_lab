
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE)

# Fourier Transform
dft = np.fft.fftshift(np.fft.fft2(img))

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# Low-pass filter for smoothing
low_pass = np.fft.ifft2(np.fft.ifftshift(dft * mask))
low_pass = np.uint8(np.abs(low_pass))

# High-pass filter for sharpening
high_pass = np.fft.ifft2(np.fft.ifftshift(dft * (1 - mask)))
high_pass = np.uint8(np.abs(high_pass))

# Contrast enhancement
enhanced = cv2.equalizeHist(img)

# Display results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(enhanced, cmap="gray")
plt.title("Enhanced Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(low_pass, cmap="gray")
plt.title("Low-pass Smoothing")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(high_pass, cmap="gray")
plt.title("High-pass Sharpening")
plt.axis("off")

plt.tight_layout()
plt.show()
