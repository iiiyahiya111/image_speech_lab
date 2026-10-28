"""
Q11: Multilevel (multi-resolution) analysis of a speech signal
Method: Discrete Wavelet Transform (DWT)

    Level k Approximation (A_k) -> low-frequency content (coarser resolution)
    Level k Detail        (D_k) -> high-frequency content lost at that level

Requires: numpy, scipy, matplotlib, PyWavelets
    pip install numpy scipy matplotlib PyWavelets
    py -m pip install numpy matplotlib scipy PyWavelets
"""
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile

# ---------- 1. Read the signal ----------
fs, x = wavfile.read('audio.wav')
if x.ndim > 1:
    x = x.mean(axis=1)
x = x.astype(float)
x = x / np.max(np.abs(x))
t = np.arange(len(x)) / fs

wavelet = 'db4'
levels = 4

# ---------- 2. Multilevel decomposition ----------
# coeffs = [cA4, cD4, cD3, cD2, cD1]
coeffs = pywt.wavedec(x, wavelet, level=levels)


def reconstruct(keep_index):
    """Rebuild a full-length signal using only ONE coefficient set."""
    c = [np.zeros_like(a) for a in coeffs]
    c[keep_index] = coeffs[keep_index]
    return pywt.waverec(c, wavelet)[:len(x)]


# Approximation at the deepest level and details at every level
approx = reconstruct(0)                                   # A4
details = {k: reconstruct(levels - k + 1) for k in range(1, levels + 1)}  # D1..D4

# Approximations at intermediate levels A1..A3 (decompose to that level, keep cA)
approximations = {}
for k in range(1, levels + 1):
    ck = pywt.wavedec(x, wavelet, level=k)
    ck = [ck[0]] + [np.zeros_like(a) for a in ck[1:]]
    approximations[k] = pywt.waverec(ck, wavelet)[:len(x)]

# ---------- 3. Plot ----------
plt.figure(figsize=(11, 12))

plt.subplot(levels + 1, 2, 1)
plt.plot(t, x, 'k'); plt.title('Original signal')

for k in range(1, levels + 1):
    plt.subplot(levels + 1, 2, 2 * k + 1)
    plt.plot(t, approximations[k], 'b')
    plt.title(f'Approximation A{k}  (0 - {fs / 2 ** (k + 1):.0f} Hz)')

    plt.subplot(levels + 1, 2, 2 * k + 2)
    plt.plot(t, details[k], 'r')
    plt.title(f'Detail D{k}  ({fs / 2 ** (k + 1):.0f} - {fs / 2 ** k:.0f} Hz)')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# ---------- 4. Check: A4 + D4 + D3 + D2 + D1 gives back the original ----------
recon = approx + sum(details.values())
print('Max reconstruction error:', np.max(np.abs(x - recon)))