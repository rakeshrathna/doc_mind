import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import time
import cv2
from preprocess import clean

image_path = "samples/invoice_01.png"

methods = ["bilateral", "gaussian", "nlmeans"]

print("==================================================")
print(" DENOISING METHOD BENCHMARK (samples/invoice_01.png)")
print("==================================================")

for m in methods:
    t0 = time.time()
    for _ in range(5):
        cleaned = clean(image_path, denoise_method=m)
    avg_ms = int(((time.time() - t0) / 5.0) * 1000)
    print(f"Method: {m:12s} | Avg Latency: {avg_ms:5d} ms | Binarized Shape: {cleaned.shape}")

print("==================================================")
