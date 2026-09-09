import os
import cv2
import numpy as np

LINES = [
    ("ACME TOOLS PVT LTD", 1.0),
    ("Invoice No: INV-77", 0.8),
    ("Date: 2024-03-02", 0.8),
    ("", 0.5),
    ("Hammer      2 x 150.00      300.00", 0.7),
    ("", 0.5),
    ("Subtotal                    300.00", 0.8),
    ("GST 18%                      54.00", 0.8),
    ("Total                       354.00", 0.9),
]

img = np.full((520, 900), 255, np.uint8)
y = 60
for text, scale in LINES:
    if text:
        cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2, cv2.LINE_AA)
    y += 50

m = cv2.getRotationMatrix2D((450, 260), 1.5, 1.0)
img = cv2.warpAffine(img, m, (900, 520), borderValue=255)
img = cv2.add(img, (np.random.randn(520, 900) * 12).astype(np.int8).astype(np.uint8))
os.makedirs("samples", exist_ok=True)
cv2.imwrite("samples/invoice_01.png", img)
print("wrote samples/invoice_01.png")

