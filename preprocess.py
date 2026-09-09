from pathlib import Path
import cv2
import numpy as np


def deskew(gray):
    inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inv > 0))
    if coords.size == 0:
        return gray
    # np.where returns (y, x) coordinates; minAreaRect requires (x, y) coordinates
    coords_xy = coords[:, ::-1]
    rect = cv2.minAreaRect(coords_xy)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
    if abs(angle) < 0.3:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def upscale(img, min_width=1200):
    h, w = img.shape[:2]
    if w >= min_width:
        return img
    s = min_width / w
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)


def clean(path_or_array, debug_path=None):
    if isinstance(path_or_array, Path):
        path_or_array = str(path_or_array)
    img = cv2.imread(path_or_array) if isinstance(path_or_array, str) else path_or_array
    if img is None:
        raise ValueError("could not read image")
    img = upscale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    binar = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 15)
    if debug_path:
        cv2.imwrite(debug_path, binar)
    return binar

