import os
from pathlib import Path
import cv2
import numpy as np
from exceptions import InvalidImageError

MAX_WIDTH_DEFAULT = int(os.getenv("MAX_IMAGE_WIDTH", "2500"))
MAX_HEIGHT_DEFAULT = int(os.getenv("MAX_IMAGE_HEIGHT", "3500"))
DENOISE_METHOD_DEFAULT = os.getenv("DENOISE_METHOD", "bilateral").lower()


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


def upscale(img, min_width=1200, max_width=MAX_WIDTH_DEFAULT):
    h, w = img.shape[:2]
    if w >= min_width:
        if w > max_width:
            s = max_width / w
            return cv2.resize(img, (max_width, int(h * s)), interpolation=cv2.INTER_AREA)
        return img
    s = min_width / w
    target_w = min(int(w * s), max_width)
    target_h = int(h * (target_w / w))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def apply_denoise(gray, method=None):
    if method is None:
        method = DENOISE_METHOD_DEFAULT
    method = method.lower()
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (3, 3), 0)
    elif method == "nlmeans":
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    else:  # default "bilateral"
        return cv2.bilateralFilter(gray, 9, 75, 75)


def clean(path_or_array, debug_path=None, denoise_method=None):
    if isinstance(path_or_array, Path):
        path_or_array = str(path_or_array)

    if isinstance(path_or_array, str):
        if not os.path.exists(path_or_array) or os.path.getsize(path_or_array) == 0:
            raise InvalidImageError("Image file does not exist or is empty.")
        img = cv2.imread(path_or_array)
    elif isinstance(path_or_array, np.ndarray):
        img = path_or_array
    else:
        raise InvalidImageError("Invalid image input type.")

    if img is None or img.size == 0:
        raise InvalidImageError("Could not read image or image data is empty.")

    h, w = img.shape[:2]
    if h > MAX_HEIGHT_DEFAULT * 2 or w > MAX_WIDTH_DEFAULT * 2:
        # Downscale extreme dimension images before processing
        s = min(MAX_WIDTH_DEFAULT / w, MAX_HEIGHT_DEFAULT / h)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    img = upscale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = deskew(gray)
    gray = apply_denoise(gray, method=denoise_method)
    binar = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 15)
    if debug_path:
        cv2.imwrite(debug_path, binar)
    return binar
