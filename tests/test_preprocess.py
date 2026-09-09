import os
import pytest
import cv2
import numpy as np
from preprocess import clean, deskew, upscale, apply_denoise
from exceptions import InvalidImageError


@pytest.fixture
def sample_image_array():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST INVOICE", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


@pytest.fixture
def sample_image_file(tmp_path, sample_image_array):
    file_path = str(tmp_path / "test_sample.png")
    cv2.imwrite(file_path, sample_image_array)
    return file_path


def test_valid_image(sample_image_file):
    binar = clean(sample_image_file)
    assert binar is not None
    assert binar.ndim == 2
    assert binar.shape[1] >= 1200  # Verify upscaling


def test_invalid_image():
    with pytest.raises(InvalidImageError):
        clean("non_existent_file.png")

    with pytest.raises(InvalidImageError):
        clean(None)


def test_deskew():
    gray = np.full((200, 300), 255, dtype=np.uint8)
    cv2.putText(gray, "SKEWED TEXT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    deskewed = deskew(gray)
    assert deskewed is not None
    assert deskewed.shape == gray.shape


def test_large_image_handling():
    large_img = np.full((4000, 6000, 3), 255, dtype=np.uint8)
    cv2.putText(large_img, "LARGE IMAGE", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    cleaned = clean(large_img)
    assert cleaned is not None
    assert cleaned.shape[1] <= 2500  # Max dimension bounds applied


def test_denoise_methods(sample_image_array):
    gray = cv2.cvtColor(sample_image_array, cv2.COLOR_BGR2GRAY)
    b_out = apply_denoise(gray, "bilateral")
    g_out = apply_denoise(gray, "gaussian")
    n_out = apply_denoise(gray, "nlmeans")

    assert b_out.shape == gray.shape
    assert g_out.shape == gray.shape
    assert n_out.shape == gray.shape
