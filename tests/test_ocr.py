import pytest
from unittest.mock import patch
from ocr import extract_text, check_tesseract_available
from exceptions import TesseractUnavailableError


def test_tesseract_missing():
    with patch("pytesseract.get_tesseract_version", side_effect=Exception("Not found")):
        with pytest.raises(TesseractUnavailableError):
            check_tesseract_available()


@patch("ocr.check_tesseract_available")
@patch("ocr.clean")
@patch("pytesseract.image_to_data")
def test_line_preservation(mock_image_to_data, mock_clean, mock_check):
    mock_clean.return_value = "fake_img_array"
    mock_image_to_data.return_value = {
        "block_num": [1, 1, 1, 1],
        "par_num":   [1, 1, 1, 1],
        "line_num":  [1, 1, 2, 2],
        "text":      ["ACME", "TOOLS", "INV-77", "2024-03-02"],
        "conf":      [95, 90, 88, 92]
    }

    result = extract_text("fake_path.png", preprocess=True)

    assert result["word_count"] == 4
    assert result["line_count"] == 2
    # Verify line preservation: Line 1 has 'ACME TOOLS', Line 2 has 'INV-77 2024-03-02'
    expected_text = "ACME TOOLS\nINV-77 2024-03-02"
    assert result["text"] == expected_text
    assert result["mean_conf"] == 91.25


@patch("ocr.check_tesseract_available")
@patch("ocr.clean")
@patch("pytesseract.image_to_data")
def test_confidence_filtering(mock_image_to_data, mock_clean, mock_check):
    mock_clean.return_value = "fake_img_array"
    mock_image_to_data.return_value = {
        "block_num": [1, 1, 1],
        "par_num":   [1, 1, 1],
        "line_num":  [1, 1, 1],
        "text":      ["GoodWord", "BadWord", "EmptyConfWord"],
        "conf":      [90, -1, "invalid"]
    }

    result = extract_text("fake_path.png")
    assert result["word_count"] == 1
    assert result["text"] == "GoodWord"
