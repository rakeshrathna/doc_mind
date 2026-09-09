import io
import pytest
from unittest.mock import patch
from PIL import Image
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("app.check_tesseract_available")
@patch("app.check_ollama_status")
def test_health_endpoint(mock_ollama, mock_tesseract, client):
    mock_tesseract.return_value = True
    mock_ollama.return_value = True

    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["tesseract_available"] is True
    assert data["ollama_available"] is True


def test_missing_file(client):
    response = client.post("/extract")
    assert response.status_code == 400
    data = response.get_json()
    assert data["error_code"] == "INVALID_IMAGE"


def test_invalid_extension(client):
    data = {"file": (io.BytesIO(b"fake data"), "test.exe")}
    response = client.post("/extract", data=data, content_type="multipart/form-data")
    assert response.status_code == 415
    res = response.get_json()
    assert res["error_code"] == "INVALID_FILE_TYPE"


def test_invalid_image_payload(client):
    data = {"file": (io.BytesIO(b"not an image payload"), "test.png")}
    response = client.post("/extract", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res = response.get_json()
    assert res["error_code"] == "INVALID_IMAGE"


@patch("app.extract_text")
@patch("app.extract")
def test_successful_extract_endpoint(mock_extract, mock_ocr, client):
    # Create valid dummy PNG image stream
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    mock_ocr.return_value = {
        "text": "ACME TOOLS INV-77 Hammer 2 x 150 = 300",
        "word_count": 10,
        "line_count": 2,
        "mean_conf": 90.0,
        "ms": 150
    }
    mock_extract.return_value = {
        "ok": True,
        "data": {
            "vendor": "ACME TOOLS",
            "invoice_number": "INV-77",
            "subtotal": 300.0,
            "tax": 54.0,
            "total": 354.0
        },
        "attempts": 1,
        "errors": []
    }

    data = {"file": (img_bytes, "invoice.png")}
    response = client.post("/extract", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["ok"] is True
    assert res["data"]["vendor"] == "ACME TOOLS"
    assert "X-Request-ID" in response.headers
