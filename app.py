import os
import time
import uuid
import tempfile
import logging
from flask import Flask, request, jsonify, g
from PIL import Image

from exceptions import (
    DocMindError, TesseractUnavailableError, OllamaUnavailableError,
    ModelNotFoundError, InvalidImageError, InvalidFileTypeError, FileTooLargeError
)
from ocr import extract_text, check_tesseract_available
from extract import extract
from llm import check_ollama_status

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@app.before_request
def start_request():
    g.rid = uuid.uuid4().hex[:8]
    g.t0 = time.time()


@app.after_request
def log_request(resp):
    ms = int((time.time() - g.t0) * 1000)
    resp.headers["X-Request-ID"] = g.rid
    resp.headers["X-Response-Time-Ms"] = str(ms)
    logging.info("%s %s %s Status:%d Duration:%dms",
                 g.rid, request.method, request.path, resp.status_code, ms)
    return resp


@app.errorhandler(DocMindError)
def handle_docmind_error(err):
    logging.warning("[%s] Domain Error (%s): %s", g.rid, err.error_code, err.message)
    return jsonify({
        "request_id": g.rid,
        "error_code": err.error_code,
        "message": err.message
    }), err.status_code


@app.errorhandler(413)
def handle_large_file(err):
    return jsonify({
        "request_id": g.rid,
        "error_code": "FILE_TOO_LARGE",
        "message": f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB."
    }), 400


@app.errorhandler(Exception)
def handle_generic_error(err):
    logging.error("[%s] Unhandled Exception: %s", g.rid, str(err), exc_info=True)
    return jsonify({
        "request_id": g.rid,
        "error_code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected internal error occurred."
    }), 500


@app.get("/health")
def health():
    tesseract_ok = True
    tesseract_err = None
    try:
        check_tesseract_available()
    except TesseractUnavailableError as e:
        tesseract_ok = False
        tesseract_err = e.message

    ollama_ok = True
    ollama_err = None
    try:
        check_ollama_status()
    except (OllamaUnavailableError, ModelNotFoundError) as e:
        ollama_ok = False
        ollama_err = e.message

    status = "ok" if (tesseract_ok and ollama_ok) else "degraded"
    return jsonify({
        "status": status,
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "tesseract_available": tesseract_ok,
        "tesseract_error": tesseract_err,
        "ollama_available": ollama_ok,
        "ollama_error": ollama_err
    }), 200 if status == "ok" else 503


@app.post("/extract")
def do_extract():
    f = request.files.get("file")
    if not f or not f.filename:
        raise InvalidImageError("Form field 'file' with a valid filename is required.")

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(list(ALLOWED_EXTENSIONS))}")

    # Validate image magic bytes / file payload integrity
    try:
        f.stream.seek(0)
        img_verify = Image.open(f.stream)
        img_verify.verify()
        f.stream.seek(0)
    except Exception as e:
        raise InvalidImageError("Uploaded file payload is corrupted or not a valid image format.") from e

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        path = tmp.name
    
    f.save(path)

    try:
        logging.info("[%s] Preprocessing & OCR started for file %s", g.rid, f.filename)
        ocr_res = extract_text(path)
        
        if ocr_res["word_count"] < 3:
            return jsonify({
                "request_id": g.rid,
                "ok": False,
                "error_code": "OCR_MINIMAL_TEXT",
                "reason": "OCR detected almost no text in document.",
                "ocr": ocr_res
            }), 422

        logging.info("[%s] OCR completed. Words:%d Lines:%d. LLM extraction starting...",
                     g.rid, ocr_res["word_count"], ocr_res["line_count"])
        
        result = extract(ocr_res["text"])
        result["ocr"] = ocr_res
        result["request_id"] = g.rid
        
        return jsonify(result), 200 if result["ok"] else 422

    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(port=5000, debug=True)
