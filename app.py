import os
import time
import uuid
import tempfile
import logging
from flask import Flask, request, jsonify, g
from ocr import extract_text
from extract import extract

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@app.before_request
def start():
    g.rid = uuid.uuid4().hex[:8]
    g.t0 = time.time()


@app.after_request
def done(resp):
    ms = int((time.time() - g.t0) * 1000)
    resp.headers["X-Request-ID"] = g.rid
    logging.info("%s %s %s %s %dms", g.rid, request.method, request.path,
                 resp.status_code, ms)
    return resp


@app.get("/health")
def health():
    return jsonify(status="ok", model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))


@app.post("/extract")
def do_extract():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="file field required"), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error=f"unsupported type {ext}"), 415

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        path = tmp.name
    f.save(path)
    try:
        ocr = extract_text(path)
        if ocr["word_count"] < 5:
            return jsonify(request_id=g.rid, ok=False,
                           reason="ocr produced almost no text", ocr=ocr), 422
        result = extract(ocr["text"])
        result["ocr"] = ocr
        result["request_id"] = g.rid
        return jsonify(result), 200 if result["ok"] else 422
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
