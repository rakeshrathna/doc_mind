import os
import time
import pytesseract
from preprocess import clean

CONFIG = "--oem 3 --psm 6"

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def extract_text(image_path, preprocess=True, debug_path=None):
    t0 = time.time()
    img = clean(image_path, debug_path) if preprocess else image_path
    data = pytesseract.image_to_data(img, config=CONFIG,
                                     output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for txt, conf in zip(data["text"], data["conf"]):
        if txt and str(txt).strip():
            try:
                conf_val = float(conf)
            except (ValueError, TypeError):
                conf_val = -1.0
            if conf_val > 0:
                words.append(str(txt).strip())
                confs.append(conf_val)
    return {
        "text": " ".join(words),
        "mean_conf": round(sum(confs) / len(confs), 2) if confs else 0.0,
        "word_count": len(words),
        "ms": int((time.time() - t0) * 1000),
    }

