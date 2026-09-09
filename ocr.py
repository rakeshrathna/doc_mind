import os
import time
import pytesseract
from preprocess import clean
from exceptions import TesseractUnavailableError

CONFIG = "--oem 3 --psm 6"


def configure_tesseract():
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def check_tesseract_available():
    configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        cmd_info = os.getenv("TESSERACT_CMD") or "system PATH"
        raise TesseractUnavailableError(
            f"Tesseract OCR executable is not available (checked: {cmd_info}). "
            "Please install Tesseract OCR and set the TESSERACT_CMD environment variable "
            "to the full executable path (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)."
        ) from e


def extract_text(image_path, preprocess=True, debug_path=None, denoise_method=None):
    t0 = time.time()
    check_tesseract_available()

    img = clean(image_path, debug_path, denoise_method) if preprocess else image_path
    try:
        data = pytesseract.image_to_data(img, config=CONFIG,
                                         output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        raise TesseractUnavailableError() from e
    except Exception as e:
        raise RuntimeError(f"OCR execution failed: {str(e)}") from e

    lines_dict = {}
    confs = []
    word_count = 0

    blocks = data.get("block_num", [])
    pars = data.get("par_num", [])
    lines = data.get("line_num", [])
    texts = data.get("text", [])
    confidence_list = data.get("conf", [])

    for block, par, line, txt, conf in zip(blocks, pars, lines, texts, confidence_list):
        txt_str = str(txt).strip() if txt else ""
        if txt_str:
            try:
                conf_val = float(conf)
            except (ValueError, TypeError):
                conf_val = -1.0
            if conf_val > 0:
                key = (block, par, line)
                if key not in lines_dict:
                    lines_dict[key] = []
                lines_dict[key].append(txt_str)
                confs.append(conf_val)
                word_count += 1

    formatted_lines = [" ".join(words) for words in lines_dict.values() if words]
    structured_text = "\n".join(formatted_lines)

    return {
        "text": structured_text,
        "mean_conf": round(sum(confs) / len(confs), 2) if confs else 0.0,
        "word_count": word_count,
        "line_count": len(formatted_lines),
        "ms": int((time.time() - t0) * 1000),
    }
