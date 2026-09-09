import json
import os
from pydantic import ValidationError
from schema import Invoice
from llm import chat, parse_json

SYSTEM = (
    "You extract invoice data from noisy OCR text.\n"
    "Return ONLY JSON matching this shape:\n"
    '{"vendor": str, "invoice_number": str, "invoice_date": "YYYY-MM-DD" or null,\n'
    ' "currency": str, "line_items": [{"description": str, "quantity": num,\n'
    ' "unit_price": num, "amount": num}], "subtotal": num, "tax": num, "total": num}\n'
    "Rules:\n"
    "- Numbers are plain numbers, no currency symbols or commas.\n"
    "- amount must equal quantity * unit_price.\n"
    "- subtotal must equal the sum of line item amounts.\n"
    "- subtotal + tax must equal total.\n"
    "- If a field is genuinely absent, use null (or 0 for tax). Never invent a vendor."
)

EXAMPLE_IN = ("ACME TOOLS PVT LTD\nInvoice No: INV-77\nDate: 2024-03-02\n\n"
              "Hammer      2 x 150.00      300.00\n\n"
              "Subtotal                    300.00\n"
              "GST 18%                      54.00\n"
              "Total                       354.00")
EXAMPLE_OUT = json.dumps({
    "vendor": "ACME TOOLS", "invoice_number": "INV-77", "invoice_date": "2024-03-02",
    "currency": "INR",
    "line_items": [{"description": "Hammer", "quantity": 2.0, "unit_price": 150.0,
                    "amount": 300.0}],
    "subtotal": 300.0, "tax": 54.0, "total": 354.0,
})


def build_prompt(ocr_text, previous_raw=None, error_msg=None):
    base = f"OCR text:\n{EXAMPLE_IN}\n\nJSON:\n{EXAMPLE_OUT}\n\nOCR text:\n{ocr_text}\n\nJSON:"
    if previous_raw and error_msg:
        return (f"{base}\n\n"
                f"[PREVIOUS ATTEMPT REJECTED BY VALIDATOR]\n"
                f"Your previous JSON was:\n{previous_raw}\n\n"
                f"Validation Errors:\n{error_msg}\n\n"
                f"Fix the calculation errors above and return valid JSON matching the exact schema.")
    return base


def extract(ocr_text, max_retries=None):
    if max_retries is None:
        max_retries = int(os.getenv("MAX_RETRIES", "2"))

    errors = []
    previous_raw = None
    error_msg = None

    for attempt in range(max_retries + 1):
        user_prompt = build_prompt(ocr_text, previous_raw, error_msg)
        raw = chat(SYSTEM, user_prompt)
        try:
            data = parse_json(raw)
            inv = Invoice(**data)
            return {
                "ok": True,
                "data": inv.model_dump(mode="json"),
                "attempts": attempt + 1,
                "errors": errors,
                "needs_human_review": False,
            }
        except (ValidationError, ValueError, KeyError, TypeError, Exception) as e:
            msg = str(e)[:800]
            errors.append(msg)
            previous_raw = raw
            error_msg = msg

    return {
        "ok": False,
        "data": None,
        "attempts": max_retries + 1,
        "errors": errors,
        "needs_human_review": True,
    }
