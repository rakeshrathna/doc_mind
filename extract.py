import json
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

EXAMPLE_IN = ("ACME TOOLS INV-77 2024-03-02 Hammer 2 x 150.00 300.00 "
              "Subtotal 300.00 GST 54.00 Total 354.00")
EXAMPLE_OUT = json.dumps({
    "vendor": "ACME TOOLS", "invoice_number": "INV-77", "invoice_date": "2024-03-02",
    "currency": "INR",
    "line_items": [{"description": "Hammer", "quantity": 2, "unit_price": 150.0,
                    "amount": 300.0}],
    "subtotal": 300.0, "tax": 54.0, "total": 354.0,
})


def extract(ocr_text, max_retries=2):
    user = f"OCR text:\n{EXAMPLE_IN}\n\nJSON:\n{EXAMPLE_OUT}\n\nOCR text:\n{ocr_text}\n\nJSON:"
    errors = []
    for attempt in range(max_retries + 1):
        raw = chat(SYSTEM, user)
        try:
            data = parse_json(raw)
            inv = Invoice(**data)
            return {"ok": True, "data": inv.model_dump(mode="json"),
                    "attempts": attempt + 1, "errors": errors}
        except (ValidationError, ValueError, KeyError, TypeError) as e:
            msg = str(e)[:800]
            errors.append(msg)
            user = (f"OCR text:\n{ocr_text}\n\nYour previous JSON was rejected:\n{raw}\n\n"
                    f"Errors:\n{msg}\n\nFix them and return corrected JSON only.")
    return {"ok": False, "data": None, "attempts": max_retries + 1,
            "errors": errors, "needs_human_review": True}
