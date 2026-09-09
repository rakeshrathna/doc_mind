import json
import pytest
from unittest.mock import patch
from extract import extract


@patch("extract.chat")
def test_successful_extraction(mock_chat):
    valid_json = json.dumps({
        "vendor": "ACME TOOLS",
        "invoice_number": "INV-77",
        "invoice_date": "2024-03-02",
        "currency": "INR",
        "line_items": [{"description": "Hammer", "quantity": 2.0, "unit_price": 150.0, "amount": 300.0}],
        "subtotal": 300.0,
        "tax": 54.0,
        "total": 354.0
    })
    mock_chat.return_value = valid_json

    res = extract("ACME TOOLS INV-77 Hammer 2 x 150 = 300")
    assert res["ok"] is True
    assert res["attempts"] == 1
    assert res["data"]["vendor"] == "ACME TOOLS"


@patch("extract.chat")
def test_validation_retry(mock_chat):
    # Attempt 1: Invalid Math (subtotal 500 != line item 300)
    bad_json = json.dumps({
        "vendor": "ACME TOOLS",
        "invoice_number": "INV-77",
        "currency": "INR",
        "line_items": [{"description": "Hammer", "quantity": 2.0, "unit_price": 150.0, "amount": 300.0}],
        "subtotal": 500.0,  # BAD SUBTOTAL
        "tax": 54.0,
        "total": 554.0
    })

    # Attempt 2: Fixed Math
    good_json = json.dumps({
        "vendor": "ACME TOOLS",
        "invoice_number": "INV-77",
        "currency": "INR",
        "line_items": [{"description": "Hammer", "quantity": 2.0, "unit_price": 150.0, "amount": 300.0}],
        "subtotal": 300.0,
        "tax": 54.0,
        "total": 354.0
    })

    mock_chat.side_effect = [bad_json, good_json]

    res = extract("ACME TOOLS INV-77 Hammer 2 x 150 = 300", max_retries=2)
    assert res["ok"] is True
    assert res["attempts"] == 2
    assert len(res["errors"]) == 1
    assert "subtotal says 500" in res["errors"][0]


@patch("extract.chat")
def test_max_retries_exceeded(mock_chat):
    bad_json = json.dumps({
        "vendor": "ACME TOOLS",
        "invoice_number": "INV-77",
        "subtotal": 999.0,
        "tax": 0.0,
        "total": 500.0,  # 999 + 0 != 500, violates invariant
        "line_items": []
    })
    mock_chat.return_value = bad_json

    res = extract("ACME TOOLS INV-77", max_retries=1)
    assert res["ok"] is False
    assert res["attempts"] == 2
    assert res["needs_human_review"] is True
