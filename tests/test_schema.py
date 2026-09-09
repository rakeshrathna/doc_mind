import pytest
from pydantic import ValidationError
from schema import Invoice, LineItem


def test_valid_invoice():
    data = {
        "vendor": "ACME TOOLS",
        "invoice_number": "INV-77",
        "invoice_date": "2024-03-02",
        "currency": "INR",
        "line_items": [
            {"description": "Hammer", "quantity": 2.0, "unit_price": 150.0, "amount": 300.0}
        ],
        "subtotal": 300.0,
        "tax": 54.0,
        "total": 354.0
    }
    inv = Invoice(**data)
    assert inv.vendor == "ACME TOOLS"
    assert inv.currency == "INR"
    assert inv.total == 354.0


def test_invalid_line_amount():
    data = {
        "vendor": "ACME",
        "invoice_number": "123",
        "subtotal": 300.0,
        "tax": 0.0,
        "total": 300.0,
        "line_items": [
            # 2 x 150 = 300, but amount is 999
            {"description": "Hammer", "quantity": 2.0, "unit_price": 150.0, "amount": 999.0}
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        Invoice(**data)
    assert "amount says 999" in str(excinfo.value)


def test_invalid_subtotal():
    data = {
        "vendor": "ACME",
        "invoice_number": "123",
        "subtotal": 500.0,  # Line items sum to 300, but subtotal is 500
        "tax": 0.0,
        "total": 500.0,
        "line_items": [
            {"description": "Hammer", "quantity": 2.0, "unit_price": 150.0, "amount": 300.0}
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        Invoice(**data)
    assert "subtotal says 500" in str(excinfo.value)


def test_invalid_total():
    data = {
        "vendor": "ACME",
        "invoice_number": "123",
        "subtotal": 300.0,
        "tax": 50.0,
        "total": 400.0,  # 300 + 50 = 350 != 400
        "line_items": [
            {"description": "Hammer", "quantity": 2.0, "unit_price": 150.0, "amount": 300.0}
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        Invoice(**data)
    assert "subtotal 300.0 + tax 50.0 = 350.0 != total 400.0" in str(excinfo.value)


def test_none_currency():
    data = {
        "vendor": "ACME",
        "invoice_number": "123",
        "currency": None,  # None currency should safely default to INR
        "subtotal": 100.0,
        "tax": 0.0,
        "total": 100.0
    }
    inv = Invoice(**data)
    assert inv.currency == "INR"


def test_invalid_currency():
    data = {
        "vendor": "ACME",
        "invoice_number": "123",
        "currency": "usdollars",  # Trimmed to USD
        "subtotal": 100.0,
        "tax": 0.0,
        "total": 100.0
    }
    inv = Invoice(**data)
    assert inv.currency == "USD"


def test_missing_optional_fields():
    data = {
        "vendor": "ACME",
        "invoice_number": "123",
        "subtotal": 100.0,
        "tax": 0.0,
        "total": 100.0
    }
    inv = Invoice(**data)
    assert inv.invoice_date is None
    assert inv.line_items == []
