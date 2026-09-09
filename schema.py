from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class LineItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float
    amount: float

    @model_validator(mode="after")
    def check_amount(self):
        if self.quantity is not None and self.unit_price is not None and self.amount is not None:
            expected = round(self.quantity * self.unit_price, 2)
            if abs(expected - round(self.amount, 2)) > 0.05:
                raise ValueError(
                    f"line '{self.description}': {self.quantity} x {self.unit_price} "
                    f"= {expected}, but amount says {self.amount}"
                )
        return self


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: Optional[date] = None
    currency: str = "INR"
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: float
    tax: float = 0.0
    total: float

    @field_validator("currency", mode="before")
    @classmethod
    def validate_currency(cls, v):
        if v is None:
            return "INR"
        if isinstance(v, str):
            v_str = v.strip().upper()
            return v_str[:3] if v_str else "INR"
        return "INR"

    @model_validator(mode="after")
    def check_totals(self):
        if self.line_items:
            s = round(sum(i.amount for i in self.line_items), 2)
            if abs(s - round(self.subtotal, 2)) > 0.05:
                raise ValueError(f"line items sum to {s}, subtotal says {self.subtotal}")
        expected_total = round(self.subtotal + self.tax, 2)
        if abs(expected_total - round(self.total, 2)) > 0.05:
            raise ValueError(
                f"subtotal {self.subtotal} + tax {self.tax} = {expected_total} != total {self.total}"
            )
        return self
