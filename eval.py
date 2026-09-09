import json
import sys
from pathlib import Path
from ocr import extract_text
from extract import extract

FIELDS = ["vendor", "invoice_number", "invoice_date", "subtotal", "tax", "total"]


def norm(v):
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return str(v).strip().lower() if v is not None else None


def main(truth_file="ground_truth.json"):
    truth = json.loads(Path(truth_file).read_text())
    hits = {f: 0 for f in FIELDS}
    n, failed, attempts = 0, 0, 0

    for item in truth:
        n += 1
        ocr = extract_text(item["image"])
        res = extract(ocr["text"])
        attempts += res["attempts"]
        if not res["ok"]:
            failed += 1
            print(f"FAIL {item['image']}: {res['errors'][-1][:120]}")
            continue
        got = res["data"]
        for f in FIELDS:
            if norm(got.get(f)) == norm(item["expected"].get(f)):
                hits[f] += 1

    print(f"\ndocs={n} unparseable={failed} avg_attempts={attempts / n:.2f}")
    for f in FIELDS:
        print(f"{f:16s} {hits[f]}/{n}  {hits[f] / n:.0%}")
    total = sum(hits.values())
    print(f"{'FIELD ACCURACY':16s} {total}/{n * len(FIELDS)}  {total / (n * len(FIELDS)):.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:])
