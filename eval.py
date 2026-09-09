import json
import sys
import time
from pathlib import Path
from ocr import extract_text
from extract import extract

FIELDS = ["vendor", "invoice_number", "invoice_date", "subtotal", "tax", "total"]


def norm(v):
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return str(v).strip().lower() if v is not None else None


def main(truth_file="ground_truth.json", denoise_method="bilateral"):
    truth_path = Path(truth_file)
    if not truth_path.exists():
        print(f"Error: Ground truth file '{truth_file}' not found.")
        sys.exit(1)

    truth = json.loads(truth_path.read_text())
    if not truth:
        print("Ground truth dataset is empty.")
        return

    hits = {f: 0 for f in FIELDS}
    n = len(truth)
    failed = 0
    total_attempts = 0

    t_preprocess_list = []
    t_ocr_list = []
    t_llm_list = []
    t_total_list = []

    print(f"\n==================================================")
    print(f" RUNNING 1-DOCMIND EVALUATION BENCHMARK ({n} docs)")
    print(f" Denoise Method: {denoise_method}")
    print(f"==================================================\n")

    for item in truth:
        img_path = item["image"]
        if not Path(img_path).exists():
            print(f"SKIP {img_path}: File not found.")
            continue

        t_doc_start = time.time()
        
        # Preprocessing & OCR
        try:
            ocr_res = extract_text(img_path, denoise_method=denoise_method)
        except Exception as e:
            failed += 1
            print(f"FAIL {img_path}: OCR error - {str(e)}")
            continue

        t_ocr_ms = ocr_res.get("ms", 0)
        t_ocr_list.append(t_ocr_ms)

        # LLM Extraction & Validation
        t_llm_start = time.time()
        res = extract(ocr_res["text"])
        t_llm_ms = int((time.time() - t_llm_start) * 1000)
        t_llm_list.append(t_llm_ms)

        t_doc_ms = int((time.time() - t_doc_start) * 1000)
        t_total_list.append(t_doc_ms)

        total_attempts += res["attempts"]

        if not res["ok"]:
            failed += 1
            print(f"FAIL {img_path}: Extraction rejected - {res['errors'][-1][:100]}")
            continue

        got = res["data"]
        for f in FIELDS:
            if norm(got.get(f)) == norm(item["expected"].get(f)):
                hits[f] += 1

    print("\n--------------------------------------------------")
    print(" ACCURACY BENCHMARK METRICS")
    print("--------------------------------------------------")
    print(f"Total Documents      : {n}")
    print(f"Failed / Unparseable : {failed}")
    print(f"Avg Retries / Doc    : {(total_attempts / n) - 1:.2f}")
    print("\nField Accuracy Breakdown:")
    for f in FIELDS:
        acc = hits[f] / n if n > 0 else 0
        print(f"  - {f:16s} : {hits[f]}/{n} ({acc:.0%})")
    
    total_hits = sum(hits.values())
    total_possible = n * len(FIELDS)
    overall_acc = total_hits / total_possible if total_possible > 0 else 0
    print(f"\nOVERALL FIELD ACCURACY : {total_hits}/{total_possible} ({overall_acc:.1%})")

    print("\n--------------------------------------------------")
    print(" PERFORMANCE & LATENCY METRICS")
    print("--------------------------------------------------")
    avg_ocr = sum(t_ocr_list) / len(t_ocr_list) if t_ocr_list else 0
    avg_llm = sum(t_llm_list) / len(t_llm_list) if t_llm_list else 0
    avg_total = sum(t_total_list) / len(t_total_list) if t_total_list else 0

    print(f"Avg OCR + Preprocess : {avg_ocr:.1f} ms")
    print(f"Avg LLM + Validation : {avg_llm:.1f} ms")
    print(f"Avg Total Latency    : {avg_total:.1f} ms")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    method = sys.argv[2] if len(sys.argv) > 2 else "bilateral"
    file_arg = sys.argv[1] if len(sys.argv) > 1 else "ground_truth.json"
    main(file_arg, method)
