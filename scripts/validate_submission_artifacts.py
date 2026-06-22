import csv
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_PDF = Path("C:/Users/wangz/Downloads/97.pdf")
DESKTOP_PDF = Path("C:/Users/wangz/Desktop/97.pdf")
MAIN_TEX = ROOT / "paper" / "main.tex"

EXPECTED_ROWS = {
    "results/dataset_summary.csv": 23040,
    "results/rollouts.csv": 322560,
    "results/ablation_rollouts.csv": 115200,
    "results/stress_sweep_raw.csv": 259200,
    "results/fixed_risk_raw.csv": 138240,
    "results/failure_cases.csv": 24,
}


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def pdf_pages(path: Path) -> int:
    completed = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    match = re.search(r"^Pages:\s+(\d+)", completed.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("could not parse pdfinfo page count")
    return int(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    if not DOWNLOADS_PDF.exists():
        raise FileNotFoundError(DOWNLOADS_PDF)
    if DESKTOP_PDF.exists():
        raise RuntimeError(f"visible Desktop PDF is not allowed: {DESKTOP_PDF}")
    pages = pdf_pages(DOWNLOADS_PDF)
    if pages < 25:
        raise RuntimeError(f"PDF is too short: {pages} pages")
    tex = MAIN_TEX.read_text(encoding="utf-8")
    for required in ["citebordercolor={0 1 0}", "linkbordercolor={1 0.55 0}", "pdfborder={0 0 1.2}"]:
        if required not in tex:
            raise RuntimeError(f"missing bright boxed citation/link setting: {required}")
    for relative, expected in EXPECTED_ROWS.items():
        observed = count_csv_rows(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"{relative}: expected {expected} rows, observed {observed}")
    digest = sha256(DOWNLOADS_PDF)
    print(f"validated Paper 97 artifacts: pages={pages}, sha256={digest}")


if __name__ == "__main__":
    main()
