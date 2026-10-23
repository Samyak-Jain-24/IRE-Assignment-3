"""Assemble report/report.html with the generated tables, figures and numbers, then print it to PDF.

Placeholders in the template:
  {{table:NAME}}       results/tables/NAME.html
  {{fig:FILE}}         results/figures/FILE, embedded as a data URI
  {{num:KEY|FMT}}      value KEY of results/tables/report_numbers.json, formatted with FMT
                       (a Python format spec such as ".3f" or ",.0f"; "x100" multiplies by 100 first)

The PDF is printed with a headless Chromium-based browser (Edge or Chrome) if one is found.
Full mode writes ../report.pdf, quick mode writes ../results_quick/report_quick.pdf.

Usage: python scripts/build_report.py [--quick]
"""
import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annbench import config

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def find_browser():
    if os.environ.get("REPORT_BROWSER"):
        return os.environ["REPORT_BROWSER"]
    for name in ("msedge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    return next((b for b in BROWSERS if Path(b).exists()), None)


def fill(template: str, P: dict, numbers: dict) -> tuple[str, list]:
    missing = []

    def table(m):
        f = P["tables"] / f"{m.group(1)}.html"
        if not f.exists():
            missing.append(str(f))
            return f"<p><b>[missing table {m.group(1)}]</b></p>"
        return f.read_text()

    def fig(m):
        f = P["figures"] / m.group(1)
        if not f.exists():
            missing.append(str(f))
            return ""
        return "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()

    def num(m):
        key, fmt = m.group(1), m.group(2) or ""
        v = numbers.get(key)
        if v is None:
            missing.append(key)
            return "n/a"
        if isinstance(v, str):
            return v
        if fmt.startswith("x100"):
            v, fmt = v * 100, fmt[4:]
        return format(v, fmt)

    html = re.sub(r"\{\{table:([\w.-]+)\}\}", table, template)
    html = re.sub(r"\{\{fig:([\w.-]+)\}\}", fig, html)
    html = re.sub(r"\{\{num:([\w.-]+)(?:\|([^}]*))?\}\}", num, html)
    return html, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    P = config.paths(args.quick)
    numbers = json.loads((P["tables"] / "report_numbers.json").read_text())
    template = (config.CODE_DIR / "report" / "report.html").read_text(encoding="utf-8")
    html, missing = fill(template, P, numbers)
    if args.quick:
        html = html.replace("<body>", '<body><p class="banner">QUICK MODE (100K-vector subset): numbers differ from the '
                                      'submitted report; the text was written for the full run.</p>', 1)
    if missing:
        print("[report] warning, missing values:", sorted(set(missing)))
    out_html = P["work"] / "report_filled.html"
    out_html.write_text(html, encoding="utf-8")
    out_pdf = (P["results"] / "report_quick.pdf") if args.quick else (config.ROOT_DIR / "report.pdf")

    browser = find_browser()
    if browser is None:
        print(f"[report] no Edge/Chrome found; open {out_html} in a browser and print to PDF")
        return
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.resolve().as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    pages = len(re.findall(rb"/Type\s*/Page(?!s)", out_pdf.read_bytes()))
    print(f"[report] {out_pdf} ({pages} pages)")
    if pages > 5:
        print("[report] WARNING: more than 5 pages")


if __name__ == "__main__":
    main()
