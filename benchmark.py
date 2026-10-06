#!/usr/bin/env python3
"""Benchmark Tesseract on a full 4K screenshot without known text locations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import statistics
import sys
import time
import unicodedata
import platform
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import pytesseract


ROOT = Path(__file__).resolve().parent
DEFAULT_TRUTH = ROOT / "ground_truth.json"
DEFAULT_IMAGE = ROOT / "test_files" / "FontQuadrantsDemo.jpg"
METHODS = ("color", "grayscale", "autocontrast", "otsu")
SCALES = (1, 2, 3)
DEFAULT_PSMS = (3, 6, 11, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OCR settings on the entire image; expected phrase locations are not used."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input screenshot.")
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH, help="JSON file containing expected phrases.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_light_dark", help="Where light/dark full-image results are written.")
    parser.add_argument("--repeats", type=int, default=2, help="Measured runs per configuration (default: 2).")
    parser.add_argument("--limit-configs", type=int, help="Run only the first N configs; useful for a quick smoke test.")
    parser.add_argument("--psm", type=int, nargs="+", choices=range(3, 14), default=list(DEFAULT_PSMS), help="Page segmentation modes to compare.")
    return parser.parse_args()


def get_host_hardware() -> dict[str, str]:
    system = platform.system()

    def run(command: list[str]) -> str:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    cpu = platform.processor().strip()
    if system == "Windows":
        for shell in ("powershell.exe", "pwsh"):
            output = run([
                shell, "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ])
            if output:
                cpu = output.splitlines()[0].strip()
                break
        cpu = cpu or os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    elif system == "Darwin":
        cpu = run(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu

    cpu_info = Path("/proc/cpuinfo") if system == "Linux" else None
    if cpu_info and cpu_info.is_file() and not cpu:
        for line in cpu_info.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                cpu = line.split(":", 1)[1].strip()
                break
    if not cpu:
        cpu = platform.machine() or "Unknown"

    gpus: list[str] = []
    if system == "Windows":
        for shell in ("powershell.exe", "pwsh"):
            output = run([
                shell, "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ])
            names = [line.strip() for line in output.splitlines() if line.strip()]
            if names:
                gpus = names
                break
    elif system == "Darwin":
        output = run(["system_profiler", "SPDisplaysDataType"])
        gpus = [
            line.split(":", 1)[1].strip()
            for line in output.splitlines()
            if line.strip().startswith("Chipset Model:")
        ]
    elif system == "Linux":
        try:
            devices = run(["lspci", "-mm"]).splitlines()
        except (OSError, subprocess.SubprocessError, csv.Error):
            devices = []
        for line in devices:
            try:
                fields = next(csv.reader([line], delimiter=" ", quotechar='"', skipinitialspace=True))
            except csv.Error:
                continue
            if len(fields) > 2 and fields[1].startswith(("VGA", "3D", "Display")):
                gpus.append(" ".join(fields[6:8]) if len(fields) > 7 and fields[6] else " ".join(fields[2:4]))

    if not gpus:
        for command in (
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            ["rocm-smi", "--showproductname"],
        ):
            output = run(command)
            names = [line.strip() for line in output.splitlines() if line.strip()]
            if names:
                gpus = names
                break

    return {
        "os": platform.platform() or system or "Unknown",
        "cpu": cpu,
        "gpu": ", ".join(gpus) if gpus else "Not detected",
    }


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def otsu_threshold(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    histogram = grayscale.histogram()
    total = sum(histogram)
    weighted_total = sum(level * count for level, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    threshold = 127
    for level, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += level * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = level
    return grayscale.point(lambda pixel: 255 if pixel > threshold else 0)


def prepare(image: Image.Image, method: str, scale: int) -> Image.Image:
    if method == "color":
        prepared = image.copy()
    elif method == "grayscale":
        prepared = ImageOps.grayscale(image)
    elif method == "autocontrast":
        prepared = ImageOps.autocontrast(ImageOps.grayscale(image))
    elif method == "otsu":
        prepared = otsu_threshold(image)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    if scale != 1:
        prepared = prepared.resize(
            (prepared.width * scale, prepared.height * scale),
            Image.Resampling.LANCZOS,
        )
    return prepared


def make_configs(psms: list[int]) -> list[dict[str, Any]]:
    return [
        {"method": method, "scale": scale, "psm": psm}
        for method in METHODS
        for scale in SCALES
        for psm in psms
    ]


def measure_config(
    source: Image.Image,
    expected: dict[str, str],
    config: dict[str, Any],
    repeats: int,
) -> dict[str, Any]:
    runs = []
    config_name = f"full_image_{config['method']}_x{config['scale']}_psm{config['psm']}"
    normalized_expected = {name: normalize(phrase).split() for name, phrase in expected.items()}
    for _ in range(repeats):
        start = time.perf_counter()
        image = prepare(source, config["method"], config["scale"])
        prep_seconds = time.perf_counter() - start

        start = time.perf_counter()
        recognized = pytesseract.image_to_string(
            image,
            lang="eng",
            config=f"--oem 3 --psm {config['psm']}",
        ).strip()
        ocr_seconds = time.perf_counter() - start

        normalized_ocr = normalize(recognized).split()
        found = {
            name: bool(phrase) and any(
                normalized_ocr[index:index + len(phrase)] == phrase
                for index in range(len(normalized_ocr) - len(phrase) + 1)
            )
            for name, phrase in normalized_expected.items()
        }
        runs.append({
            "preprocess_ms": prep_seconds * 1000,
            "ocr_ms": ocr_seconds * 1000,
            "end_to_end_ms": (prep_seconds + ocr_seconds) * 1000,
            "found": found,
            "recognized": recognized,
        })

    median_ocr = statistics.median(run["ocr_ms"] for run in runs)
    representative = min(runs, key=lambda run: abs(run["ocr_ms"] - median_ocr))
    found_per_phrase = {
        name: sum(run["found"][name] for run in runs)
        for name in expected
    }
    mode_counts = {mode: 0 for mode in ("light", "dark")}
    mode_totals = {mode: 0 for mode in mode_counts}
    font_mode_counts: dict[str, int] = {}
    font_mode_totals: dict[str, int] = {}
    size_counts: dict[int, int] = {}
    size_totals: dict[int, int] = {}
    font_mode_size_counts: dict[str, dict[int, int]] = {}
    font_mode_size_totals: dict[str, dict[int, int]] = {}
    for name, count in found_per_phrase.items():
        match = re.fullmatch(r"(.+)_(light|dark)_(\d+)px", name)
        if match:
            font, mode, size_text = match.groups()
            size = int(size_text)
            group = f"{font}_{mode}"
            mode_counts[mode] += count
            mode_totals[mode] += len(runs)
            font_mode_counts[group] = font_mode_counts.get(group, 0) + count
            font_mode_totals[group] = font_mode_totals.get(group, 0) + len(runs)
            size_counts[size] = size_counts.get(size, 0) + count
            size_totals[size] = size_totals.get(size, 0) + len(runs)
            font_mode_size_counts.setdefault(group, {})[size] = count
            font_mode_size_totals.setdefault(group, {})[size] = len(runs)
    size_phrase_accuracy = {
        f"{size}px": size_counts[size] / size_totals[size]
        for size in sorted(size_counts)
    }
    smallest_100pct_size = next(
        (size for size in sorted(size_counts) if size_counts[size] == size_totals[size]),
        None,
    )
    font_mode_smallest_100pct_size = {
        group: next(
            (
                size for size in sorted(counts)
                if counts[size] == font_mode_size_totals[group][size]
            ),
            None,
        )
        for group, counts in font_mode_size_counts.items()
    }
    exact_runs = sum(all(run["found"].values()) for run in runs)
    total_phrase_checks = len(expected) * len(runs)
    return {
        "config": config_name,
        **config,
        "preprocess_ms": statistics.median(run["preprocess_ms"] for run in runs),
        "ocr_ms": median_ocr,
        "end_to_end_ms": statistics.median(run["end_to_end_ms"] for run in runs),
        "fully_exact_runs": exact_runs,
        "runs": len(runs),
        "all_runs_exact": exact_runs == len(runs),
        "found_phrases": found_per_phrase,
        "phrase_accuracy": sum(found_per_phrase.values()) / total_phrase_checks,
        "mode_phrase_accuracy": {
            mode: mode_counts[mode] / mode_totals[mode]
            for mode in mode_counts if mode_totals[mode]
        },
        "size_phrase_accuracy": size_phrase_accuracy,
        "smallest_100pct_size_px": smallest_100pct_size,
        "font_mode_smallest_100pct_size_px": font_mode_smallest_100pct_size,
        "font_mode_phrase_accuracy": {
            group: count / font_mode_totals[group]
            for group, count in font_mode_counts.items()
        },
        "recognized": representative["recognized"],
    }


def write_csv(path: Path, results: list[dict[str, Any]], phrase_names: list[str]) -> None:
    fields = [
        "rank", "config", "method", "scale", "psm", "preprocess_ms", "ocr_ms",
        "end_to_end_ms", "fully_exact_runs", "runs", "all_runs_exact", "phrase_accuracy",
        "smallest_100pct_size_px",
    ]
    fields.extend(f"found_{name}" for name in phrase_names)
    fields.extend(("light_phrase_accuracy", "dark_phrase_accuracy"))
    sizes = sorted({
        int(match.group(1))
        for name in phrase_names
        if (match := re.search(r"_(\d+)px$", name))
    })
    fields.extend(f"size_{size}px_accuracy" for size in sizes)
    groups = sorted({
        "_".join(name.split("_")[:-2]) + "_" + name.split("_")[-2]
        for name in phrase_names
    })
    fields.extend(f"accuracy_{group}" for group in groups)
    fields.extend(f"smallest_100pct_{group}_size_px" for group in groups)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            quoting=csv.QUOTE_ALL,
            doublequote=True,
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {field: result.get(field) for field in fields}
            row["rank"] = rank
            row.update({
                f"found_{name}": result["found_phrases"][name]
                for name in phrase_names
            })
            row["light_phrase_accuracy"] = result["mode_phrase_accuracy"].get("light", 0.0)
            row["dark_phrase_accuracy"] = result["mode_phrase_accuracy"].get("dark", 0.0)
            row.update({
                f"size_{size}_accuracy": accuracy
                for size, accuracy in result["size_phrase_accuracy"].items()
            })
            row.update({
                f"accuracy_{group}": accuracy
                for group, accuracy in result["font_mode_phrase_accuracy"].items()
            })
            row.update({
                f"smallest_100pct_{group}_size_px": size
                for group, size in result["font_mode_smallest_100pct_size_px"].items()
            })
            writer.writerow(row)


def write_html(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Benchmark Results</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#121d2e;--panel2:#18263a;--line:#2a3a50;--text:#e7eef8;--muted:#9aabc1;--accent:#64d6ad;--blue:#85b8ff;--gold:#ffd166;--red:#ff8f8f}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 12% 0%,#18304a 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1600px;margin:auto;padding:38px 28px 70px}h1{font-size:clamp(1.8rem,4vw,2.8rem);margin:0;letter-spacing:-.04em}.subtitle{color:var(--muted);margin:8px 0 26px}.cards{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:14px;margin-bottom:24px}.card,.toolbar,.table-wrap{background:linear-gradient(145deg,#142238,var(--panel));border:1px solid var(--line);border-radius:15px;box-shadow:0 12px 35px #0002}.card{padding:17px 19px}.label{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.value{font-size:1.35rem;font-weight:700;margin-top:5px;overflow-wrap:anywhere}.toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap;padding:16px;margin-bottom:15px}.control{display:grid;gap:5px}.control label{font-size:.78rem;color:var(--muted)}input,select{color:var(--text);background:#0d1727;border:1px solid var(--line);border-radius:8px;padding:9px 11px;font:inherit;min-width:145px}input[type=search]{min-width:min(360px,80vw)}.count{margin-left:auto;color:var(--muted);padding:10px 0}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;min-width:1050px}th,td{text-align:left;padding:12px 13px;border-bottom:1px solid var(--line);vertical-align:top}th{position:sticky;top:0;background:#17253a;color:#bacbe0;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;z-index:1}th button{color:inherit;background:none;border:0;font:inherit;text-transform:inherit;letter-spacing:inherit;padding:0;cursor:pointer}tbody tr:hover{background:#ffffff08}.config{font-weight:650;white-space:nowrap}.pill{display:inline-block;padding:3px 8px;border-radius:99px;background:#21334b;color:#d8e7fa;font-size:.82rem}.accuracy{font-weight:750;color:var(--accent)}.small-size{font-weight:650;color:var(--gold)}.muted{color:var(--muted)}details summary{cursor:pointer;color:var(--blue);font-weight:600}.detail{min-width:440px;max-width:760px;padding:10px 0}.detail h3{margin:14px 0 7px;font-size:.95rem}.bars{display:grid;grid-template-columns:repeat(3,minmax(130px,1fr));gap:8px 16px}.baritem{font-size:.8rem}.barlabel{display:flex;justify-content:space-between;gap:8px}.track{height:6px;border-radius:8px;background:#26364a;margin-top:4px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#48c9a0,#a4edca)}.font-grid{display:grid;grid-template-columns:repeat(2,minmax(160px,1fr));gap:6px 14px}.font-item{color:#d4deeb;font-size:.83rem}.transcript{white-space:pre-wrap;overflow:auto;max-height:420px;padding:12px;background:#0b1422;border:1px solid var(--line);border-radius:8px;font:12px/1.55 ui-monospace,SFMono-Regular,monospace}.empty{padding:35px;text-align:center;color:var(--muted)}.foot{color:var(--muted);font-size:.83rem;margin-top:15px}@media(max-width:800px){main{padding:24px 14px 50px}.cards{grid-template-columns:repeat(2,minmax(130px,1fr))}.count{margin-left:0;width:100%}.bars{grid-template-columns:repeat(2,minmax(120px,1fr))}}
.best-breakdown{display:grid;grid-template-columns:minmax(280px,1fr) minmax(340px,1.4fr);gap:14px;margin:-8px 0 24px}.breakdown-panel{padding:19px;background:linear-gradient(145deg,#142238,var(--panel));border:1px solid var(--line);border-radius:15px}.breakdown-panel h2{margin:0 0 14px;font-size:1rem}.mode-pills{display:flex;gap:10px;flex-wrap:wrap}.mode-pill{background:#0d1727;border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:130px}.mode-pill span{display:block;color:var(--muted);font-size:.78rem}.mode-pill b{font-size:1.25rem;color:var(--accent)}.size-tiles{display:grid;grid-template-columns:repeat(5,minmax(70px,1fr));gap:8px}.size-tile{padding:9px 10px;border-radius:9px;background:#0d1727;border:1px solid var(--line)}.size-tile.perfect{border-color:#39866f;background:#123126}.size-tile span{display:block;color:var(--muted);font-size:.75rem}.size-tile b{color:var(--text)}.font-mode-summary{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:8px 14px}.font-mode-summary .font-item{padding:8px 10px;background:#0d1727;border-radius:8px}.font-mode-summary strong{color:var(--gold)}@media(max-width:800px){.best-breakdown{grid-template-columns:1fr}}
.candidate-section{margin:0 0 25px}.candidate-section h2{font-size:1.2rem;margin:0 0 5px}.candidate-section>p{color:var(--muted);margin:0 0 12px}.candidate-tools{display:flex;gap:10px;align-items:end;flex-wrap:wrap;padding:13px 15px;margin-bottom:10px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.candidate-tools label{display:grid;gap:4px;color:var(--muted);font-size:.78rem}.candidate-tools input,.candidate-tools select{min-width:190px}.candidate-tools .count{padding:8px 0}.candidate-table{max-height:570px}.candidate-table table{min-width:850px}.candidate-table th{top:0}.candidate-table td{padding:9px 12px}.candidate-table tr.top-choice{background:#153126}
.hardware{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:-8px 0 24px}.hardware-item{padding:13px 17px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.hardware-item span{display:block;color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.07em}.hardware-item strong{display:block;margin-top:3px;overflow-wrap:anywhere}@media(max-width:600px){.hardware{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<h1>OCR Benchmark Results</h1>
<p class="subtitle" id="meta"></p>
<section class="hardware" id="hardware"></section>
<section class="cards" id="summary"></section>
<section class="best-breakdown" id="best-breakdown"></section>
<section class="candidate-section"><h2>All configuration results</h2><p>Compare phrase detection, smallest fully detected text size, and timing for every benchmark configuration.</p></section>
<section class="toolbar" aria-label="Filter benchmark results">
<div class="control"><label for="search">Search configuration</label><input id="search" type="search" placeholder="e.g. grayscale, PSM 6"></div>
<div class="control"><label for="method">Preprocessing</label><select id="method"><option value="">All methods</option></select></div>
<div class="control"><label for="scale">Scale</label><select id="scale"><option value="">All scales</option></select></div>
<div class="control"><label for="minAccuracy">Minimum phrase accuracy</label><select id="minAccuracy"><option value="0">Any</option><option value="0.5">50%+</option><option value="0.7">70%+</option><option value="0.8">80%+</option><option value="0.9">90%+</option></select></div>
<div class="count" id="count"></div>
</section>
<section class="table-wrap candidate-table"><table><thead><tr id="headers"></tr></thead><tbody id="rows"></tbody></table></section>
<p class="foot">Text-size 100% means every phrase at that size was found. OCR text is available by expanding a row.</p>
<section class="candidate-section"><h2>Configuration accuracy by font and mode</h2><p>This breakdown is mainly for comparing which fonts and light/dark modes were easier for OCR to read. Font designs vary widely, so treat these differences as interesting observations about this particular test image—not a definitive ranking of font readability. It’s here for anyone curious about which fonts gave the OCR more trouble. Accuracy is shown for each configuration; the smallest-100% column separately shows the smallest tested size where every phrase was detected, if any.</p><div class="candidate-tools"><label>Font / mode<select id="candidateGroup"><option value="">All font/modes</option></select></label><label>Minimum accuracy<select id="candidateAccuracy"><option value="0">Any accuracy</option><option value="0.5">50%+</option><option value="0.7">70%+</option><option value="0.8">80%+</option><option value="0.9">90%+</option><option value="1">100%</option></select></label><label>Find configuration<input id="candidateSearch" type="search" placeholder="method, scale, PSM"></label><div class="count" id="candidateCount"></div></div><div class="table-wrap candidate-table"><table><thead><tr id="candidateHeaders"></tr></thead><tbody id="candidateRows"></tbody></table></div></section>
</main>
<script>
const report = __PAYLOAD__;
const results = report.results || [];
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const pct = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const ms = value => `${(Number(value || 0) / 1000).toFixed(2)} s`;
const best = results[0] || {};
document.getElementById("meta").textContent = `${report.image || "Image"} · ${(report.image_size || []).join(" × ")} · Tesseract ${report.tesseract_version || "unknown"} · ${report.repeats || 0} measured repeat(s)`;
document.getElementById("hardware").innerHTML = `<div class="hardware-item"><span>Operating system</span><strong>${esc(report.host_hardware?.os || "Unknown")}</strong></div><div class="hardware-item"><span>Benchmark CPU</span><strong>${esc(report.host_hardware?.cpu || "Not detected")}</strong></div><div class="hardware-item"><span>Benchmark GPU</span><strong>${esc(report.host_hardware?.gpu || "Not detected")}</strong></div>`;
const smallest = best.smallest_100pct_size_px == null ? "Not reached" : `${best.smallest_100pct_size_px}px`;
const summary = [
 ["Top configuration", best.config || "No results"],
 ["Phrase detection", pct(best.phrase_accuracy)],
 ["Smallest 100% size", smallest],
 ["Median end-to-end", ms(best.end_to_end_ms)]
];
document.getElementById("summary").innerHTML = summary.map(([label,value]) => `<article class="card"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></article>`).join("");
const modePills = Object.entries(best.mode_phrase_accuracy || {}).map(([mode,accuracy]) => `<div class="mode-pill"><span>${esc(mode)} mode detection</span><b>${pct(accuracy)}</b></div>`).join("");
const sizeTiles = Object.entries(best.size_phrase_accuracy || {}).sort((a,b)=>parseInt(a[0])-parseInt(b[0])).map(([size,accuracy]) => `<div class="size-tile${accuracy===1?" perfect":""}"><span>${esc(size)}</span><b>${pct(accuracy)}</b></div>`).join("");
const fontModeSizes = Object.entries(best.font_mode_smallest_100pct_size_px || {}).map(([name,size]) => `<div class="font-item">${esc(name.replaceAll("_"," "))}<br><strong>${size==null?"No 100% size":`${esc(size)}px`}</strong></div>`).join("");
document.getElementById("best-breakdown").innerHTML = `<article class="breakdown-panel"><h2>Light / dark mode detection</h2><div class="mode-pills">${modePills}</div><h2 style="margin-top:20px">Smallest 100% size by font and mode</h2><div class="font-mode-summary">${fontModeSizes}</div></article><article class="breakdown-panel"><h2>Detection by text size</h2><div class="size-tiles">${sizeTiles}</div></article>`;
const candidates = results.flatMap(r=>Object.entries(r.font_mode_phrase_accuracy||{}).map(([group,accuracy])=>({group,size:r.font_mode_smallest_100pct_size_px?.[group]??null,config:r.config,method:r.method,scale:r.scale,psm:r.psm,ocr_ms:r.ocr_ms,end_to_end_ms:r.end_to_end_ms,phrase_accuracy:accuracy}))).sort((a,b)=>b.phrase_accuracy-a.phrase_accuracy||a.end_to_end_ms-b.end_to_end_ms);
const candidateGroup=document.getElementById("candidateGroup"), candidateSearch=document.getElementById("candidateSearch"), candidateAccuracy=document.getElementById("candidateAccuracy");
[...new Set(candidates.map(c=>c.group))].sort().forEach(g=>candidateGroup.insertAdjacentHTML("beforeend",`<option value="${esc(g)}">${esc(g.replaceAll("_"," "))}</option>`));
const candidateColumns=[["group","Font / mode"],["phrase_accuracy","Accuracy"],["size","Smallest 100% size"],["config","Configuration"],["ocr_ms","Median OCR"],["end_to_end_ms","End-to-end"]];
document.getElementById("candidateHeaders").innerHTML=candidateColumns.map(([key,label])=>`<th><button data-candidate-sort="${key}">${esc(label)} ↕</button></th>`).join("");
 let candidateSortKey="phrase_accuracy",candidateSortDir=-1;
function renderCandidates(){
 const group=candidateGroup.value,query=candidateSearch.value.trim().toLowerCase(),minimum=Number(candidateAccuracy.value);
 const shown=candidates.filter(c=>c.phrase_accuracy>=minimum&&(!group||c.group===group)&&(!query||`${c.config} ${c.method} ${c.scale} ${c.psm} ${c.group}`.toLowerCase().includes(query))).sort((a,b)=>{const av=a[candidateSortKey],bv=b[candidateSortKey];if(av==null)return 1;if(bv==null)return -1;return (typeof av==="string"?av.localeCompare(bv):av-bv)*candidateSortDir;});
 document.getElementById("candidateCount").textContent=`${shown.length} configuration/font-mode pairs`;
 document.getElementById("candidateRows").innerHTML=shown.length?shown.map((c,i)=>`<tr${i===0&&!group&&!query&&minimum===0?" class=\"top-choice\"":""}><td>${esc(c.group.replaceAll("_"," "))}</td><td class="accuracy">${pct(c.phrase_accuracy)}</td><td class="small-size">${c.size==null?"—":`${c.size}px`}</td><td class="config">${esc(c.config)}</td><td>${ms(c.ocr_ms)}</td><td>${ms(c.end_to_end_ms)}</td></tr>`).join(""):`<tr><td class="empty" colspan="7">No configurations match these filters.</td></tr>`;
}
document.querySelectorAll("#candidateHeaders button").forEach(button=>button.addEventListener("click",()=>{const key=button.dataset.candidateSort;if(candidateSortKey===key)candidateSortDir*=-1;else{candidateSortKey=key;candidateSortDir=(key==="config"||key==="group")?1:-1;}renderCandidates();}));
candidateGroup.addEventListener("change",renderCandidates);candidateAccuracy.addEventListener("change",renderCandidates);candidateSearch.addEventListener("input",renderCandidates);renderCandidates();
const methodSelect = document.getElementById("method"), scaleSelect = document.getElementById("scale");
[...new Set(results.map(r=>r.method))].sort().forEach(v=>methodSelect.insertAdjacentHTML("beforeend",`<option>${esc(v)}</option>`));
[...new Set(results.map(r=>r.scale))].sort((a,b)=>a-b).forEach(v=>scaleSelect.insertAdjacentHTML("beforeend",`<option value="${v}">${v}×</option>`));
const columns = [["rank","Rank"],["config","Configuration"],["method","Method"],["scale","Scale"],["psm","PSM"],["phrase_accuracy","Phrase accuracy"],["smallest_100pct_size_px","Smallest 100%"],["ocr_ms","OCR time"],["end_to_end_ms","End-to-end"],["fully_exact_runs","Exact runs"],["details","Details"]];
document.getElementById("headers").innerHTML = columns.map(([key,label])=>`<th>${key==="details"?esc(label):`<button data-key="${key}">${esc(label)} ↕</button>`}</th>`).join("");
let sortKey="phrase_accuracy", sortDir=-1;
function detailHtml(r){
 const sizes=Object.entries(r.size_phrase_accuracy||{}).sort((a,b)=>parseInt(a[0])-parseInt(b[0]));
 const sizeBars=sizes.map(([name,score])=>`<div class="baritem"><div class="barlabel"><span>${esc(name)}</span><b>${pct(score)}</b></div><div class="track"><div class="fill" style="width:${Math.max(0,Math.min(100,score*100))}%"></div></div></div>`).join("");
 const modes=Object.entries(r.font_mode_phrase_accuracy||{}).map(([name,score])=>`<div class="font-item">${esc(name.replaceAll("_"," "))}: <b>${pct(score)}</b> · 100% from ${r.font_mode_smallest_100pct_size_px?.[name] == null ? "not reached" : `${esc(r.font_mode_smallest_100pct_size_px[name])}px`}</div>`).join("");
 const phrases=Object.entries(r.found_phrases||{}).map(([name,count])=>`<div class="font-item">${esc(name.replaceAll("_"," "))}: <b>${count}/${r.runs}</b></div>`).join("");
 return `<div class="detail"><h3>Detection by text size</h3><div class="bars">${sizeBars}</div><h3>Detection by font and mode</h3><div class="font-grid">${modes}</div><h3>Phrase matches</h3><div class="font-grid">${phrases}</div><h3>Recognized text</h3><pre class="transcript">${esc(r.recognized)}</pre></div>`;
}
function render(){
 const query=document.getElementById("search").value.trim().toLowerCase(), method=methodSelect.value, scale=scaleSelect.value, minimum=Number(document.getElementById("minAccuracy").value);
 let shown=results.map((r,index)=>({...r,_rank:index+1})).filter(r=>(!method||r.method===method)&&(!scale||String(r.scale)===scale)&&r.phrase_accuracy>=minimum&&(!query||`${r.config} ${r.method} ${r.scale} ${r.psm}`.toLowerCase().includes(query)));
 shown.sort((a,b)=>{const av=a[sortKey],bv=b[sortKey]; if(av==null)return 1;if(bv==null)return -1;return (typeof av==="string"?av.localeCompare(bv):av-bv)*sortDir;});
 document.getElementById("count").textContent=`Showing ${shown.length} of ${results.length} configurations`;
  document.getElementById("rows").innerHTML=shown.length?shown.map((r,i)=>`<tr><td>${i+1}</td><td class="config">${esc(r.config)}</td><td><span class="pill">${esc(r.method)}</span></td><td>${r.scale}×</td><td>${r.psm}</td><td class="accuracy">${pct(r.phrase_accuracy)}</td><td class="small-size">${r.smallest_100pct_size_px==null?"—":`${r.smallest_100pct_size_px}px`}</td><td>${ms(r.ocr_ms)}</td><td>${ms(r.end_to_end_ms)}</td><td>${r.fully_exact_runs}/${r.runs}</td><td><details><summary>Explore</summary>${detailHtml(r)}</details></td></tr>`).join(""):`<tr><td class="empty" colspan="11">No configurations match these filters.</td></tr>`;
}
document.querySelectorAll("#headers button").forEach(button=>button.addEventListener("click",()=>{const key=button.dataset.key;if(sortKey===key)sortDir*=-1;else{sortKey=key;sortDir=(key==="config"||key==="method")?1:-1;}render();}));
["search","method","scale","minAccuracy"].forEach(id=>document.getElementById(id).addEventListener(id==="search"?"input":"change",render));
render();
</script></body></html>'''.replace("__PAYLOAD__", payload)
    path.write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.limit_configs is not None and args.limit_configs < 1:
        raise SystemExit("--limit-configs must be at least 1")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}\nPass its path using --image.")
    if not args.truth.is_file():
        raise SystemExit(f"Ground-truth file not found: {args.truth}")

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    expected = truth["phrases"]
    if not expected or not all(isinstance(name, str) and isinstance(text, str) and text for name, text in expected.items()):
        raise SystemExit("Ground truth must define a non-empty 'phrases' object of phrase names and text.")
    image = Image.open(args.image).convert("RGB")

    configs = make_configs(args.psm)
    if args.limit_configs is not None:
        configs = configs[:args.limit_configs]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tesseract_version = str(pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError as error:
        raise SystemExit(f"Tesseract executable not found. Install Tesseract or add it to PATH.\n{error}") from error
    host_hardware = get_host_hardware()

    # Pay the one-time Python/Tesseract startup cost before taking timings.
    pytesseract.image_to_string(image, lang="eng", config="--oem 3 --psm 3")

    print(f"Image: {args.image} ({image.width}x{image.height})")
    print(f"Tesseract: {tesseract_version}")
    print(
        f"Testing {len(configs)} full-image configurations, {args.repeats} measured repeat(s); "
        "phrase positions and ground-truth coordinates are not used."
    )
    results = []
    for index, config in enumerate(configs, start=1):
        result = measure_config(image, expected, config, args.repeats)
        results.append(result)
        print(
            f"[{index:>3}/{len(configs)}] {result['config']:<32} "
            f"exact runs {result['fully_exact_runs']}/{result['runs']}  "
            f"phrases {sum(result['found_phrases'].values())}/{len(expected) * result['runs']}  "
            f"OCR {result['ocr_ms']:.1f} ms",
            flush=True,
        )

    results.sort(
        key=lambda item: (
            not item["all_runs_exact"],
            -item["fully_exact_runs"],
            -item["phrase_accuracy"],
            item["end_to_end_ms"],
        )
    )
    csv_path = args.output_dir / "results.csv"
    json_path = args.output_dir / "results.json"
    html_path = args.output_dir / "results.html"
    write_csv(csv_path, results, list(expected))
    report = {
        "image": str(args.image),
        "image_size": list(image.size),
        "tesseract_version": tesseract_version,
        "host_hardware": host_hardware,
        "truth_file": str(args.truth),
        "repeats": args.repeats,
        "host_hardware": host_hardware,
        "ground_truth_phrases": expected,
        "scoring": "Each expected phrase is found when its contiguous case- and punctuation-insensitive word sequence occurs anywhere in the full-image OCR output. Results include aggregate detection accuracy for light versus dark mode, font/mode pairs, and each text size. The smallest 100%-readable size is the smallest tested size at which every phrase at that size is found on every repeat; it is also reported per font/mode pair.",
        "ranking": "all repeats containing every phrase first, then most fully exact repeats, phrase accuracy, and end-to-end milliseconds",
        "results": results,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(html_path, report)

    exact = [result for result in results if result["all_runs_exact"]]
    best = min(exact, key=lambda item: item["end_to_end_ms"]) if exact else results[0]
    found_summary = ", ".join(
        f"{name} {count}/{best['runs']}"
        for name, count in best["found_phrases"].items()
    )
    mode_summary = ", ".join(
        f"{mode} {accuracy:.1%}"
        for mode, accuracy in best["mode_phrase_accuracy"].items()
    )
    font_mode_summary = "\n".join(
        f"  {group}: {accuracy:.1%}"
        for group, accuracy in best["font_mode_phrase_accuracy"].items()
    )
    smallest_size_summary = (
        f"{best['smallest_100pct_size_px']}px"
        if best["smallest_100pct_size_px"] is not None
        else "none of the tested sizes"
    )
    font_mode_size_summary = ", ".join(
        f"{group} {size}px" if size is not None else f"{group} none"
        for group, size in best["font_mode_smallest_100pct_size_px"].items()
    )
    (args.output_dir / "best.txt").write_text(
        f"Configuration: {best['config']}\n"
        f"Benchmark host OS: {host_hardware['os']}\n"
        f"Benchmark host CPU: {host_hardware['cpu']}\n"
        f"Benchmark host GPU: {host_hardware['gpu']}\n"
        f"Fully exact runs: {best['fully_exact_runs']}/{best['runs']}\n"
        f"Phrase accuracy across runs: {best['phrase_accuracy']:.2%}\n"
        f"Median OCR: {best['ocr_ms']:.2f} ms\n"
        f"Median end-to-end: {best['end_to_end_ms']:.2f} ms\n"
        f"Smallest tested size with 100% phrase detection: {smallest_size_summary}\n"
        f"Smallest 100% size by font/mode: {font_mode_size_summary}\n"
        f"Detection by size: {best['size_phrase_accuracy']}\n"
        f"Mode detection: {mode_summary}\n"
        f"Per-font/mode detection:\n{font_mode_summary}\n"
        f"Phrase detections across runs: {found_summary}\n\n"
        f"Full-image OCR output (representative run):\n{best['recognized']}\n",
        encoding="utf-8",
    )

    print(
        "\nFastest full-image configuration that found every phrase on every repeat:"
        if exact
        else "\nNo full-image configuration found every phrase on every repeat; best phrase-accuracy result:"
    )
    print(
        f"  {best['config']} - fully exact {best['fully_exact_runs']}/{best['runs']} runs, "
        f"all-phrase accuracy {best['phrase_accuracy']:.1%}; "
        f"light {best['mode_phrase_accuracy'].get('light', 0):.1%}, "
        f"dark {best['mode_phrase_accuracy'].get('dark', 0):.1%}"
    )
    print(f"  Smallest tested size with 100% phrase detection: {smallest_size_summary}")
    print(f"  {best['ocr_ms']:.1f} ms OCR; {best['end_to_end_ms']:.1f} ms including preprocessing (median)")
    print(f"  Results: {csv_path}\n           {json_path}\n           {html_path}\n           {args.output_dir / 'best.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
