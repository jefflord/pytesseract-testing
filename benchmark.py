#!/usr/bin/env python3
"""Benchmark Tesseract on a full 4K screenshot without known text locations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import pytesseract


ROOT = Path(__file__).resolve().parent
DEFAULT_TRUTH = ROOT / "ground_truth.json"
DEFAULT_IMAGE = ROOT / "test_files" / "FontQuadrantsDemo.jpg"
METHODS = ("color", "grayscale", "autocontrast", "otsu")
SCALES = (1, 2)
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
    for name, count in found_per_phrase.items():
        match = re.fullmatch(r"(.+)_(light|dark)_(\d+)px", name)
        if match:
            font, mode, _size = match.groups()
            group = f"{font}_{mode}"
            mode_counts[mode] += count
            mode_totals[mode] += len(runs)
            font_mode_counts[group] = font_mode_counts.get(group, 0) + count
            font_mode_totals[group] = font_mode_totals.get(group, 0) + len(runs)
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
    ]
    fields.extend(f"found_{name}" for name in phrase_names)
    fields.extend(("light_phrase_accuracy", "dark_phrase_accuracy"))
    groups = sorted({
        "_".join(name.split("_")[:-2]) + "_" + name.split("_")[-2]
        for name in phrase_names
    })
    fields.extend(f"accuracy_{group}" for group in groups)
    fields.append("ocr_text")
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
                f"accuracy_{group}": accuracy
                for group, accuracy in result["font_mode_phrase_accuracy"].items()
            })
            row["ocr_text"] = result["recognized"]
            writer.writerow(row)


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
    write_csv(csv_path, results, list(expected))
    report = {
        "image": str(args.image),
        "image_size": list(image.size),
        "tesseract_version": tesseract_version,
        "truth_file": str(args.truth),
        "repeats": args.repeats,
        "ground_truth_phrases": expected,
        "scoring": "Each expected phrase is found when its contiguous case- and punctuation-insensitive word sequence occurs anywhere in the full-image OCR output. Results include aggregate detection accuracy for light versus dark mode and each font/mode pair.",
        "ranking": "all repeats containing every phrase first, then most fully exact repeats, phrase accuracy, and end-to-end milliseconds",
        "results": results,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

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
    (args.output_dir / "best.txt").write_text(
        f"Configuration: {best['config']}\n"
        f"Fully exact runs: {best['fully_exact_runs']}/{best['runs']}\n"
        f"Phrase accuracy across runs: {best['phrase_accuracy']:.2%}\n"
        f"Median OCR: {best['ocr_ms']:.2f} ms\n"
        f"Median end-to-end: {best['end_to_end_ms']:.2f} ms\n"
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
    print(f"  {best['ocr_ms']:.1f} ms OCR; {best['end_to_end_ms']:.1f} ms including preprocessing (median)")
    print(f"  Results: {csv_path}\n           {json_path}\n           {args.output_dir / 'best.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
