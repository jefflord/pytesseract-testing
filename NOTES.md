# Pytesseract 4K screenshot benchmark — notes and findings

## Goal

Find a fast and reliable Tesseract setup for recognizing four known phrases in a 4K screenshot, without assuming where in the image those phrases appear. The current image is `test_files/FontQuadrantsDemo.jpg` (3840×2160). The source HTML randomizes its words when rendered, so `ground_truth.json` lists the expected phrases for this particular screenshot.

## Location-independent benchmark

`benchmark.py` OCRs the full image for each configuration. It does not crop the image, use bounding boxes, or read phrase locations from ground truth. A phrase is counted as found when its contiguous normalized word sequence appears anywhere in Tesseract's output. Normalization ignores case and punctuation but preserves word boundaries and order. Extra text and unrelated OCR errors are not penalized by this detection metric.

The default sweep compares four image preparations (color, grayscale, grayscale with autocontrast, Otsu), 1× and 2× scales, and PSM 3, 6, 11, and 12. Each configuration is repeated; one warmup is excluded from timing. Results rank fully exact repeats first, then repeat-level phrase detection rate and end-to-end time.

## Files

- `benchmark.py` — full-image configuration sweep, scoring, timing, and result generation.
- `ground_truth.json` — expected phrases only; intentionally contains no coordinates.
- `test_files/FontQuadrantsDemo.jpg` — default benchmark image.
- `results_full_image/results.csv` — ranked metrics, phrase detections, and full OCR output.
- `results_full_image/results.json` — detailed output and run metadata.
- `results_full_image/best.txt` — fastest configuration that found every expected phrase on every repeat, or best phrase detection rate otherwise.

Latest full-image run (`--repeats 2`): `full_image_autocontrast_x2_psm6` found all four phrases in both repeats. Median OCR time was about 924 ms; median end-to-end time was about 1074 ms. These timings apply to the full-image task and are not comparable to the earlier crop-only benchmark.

CSV output quotes every field and uses standard escaped double quotes. Python syntax compilation and full-image benchmark runs should be checked after changes.

## Run

```powershell
python -m pip install -r requirements.txt
python benchmark.py --repeats 5
```

Useful options include `--limit-configs 3`, `--psm 3 6 11 12`, `--image <path>`, and `--truth <path>`.
