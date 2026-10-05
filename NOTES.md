# Pytesseract 4K screenshot benchmark — notes and findings

## Goal

Find a fast and reliable Tesseract setup for recognizing known phrases in a 4K screenshot, without assuming where in the image those phrases appear. The current image is `test_files/FontQuadrantsDemo.jpg` (3840×2160). The companion HTML has fixed text; `ground_truth.json` lists the expected phrases for its current screenshot.

## Location-independent benchmark

`benchmark.py` OCRs the full image for each configuration. It does not crop the image, use bounding boxes, or read phrase locations from ground truth. A phrase is counted as found when its contiguous normalized word sequence appears anywhere in Tesseract's output. Normalization ignores case and punctuation but preserves word boundaries and order. Extra text and unrelated OCR errors are not penalized by this detection metric.

The HTML has four font quadrants (Arimo, Courier Prime, Atkinson Hyperlegible, JetBrains Mono), each containing light and dark samples side by side. Every mode has nine fixed rows at 8, 10, 12, 14, 16, 18, 20, 22, and 24 pixels; each phrase has three words and a seven-digit number as its fourth token. Dark-mode text is different from its light-mode counterpart. `ground_truth.json` records all 72 font/mode/size phrases; keys are for reference only.

The default sweep compares four image preparations (color, grayscale, grayscale with autocontrast, Otsu), 1× and 2× scales, and PSM 3, 6, 11, and 12. Each configuration is repeated; one warmup is excluded from timing. Results report total detection as well as independent light/dark and per-font/mode rates. The latest updated-image run found 122 of 144 phrase instances (84.72%) with grayscale 2× PSM 6: light detection was 86.1%, dark detection was 83.3%, and no configuration found all 72 phrases in both repeats.

## Files

- `benchmark.py` — full-image configuration sweep, scoring, timing, and result generation.
- `ground_truth.json` — expected phrases only; intentionally contains no coordinates.
- `test_files/FontQuadrantsDemo.jpg` — default benchmark image.
- `results_light_dark/results.csv` — ranked metrics, per-phrase detections, light-vs-dark accuracy, and full OCR output.
- `results_light_dark/results.json` — detailed output and run metadata.
- `results_light_dark/best.txt` — fastest configuration that found every expected phrase on every repeat, or best phrase detection rate otherwise.

Latest updated-image run (`--repeats 2`): `full_image_grayscale_x2_psm6` ranked highest by phrase detection. Median OCR time was about 1825 ms; median end-to-end time was about 1961 ms. These timings apply to this full-image, 72-phrase light/dark test.

CSV output quotes every field and uses standard escaped double quotes. Python syntax compilation and full-image benchmark runs should be checked after changes.

## Run

```powershell
python -m pip install -r requirements.txt
python benchmark.py --repeats 5
```

Useful options include `--limit-configs 3`, `--psm 3 6 11 12`, `--image <path>`, and `--truth <path>`.
