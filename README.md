# Pytesseract full-image screenshot benchmark

Benchmarks Tesseract on the **entire** 4K screenshot at `test_files/FontQuadrantsDemo.jpg`. The benchmark does not crop around known text, pass bounding boxes to Tesseract, or use phrase coordinates. Its only ground truth is the set of expected phrases in `ground_truth.json`; a phrase is counted as found if it appears anywhere in Tesseract's full-image OCR output. The companion `test_files/FontQuadrantsDemo.html` contains fixed phrases (no randomization), text sizes from 8px through 24px, and four font quadrants. Each quadrant places light and dark samples side by side, with different phrases for the two modes; each phrase ends with a seven-digit number.

The benchmark compares original color, grayscale, grayscale with autocontrast, and Otsu thresholding; 1× and 2× image scales; and Tesseract page segmentation modes 3, 6, 11, and 12 by default. Adjust the PSM set with `--psm`.

## Run

Prerequisites: Python 3.10+, [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and available on `PATH`.

```powershell
python -m pip install -r requirements.txt
python benchmark.py --repeats 5
```

The default image path is `test_files/FontQuadrantsDemo.jpg`. To use a different image or phrase list:

```powershell
python benchmark.py --image "C:\path\to\screenshot.jpg" --truth .\ground_truth.json
```

Use `--limit-configs 3` for a quick smoke test, `--repeats 5` for more stable medians, or `--psm 3 6 11 12` to explicitly select page segmentation modes.

## Scoring and outputs

A phrase matches if its contiguous word sequence occurs in the OCR text from the full image; case and punctuation are ignored, while word boundaries and order are kept. This lets a phrase be detected anywhere without relying on its known location or accidentally matching the letters across unrelated words. Results show total detection plus separate light/dark rates and rates for each font/mode pair. The benchmark ranks configurations that find all phrases on every measured repeat first, then by how many repeats found all phrases, phrase detection rate, and median end-to-end time. This is measured detection against the listed expected phrases; it does not assess extra OCR text or prove that arbitrary text was transcribed perfectly.

The `results_light_dark` folder contains:

- `results.csv` — ranked measurements, per-phrase detection counts, and the full OCR output for the representative run. All fields are quoted using standard CSV escaping.
- `results.json` — detailed measurements, expected phrases, scoring definition, and run metadata.
- `best.txt` — fastest configuration that found all phrases on every repeat, or the highest detection-rate result when none did.

Each reported timing is a median over the measured repeats. A one-time Tesseract warmup is excluded. End-to-end time includes image preprocessing and OCR.

## Ground truth

`ground_truth.json` lists all 72 font/mode/size phrases, named by font, mode, and pixel size for reference. It contains no boxes or coordinates: phrase location is intentionally unknown to the scoring code. Keep its phrase entries synchronized with the fixed HTML text. Capture a new screenshot after changing the HTML.
