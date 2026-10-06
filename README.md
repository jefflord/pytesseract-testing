# Pytesseract full-image screenshot benchmark

Benchmarks Tesseract on the **entire** 4K screenshot at `test_files/FontQuadrantsDemo.jpg`. The benchmark does not crop around known text, pass bounding boxes to Tesseract, or use phrase coordinates. Its only ground truth is the set of expected phrases in `ground_truth.json`; a phrase is counted as found if it appears anywhere in Tesseract's full-image OCR output. The companion `test_files/FontQuadrantsDemo.html` contains fixed phrases (no randomization), text sizes from 8px through 24px, and four font quadrants. Each quadrant places light and dark samples side by side, with different phrases for the two modes; each phrase ends with a seven-digit number.

The benchmark compares original color, grayscale, grayscale with autocontrast, and Otsu thresholding; 1×, 2×, and 3× image scales; and Tesseract page segmentation modes 3, 6, 11, and 12 by default. Adjust the PSM set with `--psm`.

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

Results also report phrase-detection accuracy by text size and the smallest tested size with 100% detection. A size counts as 100% only when every expected phrase at that size is found on every measured repeat. The CSV and summary include this overall and separately for each font/mode pair; `null`/blank means no tested size reached 100%.

The `results_light_dark` folder contains:

- `results.csv` — ranked configuration metrics, phrase detections, accuracy by text size and font/mode, and smallest 100%-detected size. It omits the long recognized-text transcript to keep the table manageable.
- `results.json` — complete machine-readable report, including per-configuration OCR text, detection counts, accuracy summaries, expected phrases, scoring details, and run metadata.
- `results.html` — self-contained interactive browser report. Its summary highlights the best-ranked configuration and its light/dark accuracy, accuracy by text size, and smallest 100%-detected sizes by font/mode. The **All configuration results** table can be searched, filtered by preprocessing, scale, and minimum phrase accuracy, and sorted by its columns; expand a row to see size/font breakdowns, phrase matches, and OCR text. The **Configuration accuracy by font and mode** table compares font/mode detection across configurations and supports font/mode, minimum-accuracy, and configuration-search filters. That comparison is exploratory and reflects only the fonts and image in this benchmark.
- `best.txt` — summary of the top-ranked configuration, its detection and timing metrics, smallest fully detected sizes, and representative OCR text.

Each reported timing is a median over the measured repeats. A one-time Tesseract warmup is excluded. End-to-end time includes image preprocessing and OCR. The default sweep has 48 configurations (four preprocessing methods × three scales × four PSMs); `--limit-configs` can shorten it for a smoke test.

## Ground truth

`ground_truth.json` lists all 72 font/mode/size phrases, named by font, mode, and pixel size for reference. It contains no boxes or coordinates: phrase location is intentionally unknown to the scoring code. Keep its phrase entries synchronized with the fixed HTML text. Capture a new screenshot after changing the HTML.
