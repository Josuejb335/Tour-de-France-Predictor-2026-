# Tour de France 2026 Stage Winner Predictions

Machine learning system for predicting Tour de France stage winners using XGBoost Ranker trained on historical race data (2020-2025).

## Overview

This project scrapes historical Tour de France stage results from ProCyclingStats, enriches them with rider statistics, trains a learning-to-rank model (XGBRanker), and generates stage-by-stage win probability predictions for the 2026 edition.

## Project Structure

```
.
├── predict_winners.py          # Main prediction script
├── scraper_complete.py         # Scrape historical stage results (2020-2025)
├── scrape_riders.py            # Scrape rider stats (PCS ranking, wins, points)
├── requirements.txt            # Python dependencies
├── 2026_riders.csv             # 2026 Tour de France start list
├── tdf_stage_results_2020_2025.csv  # Combined historical stage data
├── rider_data.json             # Cached rider statistics
├── predictions_2026.csv        # Generated predictions (output)
└── cache/                      # Per-year cached stage results
```

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

Additional dependencies for ML:
```bash
pip install xgboost scikit-learn numpy pandas
```

## Usage

### 1. Scrape Historical Data (one-time)

```bash
python scraper_complete.py
```
Fetches stage results for 2020-2025 from ProCyclingStats. Outputs `tdf_stage_results_2020_2025.csv`.

### 2. Scrape Rider Statistics (one-time)

```bash
python scrape_riders.py
```
Enriches historical data with PCS rankings, career wins, and PCS points. Updates `rider_data.json` and the main CSV.

### 3. Generate 2026 Predictions

```bash
python predict_winners.py
```
Trains XGBRanker on historical data and outputs top-5 predictions per stage to `predictions_2026.csv`.

## Data Sources

- **ProCyclingStats** — Stage results, rider profiles, rankings
- **2026_riders.csv** — Manual start list for 2026 Tour de France

## Model Features

- Rider encoding (name, type, PCS ranking, historical wins, PCS points)
- Stage encoding (type, distance, elevation, climbing intensity)
- Rider-stage interaction (type matching, breakaway index)
- Temporal features (previous year wins, 3-year rolling average)

## Output

`predictions_2026.csv` contains:
- Stage number and type
- Top 5 predicted riders per stage
- Win probability, rider type, PCS ranking, historical wins

## Requirements

- Python 3.10+
- Playwright (Chromium)
- XGBoost, scikit-learn, pandas, numpy
- beautifulsoup4, lxml, tqdm

## License

MIT