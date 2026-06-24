# Tour de France 2026 Stage Winner Predictions

Machine learning system for predicting Tour de France stage winners using XGBoost Ranker trained on historical race data (2020-2025).

## Working pipeline in Google Colab
[To the Colab](https://colab.research.google.com/drive/1WzqYTJl0ChTXazbUqSu2PaEQsKIjIaIv?usp=sharing)
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

## Results

### Summary Statistics

| Metric | Value |
|--------|-------|
| 2026 Riders | 110 |
| Historical Data | 19,680 rows × 13 features |
| Training Winners | 120 stage wins (2020-2025) |
| Unmatched Riders | 33 (not in historical data) |

### Top Stage Win Probabilities

| Stage | Type | Distance | Top Contender | Prob | Runner-up | Prob |
|-------|------|----------|---------------|------|-----------|------|
| 1 | TTT | 19.6 km | Remco Evenepoel | 12.9% | Jasper Philipsen | 9.0% |
| 2 | Hilly | 168.5 km | Jasper Philipsen | 5.6% | Remco Evenepoel | 5.2% |
| 3 | Mountain | 195.9 km | **Tadej Pogačar** | **15.7%** | Jonas Vingegaard | 5.4% |
| 4 | Hilly | 181.9 km | Tadej Pogačar | 14.4% | Remco Evenepoel | 8.9% |
| 5 | Flat | 158.3 km | Jasper Philipsen | 20.0% | Tim Merlier | 18.2% |
| 6 | Mountain | 186.2 km | **Tadej Pogačar** | **36.8%** | Jonas Vingegaard | 8.5% |
| 7 | Flat | 175.1 km | Jasper Philipsen | 31.5% | Tim Merlier | 27.0% |
| 8 | Flat | 180.4 km | Jasper Philipsen | 29.3% | Tim Merlier | 25.1% |
| 9 | Hilly | 185.5 km | Mads Pedersen | 5.1% | Biniam Girmay | 5.1% |
| 10 | Mountain | 166.6 km | **Tadej Pogačar** | **42.4%** | Jonas Vingegaard | 7.9% |
| 11 | Flat | 161.3 km | Jasper Philipsen | 25.9% | Tim Merlier | 15.4% |
| 12 | Flat | 179.1 km | Jasper Philipsen | 26.3% | Tim Merlier | 14.7% |
| 13 | Hilly | 205.8 km | Jasper Philipsen | 6.8% | Mads Pedersen | 6.0% |
| 14 | Mountain | 155.3 km | **Tadej Pogačar** | **47.0%** | Jonas Vingegaard | 7.0% |
| 15 | Mountain | 183.9 km | **Tadej Pogačar** | **36.7%** | Jonas Vingegaard | 7.8% |
| 16 | ITT | 26.1 km | Tadej Pogačar | 23.6% | Remco Evenepoel | 16.4% |
| 17 | Flat | 174.7 km | Jasper Philipsen | 9.3% | Tim Merlier | 5.6% |
| 18 | Mountain | 185.2 km | **Tadej Pogačar** | **38.7%** | Jonas Vingegaard | 7.0% |
| 19 | Mountain | 127.9 km | **Tadej Pogačar** | **55.2%** | Thymen Arensman | 7.5% |
| 20 | Mountain | 170.9 km | **Tadej Pogačar** | **40.4%** | Thymen Arensman | 10.3% |
| 21 | Flat | 133.0 km | Tim Merlier | 8.5% | Jasper Philipsen | 8.4% |

### Key Insights

- **Tadej Pogačar** dominates mountain stages (Stages 3, 6, 10, 14, 15, 18, 19, 20) with probabilities 15–55%
- **Jasper Philipsen** and **Tim Merlier** are the clear sprint favorites on flat stages
- **Remco Evenepoel** is strongest on the TTT (Stage 1) and ITT (Stage 16)
- Mountain stages show higher concentration of probability on top riders vs flat/hilly stages

### Full Predictions

See `predictions_2026.csv` for complete top-5 predictions per stage with probabilities, rider types, PCS rankings, and historical wins.

## License

MIT
