# Hotel Bookings: Cancellation Analysis

Python analysis of hotel booking cancellations for two hotels: a resort hotel (`H1`) and a city hotel (`H2`). It estimates what cancellations cost, finds which booking characteristics are most associated with cancelling, and fits a logistic regression to predict cancellations.

> **Sample data only.** This repository ships a random sample (1,000 bookings per hotel) in [data/sample](data/sample). The full dataset is not included, so numbers from the sample will differ from the full-data results.

## Project structure

```
hotel_bookings/
├── src/hotel_bookings/
│   ├── data.py          # load, combine, clean, create sample
│   └── analysis.py      # cost estimate, association analysis, logistic model
├── scripts/
│   ├── run_analysis.py  # runs the full analysis
│   └── make_sample.py   # rebuilds data/sample from the full data
├── tests/               # pytest checks
├── data/
│   ├── sample/          # H1_sample.csv, H2_sample.csv (committed)
│   └── raw/             # put full H1.csv / H2.csv here (git-ignored)
└── 000_environment_setup/   # VSCode / Anaconda install notes and conda env file
```

## Setup

Install Miniconda or Anaconda and VSCode (see [000_environment_setup/00_install_vscode_anaconda.md](000_environment_setup/00_install_vscode_anaconda.md) for the recommended extensions), then create the environment:

```bash
conda env create -f 000_environment_setup/01_conda_environment.yml
conda activate hotel_bookings
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_analysis.py
python -m pytest
```

The script uses the full data in `data/raw/` when `H1.csv` and `H2.csv` are there, and the bundled sample otherwise.

## Data

Each row is one booking: hotel type, lead time, arrival date, stay length, guests, meal plan, country, market segment, deposit type, customer type, ADR (average daily rate), special requests, and whether the booking was cancelled (`is_canceled`). Column names are converted to snake_case on load (for example `ADR` becomes `a_d_r`).

The full data is the public *Hotel booking demand* dataset (Antonio, Almeida & Nunes, 2019). To use it, download `H1.csv` and `H2.csv` and place them in `data/raw/`. It contains no names, contact details or other personal identifiers.

## Analysis

1. **Cost of cancellations**: cancelled bookings × $200 per night × 3.4 nights (assumptions from the original analysis, set in `analysis.py`). The original full-data figure used 44,224 cancellations, about $30.1M, with a 10% reduction worth about $3.0M.
2. **Cleaning**: trim padded text, convert `'NULL'` to missing, combine weekend and week nights into `total_stay`, combine adults, children and babies into `total_guests`, flag `different_room_assigned`, drop `agent`, `company` and `reservation_status_date`, and fill missing `country` with `PRT`, the most common value.
3. **Association**: numeric features are binned into quartiles and all features one-hot encoded, then ranked by correlation with `is_canceled`. Deposit type, lead time and country are the strongest signals on the sample.
4. **Model**: a binomial GLM (logistic regression) with in-sample AUC. `reservation_status` is excluded because it reveals the outcome.

## Notes

- The model's AUC is measured on the data it was fitted on, so it is optimistic. A train/test split is the natural next step.
- Results on the sample are indicative only.
