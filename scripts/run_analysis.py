"""Run the full analysis. Uses data/raw if present, otherwise data/sample."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotel_bookings import analysis, data  # noqa: E402

data_dir = data.find_data_dir()
print(f"Data source: {data_dir}")

raw = data.load_raw(data_dir)
df = data.clean(raw)
print(f"Raw rows: {len(raw):,} | clean rows: {len(df):,}")

# COST ANALYSIS
counts = analysis.cancellation_counts(raw)
print("\nCancellations:\n", counts.to_string(index=False))
n_cancelled = int((raw["is_canceled"] == 1).sum())
cost = analysis.cancellation_cost(n_cancelled)
print(f"\nEstimated cost of {n_cancelled:,} cancellations: ${cost:,.0f}")
print(f"10% reduction in cancellations would save: ${0.1 * cost:,.0f}")

# ASSOCIATION
corr = analysis.correlate_with_target(df.drop(columns=["reservation_status"]))
print("\nTop 10 features associated with cancellation:\n", corr.head(10).round(3).to_string())

# MODEL
model, auc = analysis.fit_logit(df)
print(f"\nLogistic regression in-sample AUC: {auc:.3f}")
