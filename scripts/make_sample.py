"""Regenerate data/sample from the full files in data/raw (not committed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotel_bookings import data  # noqa: E402

data.make_sample(n_per_hotel=1000, seed=42)
print("Wrote sample files to", data.SAMPLE_DIR)
