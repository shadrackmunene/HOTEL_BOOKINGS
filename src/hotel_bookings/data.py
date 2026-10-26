"""Loading, combining and cleaning the hotel booking data."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"

HOTEL_NAMES = {"H1": "Resort Hotel", "H2": "City Hotel"}


def to_snake_case(name: str) -> str:
    """IsCanceled -> is_canceled, ADR -> a_d_r (kept for compatibility with earlier outputs)."""
    return re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")


def find_data_dir() -> Path:
    """Use the full data in data/raw if present, otherwise the bundled sample."""
    if (RAW_DIR / "H1.csv").exists() and (RAW_DIR / "H2.csv").exists():
        return RAW_DIR
    return SAMPLE_DIR


def _read(path: Path, hotel: str) -> pd.DataFrame:
    suffix = "_sample" if path == SAMPLE_DIR else ""
    df = pd.read_csv(path / f"{hotel}{suffix}.csv")
    return df.assign(hotel=HOTEL_NAMES[hotel])


def load_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Combine H1 (resort) and H2 (city) bookings with snake_case columns."""
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    df = pd.concat([_read(data_dir, "H1"), _read(data_dir, "H2")], ignore_index=True)
    df.columns = [to_snake_case(c) for c in df.columns]
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer features, tidy strings and handle missing values."""
    df = raw.copy()

    # Source files pad strings with spaces and use the text 'NULL' for missing.
    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    df = df.replace("NULL", np.nan)

    df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["different_room_assigned"] = (
        df["reserved_room_type"] != df["assigned_room_type"]
    ).astype(int)

    df = df.drop(
        columns=[
            "stays_in_weekend_nights", "stays_in_week_nights",
            "adults", "children", "babies",
            "reserved_room_type", "assigned_room_type",
            "company", "agent",  # mostly missing
            "reservation_status_date",
        ]
    )
    df["country"] = df["country"].fillna("PRT")  # most common country
    return df.dropna().reset_index(drop=True)


def make_sample(n_per_hotel: int = 1000, seed: int = 42) -> None:
    """Write a reproducible random sample of each raw file to data/sample."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for hotel in HOTEL_NAMES:
        df = pd.read_csv(RAW_DIR / f"{hotel}.csv")
        df.sample(n=n_per_hotel, random_state=seed).sort_index().to_csv(
            SAMPLE_DIR / f"{hotel}_sample.csv", index=False
        )
