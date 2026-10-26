import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotel_bookings import analysis, data  # noqa: E402


def test_clean_has_no_missing_and_no_padding():
    df = data.clean(data.load_raw(data.SAMPLE_DIR))
    assert not df.isna().any().any()
    assert not df["meal"].str.contains(" ").any()
    assert {"total_stay", "total_guests", "different_room_assigned"} <= set(df.columns)


def test_cancellation_cost_matches_original_estimate():
    assert analysis.cancellation_cost(44224) == 44224 * 200 * 3.4


def test_model_runs_on_sample():
    df = data.clean(data.load_raw(data.SAMPLE_DIR))
    _, auc = analysis.fit_logit(df)
    assert 0.5 < auc <= 1
