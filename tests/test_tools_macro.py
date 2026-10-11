"""
test_tools_macro.py
====================
Offline tests for the FRED-backed macro indicators tool. The ``fredapi`` client is
replaced with a fake, so these tests check series selection and YoY math without
network access.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from research_agent.tools import cache, macro


class FakeFred:
    """Stand-in for ``fredapi.Fred``; returns a preset pandas Series per series id."""

    def __init__(self, series_by_id: dict[str, pd.Series]) -> None:
        self._series_by_id = series_by_id

    def get_series(self, series_id: str) -> pd.Series:
        return self._series_by_id[series_id]


def _monthly_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=index)


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)


class TestClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.setattr(macro, "FRED_API_KEY", "")
        with pytest.raises(ValueError, match="not set"):
            macro._client()

    def test_builds_fred_client_with_key(self, monkeypatch):
        monkeypatch.setattr(macro, "FRED_API_KEY", "test-key")
        client = macro._client()
        assert client.api_key == "test-key"


class TestLatestValue:
    def test_returns_most_recent_non_missing_observation(self):
        series = _monthly_series([1.0, 2.0, float("nan"), 3.5])
        fred = FakeFred({"X": series})
        _date, value = macro._latest_value(fred, "X")
        assert value == 3.5

    def test_raises_when_series_is_all_missing(self):
        series = _monthly_series([float("nan")])
        fred = FakeFred({"X": series})
        with pytest.raises(ValueError, match="no observations"):
            macro._latest_value(fred, "X")


class TestLatestYoyPctChange:
    def test_computes_pct_change_13_points_back(self):
        values = [100.0] * 12 + [110.0]  # 13th point vs. the one 12 months earlier
        series = _monthly_series(values)
        fred = FakeFred({"X": series})
        _, pct = macro._latest_yoy_pct_change(fred, "X")
        assert pct == pytest.approx(10.0)

    def test_raises_with_fewer_than_13_points(self):
        series = _monthly_series([100.0] * 5)
        fred = FakeFred({"X": series})
        with pytest.raises(ValueError, match="Not enough history"):
            macro._latest_yoy_pct_change(fred, "X")

    def test_raises_when_base_value_is_zero(self):
        values = [0.0] * 12 + [5.0]
        series = _monthly_series(values)
        fred = FakeFred({"X": series})
        with pytest.raises(ValueError, match="base value is 0"):
            macro._latest_yoy_pct_change(fred, "X")


class TestFetchIndicator:
    def test_raises_on_unknown_category(self, monkeypatch):
        monkeypatch.setattr(macro, "_client", lambda: FakeFred({}))
        with pytest.raises(ValueError, match="Unknown macro category"):
            macro._fetch_indicator("not_a_real_category")

    def test_fetches_a_level_series(self, monkeypatch):
        series = _monthly_series([1.0, 2.0, 5.25])
        monkeypatch.setattr(macro, "_client", lambda: FakeFred({"FEDFUNDS": series}))
        result = macro._fetch_indicator("fed_funds_rate")
        assert result["series_id"] == "FEDFUNDS"
        assert result["value"] == 5.25

    def test_fetches_a_yoy_series(self, monkeypatch):
        values = [100.0] * 12 + [108.0]
        series = _monthly_series(values)
        monkeypatch.setattr(macro, "_client", lambda: FakeFred({"CPIAUCSL": series}))
        result = macro._fetch_indicator("cpi_yoy")
        assert result["series_id"] == "CPIAUCSL"
        assert result["value"] == pytest.approx(8.0)

    def test_wraps_unexpected_errors_as_value_error(self, monkeypatch):
        class ExplodingFred:
            def get_series(self, series_id):
                raise RuntimeError("fredapi internals blew up")

        monkeypatch.setattr(macro, "_client", ExplodingFred)
        with pytest.raises(ValueError, match="FRED request failed"):
            macro._fetch_indicator("fed_funds_rate")


class TestGetMacroIndicators:
    def test_single_known_category(self, monkeypatch):
        series = _monthly_series([1.0, 2.0, 5.25])
        monkeypatch.setattr(macro, "_client", lambda: FakeFred({"FEDFUNDS": series}))
        data = json.loads(macro.get_macro_indicators.func("fed_funds_rate"))
        assert data["fed_funds_rate"]["value"] == 5.25

    def test_all_categories_with_one_failure_reports_partial_errors(self, monkeypatch):
        good_series = _monthly_series([1.0, 2.0, 5.25])

        class PartialFred:
            def get_series(self, series_id):
                if series_id == "FEDFUNDS":
                    return good_series
                raise RuntimeError("series unavailable")

        monkeypatch.setattr(macro, "_client", PartialFred)
        data = json.loads(macro.get_macro_indicators.func("all"))
        assert data["fed_funds_rate"]["value"] == 5.25
        assert "_partial_errors" in data

    def test_unknown_category_falls_back_to_all(self, monkeypatch):
        series = _monthly_series([100.0] * 12 + [108.0])  # long enough for the YoY series too
        monkeypatch.setattr(
            macro, "_client", lambda: FakeFred(dict.fromkeys(macro._SERIES.values(), series))
        )
        data = json.loads(macro.get_macro_indicators.func("not a real one"))
        assert set(data) >= set(macro._SERIES)

    def test_total_failure_returns_error(self, monkeypatch):
        def failing_client():
            raise ValueError("FRED_API_KEY is not set.")

        monkeypatch.setattr(macro, "_client", failing_client)
        data = json.loads(macro.get_macro_indicators.func("fed_funds_rate"))
        assert "error" in data
