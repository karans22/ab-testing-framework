"""Unit tests for statistical hypothesis test functions."""
import numpy as np
import pytest
from src.stats.hypothesis_tests import two_proportion_ztest, welch_ttest, mann_whitney_test
from src.stats.power_analysis import sample_size_for_proportions, sample_size_for_means


# ── Proportion z-test ─────────────────────────────────────────────────────────
def test_ztest_significant():
    """Clear effect should be significant."""
    result = two_proportion_ztest(
        control_conversions=500, control_n=5000,
        treatment_conversions=650, treatment_n=5000,
    )
    assert result.is_significant
    assert result.p_value < 0.05
    assert result.absolute_lift > 0


def test_ztest_not_significant():
    """No effect should not be significant."""
    rng = np.random.default_rng(42)
    result = two_proportion_ztest(
        control_conversions=500, control_n=5000,
        treatment_conversions=502, treatment_n=5000,
    )
    assert not result.is_significant


def test_ztest_relative_lift():
    result = two_proportion_ztest(100, 1000, 120, 1000)
    assert abs(result.relative_lift_pct - 20.0) < 0.5


# ── Welch t-test ──────────────────────────────────────────────────────────────
def test_ttest_detects_effect():
    rng = np.random.default_rng(0)
    ctrl = rng.normal(50, 10, 500)
    trt  = rng.normal(55, 10, 500)   # 5-unit lift
    result = welch_ttest(ctrl, trt)
    assert result.is_significant
    assert result.absolute_lift > 0


def test_ttest_no_effect():
    rng = np.random.default_rng(1)
    ctrl = rng.normal(50, 10, 200)
    trt  = rng.normal(50, 10, 200)
    result = welch_ttest(ctrl, trt)
    # With identical distributions, should usually not be significant
    # (may occasionally fail — that's fine for a random test)
    assert result.p_value > 0.001


# ── Power analysis ────────────────────────────────────────────────────────────
def test_sample_size_increases_with_smaller_mde():
    r1 = sample_size_for_proportions(0.10, 0.02)   # larger MDE
    r2 = sample_size_for_proportions(0.10, 0.01)   # smaller MDE
    assert r2["n_per_variant"] > r1["n_per_variant"]


def test_sample_size_increases_with_higher_power():
    r1 = sample_size_for_proportions(0.10, 0.02, power=0.80)
    r2 = sample_size_for_proportions(0.10, 0.02, power=0.90)
    assert r2["n_per_variant"] > r1["n_per_variant"]


def test_sample_size_for_means_positive():
    result = sample_size_for_means(50.0, 15.0, 3.0)
    assert result["n_per_variant"] > 0
    assert result["total_n"] == result["n_per_variant"] * 2


def test_ci_contains_true_effect():
    """95% CI should contain the true effect most of the time."""
    result = two_proportion_ztest(100, 1000, 130, 1000)
    true_effect = 0.130 - 0.100   # 0.030
    lo, hi = result.confidence_interval
    assert lo < true_effect < hi
