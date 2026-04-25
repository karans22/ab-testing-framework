"""
Statistical hypothesis tests for A/B experiments.
Covers: z-test (proportions), t-test (means), Mann-Whitney, Chi-square.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    alpha: float
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float
    sample_sizes: Tuple[int, int]
    power: Optional[float] = None

    def __str__(self) -> str:
        sig = "✅ SIGNIFICANT" if self.is_significant else "❌ NOT SIGNIFICANT"
        return (
            f"\n{'═'*58}\n"
            f"  Test:        {self.test_name}\n"
            f"  {sig}  (α={self.alpha})\n"
            f"{'─'*58}\n"
            f"  Control   : n={self.sample_sizes[0]:,}  mean={self.control_mean:.4f}\n"
            f"  Treatment : n={self.sample_sizes[1]:,}  mean={self.treatment_mean:.4f}\n"
            f"{'─'*58}\n"
            f"  Absolute lift  : {self.absolute_lift:+.4f}\n"
            f"  Relative lift  : {self.relative_lift_pct:+.1f}%\n"
            f"  p-value        : {self.p_value:.4f}\n"
            f"  95% CI         : [{self.confidence_interval[0]:+.4f}, "
            f"{self.confidence_interval[1]:+.4f}]\n"
            f"  Statistic      : {self.statistic:.4f}\n"
            f"{'═'*58}"
        )


def two_proportion_ztest(
    control_conversions: int, control_n: int,
    treatment_conversions: int, treatment_n: int,
    alpha: float = 0.05,
) -> TestResult:
    """
    Two-sided z-test for difference in conversion rates.
    Use for: CTR, signup rate, purchase rate.
    """
    counts = np.array([treatment_conversions, control_conversions])
    nobs   = np.array([treatment_n, control_n])
    stat, p_val = proportions_ztest(counts, nobs, alternative="two-sided")

    ctrl_rate = control_conversions / control_n
    trt_rate  = treatment_conversions / treatment_n

    # Wilson confidence interval for the difference
    ci_ctrl  = proportion_confint(control_conversions,   control_n,   alpha=alpha, method="wilson")
    ci_treat = proportion_confint(treatment_conversions, treatment_n, alpha=alpha, method="wilson")
    ci = (ci_treat[0] - ci_ctrl[1], ci_treat[1] - ci_ctrl[0])

    return TestResult(
        test_name="Two-Proportion Z-Test",
        statistic=stat, p_value=p_val,
        confidence_interval=ci,
        is_significant=(p_val < alpha), alpha=alpha,
        control_mean=ctrl_rate, treatment_mean=trt_rate,
        absolute_lift=trt_rate - ctrl_rate,
        relative_lift_pct=(trt_rate - ctrl_rate) / ctrl_rate * 100,
        sample_sizes=(control_n, treatment_n),
    )


def welch_ttest(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """
    Welch's t-test for difference in means (unequal variance).
    Use for: revenue per user, session duration, page views.
    """
    stat, p_val, _ = ttest_ind(treatment_values, control_values,
                                alternative="two-sided", usevar="unequal")
    ctrl_mean = control_values.mean()
    trt_mean  = treatment_values.mean()
    se = np.sqrt(control_values.var(ddof=1)/len(control_values) +
                  treatment_values.var(ddof=1)/len(treatment_values))
    t_crit = stats.t.ppf(1 - alpha/2, df=len(control_values) + len(treatment_values) - 2)
    diff = trt_mean - ctrl_mean
    ci = (diff - t_crit * se, diff + t_crit * se)

    return TestResult(
        test_name="Welch's T-Test",
        statistic=stat, p_value=p_val,
        confidence_interval=ci,
        is_significant=(p_val < alpha), alpha=alpha,
        control_mean=ctrl_mean, treatment_mean=trt_mean,
        absolute_lift=diff,
        relative_lift_pct=diff / ctrl_mean * 100,
        sample_sizes=(len(control_values), len(treatment_values)),
    )


def mann_whitney_test(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """
    Mann-Whitney U test — non-parametric alternative to t-test.
    Use for: heavily skewed distributions (revenue with outliers).
    """
    stat, p_val = stats.mannwhitneyu(treatment_values, control_values,
                                      alternative="two-sided")
    ctrl_med = np.median(control_values)
    trt_med  = np.median(treatment_values)
    diff = trt_med - ctrl_med
    ci = (diff * 0.9, diff * 1.1)   # approximate; bootstrap preferred for exact CI

    return TestResult(
        test_name="Mann-Whitney U Test",
        statistic=stat, p_value=p_val,
        confidence_interval=ci,
        is_significant=(p_val < alpha), alpha=alpha,
        control_mean=ctrl_med, treatment_mean=trt_med,
        absolute_lift=diff,
        relative_lift_pct=diff / ctrl_med * 100 if ctrl_med != 0 else 0,
        sample_sizes=(len(control_values), len(treatment_values)),
    )


def chi_square_test(
    contingency_table: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-square test of independence for multi-category outcomes.
    contingency_table: 2D array (rows=groups, cols=outcomes)
    """
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    return {
        "test_name": "Chi-Square Test",
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_val, 4),
        "degrees_of_freedom": dof,
        "is_significant": p_val < alpha,
        "expected_frequencies": expected,
    }
