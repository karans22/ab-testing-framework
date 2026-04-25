"""
Sample size and statistical power calculations for A/B test design.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

logger = logging.getLogger(__name__)


def sample_size_for_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> dict:
    """
    Calculate required sample size per variant for a conversion rate test.

    Args:
        baseline_rate: Current conversion rate (e.g. 0.05 for 5%)
        minimum_detectable_effect: Absolute lift to detect (e.g. 0.01 for +1pp)
        alpha: Type I error rate
        power: 1 - Type II error rate
        two_tailed: Whether to use two-sided test
    """
    treatment_rate = baseline_rate + minimum_detectable_effect
    effect_size = proportion_effectsize(baseline_rate, treatment_rate)

    analysis = NormalIndPower()
    n_per_variant = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        alternative="two-sided" if two_tailed else "larger",
    )
    n_per_variant = int(np.ceil(n_per_variant))

    return {
        "baseline_rate":     baseline_rate,
        "treatment_rate":    treatment_rate,
        "mde_absolute":      minimum_detectable_effect,
        "mde_relative_pct":  minimum_detectable_effect / baseline_rate * 100,
        "alpha":             alpha,
        "power":             power,
        "n_per_variant":     n_per_variant,
        "total_n":           n_per_variant * 2,
        "effect_size":       abs(effect_size),
    }


def sample_size_for_means(
    baseline_mean: float,
    baseline_std: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate required sample size per variant for a continuous metric test
    (e.g., revenue per user, session duration).
    """
    effect_size = minimum_detectable_effect / baseline_std  # Cohen's d
    analysis = TTestIndPower()
    n_per_variant = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    n_per_variant = int(np.ceil(n_per_variant))

    return {
        "baseline_mean":    baseline_mean,
        "baseline_std":     baseline_std,
        "mde_absolute":     minimum_detectable_effect,
        "mde_relative_pct": minimum_detectable_effect / baseline_mean * 100,
        "cohens_d":         effect_size,
        "alpha":            alpha,
        "power":            power,
        "n_per_variant":    n_per_variant,
        "total_n":          n_per_variant * 2,
    }


def plot_power_curve(
    baseline_rate: float,
    alpha: float = 0.05,
    power_target: float = 0.80,
    mde_range=(0.005, 0.05),
    save_path: str = None,
):
    """Plot sample size vs MDE power curve."""
    mdes = np.linspace(mde_range[0], mde_range[1], 50)
    sample_sizes = []
    for mde in mdes:
        result = sample_size_for_proportions(baseline_rate, mde, alpha, power_target)
        sample_sizes.append(result["n_per_variant"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mdes * 100, sample_sizes, linewidth=2.5, color="#2E75B6")
    ax.fill_between(mdes * 100, sample_sizes, alpha=0.12, color="#2E75B6")
    ax.axhline(y=1000, color="red", linestyle="--", linewidth=1, label="n=1,000")
    ax.axhline(y=5000, color="orange", linestyle="--", linewidth=1, label="n=5,000")
    ax.set_xlabel("Minimum Detectable Effect (pp)", fontsize=12)
    ax.set_ylabel("Required Sample Size per Variant", fontsize=12)
    ax.set_title(f"Power Curve (baseline={baseline_rate:.1%}, α={alpha}, power={power_target:.0%})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def runtime_estimator(n_per_variant: int, daily_traffic: int) -> dict:
    """Estimate experiment runtime given sample size and daily traffic."""
    days = np.ceil(n_per_variant / (daily_traffic / 2))  # 50/50 split
    return {
        "days_required":   int(days),
        "weeks_required":  round(days / 7, 1),
        "start_date_hint": f"Run for at least {int(days)} days, ending on a full week boundary.",
    }
