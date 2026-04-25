"""
Experiment runner: end-to-end A/B experiment simulation and analysis.
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import date, timedelta

from src.stats.hypothesis_tests import two_proportion_ztest, welch_ttest
from src.stats.power_analysis import sample_size_for_proportions, runtime_estimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("experiment")


@dataclass
class ExperimentConfig:
    name: str
    hypothesis: str
    metric_type: str          # "binary" | "continuous"
    baseline_value: float     # conversion rate or mean
    mde: float                # minimum detectable effect (absolute)
    alpha: float = 0.05
    power: float = 0.80
    daily_traffic: int = 2000
    std_dev: Optional[float] = None   # required for continuous metrics


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._design = None

    def design(self) -> dict:
        """Calculate required sample size and runtime."""
        cfg = self.config
        if cfg.metric_type == "binary":
            design = sample_size_for_proportions(
                cfg.baseline_value, cfg.mde, cfg.alpha, cfg.power
            )
        else:
            if cfg.std_dev is None:
                raise ValueError("std_dev required for continuous metric experiments.")
            from src.stats.power_analysis import sample_size_for_means
            design = sample_size_for_means(
                cfg.baseline_value, cfg.std_dev, cfg.mde, cfg.alpha, cfg.power
            )

        runtime = runtime_estimator(design["n_per_variant"], cfg.daily_traffic)
        design.update(runtime)
        self._design = design

        logger.info(f"\n{'='*55}")
        logger.info(f"  Experiment Design: {cfg.name}")
        logger.info(f"{'='*55}")
        logger.info(f"  Hypothesis   : {cfg.hypothesis}")
        logger.info(f"  Baseline     : {cfg.baseline_value}")
        logger.info(f"  MDE          : {cfg.mde:+} ({design['mde_relative_pct']:.1f}% relative)")
        logger.info(f"  Sample needed: {design['n_per_variant']:,} / variant  "
                    f"({design['total_n']:,} total)")
        logger.info(f"  Runtime      : ~{design['days_required']} days "
                    f"({design['weeks_required']} weeks)")
        return design

    def simulate(self, seed: int = 42, true_effect: Optional[float] = None) -> dict:
        """
        Simulate experiment data and run statistical test.
        true_effect: actual effect (None = no effect / null hypothesis is true)
        """
        if self._design is None:
            self.design()

        rng = np.random.default_rng(seed)
        n = self._design["n_per_variant"]
        cfg = self.config
        effect = true_effect if true_effect is not None else 0.0

        if cfg.metric_type == "binary":
            ctrl_conv = rng.binomial(n, cfg.baseline_value)
            trt_conv  = rng.binomial(n, cfg.baseline_value + effect)
            result = two_proportion_ztest(ctrl_conv, n, trt_conv, n, alpha=cfg.alpha)

        else:
            ctrl_vals = rng.normal(cfg.baseline_value, cfg.std_dev, n)
            trt_vals  = rng.normal(cfg.baseline_value + effect, cfg.std_dev, n)
            result = welch_ttest(ctrl_vals, trt_vals, alpha=cfg.alpha)

        logger.info(result)

        # Business impact estimate
        if cfg.metric_type == "binary" and result.is_significant:
            monthly_users = cfg.daily_traffic * 30
            extra_conversions = int(result.absolute_lift * monthly_users / 2)
            logger.info(f"\n💼 Business Impact: ~{extra_conversions:,} extra conversions/month")

        return {"design": self._design, "result": result}


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Experiment 1: Homepage CTA colour change
    exp = Experiment(ExperimentConfig(
        name="homepage_cta_colour_v2",
        hypothesis="Changing CTA button from grey to green increases sign-up rate",
        metric_type="binary",
        baseline_value=0.089,   # 8.9% baseline sign-up rate
        mde=0.015,              # detect +1.5pp lift
        alpha=0.05, power=0.80,
        daily_traffic=3000,
    ))
    exp.design()
    exp.simulate(seed=42, true_effect=0.019)   # true effect slightly above MDE

    print("\n" + "─"*55)

    # Experiment 2: Checkout flow simplification (revenue per user)
    exp2 = Experiment(ExperimentConfig(
        name="checkout_simplification_v1",
        hypothesis="Simplified checkout increases average order value",
        metric_type="continuous",
        baseline_value=52.40,   # $52.40 avg order value
        mde=4.0,                # detect +$4 lift
        std_dev=28.0,
        alpha=0.05, power=0.80,
        daily_traffic=1500,
    ))
    exp2.design()
    exp2.simulate(seed=7, true_effect=3.8)
