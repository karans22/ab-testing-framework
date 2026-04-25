# 🧪 A/B Testing Framework

A rigorous statistical A/B testing framework covering experiment design, sample size calculation, multiple testing correction, and results interpretation. Designed for product analysts and data scientists running experiments in e-commerce and SaaS contexts.

## 🎯 Objective
Build a reusable, statistically rigorous A/B testing library that handles the full experiment lifecycle — from power analysis to results reporting — including common pitfalls like p-hacking and peeking.

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| Statistical Engine | SciPy, Statsmodels |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Bayesian Testing | PyMC (optional) |
| Notebooks | Jupyter |
| Testing | pytest |

## 📁 Project Structure
```
ab-testing-framework/
├── src/
│   ├── stats/
│   │   ├── power_analysis.py       # Sample size & power calculations
│   │   ├── hypothesis_tests.py     # z-test, t-test, chi-square, Mann-Whitney
│   │   ├── multiple_testing.py     # Bonferroni, BH correction
│   │   └── sequential_testing.py   # Always-valid inference (alpha-spending)
│   └── experiments/
│       ├── experiment.py           # Experiment runner class
│       └── result_reporter.py      # Formatted results + business impact
├── notebooks/
│   ├── 01_power_analysis.ipynb
│   ├── 02_conversion_rate_test.ipynb
│   ├── 03_revenue_test.ipynb
│   └── 04_multi_armed_bandit.ipynb
├── reports/                        # Auto-generated experiment reports
├── tests/
│   └── test_stats.py
├── requirements.txt
└── README.md
```

## 📌 Tests Supported
| Test | Use Case | Metric Type |
|---|---|---|
| Two-proportion z-test | Conversion rates (CTR, signup) | Binary |
| Welch's t-test | Revenue, session duration | Continuous |
| Mann-Whitney U | Non-normal distributions | Ordinal/skewed |
| Chi-square test | Multi-category outcomes | Categorical |
| CUPED | Variance reduction with pre-experiment data | Any |
| Bonferroni / BH | Multiple variants / metrics | Correction |

## 📊 Example Results Output
```
════════════════════════════════════════════════════════
  A/B TEST RESULTS — Homepage CTA Colour Change
════════════════════════════════════════════════════════
  Experiment:   homepage_cta_v2         Status: SIGNIFICANT
  Start Date:   2023-11-01             End Date: 2023-11-15
  ─────────────────────────────────────────────────────
  Control   (A): n=4,821  conversions=432  rate=8.96%
  Treatment (B): n=4,793  conversions=511  rate=10.66%
  ─────────────────────────────────────────────────────
  Absolute Lift:   +1.70 pp
  Relative Lift:   +19.0%
  p-value:         0.0018   ✅ (α=0.05)
  95% CI:          [+0.63pp, +2.77pp]
  Statistical Power: 91.2%
  ─────────────────────────────────────────────────────
  💼 Business Impact (annualised):
     Est. additional conversions/month: ~816
     Est. revenue impact (@ $45 AOV):  +$36,720/month
════════════════════════════════════════════════════════
```

## 🚀 How to Run
```bash
git clone https://github.com/karans22/ab-testing-framework.git
cd ab-testing-framework
pip install -r requirements.txt

# Run an experiment
python src/experiments/experiment.py

# Run notebooks
jupyter notebook notebooks/02_conversion_rate_test.ipynb

# Run tests
pytest tests/ -v
```

## 👤 Author
**Karan S** | Aspiring Data Scientist  
[LinkedIn](https://linkedin.com/in/karans22) | [GitHub](https://github.com/karans22)
