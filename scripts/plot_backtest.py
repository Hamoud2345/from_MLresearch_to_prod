"""Generate the backtest figures used in the README.

Runs a walk-forward backtest for the LightGBM model and the seasonal-naive
baseline, then saves:
  - assets/backtest_pnl.png        cumulative trading PnL of the ML strategy
  - assets/forecast_vs_actual.png  forecast with its uncertainty band

Run with:  python scripts/plot_backtest.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np

from power_forecaster.backtest import DirectionalStrategy, WalkForwardBacktester
from power_forecaster.data import get_data_source
from power_forecaster.features import build_default_pipeline
from power_forecaster.features.base import TARGET
from power_forecaster.models import create_model

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ASSETS.mkdir(exist_ok=True)

HISTORY_DAYS = 500
TRAIN_DAYS = 180
TEST_DAYS = 30


def main() -> None:
    pipeline = build_default_pipeline()
    raw = get_data_source("synthetic", seed=42).load(HISTORY_DAYS)

    bt = WalkForwardBacktester(
        pipeline=pipeline,
        train_window_days=TRAIN_DAYS,
        test_window_days=TEST_DAYS,
        strategy=DirectionalStrategy(),
    )
    report = bt.run(raw, lambda: create_model("lightgbm"))
    agg = report.aggregate()

    # --- Figure 1: cumulative PnL ------------------------------------------
    plt.figure(figsize=(9, 4.5))
    plt.plot(report.cum_pnl, color="#1b9e77", lw=1.6)
    plt.fill_between(range(len(report.cum_pnl)), 0, report.cum_pnl, color="#1b9e77", alpha=0.15)
    plt.title(
        f"Walk-forward backtest — cumulative PnL (LightGBM)\n"
        f"Sharpe={agg['sharpe']:.1f}  |  RMSE={agg['rmse']:.2f} €/MWh  |  "
        f"{agg['n_folds']} folds",
        fontsize=11,
    )
    plt.xlabel("Hour (chronological, out-of-sample)")
    plt.ylabel("Cumulative PnL (€)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / "backtest_pnl.png", dpi=130)
    plt.close()

    # --- Figure 2: forecast vs actual with uncertainty band ----------------
    data = pipeline.transform(raw).dropna().reset_index(drop=True)
    cols = [c for c in data.columns if c not in {"timestamp", TARGET}]
    split = len(data) - TEST_DAYS * 24
    model = create_model("lightgbm").fit(data[cols].iloc[:split], data[TARGET].iloc[:split])
    test = data.iloc[split : split + 24 * 7]  # one week out-of-sample
    pred = model.predict(test[cols])

    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(test))
    plt.plot(x, test[TARGET].to_numpy(), color="#262626", lw=1.4, label="Actual price")
    plt.plot(x, pred.median, color="#d95f02", lw=1.4, label="Forecast (median)")
    plt.fill_between(x, pred.lower, pred.upper, color="#d95f02", alpha=0.2, label="80% interval")
    plt.title("Day-ahead price: forecast vs actual (1 week out-of-sample)", fontsize=11)
    plt.xlabel("Hour")
    plt.ylabel("Price (€/MWh)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / "forecast_vs_actual.png", dpi=130)
    plt.close()

    print(f"Saved figures to {ASSETS}/  | metrics: {agg}")


if __name__ == "__main__":
    main()
