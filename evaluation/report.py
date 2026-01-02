from pathlib import Path
import pandas as pd


def generate_monitoring_report(
    metrics,
    backtest_csv,
    drift_csv,
    ks_result,
    output_path
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backtest = pd.read_csv(backtest_csv)
    drift = pd.read_csv(drift_csv)

    drift_rate = drift["drift_flag"].mean() * 100

    lines = []

    lines.append("# 📊 Model Monitoring Report\n")

    lines.append("## 1️⃣ Model Accuracy\n")
    lines.append(f"- RMSE: **{metrics['rmse']:.2f}**\n")
    lines.append(f"- MAE: **{metrics['mae']:.2f}**\n")
    lines.append(f"- MAPE: **{metrics['mape']:.2f}%**\n")

    lines.append("## 2️⃣ Backtesting vs Naive Baseline\n")
    lines.append(
        f"- RMSE Improvement: **{backtest['rmse_improvement_pct'].iloc[0]:.2f}%**\n"
    )
    lines.append(
        f"- MAE Improvement: **{backtest['mae_improvement_pct'].iloc[0]:.2f}%**\n"
    )

    lines.append("## 3️⃣ Drift Monitoring\n")
    lines.append(f"- Drift flag rate: **{drift_rate:.2f}%**\n")
    lines.append(
        f"- KS-test p-value: **{ks_result['p_value']:.4f}**\n"
    )

    lines.append("## 4️⃣ Model Health Assessment\n")

    if ks_result["drift_detected"] or drift_rate > 20:
        lines.append("⚠️ **Warning:** Performance degradation detected.\n")
        lines.append("Recommendation: Schedule model retraining.\n")
    else:
        lines.append("✅ **Healthy:** Model performance is stable.\n")
        lines.append("Recommendation: Continue monitoring.\n")

    report_text = "\n".join(lines)
    output_path.write_text(report_text)

    return report_text
