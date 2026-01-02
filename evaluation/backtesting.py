import pandas as pd
from evaluation.metrics import rmse, mae, mape


def naive_forecast(y):
    """
    Predict y[t+1] = y[t]
    """
    return y.shift(1)


def compare_models(y_true, y_pred_model):
    """
    Compare ML model vs naive baseline
    """
    y_naive_pred = naive_forecast(y_true)

    # Align
    valid_idx = y_naive_pred.notna()
    y_true = y_true[valid_idx]
    y_naive_pred = y_naive_pred[valid_idx]
    y_pred_model = y_pred_model[valid_idx]

    results = {
        "rmse_naive": rmse(y_true, y_naive_pred),
        "rmse_model": rmse(y_true, y_pred_model),
        "mae_naive": mae(y_true, y_naive_pred),
        "mae_model": mae(y_true, y_pred_model),
        "mape_naive": mape(y_true, y_naive_pred),
        "mape_model": mape(y_true, y_pred_model),
    }

    results["rmse_improvement_pct"] = (
        (results["rmse_naive"] - results["rmse_model"])
        / results["rmse_naive"]
        * 100
    )

    results["mae_improvement_pct"] = (
        (results["mae_naive"] - results["mae_model"])
        / results["mae_naive"]
        * 100
    )

    results["mape_improvement_pct"] = (
        (results["mape_naive"] - results["mape_model"])
        / results["mape_naive"]
        * 100
    )

    return results
