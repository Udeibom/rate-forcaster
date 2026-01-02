import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def rolling_error_detector(
    y_true,
    y_pred,
    window=30,
    spike_threshold=2.0
):
    """
    Detects drift if rolling RMSE exceeds
    spike_threshold * historical RMSE
    """
    errors = y_true - y_pred
    rolling_rmse = (
        errors.rolling(window)
        .apply(lambda x: np.sqrt(np.mean(x**2)))
    )

    baseline_rmse = rolling_rmse.mean()

    drift_flag = rolling_rmse > spike_threshold * baseline_rmse

    return pd.DataFrame({
        "rolling_rmse": rolling_rmse,
        "drift_flag": drift_flag
    })

def ks_drift_test(
    reference_errors,
    recent_errors,
    alpha=0.05
):
    """
    Kolmogorov–Smirnov test
    H0: both samples come from same distribution
    """
    statistic, p_value = ks_2samp(reference_errors, recent_errors)

    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha)
    }
