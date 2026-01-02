import pandas as pd
from pathlib import Path
from typing import Tuple, List

from features.build_features import (
    create_lag_features,
    create_rolling_features,
    create_calendar_features,
    create_return_features,
    create_volatility_features,
    create_target,
)


class FeaturePipeline:
    """
    Shared feature engineering pipeline for training and inference.
    Single source of truth.
    """

    def __init__(
        self,
        lags=[1, 7, 14, 30],
        rolling_windows=[7, 14, 30],
        volatility_windows=[7, 14, 30],
    ):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.volatility_windows = volatility_windows
        self.feature_names_ = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = create_lag_features(df, self.lags)
        df = create_rolling_features(df, self.rolling_windows)
        df = create_calendar_features(df)
        df = create_return_features(df)
        df = create_volatility_features(df, self.volatility_windows)
        df = create_target(df)

        drop_cols = [
            "Date",
            "Month",
            "Weekday",
            "USD_Rate_Category",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df = df.dropna().reset_index(drop=True)
        return df

    # --------------------
    # SKLEARN-COMPATIBLE API
    # --------------------

    def fit(self, df: pd.DataFrame):
        df_feat = self._engineer_features(df)
        X = df_feat.drop(columns=["target"])
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("FeaturePipeline is not fitted")

        df_feat = self._engineer_features(df)
        return df_feat[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        df_feat = self._engineer_features(df)
        X = df_feat.drop(columns=["target"])
        y = df_feat["target"]
        return X, y
