from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class RidgeBaselineRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.feature_means: pd.Series | None = None
        self.feature_stds: pd.Series | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.feature_means = x.mean()
        self.feature_stds = x.std().replace(0, 1)
        x_scaled = (x - self.feature_means) / self.feature_stds
        design = np.c_[np.ones(len(x_scaled)), x_scaled.to_numpy()]
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0
        self.weights = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ y.to_numpy(),
        )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.feature_means is None or self.feature_stds is None or self.weights is None:
            raise RuntimeError("Model is not fitted.")
        x_scaled = (x - self.feature_means) / self.feature_stds
        design = np.c_[np.ones(len(x_scaled)), x_scaled.to_numpy()]
        return design @ self.weights


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "sample_sales.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
FORECAST_PATH = OUTPUT_DIR / "forecast_results.csv"
METRICS_PATH = OUTPUT_DIR / "forecast_metrics.csv"


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denominator


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "sku_id", "date"]).copy()
    group_cols = ["store_id", "sku_id"]

    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df.groupby(group_cols)["quantity"].shift(lag)

    shifted_sales = df.groupby(group_cols)["quantity"].shift(1)
    df["rolling_mean_7"] = (
        shifted_sales.groupby([df["store_id"], df["sku_id"]])
        .rolling(7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_std_7"] = (
        shifted_sales.groupby([df["store_id"], df["sku_id"]])
        .rolling(7)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    df["baseline_7d"] = df["rolling_mean_7"]
    return df.dropna().reset_index(drop=True)


def train_model(train_df: pd.DataFrame, features: list[str]):
    if lgb is None:
        model = RidgeBaselineRegressor(alpha=10.0)
        model.fit(train_df[features], train_df["quantity"])
        return model, "ridge_fallback"

    train_dataset = lgb.Dataset(train_df[features], label=train_df["quantity"])
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": 42,
        "verbose": -1,
    }
    return lgb.train(params, train_dataset, num_boost_round=250), "lightgbm"


def evaluate_model(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | str]:
    return {
        "model": name,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def build_forecast() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. Run src/generate_sample_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df_model = add_time_series_features(df)

    split_date = df_model["date"].max() - pd.Timedelta(days=21)
    train_df = df_model[df_model["date"] <= split_date].copy()
    test_df = df_model[df_model["date"] > split_date].copy()

    features = [
        "store_id",
        "temperature",
        "is_weekend",
        "day_of_week",
        "week_of_year",
        "month",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_std_7",
    ]

    model, model_name = train_model(train_df, features)
    test_df["prediction_model"] = np.maximum(model.predict(test_df[features]), 0)
    test_df["prediction_baseline_7d"] = np.maximum(test_df["baseline_7d"], 0)

    metrics = pd.DataFrame(
        [
            evaluate_model("baseline_7d", test_df["quantity"], test_df["prediction_baseline_7d"]),
            evaluate_model(model_name, test_df["quantity"], test_df["prediction_model"]),
        ]
    )

    test_df.to_csv(FORECAST_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)

    print("Forecast completed.")
    print(f"Forecast file: {FORECAST_PATH}")
    print(f"Metrics file: {METRICS_PATH}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    build_forecast()
