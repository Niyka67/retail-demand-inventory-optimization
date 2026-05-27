from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FORECAST_PATH = ROOT_DIR / "outputs" / "forecast_results.csv"
OUTPUT_PATH = ROOT_DIR / "outputs" / "inventory_recommendations.csv"


CATEGORY_ECONOMICS = {
    "ultra_fresh": {"selling_price": 280, "cost_price": 180, "salvage_price": 40},
    "drinks": {"selling_price": 360, "cost_price": 230, "salvage_price": 120},
    "bakery": {"selling_price": 220, "cost_price": 130, "salvage_price": 20},
    "grocery": {"selling_price": 420, "cost_price": 310, "salvage_price": 220},
}

STANDARD_NORMAL = NormalDist()


def critical_ratio(selling_price: float, cost_price: float, salvage_price: float) -> float:
    understock_cost = selling_price - cost_price
    overstock_cost = cost_price - salvage_price
    return understock_cost / (understock_cost + overstock_cost)


def add_inventory_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    residuals = df["quantity"] - df["prediction_model"]
    global_rmse = float(np.sqrt(np.mean(residuals**2)))

    recommendations = []
    for _, row in df.iterrows():
        economics = CATEGORY_ECONOMICS[row["category"]]
        ratio = critical_ratio(**economics)

        raw_order = row["prediction_model"] + STANDARD_NORMAL.inv_cdf(ratio) * max(global_rmse, 1.0)
        recommended_order = max(0, int(np.ceil(raw_order)))

        recommendations.append(
            {
                "critical_ratio": ratio,
                "forecast_error_rmse": global_rmse,
                "recommended_order": recommended_order,
            }
        )

    rec_df = pd.DataFrame(recommendations)
    result = pd.concat([df.reset_index(drop=True), rec_df], axis=1)

    result["expected_leftover"] = np.maximum(
        result["recommended_order"] - result["quantity"], 0
    )
    result["expected_stockout"] = np.maximum(
        result["quantity"] - result["recommended_order"], 0
    )
    return result


def optimize_inventory() -> None:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}. Run src/forecast_demand.py first."
        )

    df = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    result = add_inventory_recommendations(df)
    result.to_csv(OUTPUT_PATH, index=False)

    summary = (
        result.groupby("category")
        .agg(
            rows=("sku_id", "count"),
            avg_forecast=("prediction_model", "mean"),
            avg_recommended_order=("recommended_order", "mean"),
            avg_stockout=("expected_stockout", "mean"),
            avg_leftover=("expected_leftover", "mean"),
        )
        .round(2)
        .reset_index()
    )

    print("Inventory optimization completed.")
    print(f"Recommendations file: {OUTPUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    optimize_inventory()
