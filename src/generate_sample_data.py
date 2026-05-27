from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "sample_sales.csv"


def generate_sample_sales(
    n_stores: int = 42,
    n_skus: int = 12,
    start_date: str = "2024-01-01",
    periods: int = 180,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start_date, periods=periods, freq="D")

    sku_categories = {
        sku_id: rng.choice(["ultra_fresh", "drinks", "bakery", "grocery"])
        for sku_id in range(1, n_skus + 1)
    }

    rows = []
    for date in dates:
        day_of_week = date.dayofweek
        is_weekend = int(day_of_week >= 5)
        seasonal_temp = 18 + 13 * np.sin(2 * np.pi * date.dayofyear / 365)
        temperature = seasonal_temp + rng.normal(0, 4)

        for store_id in range(1, n_stores + 1):
            store_multiplier = rng.normal(1.0, 0.12)

            for sku_id in range(1, n_skus + 1):
                category = sku_categories[sku_id]
                base_demand = {
                    "ultra_fresh": 28,
                    "drinks": 34,
                    "bakery": 24,
                    "grocery": 18,
                }[category]

                weekend_effect = 1.15 if is_weekend else 1.0
                temp_effect = 1.0
                if category == "drinks":
                    temp_effect += max(temperature - 22, 0) * 0.035
                if category == "ultra_fresh":
                    temp_effect += max(temperature - 20, 0) * 0.015

                expected_demand = base_demand * store_multiplier * weekend_effect * temp_effect
                quantity = max(0, rng.poisson(expected_demand))

                rows.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "sku_id": f"SKU_{sku_id:03d}",
                        "category": category,
                        "temperature": round(float(temperature), 1),
                        "is_weekend": is_weekend,
                        "quantity": int(quantity),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_sample_sales()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Sample data saved to {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Stores: {df['store_id'].nunique()}")
    print(f"SKUs: {df['sku_id'].nunique()}")


if __name__ == "__main__":
    main()
