from pathlib import Path
import pandas as pd
import numpy as np


class MonthlyKwhTableFromFormula:
    def __init__(
        self,
        forecast_csv="us_10yr_monthly_cluster_forecast_with_irridance.csv",
        performance_ratio=0.75,
        temp_coeff=0.004,
        min_temp_loss=0.80,
        time_col="time",
        temp_col="pred_tavg",
        irr_col="irradiance",
        lat_col="cluster_lat",
        lon_col="cluster_lon",
        cluster_col="cluster_id",
    ):
        base_dir = Path(__file__).resolve().parent
        self.csv_path = base_dir / forecast_csv

        print("Forecast CSV:", self.csv_path, "Exists:", self.csv_path.exists())

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Forecast CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.df[time_col] = pd.to_datetime(self.df[time_col])

        self.performance_ratio = performance_ratio
        self.temp_coeff = temp_coeff
        self.min_temp_loss = min_temp_loss

        self.time_col = time_col
        self.temp_col = temp_col
        self.irr_col = irr_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.cluster_col = cluster_col

    def _estimate_monthly_kwh(self, kw_capacity, irradiance, avg_temp, days_in_month):
        temp_loss = 1 - np.maximum(avg_temp - 25, 0) * self.temp_coeff
        temp_loss = np.maximum(temp_loss, self.min_temp_loss)
        return kw_capacity * irradiance * days_in_month * self.performance_ratio * temp_loss

    def predict_table(self, kw_capacity, month, year):
        subset = self.df[
            (self.df[self.time_col].dt.month == month) &
            (self.df[self.time_col].dt.year == year)
        ].copy()

        if subset.empty:
            raise ValueError(f"No forecast data found for {year}-{month:02d}")

        days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month

        subset["predicted_monthly_kwh"] = self._estimate_monthly_kwh(
            kw_capacity=kw_capacity,
            irradiance=subset[self.irr_col].values,
            avg_temp=subset[self.temp_col].values,
            days_in_month=days_in_month
        )

        result = (
            subset.groupby(
                [self.cluster_col, self.lat_col, self.lon_col],
                as_index=False
            )
            .agg({
                self.irr_col: "mean",
                self.temp_col: "mean",
                "predicted_monthly_kwh": "mean"
            })
            .sort_values([self.lat_col, self.lon_col])
            .reset_index(drop=True)
        )

        return result
    
if __name__ == "__main__":
    formula_predictor = MonthlyKwhTableFromFormula()

    jan_2026_formula = formula_predictor.predict_table(
        kw_capacity=5.0,
        month=1,
        year=2026
    )

    print(jan_2026_formula.head())
