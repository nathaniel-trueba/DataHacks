from pathlib import Path
import pandas as pd
import numpy as np
import joblib


class MonthlyKwhTableFromModel:
    def __init__(
        self,
        forecast_csv="us_10yr_monthly_cluster_forecast_with_irridance.csv",
        model_file="monthly_kwh_model.joblib",
        features_file="monthly_kwh_features.joblib",
        time_col="time",
        temp_col="pred_tavg",
        irr_col="irradiance",
        lat_col="cluster_lat",
        lon_col="cluster_lon",
        cluster_col="cluster_id",
    ):
        base_dir = Path(__file__).resolve().parent

        self.csv_path = base_dir / forecast_csv
        self.model_path = base_dir / model_file
        self.features_path = base_dir / features_file

        print("Forecast CSV:", self.csv_path, "Exists:", self.csv_path.exists())
        print("Model file  :", self.model_path, "Exists:", self.model_path.exists())
        print("Features file:", self.features_path, "Exists:", self.features_path.exists())

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Forecast CSV not found: {self.csv_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        self.df = pd.read_csv(self.csv_path)
        self.df[time_col] = pd.to_datetime(self.df[time_col])

        self.model = joblib.load(self.model_path)
        self.features = joblib.load(self.features_path)

        self.time_col = time_col
        self.temp_col = temp_col
        self.irr_col = irr_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.cluster_col = cluster_col

    def _build_features(self, subset, kw_capacity, month, year):
        days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month

        X_pred = pd.DataFrame({
            "kilowatt_value": kw_capacity,
            "latitude": subset[self.lat_col].values,
            "longitude": subset[self.lon_col].values,
            "irradiance": subset[self.irr_col].values,
            "avg_temp": subset[self.temp_col].values,
            "days_in_month": days_in_month,
        })

        X_pred["kw_x_irradiance"] = X_pred["kilowatt_value"] * X_pred["irradiance"]
        X_pred["temp_above_25"] = np.maximum(X_pred["avg_temp"] - 25, 0)
        X_pred["month_sin"] = np.sin(2 * np.pi * month / 12)
        X_pred["month_cos"] = np.cos(2 * np.pi * month / 12)

        return X_pred[self.features]

    def predict_table(self, kw_capacity, month, year):
        subset = self.df[
            (self.df[self.time_col].dt.month == month) &
            (self.df[self.time_col].dt.year == year)
        ].copy()

        if subset.empty:
            raise ValueError(f"No forecast data found for {year}-{month:02d}")

        X_pred = self._build_features(subset, kw_capacity, month, year)
        subset["predicted_monthly_kwh"] = self.model.predict(X_pred)

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
    
model_predictor = MonthlyKwhTableFromModel()

jan_2026_model = model_predictor.predict_table(
    kw_capacity=5.0,
    month=1,
    year=2026
)

print(jan_2026_model.head())