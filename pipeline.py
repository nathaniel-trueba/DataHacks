from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "heat_trace_state_timeseries.parquet"


STATE_NAME_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District Of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}
STATE_ABBR_TO_NAME = {abbr: name for name, abbr in STATE_NAME_TO_ABBR.items()}


@dataclass(frozen=True)
class SourceFiles:
    eia: list[Path]
    solar: list[Path]
    epa: list[Path]


def get_state_data() -> list[dict[str, object]]:
    """Return the app-ready state/year panel as JSON-serializable records."""
    return load_state_dataset().to_dict(orient="records")


def load_state_dataset(prefer_raw: bool = True) -> pd.DataFrame:
    """Load the processed dataset, building it from raw files when available."""
    if prefer_raw:
        raw_files = discover_source_files()
        if raw_files.eia and raw_files.solar and raw_files.epa:
            return build_unified_dataset(raw_files)

    if OUTPUT_PATH.exists():
        return finalize_dataset(pd.read_parquet(OUTPUT_PATH))

    from scripts.build_mock_data import save_mock_dataset

    save_mock_dataset(OUTPUT_PATH)
    return finalize_dataset(pd.read_parquet(OUTPUT_PATH))


def build_and_save_dataset() -> Path:
    """Create the canonical processed parquet file from raw data when possible."""
    df = load_state_dataset(prefer_raw=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


def discover_source_files() -> SourceFiles:
    return SourceFiles(
        eia=_existing_files(["Complete_SEDS.csv"]),
        solar=_existing_files(
            [
                "records.csv",
                "Sullivan-Solar.csv",
                "Titan_All_Addresses.csv",
                "solar-city-permits.csv",
                "freedom-forever.csv",
                "sunrun.csv",
            ]
        ),
        epa=sorted(RAW_DIR.glob("annual_aqi_by_county*.csv")),
    )


def _existing_files(names: Iterable[str]) -> list[Path]:
    return [path for name in names if (path := RAW_DIR / name).exists()]


def build_unified_dataset(source_files: SourceFiles | None = None) -> pd.DataFrame:
    source_files = source_files or discover_source_files()
    eia_df = ingest_eia(source_files.eia)
    solar_df = ingest_solar(source_files.solar)
    epa_df = ingest_epa(source_files.epa)

    merged = eia_df.merge(solar_df, on=["state_abbr", "year"], how="outer")
    merged = merged.merge(epa_df, on=["state_abbr", "year"], how="outer")

    merged["state"] = merged["state_abbr"].map(STATE_ABBR_TO_NAME)
    merged["date"] = pd.to_datetime(merged["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")

    for column in [
        "energy_consumption",
        "energy_production",
        "solar_capacity_added",
        "co2_emissions",
        "air_quality_index",
    ]:
        if column not in merged.columns:
            merged[column] = np.nan

    return finalize_dataset(merged)


def ingest_eia(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    metric_map = {
        "TETCB": "energy_consumption",
        "TEPRB": "energy_production",
        "EMTCB": "co2_emissions",
    }

    for path in paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if not {"MSN", "StateCode", "Year", "Data"}.issubset(df.columns):
            continue

        filtered = df[df["MSN"].isin(metric_map)].copy()
        filtered["metric"] = filtered["MSN"].map(metric_map)
        filtered["Data"] = pd.to_numeric(filtered["Data"], errors="coerce")
        filtered["Year"] = pd.to_numeric(filtered["Year"], errors="coerce")
        filtered = filtered.dropna(subset=["StateCode", "Year", "Data", "metric"])

        grouped = (
            filtered.groupby(["StateCode", "Year", "metric"], as_index=False)["Data"]
            .sum()
            .pivot(index=["StateCode", "Year"], columns="metric", values="Data")
            .reset_index()
        )
        grouped.columns.name = None
        grouped = grouped.rename(columns={"StateCode": "state_abbr", "Year": "year"})
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=["state_abbr", "year", "energy_consumption", "energy_production", "co2_emissions"])

    result = pd.concat(frames, ignore_index=True)
    result["state_abbr"] = result["state_abbr"].astype(str).str.upper().str.strip()
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    return result.groupby(["state_abbr", "year"], as_index=False).sum(numeric_only=True)


def ingest_solar(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        if path.name == "records.csv":
            df = pd.read_csv(path)
            if {"state", "kilowatt_value", "issue_date"}.issubset(df.columns):
                current = df[["state", "kilowatt_value", "issue_date"]].rename(
                    columns={"kilowatt_value": "kw", "issue_date": "date"}
                )
                frames.append(current)
            continue

        if path.name in {"Sullivan-Solar.csv", "Titan_All_Addresses.csv"}:
            df = pd.read_csv(path)
            state_col = "STATE"
            date_col = "PERMIT_DATE"
            if {state_col, date_col}.issubset(df.columns):
                current = df[[state_col, date_col]].rename(columns={state_col: "state", date_col: "date"})
                current["kw"] = np.nan
                frames.append(current)
            continue

        if path.name == "solar-city-permits.csv":
            df = pd.read_csv(path)
            if {"STATE", "ISSUE_DATE"}.issubset(df.columns):
                current = df[["STATE", "ISSUE_DATE"]].rename(columns={"STATE": "state", "ISSUE_DATE": "date"})
                current["kw"] = np.nan
                frames.append(current)
            continue

        if path.name in {"freedom-forever.csv", "sunrun.csv"}:
            df = pd.read_csv(path)
            if "PROJECT_ADDRESS" not in df.columns:
                continue
            current = pd.DataFrame(
                {
                    "state": df["PROJECT_ADDRESS"].astype(str).str.strip().str[-8:-6].str.strip(),
                    "date": pd.to_datetime(df.get("INSTALL_DATE"), errors="coerce"),
                    "kw": np.nan,
                }
            )
            frames.append(current)

    if not frames:
        return pd.DataFrame(columns=["state_abbr", "year", "solar_capacity_added"])

    solar = pd.concat(frames, ignore_index=True)
    solar["date"] = pd.to_datetime(solar["date"], errors="coerce", utc=True).dt.tz_localize(None)
    solar["kw"] = pd.to_numeric(solar["kw"], errors="coerce")
    solar["state_abbr"] = solar["state"].astype(str).str.upper().str.strip()
    solar["year"] = solar["date"].dt.year.astype("Int64")
    solar = solar.dropna(subset=["state_abbr", "year"])

    grouped = solar.groupby(["state_abbr", "year"], as_index=False).agg(
        solar_permits=("state_abbr", "size"),
        solar_kw_total=("kw", "sum"),
    )
    grouped["solar_capacity_added"] = grouped["solar_kw_total"].where(grouped["solar_kw_total"].gt(0), grouped["solar_permits"])
    return grouped[["state_abbr", "year", "solar_capacity_added"]]


def ingest_epa(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if not {"State", "Year"}.issubset(df.columns):
            continue

        current = pd.DataFrame(
            {
                "state_abbr": df["State"].map(STATE_NAME_TO_ABBR).fillna(df["State"]),
                "year": pd.to_numeric(df["Year"], errors="coerce"),
                "median_aqi": pd.to_numeric(df.get("Median AQI"), errors="coerce"),
                "max_aqi": pd.to_numeric(df.get("Max AQI"), errors="coerce"),
            }
        )
        current["air_quality_index"] = current["median_aqi"].fillna(current["max_aqi"])
        current = current.dropna(subset=["state_abbr", "year", "air_quality_index"])
        frames.append(current[["state_abbr", "year", "air_quality_index"]])

    if not frames:
        return pd.DataFrame(columns=["state_abbr", "year", "air_quality_index"])

    result = pd.concat(frames, ignore_index=True)
    result["state_abbr"] = result["state_abbr"].astype(str).str.upper().str.strip()
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    return result.groupby(["state_abbr", "year"], as_index=False)["air_quality_index"].mean()


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "state_abbr" not in df.columns and "state" in df.columns:
        df["state_abbr"] = df["state"].map(STATE_NAME_TO_ABBR)
    if "state" not in df.columns and "state_abbr" in df.columns:
        df["state"] = df["state_abbr"].map(STATE_ABBR_TO_NAME)

    df["state_abbr"] = df["state_abbr"].astype(str).str.upper().str.strip()
    df["state"] = df["state"].fillna(df["state_abbr"].map(STATE_ABBR_TO_NAME))
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01", errors="coerce")

    for column in [
        "energy_consumption",
        "energy_production",
        "solar_capacity_added",
        "co2_emissions",
        "air_quality_index",
    ]:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")

    df = df.dropna(subset=["state", "state_abbr", "year"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)
    df = compute_derived_metrics(df)

    base_columns = [
        "state",
        "state_abbr",
        "date",
        "year",
        "energy_consumption",
        "energy_production",
        "solar_capacity_added",
        "co2_emissions",
        "air_quality_index",
        "clean_ratio",
        "emissions_intensity",
        "solar_growth_rate",
        "impact_gap_flag",
    ]
    return df[base_columns]


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_ratio"] = safe_divide(df["solar_capacity_added"], df["energy_production"])
    df["emissions_intensity"] = safe_divide(df["co2_emissions"], df["energy_consumption"])
    df["solar_growth_rate"] = (
        df.groupby("state")["solar_capacity_added"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    co2_change = df.groupby("state")["co2_emissions"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df["impact_gap_flag"] = (df["solar_growth_rate"] > 0.05) & (co2_change >= -0.005)
    return df


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


if __name__ == "__main__":
    path = build_and_save_dataset()
    df = pd.read_parquet(path)
    print(f"Wrote {len(df):,} rows across {df['state'].nunique()} states to {path}")
