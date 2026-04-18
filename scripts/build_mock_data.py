from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "heat_trace_state_timeseries.parquet"


STATES = [
    ("Alabama", "AL"),
    ("Alaska", "AK"),
    ("Arizona", "AZ"),
    ("Arkansas", "AR"),
    ("California", "CA"),
    ("Colorado", "CO"),
    ("Connecticut", "CT"),
    ("Delaware", "DE"),
    ("Florida", "FL"),
    ("Georgia", "GA"),
    ("Hawaii", "HI"),
    ("Idaho", "ID"),
    ("Illinois", "IL"),
    ("Indiana", "IN"),
    ("Iowa", "IA"),
    ("Kansas", "KS"),
    ("Kentucky", "KY"),
    ("Louisiana", "LA"),
    ("Maine", "ME"),
    ("Maryland", "MD"),
    ("Massachusetts", "MA"),
    ("Michigan", "MI"),
    ("Minnesota", "MN"),
    ("Mississippi", "MS"),
    ("Missouri", "MO"),
    ("Montana", "MT"),
    ("Nebraska", "NE"),
    ("Nevada", "NV"),
    ("New Hampshire", "NH"),
    ("New Jersey", "NJ"),
    ("New Mexico", "NM"),
    ("New York", "NY"),
    ("North Carolina", "NC"),
    ("North Dakota", "ND"),
    ("Ohio", "OH"),
    ("Oklahoma", "OK"),
    ("Oregon", "OR"),
    ("Pennsylvania", "PA"),
    ("Rhode Island", "RI"),
    ("South Carolina", "SC"),
    ("South Dakota", "SD"),
    ("Tennessee", "TN"),
    ("Texas", "TX"),
    ("Utah", "UT"),
    ("Vermont", "VT"),
    ("Virginia", "VA"),
    ("Washington", "WA"),
    ("West Virginia", "WV"),
    ("Wisconsin", "WI"),
    ("Wyoming", "WY"),
]


HIGH_SOLAR_STATES = {"CA", "TX", "FL", "AZ", "NV", "NC", "NY", "NJ", "MA", "CO", "NM", "UT"}
FOSSIL_HEAVY_STATES = {"TX", "LA", "WY", "WV", "ND", "OK", "PA", "OH", "IN", "KY", "AL"}
LOW_POLLUTION_STATES = {"VT", "ME", "NH", "OR", "WA", "ID", "HI", "RI"}


def build_mock_dataset(seed: int = 42) -> pd.DataFrame:
    """Create a realistic-enough state/year panel for a hackathon prototype."""
    rng = np.random.default_rng(seed)
    years = np.arange(2015, 2026)
    rows: list[dict[str, object]] = []

    for state, abbr in STATES:
        population_scale = rng.lognormal(mean=1.2, sigma=0.65)
        fossil_bias = 1.25 if abbr in FOSSIL_HEAVY_STATES else rng.uniform(0.75, 1.05)
        solar_bias = 1.8 if abbr in HIGH_SOLAR_STATES else rng.uniform(0.45, 1.05)
        air_bias = -8 if abbr in LOW_POLLUTION_STATES else (8 if abbr in FOSSIL_HEAVY_STATES else rng.uniform(-4, 5))

        base_consumption = rng.uniform(420, 2_800) * population_scale
        base_production = base_consumption * rng.uniform(0.65, 1.45) * fossil_bias
        base_solar = rng.uniform(15, 180) * solar_bias
        base_co2 = base_consumption * rng.uniform(0.28, 0.55) * fossil_bias
        base_aqi = np.clip(rng.normal(48 + air_bias, 9), 20, 95)

        solar_momentum = rng.uniform(0.10, 0.27) * solar_bias
        efficiency_trend = rng.uniform(-0.018, 0.006)
        production_trend = rng.uniform(-0.004, 0.018)
        consumption_trend = rng.uniform(-0.002, 0.016)
        aqi_trend = rng.uniform(-1.4, 0.8)

        # A small set of states deliberately create "impact gap" examples:
        # solar grows, but emissions do not yet fall.
        gap_state = abbr in {"TX", "LA", "OH", "PA", "IN", "KY"} or rng.random() < 0.12
        co2_trend = rng.uniform(0.002, 0.018) if gap_state else efficiency_trend

        for idx, year in enumerate(years):
            noise = lambda scale: rng.normal(1.0, scale)
            consumption = base_consumption * ((1 + consumption_trend) ** idx) * noise(0.035)
            production = base_production * ((1 + production_trend) ** idx) * noise(0.045)
            solar = base_solar * ((1 + solar_momentum) ** idx) * noise(0.075)
            co2 = base_co2 * ((1 + co2_trend) ** idx) * noise(0.035)
            aqi = base_aqi + (aqi_trend * idx) + rng.normal(0, 3.0)

            rows.append(
                {
                    "state": state,
                    "state_abbr": abbr,
                    "date": pd.Timestamp(year=year, month=1, day=1),
                    "year": int(year),
                    "energy_consumption": max(consumption, 50),
                    "energy_production": max(production, 25),
                    "solar_capacity_added": max(solar, 1),
                    "co2_emissions": max(co2, 5),
                    "air_quality_index": float(np.clip(aqi, 12, 135)),
                }
            )

    df = pd.DataFrame(rows).sort_values(["state", "date"]).reset_index(drop=True)
    return compute_derived_metrics(df)


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["state", "date"])
    df["clean_ratio"] = df["solar_capacity_added"] / df["energy_production"]
    df["emissions_intensity"] = df["co2_emissions"] / df["energy_consumption"]
    df["solar_growth_rate"] = (
        df.groupby("state")["solar_capacity_added"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    co2_change = df.groupby("state")["co2_emissions"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df["impact_gap_flag"] = (df["solar_growth_rate"] > 0.05) & (co2_change >= -0.005)

    numeric_cols = [
        "energy_consumption",
        "energy_production",
        "solar_capacity_added",
        "co2_emissions",
        "air_quality_index",
        "clean_ratio",
        "emissions_intensity",
        "solar_growth_rate",
    ]
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


def save_mock_dataset(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_mock_dataset()
    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    path = save_mock_dataset()
    df = pd.read_parquet(path)
    print(f"Wrote {len(df):,} rows for {df['state'].nunique()} states to {path}")


if __name__ == "__main__":
    main()
