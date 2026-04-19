from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_energy_data.csv"
HOME_MAP_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "homepage_mapdata.csv"
BTU_PER_KWH = 3412.142

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


METRIC_LABELS = {
    "energy_btu": "Energy consumption (BTU)",
    "energy_kwh": "Energy consumption (kWh)",
    "year_over_year_change": "Year-over-year change",
}

CHART_METRICS = ["energy_btu", "energy_kwh", "year_over_year_change"]

STATE_NAME_BY_ABBR = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "US": "United States",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}

US_STATE_ABBRS = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


def apply_light_mode_background() -> None:
    """Use a warmer page background in light mode while preserving dark mode."""
    import streamlit as st

    is_dark = st.context.theme.get("type") == "dark"
    page_bg = "var(--background-color, #0e1117)" if is_dark else "hsl(38 48% 88%)"
    surface_bg = "var(--background-color, #0e1117)" if is_dark else "hsl(38 48% 92% / 0.94)"
    text_color = "var(--text-color, #fafafa)" if is_dark else "hsl(222 47% 11%)"
    border_color = "rgba(250, 250, 250, 0.14)" if is_dark else "hsl(34 24% 72%)"

    st.markdown(
        f"""
        <style>
        :root {{
            --heat-page-bg: {page_bg};
            --heat-surface-bg: {surface_bg};
            --heat-text-color: {text_color};
            --heat-border-color: {border_color};
            --heat-theme-transition:
                background-color 240ms ease-in-out,
                color 240ms ease-in-out,
                border-color 240ms ease-in-out;
        }}

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"] {{
            background-color: var(--heat-page-bg);
            color: var(--heat-text-color);
            transition: var(--heat-theme-transition);
        }}

        [data-testid="stHeader"],
        [data-testid="stToolbar"] {{
            background-color: var(--heat-surface-bg);
            transition: var(--heat-theme-transition);
        }}

        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {{
            background-color: var(--heat-surface-bg);
            color: var(--heat-text-color);
            transition: var(--heat-theme-transition);
        }}

        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stSelectbox"] {{
            transition: var(--heat-theme-transition);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_energy_data() -> Path:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected energy dataset at {DATA_PATH}")
    return DATA_PATH


def load_state_timeseries() -> pd.DataFrame:
    ensure_energy_data()
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"state": "state_abbr"})
    df = df[df["state_abbr"] != "US"].copy()
    df["state"] = df["state_abbr"].map(STATE_NAME_BY_ABBR).fillna(df["state_abbr"])
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01")
    df["energy_btu"] = pd.to_numeric(df["energy_btu"], errors="coerce")
    df = df.dropna(subset=["energy_btu", "year"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    return compute_metrics(df)


def load_homepage_map_data() -> pd.DataFrame:
    ensure_energy_data()

    energy_df = pd.read_csv(DATA_PATH)
    energy_df = energy_df.rename(columns={"state": "state_abbr"}).copy()
    energy_df["state_abbr"] = energy_df["state_abbr"].str.upper().str.strip()
    energy_df["energy_consumption"] = pd.to_numeric(energy_df["energy_btu"], errors="coerce")
    latest_year = int(energy_df["year"].max())
    energy_latest = energy_df[
        (energy_df["year"] == latest_year) & (energy_df["state_abbr"].isin(US_STATE_ABBRS))
    ][["state_abbr", "energy_consumption"]]

    solar_df = pd.read_csv(HOME_MAP_DATA_PATH)
    solar_df = solar_df.rename(columns={"state": "state_abbr"}).copy()
    solar_df["state_abbr"] = solar_df["state_abbr"].str.upper().str.strip()
    solar_df["solar_production"] = pd.to_numeric(solar_df["solar_production"], errors="coerce")
    solar_df = solar_df[["state_abbr", "solar_production"]].dropna(subset=["state_abbr"])

    map_df = energy_latest.merge(solar_df, on="state_abbr", how="outer")
    map_df = map_df[map_df["state_abbr"].isin(US_STATE_ABBRS)].copy()
    map_df["state"] = map_df["state_abbr"].map(STATE_NAME_BY_ABBR).fillna(map_df["state_abbr"])
    map_df["has_energy"] = map_df["energy_consumption"].notna()
    map_df["has_solar"] = map_df["solar_production"].notna()
    return map_df.sort_values("state").reset_index(drop=True)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["state", "date"])
    df["energy_kwh"] = df["energy_btu"] / BTU_PER_KWH
    df["year_over_year_change"] = df.groupby("state")["energy_btu"].pct_change().fillna(0)
    return df


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(df["year"].max())
    return df[df["year"] == latest_year].copy()


def format_metric(value: float, metric: str) -> str:
    if metric == "year_over_year_change":
        return f"{value:.1%}"
    return f"{value:,.0f}"


def metric_help(metric: str) -> str:
    descriptions = {
        "energy_btu": "Annual state energy consumption from the clean dataset, measured in BTUs.",
        "energy_kwh": "Annual state energy consumption converted from BTUs into kilowatt-hours.",
        "year_over_year_change": "Percent change in annual state energy consumption versus the prior year.",
    }
    return descriptions.get(metric, "")


def render_metric_cards(latest: pd.DataFrame) -> None:
    import streamlit as st

    total_energy = latest["energy_btu"].sum()
    avg_energy = latest["energy_btu"].mean()
    top_state = latest.nlargest(1, "energy_btu").iloc[0]
    median_yoy = latest["year_over_year_change"].median()

    cols = st.columns(4)
    cols[0].metric("Total energy", f"{total_energy:,.0f}")
    cols[1].metric("Average by state", f"{avg_energy:,.0f}")
    cols[2].metric("Highest state", top_state["state"])
    cols[3].metric("Median YoY change", f"{median_yoy:.1%}")


def render_homepage_map_cards(map_df: pd.DataFrame) -> None:
    import streamlit as st

    energy_df = map_df.dropna(subset=["energy_consumption"])
    solar_df = map_df.dropna(subset=["solar_production"])
    overlap_count = int((map_df["has_energy"] & map_df["has_solar"]).sum())
    top_consumption = energy_df.nlargest(1, "energy_consumption").iloc[0]
    top_solar = solar_df.nlargest(1, "solar_production").iloc[0]

    top_row = st.columns(4)
    top_row[0].metric("States with energy data", f"{len(energy_df)}")
    top_row[1].metric("States with solar data", f"{len(solar_df)}")
    top_row[2].metric("States with both", f"{overlap_count}")
    top_row[3].metric("Total consumption (BTU)", f"{energy_df['energy_consumption'].sum():,.0f}")

    bottom_row = st.columns(2)
    bottom_row[0].metric("Top consumption", f"{top_consumption['state_abbr']} - {top_consumption['energy_consumption']:,.0f} BTU")
    bottom_row[1].metric("Top solar production", f"{top_solar['state_abbr']} - {top_solar['solar_production']:,.0f} kWh")


def energy_solar_overlay_map(map_df: pd.DataFrame) -> go.Figure:
    energy_df = map_df.dropna(subset=["energy_consumption"]).copy()
    solar_df = map_df.dropna(subset=["solar_production"]).copy()
    energy_df["solar_display"] = energy_df["solar_production"].map(
        lambda value: f"{value:,.0f}" if pd.notna(value) else "Not available"
    )
    solar_df["energy_display"] = solar_df["energy_consumption"].map(
        lambda value: f"{value:,.0f}" if pd.notna(value) else "Not available"
    )
    data_states = set(energy_df["state_abbr"]) | set(solar_df["state_abbr"])
    missing_states = [abbr for abbr in US_STATE_ABBRS if abbr not in data_states]

    fig = go.Figure()

    fig.add_trace(
        go.Choropleth(
            locations=missing_states,
            locationmode="USA-states",
            z=[1] * len(missing_states),
            colorscale=[[0, "#9ca3af"], [1, "#9ca3af"]],
            showscale=False,
            marker_line_color="rgba(255, 255, 255, 0.75)",
            marker_line_width=0.8,
            hovertemplate="%{location}<br>No overlapping data<extra></extra>",
            name="No overlapping data",
        )
    )

    fig.add_trace(
        go.Choropleth(
            locations=energy_df["state_abbr"],
            locationmode="USA-states",
            z=energy_df["energy_consumption"],
            colorscale="YlOrRd",
            marker_line_color="rgba(255, 255, 255, 0.85)",
            marker_line_width=0.9,
            colorbar=dict(title="Energy consumption (BTU)"),
            customdata=np.stack([energy_df["solar_display"]], axis=-1),
            hovertemplate=(
                "%{location}<br>"
                "Energy consumption: %{z:,.0f}<br>"
                "Solar production: %{customdata[0]}<extra></extra>"
            ),
            name="Energy consumption",
        )
    )

    if not solar_df.empty:
        max_solar = solar_df["solar_production"].max()
        bubble_sizes = 10 + 42 * np.sqrt(solar_df["solar_production"] / max_solar)
        fig.add_trace(
            go.Scattergeo(
                locations=solar_df["state_abbr"],
                locationmode="USA-states",
                mode="markers",
                marker=dict(
                    size=bubble_sizes,
                    color="rgba(34, 197, 94, 0.62)",
                    line=dict(color="rgba(17, 24, 39, 0.85)", width=1.2),
                ),
                customdata=np.stack([solar_df["solar_production"], solar_df["energy_display"]], axis=-1),
                hovertemplate=(
                    "%{location}<br>"
                    "Solar production: %{customdata[0]:,.0f}<br>"
                    "Energy consumption: %{customdata[1]}<extra></extra>"
                ),
                name="Solar production",
            )
        )

    fig.update_layout(
        dragmode=False,
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0.0),
    )
    fig.update_geos(
        scope="usa",
        bgcolor="rgba(0,0,0,0)",
        showland=False,
        showocean=False,
        showlakes=False,
        showcoastlines=False,
        showcountries=False,
        showsubunits=False,
        showframe=False,
    )
    return fig


def homepage_rankings(map_df: pd.DataFrame, metric: str, n: int = 5) -> pd.DataFrame:
    return (
        map_df.dropna(subset=[metric])
        .nlargest(n, metric)[["state_abbr", metric]]
        .rename(columns={"state_abbr": "State", metric: "Value"})
        .reset_index(drop=True)
    )


def choropleth_map(latest: pd.DataFrame, metric: str) -> go.Figure:
    color_scale = "RdYlGn" if metric == "year_over_year_change" else "Viridis"
    fig = px.choropleth(
        latest,
        locations="state_abbr",
        locationmode="USA-states",
        scope="usa",
        color=metric,
        hover_name="state",
        hover_data={
            "state_abbr": False,
            metric: ":.1%" if metric == "year_over_year_change" else ":,.0f",
        },
        color_continuous_scale=color_scale,
        labels={metric: METRIC_LABELS.get(metric, metric)},
    )
    fig.update_traces(marker_line_color="rgba(255, 255, 255, 0.8)", marker_line_width=0.8)
    fig.update_layout(
        dragmode=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showland=False,
        showocean=False,
        showlakes=False,
        showcoastlines=False,
        showcountries=False,
        showsubunits=False,
        showframe=False,
    )
    return fig


def ranked_states(latest: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    top = latest.nlargest(n, metric).assign(rank_group="Top")
    bottom = latest.nsmallest(n, metric).assign(rank_group="Bottom")
    ranked = pd.concat([top, bottom], ignore_index=True)
    return ranked[["rank_group", "state", "state_abbr", metric]].rename(
        columns={
            "rank_group": "Group",
            "state": "State",
            "state_abbr": "Abbr",
            metric: METRIC_LABELS.get(metric, metric),
        }
    )


def time_series_chart(state_df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.line(
        state_df,
        x="date",
        y=metric,
        markers=True,
        labels={"date": "Year", metric: METRIC_LABELS.get(metric, metric)},
        title=METRIC_LABELS.get(metric, metric),
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(height=310, margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified")
    return fig


def state_summary(state_df: pd.DataFrame) -> str:
    state_df = state_df.sort_values("year")
    state = state_df["state"].iloc[0]
    first = state_df.iloc[0]
    latest = state_df.iloc[-1]
    energy_change = pct_change(first["energy_btu"], latest["energy_btu"])
    avg_growth = state_df["year_over_year_change"].iloc[1:].mean() if len(state_df) > 1 else 0.0
    peak_year = int(state_df.loc[state_df["energy_btu"].idxmax(), "year"])
    latest_rank = int(
        state_df[state_df["year"] == latest["year"]]["energy_btu"].rank(method="dense", ascending=False).iloc[0]
    )

    return (
        f"{state} used {energy_change:.0%} more energy in {int(latest['year'])} than in {int(first['year'])}. "
        f"Its average year-over-year change across the series is {avg_growth:.1%}, and its peak energy use "
        f"occurred in {peak_year} at about {state_df['energy_kwh'].max():,.0f} kWh."
    )


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old

def build_insights(df: pd.DataFrame) -> pd.DataFrame:
    latest = latest_snapshot(df)
    national_median = latest["energy_btu"].median()
    records = []

    for state, state_df in df.groupby("state", sort=True):
        state_df = state_df.sort_values("year")
        latest_row = state_df.iloc[-1]
        recent_window = state_df.tail(min(5, len(state_df)))
        energy_change = pct_change(recent_window.iloc[0]["energy_btu"], recent_window.iloc[-1]["energy_btu"])

        if energy_change > 0.15:
            records.append(
                {
                    "State": state,
                    "Abbr": latest_row["state_abbr"],
                    "Insight": "Fast recent energy growth",
                    "Energy change, recent": energy_change,
                    "Latest energy": latest_row["energy_btu"],
                }
            )
        if latest_row["energy_btu"] > national_median * 2:
            records.append(
                {
                    "State": state,
                    "Abbr": latest_row["state_abbr"],
                    "Insight": "Very high current energy use",
                    "Energy change, recent": energy_change,
                    "Latest energy": latest_row["energy_btu"],
                }
            )

    return pd.DataFrame(records)


def insight_count_chart(insights: pd.DataFrame) -> go.Figure:
    if insights.empty:
        fig = go.Figure()
        fig.update_layout(title="No insight flags found", height=320)
        return fig

    counts = insights["Insight"].value_counts().reset_index()
    counts.columns = ["Insight", "States"]
    fig = px.bar(counts, x="States", y="Insight", orientation="h", text="States", color="Insight")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def dataframe_with_formats(df: pd.DataFrame, percent_columns: Iterable[str] = ()) -> object:
    styled = df.style
    for col in percent_columns:
        if col in df.columns:
            styled = styled.format({col: "{:.1%}"})
    numeric_formats = {
        "Latest energy": "{:,.0f}",
    }
    existing = {key: value for key, value in numeric_formats.items() if key in df.columns}
    if existing:
        styled = styled.format(existing)
    return styled
