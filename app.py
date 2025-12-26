import re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Education Indicators Dashboard", layout="wide")
st.title("Education Indicators Dashboard")

# Altair default: limit rows; disable so it works with larger data
alt.data_transformers.disable_max_rows()

# ---------- Load + reshape ----------
@st.cache_data
def load_edstats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Detect year columns like "1970"..."2015"
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    # Keep only 1970–2015
    year_cols = [c for c in year_cols if 1970 <= int(c) <= 2015]
    year_cols = sorted(year_cols, key=lambda x: int(x))

    required_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing columns in CSV: {missing_required}")

    # Wide -> long
    long_df = df[required_cols + year_cols].melt(
        id_vars=required_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )

    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce").astype("Int64")

    # Convert values, treat ".." and non-numeric as missing
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")

    long_df["Missing"] = long_df["Value"].isna()

    # Drop rows with no year (safety)
    long_df = long_df.dropna(subset=["Year"])
    long_df["Year"] = long_df["Year"].astype(int)

    return long_df


# ---------- Input ----------
# Put your CSV in the same folder as app.py, or update this path
CSV_URL = "https://drive.google.com/uc?export=download&id=1xUH-nEyU-yTTvrgr5f1xQ0izhCHo3JAo"

try:
    data = load_edstats(CSV_URL)
except Exception as e:
    st.error(f"Could not load CSV from Google Drive. Details: {e}")
    st.stop()


# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

countries = sorted(data["Country Name"].dropna().unique().tolist())
indicators = sorted(data["Indicator Name"].dropna().unique().tolist())

default_country = "Arab World" if "Arab World" in countries else countries[0]
country = st.sidebar.selectbox("Country", countries, index=countries.index(default_country))

# narrow indicators list based on country to reduce “no data”
ind_for_country = sorted(data.loc[data["Country Name"] == country, "Indicator Name"].dropna().unique().tolist())
default_indicator = None
for candidate in [
    "School enrollment, primary, female (% gross)",
    "School enrollment, primary, female (% net)",
    "Adjusted net enrolment rate, primary, female (%)",
]:
    if candidate in ind_for_country:
        default_indicator = candidate
        break

indicator = st.sidebar.selectbox(
    "Indicator",
    ind_for_country,
    index=ind_for_country.index(default_indicator) if default_indicator in ind_for_country else 0
)

year_min, year_max = st.sidebar.slider("Year range", 1970, 2015, (1970, 2015))

df = data[
    (data["Country Name"] == country) &
    (data["Indicator Name"] == indicator) &
    (data["Year"].between(year_min, year_max))
].copy()

# ---------- Basic checks ----------
total_years = (year_max - year_min + 1)
non_missing = df.loc[~df["Missing"]].shape[0]

# ---------- Brushing selection ----------
# Brush on year axis; this is the “brushing” that links charts
brush = alt.selection_interval(encodings=["x"], name="brush")

# Point selection (click a year point)
pt = alt.selection_point(encodings=["x"], name="pt", toggle=True)

# ---------- Base charts data ----------
base = alt.Chart(df).properties(height=280)

# 1) Trend line with brush + point selection + hover
trend = (
    base.mark_line(point=True)
    .encode(
        x=alt.X("Year:Q", title="Year"),
        y=alt.Y("Value:Q", title="Value"),
        tooltip=[
            alt.Tooltip("Year:Q"),
            alt.Tooltip("Value:Q", format=".2f"),
            alt.Tooltip("Missing:N"),
        ],
        opacity=alt.condition(brush, alt.value(1.0), alt.value(0.25)),
    )
    .add_params(brush, pt)
    .interactive()  # zoom + pan
    .properties(title=f"Trend (click or brush years): {indicator} — {country}")
)

# 2) Regression model view (Altair computes regression in chart)
# Uses only non-missing values, and filters by brush (linking)
scatter = (
    alt.Chart(df.dropna(subset=["Value"]))
    .mark_circle(size=60)
    .encode(
        x=alt.X("Year:Q", title="Year"),
        y=alt.Y("Value:Q", title="Value"),
        tooltip=[alt.Tooltip("Year:Q"), alt.Tooltip("Value:Q", format=".2f")],
        opacity=alt.condition(brush, alt.value(1.0), alt.value(0.15)),
    )
    .transform_filter(brush)
    .properties(height=280, title="Model view: linear regression on brushed years")
)

reg_line = (
    alt.Chart(df.dropna(subset=["Value"]))
    .transform_filter(brush)
    .transform_regression("Year", "Value", method="linear")
    .mark_line()
    .encode(x="Year:Q", y="Value:Q")
)

model_chart = (scatter + reg_line).interactive()

# 3) Missing rate by year (linked to brush)
missing_rate = (
    alt.Chart(df.assign(MissingInt=df["Missing"].astype(int)))
    .mark_bar()
    .encode(
        x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean(MissingInt):Q", title="Missing rate (0–1)"),
        tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("mean(MissingInt):Q", format=".2f")],
        opacity=alt.condition(brush, alt.value(1.0), alt.value(0.25)),
    )
    .properties(height=240, title="Missing rate by year")
)


# 4) Distribution of values (linked to brush)
hist = (
    alt.Chart(df.dropna(subset=["Value"]))
    .transform_filter(brush)
    .mark_bar()
    .encode(
        x=alt.X("Value:Q", bin=alt.Bin(maxbins=30), title="Value (binned)"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q")],
    )
    .properties(height=240, title="Value distribution (non-missing, brushed)")
)

# Optional: A small KPI row
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Selected years", f"{year_min}–{year_max}", f"{total_years} years")
kpi2.metric("Non-missing points", f"{non_missing}", f"{(non_missing/total_years)*100:.1f}% coverage")
kpi3.metric("Missing points", f"{total_years - non_missing}", f"{((total_years-non_missing)/total_years)*100:.1f}% missing")

# ---------- Layout (one screen) ----------
colA, colB = st.columns([1.2, 1.0])
with colA:
    st.altair_chart(trend, use_container_width=True)
with colB:
    st.altair_chart(model_chart, use_container_width=True)

colC, colD = st.columns(2)
with colC:
    st.altair_chart(missing_rate, use_container_width=True)
with colD:
    st.altair_chart(hist, use_container_width=True)

st.caption(
    "How to use: drag on the top chart to brush a year range. All other charts update (brushing & linking). "
    "Use mouse wheel/trackpad to zoom and drag to pan."
)
