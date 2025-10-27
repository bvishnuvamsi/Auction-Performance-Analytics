import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- DATA ----------
@st.cache_data
def load_data():
    # prefer processed path; fall back to cwd
    candidates = [
        "data/processed/auction_cleaned.csv",
        "auction_cleaned.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("auction_cleaned.csv not found in data/processed/ or project root.")

    # ensure column names expected from your pipeline
    # sold_year may be float with -1 sentinel
    if "sold_year" in df.columns:
        # normalize types
        df["sold_year"] = pd.to_numeric(df["sold_year"], errors="coerce")
    else:
        # derive if not present
        if "soldtime" in df.columns:
            df["soldtime"] = pd.to_datetime(df["soldtime"], errors="coerce")
            df["sold_year"] = df["soldtime"].dt.year
        else:
            df["sold_year"] = np.nan

    # standardize strings if needed (your cleaned file should already be tidy)
    for c in ["artist", "material", "country", "dominantcolor"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # filter out invalid/placeholder years for the slider domain
    valid_years = df.loc[(df["sold_year"].notna()) & (df["sold_year"] > 0), "sold_year"]
    year_min = int(valid_years.min()) if not valid_years.empty else 1800
    year_max = int(valid_years.max()) if not valid_years.empty else 2025

    return df, year_min, year_max

df, YEAR_MIN, YEAR_MAX = load_data()

def ensure_area(df_):
    if "area" not in df_.columns and {"height","width"}.issubset(df_.columns):
        df_ = df_.copy()
        df_["area"] = df_["height"] * df_["width"]
    return df_


# ---------- UI ----------
st.set_page_config(page_title="Auction Performance Dashboard", layout="wide")
st.title("🎨 Auction Performance Dashboard")

with st.sidebar:
    st.header("Filters")

    # Year filter (sold_year only; rows without sold_year are excluded from year filtering)
    year_range = st.slider(
        "Sold Year Range",
        min_value=YEAR_MIN, max_value=YEAR_MAX,
        value=(YEAR_MIN, YEAR_MAX), step=1
    )

    # Artist multiselect (top by count for usability)
    if "artist" in df.columns:
        top_artists = (
            df["artist"].value_counts().head(50).index.tolist()
        )
        selected_artists = st.multiselect(
            "Artists (Top 50 by Count)",
            options=top_artists,
            default=[],
            placeholder="Leave empty = All"
        )
    else:
        selected_artists = []

    # Material multiselect
    if "material" in df.columns:
        materials = sorted(df["material"].dropna().unique().tolist())
        selected_materials = st.multiselect(
            "Material",
            options=materials,
            default=[],
            placeholder="Leave empty = All"
        )
    else:
        selected_materials = []

# ---------- FILTERING ----------
mask = pd.Series(True, index=df.index)

# Year range filter: apply only to valid sold_year
valid_year_mask = df["sold_year"].between(year_range[0], year_range[1], inclusive="both")
has_valid_year = df["sold_year"].notna() & (df["sold_year"] > 0)
mask &= (~has_valid_year) | (has_valid_year & valid_year_mask)  # keep rows with no year; filter valid ones by range

# Artist filter
if selected_artists and "artist" in df.columns:
    mask &= df["artist"].isin(selected_artists)

# Material filter
if selected_materials and "material" in df.columns:
    mask &= df["material"].isin(selected_materials)

dff = df.loc[mask].copy()

if dff.empty:
    st.warning("No data after filters. Adjust filters to see results.")
    st.stop()

# ---------- KPIs ----------
total_sales = dff["price"].sum()
lots = len(dff)
avg_price = dff["price"].mean()
med_price = dff["price"].median()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Sales ($)", f"{total_sales:,.0f}")
kpi2.metric("Lots", f"{lots:,}")
kpi3.metric("Average Price ($)", f"{avg_price:,.0f}")
kpi4.metric("Median Price ($)", f"{med_price:,.0f}")

st.markdown("---")

# ---------- Top 10 Artists by Sales ----------
if "artist" in dff.columns:
    top_sales_by_artist = (
        dff.groupby("artist", as_index=False)["price"].sum()
          .sort_values("price", ascending=False)
          .head(10)
    )
    fig_artist = px.bar(
        top_sales_by_artist,
        x="price", y="artist",
        orientation="h",
        title="Top 10 Artists by Total Sales",
        labels={"price": "Total Sales ($)", "artist": "Artist"},
    )
    fig_artist.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig_artist, use_container_width=True)
else:
    st.info("Artist column not found; skipping 'Top Artists' chart.")

# ---------- Average Price by Material ----------
if "material" in dff.columns:
    avg_by_material = (
        dff.groupby("material", as_index=False)["price"].mean()
          .sort_values("price", ascending=False)
    )
    fig_material = px.bar(
        avg_by_material.head(20),
        x="material", y="price",
        title="Average Price by Material",
        labels={"price": "Average Price ($)", "material": "Material"},
    )
    fig_material.update_xaxes(tickangle=45)
    st.plotly_chart(fig_material, use_container_width=True)
else:
    st.info("Material column not found; skipping 'Average Price by Material'.")

# ---------- Geographic Distribution (by Country) ----------
if "country" in dff.columns:
    sales_by_country = (
        dff.groupby("country", as_index=False)["price"].sum()
          .sort_values("price", ascending=False)
          .head(20)
    )
    fig_country = px.bar(
        sales_by_country,
        x="price", y="country",
        orientation="h",
        title="Geographic Distribution (Top Countries by Sales)",
        labels={"price": "Total Sales ($)", "country": "Country"},
    )
    fig_country.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig_country, use_container_width=True)
else:
    st.info("Country column not found; skipping 'Geographic Distribution'.")

# ---------- Value vs Brightness Scatter ----------
if {"brightness","price"}.issubset(dff.columns):
    # clamp extreme prices for visual readability (optional safety)
    fig_scatter = px.scatter(
        dff.sample(min(len(dff), 8000), random_state=42),
        x="brightness", y="price",
        color="dominantcolor" if "dominantcolor" in dff.columns else None,
        hover_data=["artist","material","country","sold_year"] if {"artist","material","country","sold_year"}.issubset(dff.columns) else None,
        title="Artwork Value vs. Brightness",
        labels={"brightness":"Brightness (0–255)", "price":"Price ($)"},
        opacity=0.4,
        trendline=None
    )
    fig_scatter.update_yaxes(type="log")
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Brightness or price column not found; skipping 'Value vs Brightness'.")

dff = ensure_area(dff)


# ===== Revenue Concentration – Top Artists’ Share =====
if {"artist","price"}.issubset(dff.columns):
    st.subheader("Revenue Concentration – Top Artists’ Share of Total Sales")

    top_n = st.slider("Top N artists", 5, 30, 10, step=1, key="artist_topn")
    chart_type = st.radio("Chart type", ["Donut", "Bar"], horizontal=True, key="artist_charttype")

    sales_by_artist = (
        dff.groupby("artist", as_index=False)["price"].sum()
          .sort_values("price", ascending=False)
          .head(top_n)
    )
    total_sel_sales = sales_by_artist["price"].sum()
    sales_by_artist["share"] = sales_by_artist["price"] / total_sel_sales

    if chart_type == "Donut":
        fig_artist_share = px.pie(
            sales_by_artist, values="price", names="artist",
            title=f"Top {top_n} Artists – Share of Total Sales", hole=0.4
        )
        fig_artist_share.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_artist_share, use_container_width=True)
    else:
        fig_artist_bar = px.bar(
            sales_by_artist, x="price", y="artist", orientation="h",
            title=f"Top {top_n} Artists – Total Sales",
            labels={"price":"Total Sales ($)","artist":"Artist"}
        )
        fig_artist_bar.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_artist_bar, use_container_width=True)
else:
    st.info("Artist column not found; skipping artist concentration.")

# ===== Country × Material – Average Price Heatmap =====
if {"country","material","price"}.issubset(dff.columns):
    st.subheader("Country × Material – Average Price")

    top_countries = st.slider("Number of countries", 5, 30, 15, step=1, key="heatmap_countries")
    top_materials = st.slider("Number of materials", 5, 30, 15, step=1, key="heatmap_materials")

    # stabilize averages by limiting to most represented categories
    top_country_list = dff["country"].value_counts().head(top_countries).index.tolist()
    top_material_list = dff["material"].value_counts().head(top_materials).index.tolist()

    df_hm = dff[dff["country"].isin(top_country_list) & dff["material"].isin(top_material_list)]
    pivot = df_hm.pivot_table(index="country", columns="material", values="price", aggfunc="mean")

    fig_hm = px.imshow(
        pivot, aspect="auto", color_continuous_scale="Viridis",
        labels=dict(color="Avg Price ($)"),
        title="Average Price by Country × Material"
    )
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Need 'country', 'material', and 'price' for heatmap.")

tab_mat, tab_area, tab_artist_avg, tab_country_avg, tab_brightness = st.tabs(
    ["Price by Material", "Area vs Price", "Top Artists (Avg Price)", "Country-wise Avg Price", "Price vs Brightness"]
)


# ---- Price by Material (total + average toggle) ----
with tab_mat:
    if "material" in dff.columns:
        mode = st.radio("Aggregate", ["Total Sales", "Average Price"], horizontal=True)
        if mode == "Total Sales":
            g = (dff.groupby("material", as_index=False)["price"].sum()
                   .sort_values("price", ascending=False))
            fig = px.bar(g.head(30), x="material", y="price",
                         title="Total Sales by Material",
                         labels={"price":"Total Sales ($)","material":"Material"})
        else:
            g = (dff.groupby("material", as_index=False)["price"].mean()
                   .sort_values("price", ascending=False))
            fig = px.bar(g.head(30), x="material", y="price",
                         title="Average Price by Material",
                         labels={"price":"Average Price ($)","material":"Material"})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Material column not found.")

# ---- Area vs Price ----
with tab_area:
    if {"area","price"}.issubset(dff.columns):
        sample_n = min(len(dff), 10000)
        fig = px.scatter(
            dff.sample(sample_n, random_state=42),
            x="area", y="price",
            hover_data=[c for c in ["artist","material","country","sold_year"] if c in dff.columns],
            title="Area vs Price",
            labels={"area":"Area (sq units)","price":"Price ($)"},
            opacity=0.4
        )
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'area' and 'price' columns for Area vs Price.")

# ---- Top Artists by Average Sale Price ----
with tab_artist_avg:
    if {"artist","price"}.issubset(dff.columns):
        g = (dff.groupby("artist", as_index=False)["price"].mean()
               .sort_values("price", ascending=False)
               .head(15))
        fig = px.bar(g, x="price", y="artist", orientation="h",
                     title="Top Artists by Average Sale Price",
                     labels={"price":"Average Price ($)","artist":"Artist"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'artist' and 'price' columns.")

# ---- Country-wise Average Price ----
with tab_country_avg:
    if {"country","price"}.issubset(dff.columns):
        g = (dff.groupby("country", as_index=False)["price"].mean()
               .sort_values("price", ascending=False)
               .head(20))
        fig = px.bar(g, x="price", y="country", orientation="h",
                     title="Country-wise Average Price",
                     labels={"price":"Average Price ($)","country":"Country"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'country' and 'price' columns.")

# ---- Price vs Brightness (keep log-y) ----
with tab_brightness:
    if {"brightness","price"}.issubset(dff.columns):
        sample_n = min(len(dff), 10000)
        color_col = "dominantcolor" if "dominantcolor" in dff.columns else None
        fig = px.scatter(
            dff.sample(sample_n, random_state=42),
            x="brightness", y="price",
            color=color_col,
            hover_data=[c for c in ["artist","material","country","sold_year"] if c in dff.columns],
            title="Price vs Brightness",
            labels={"brightness":"Brightness (0–255)","price":"Price ($)"},
            opacity=0.4
        )
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'brightness' and 'price' columns.")
