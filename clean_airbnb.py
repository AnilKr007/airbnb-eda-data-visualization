"""
Airbnb Chicago vs New Orleans - data cleaning & transformation
----------------------------------------------------------------
Input : Inside Airbnb "listings.csv" for each city (summary OR detailed file,
        plain .csv or .csv.gz) from http://insideairbnb.com/get-the-data/
Output: airbnb_clean.csv      -> load this into Power BI Desktop
        key_numbers.md        -> real figures to quote in your presentation
        cleaning_log.txt      -> what was removed/fixed (show this in the video)

Usage:
    python clean_airbnb.py --chicago chicago_listings.csv --neworleans neworleans_listings.csv
"""
import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------- loading
def load_city(path, city):
    df = pd.read_csv(path, low_memory=False)
    log(f"[{city}] loaded {len(df):,} rows, {df.shape[1]} columns from {Path(path).name}")
    df["city"] = city
    return df


# ---------------------------------------------------------------- price
def clean_price(s):
    # detailed file stores "$1,250.00"; summary file stores 1250
    return pd.to_numeric(
        s.astype(str).str.replace(r"[$,\s]", "", regex=True).replace({"": np.nan, "nan": np.nan}),
        errors="coerce",
    )


# ---------------------------------------------------------------- name parsing
# Newer Inside Airbnb summary files put structured text in `name`, e.g.
# "Rental unit in Chicago · ★4.92 · 2 bedrooms · 3 beds · 1.5 baths"
RX_PROP = re.compile(r"^\s*(.+?)\s+in\s+", re.I)
RX_RATING = re.compile(r"★\s*(\d+(?:\.\d+)?|New)", re.I)
RX_BEDROOMS = re.compile(r"(\d+)\s+bedrooms?", re.I)
RX_STUDIO = re.compile(r"\bstudio\b", re.I)
RX_BEDS = re.compile(r"(\d+)\s+beds?\b", re.I)
RX_BATHS = re.compile(r"(\d+(?:\.\d+)?)\s+(?:shared\s+|private\s+)?(?:half-)?baths?", re.I)


def parse_name(name):
    name = "" if pd.isna(name) else str(name)
    out = {}
    m = RX_PROP.search(name)
    out["name_property_type"] = m.group(1).strip() if (m and "·" in name) else np.nan
    m = RX_RATING.search(name)
    if m and m.group(1).lower() != "new":
        out["name_rating"] = float(m.group(1))
    else:
        out["name_rating"] = np.nan
    m = RX_BEDROOMS.search(name)
    out["name_bedrooms"] = 0 if RX_STUDIO.search(name) else (int(m.group(1)) if m else np.nan)
    m = RX_BEDS.search(name)
    out["name_beds"] = int(m.group(1)) if m else np.nan
    m = RX_BATHS.search(name)
    out["name_baths"] = float(m.group(1)) if m else np.nan
    return out


# ---------------------------------------------------------------- neighbourhood "group & replace"
def normalise_neighbourhood(s):
    base = (
        s.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
        .replace({"Nan": np.nan})
    )
    # fuzzy group: map rare spellings onto the most frequent close match
    counts = base.value_counts()
    canonical = list(counts.index)
    mapping = {}
    for name in canonical:
        if name in mapping:
            continue
        matches = difflib.get_close_matches(name, canonical, n=10, cutoff=0.92)
        best = max(matches, key=lambda x: counts[x])
        for mname in matches:
            if mname not in mapping:
                mapping[mname] = best
    changed = {k: v for k, v in mapping.items() if k != v}
    for k, v in changed.items():
        log(f"  neighbourhood grouped: '{k}' -> '{v}'")
    return base.map(lambda x: mapping.get(x, x))


def property_group(pt):
    if pd.isna(pt):
        return "Unknown"
    p = str(pt).lower()
    if any(k in p for k in ["hotel", "hostel", "boutique"]):
        return "Hotel / Hostel"
    if any(k in p for k in ["condo", "apartment", "rental unit", "loft", "serviced"]):
        return "Apartment / Condo"
    if any(k in p for k in ["townhouse", "home", "house", "bungalow", "cottage", "villa", "cabin"]):
        return "House / Townhouse"
    if "guest" in p:
        return "Guest Suite / Guesthouse"
    return "Other"


# ---------------------------------------------------------------- main transform
def transform(df):
    n0 = len(df)

    # 1. duplicates
    df = df.drop_duplicates(subset=["city", "id"])
    log(f"duplicate listing ids removed: {n0 - len(df):,}")

    # 2. price
    df["price"] = clean_price(df["price"])
    miss = df["price"].isna().sum()
    zero = (df["price"] <= 0).sum()
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    log(f"rows with missing price removed: {miss:,}; zero/negative price removed: {zero:,}")

    # 3. outliers flagged per city (99th pct) - kept, but excluded from averages via filter
    p99 = df.groupby("city")["price"].transform(lambda s: s.quantile(0.99))
    df["is_price_outlier"] = df["price"] > p99
    log(f"price outliers flagged (> city 99th percentile): {int(df['is_price_outlier'].sum()):,}")

    # 4. types & nulls
    df["last_review"] = pd.to_datetime(df.get("last_review"), errors="coerce")
    df["reviews_per_month"] = pd.to_numeric(df.get("reviews_per_month"), errors="coerce").fillna(0)
    for c in ["number_of_reviews", "minimum_nights", "availability_365", "calculated_host_listings_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["host_name"] = df.get("host_name", pd.Series(index=df.index)).fillna("Unknown").astype(str).str.strip()
    df["room_type"] = df["room_type"].astype(str).str.strip()

    # 5. neighbourhood clean (detailed file uses neighbourhood_cleansed)
    nb = df["neighbourhood"] if "neighbourhood" in df.columns else pd.Series(np.nan, index=df.index)
    if "neighbourhood_cleansed" in df.columns:
        nb = df["neighbourhood_cleansed"].fillna(nb)
    df["neighbourhood"] = normalise_neighbourhood(nb)

    # 6. derived columns from descriptive text
    parsed = pd.DataFrame([parse_name(x) for x in df["name"]], index=df.index)
    df = pd.concat([df, parsed], axis=1)
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].fillna(df["name_property_type"])
    else:
        df["property_type"] = df["name_property_type"]
    df["property_type"] = df["property_type"].fillna("Unknown").astype(str).str.strip()
    df["property_group"] = df["property_type"].map(property_group)

    if "review_scores_rating" in df.columns:
        df["rating"] = pd.to_numeric(df["review_scores_rating"], errors="coerce")
        df["rating"] = df["rating"].where(df["rating"] <= 5, df["rating"] / 20)  # old 0-100 scale
        df["rating"] = df["rating"].fillna(df["name_rating"])
    else:
        df["rating"] = df["name_rating"]
    for src, dst in [("bedrooms", "bedrooms"), ("beds", "beds")]:
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce").fillna(df["name_" + dst])
        else:
            df[dst] = df["name_" + dst]
    baths = pd.to_numeric(df["bathrooms"], errors="coerce") if "bathrooms" in df.columns else pd.Series(np.nan, index=df.index)
    if "bathrooms_text" in df.columns:
        txt = df["bathrooms_text"].astype(str).str.lower()
        from_txt = txt.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
        from_txt = from_txt.where(~txt.str.contains("half-bath"), 0.5)
        baths = baths.fillna(from_txt)
    df["bathrooms"] = baths.fillna(df["name_baths"])

    # 7. host classification
    if "calculated_host_listings_count" not in df.columns or (df["calculated_host_listings_count"] == 0).all():
        df["calculated_host_listings_count"] = df.groupby("host_id")["id"].transform("count")
    hl = df["calculated_host_listings_count"]
    df["host_type"] = np.select(
        [hl == 1, hl.between(2, 5), hl > 5],
        ["Single-listing host", "Small multi-host (2-5)", "Commercial host (6+)"],
        default="Unknown",
    )

    # 8. bands for visuals
    df["price_band"] = pd.cut(
        df["price"], [0, 50, 100, 150, 250, 500, np.inf],
        labels=["<$50", "$50-100", "$100-150", "$150-250", "$250-500", "$500+"],
    ).astype(str)
    df["review_band"] = pd.cut(
        df["number_of_reviews"], [-1, 0, 10, 50, 100, np.inf],
        labels=["0", "1-10", "11-50", "51-100", "100+"],
    ).astype(str)
    df["availability_band"] = pd.cut(
        df["availability_365"], [-1, 0, 90, 180, 270, 365],
        labels=["Not available", "1-90 days", "91-180 days", "181-270 days", "271-365 days"],
    ).astype(str)
    df["stay_type"] = np.where(df["minimum_nights"] >= 30, "Long-term (30+ nights)", "Short-term (<30 nights)")
    df["rating_band"] = pd.cut(
        df["rating"], [0, 4.0, 4.5, 4.8, 5.01], labels=["<4.0", "4.0-4.5", "4.5-4.8", "4.8-5.0"], right=False
    ).astype(object).where(df["rating"].notna(), "No rating").astype(str)

    keep = [
        "id", "city", "name", "host_id", "host_name", "neighbourhood", "latitude", "longitude",
        "room_type", "property_type", "property_group", "price", "is_price_outlier", "price_band",
        "minimum_nights", "stay_type", "number_of_reviews", "review_band", "last_review",
        "reviews_per_month", "availability_365", "availability_band", "calculated_host_listings_count",
        "host_type", "rating", "rating_band", "bedrooms", "beds", "bathrooms",
    ]
    if "license" in df.columns:
        df["has_license"] = df["license"].notna() & (df["license"].astype(str).str.strip() != "")
        keep.append("has_license")
    log(f"final clean rows: {len(df):,}")
    return df[keep]


# ---------------------------------------------------------------- headline numbers
def key_numbers(df):
    d = df[~df["is_price_outlier"]]
    lines = ["# Key numbers (computed from your cleaned data — outliers excluded from price stats)\n"]
    for city, g in d.groupby("city"):
        full = df[df["city"] == city]
        lines.append(f"## {city}")
        lines.append(f"- Listings: {len(full):,} | Hosts: {full['host_id'].nunique():,} | Neighbourhoods: {full['neighbourhood'].nunique()}")
        lines.append(f"- Average price: ${g['price'].mean():,.0f} | Median price: ${g['price'].median():,.0f}")
        lines.append(f"- Avg availability: {full['availability_365'].mean():.0f} days/yr | Avg reviews: {full['number_of_reviews'].mean():.1f}")
        rt = full["room_type"].value_counts(normalize=True).mul(100).round(1)
        lines.append("- Room type share: " + ", ".join(f"{k} {v}%" for k, v in rt.items()))
        pg = full["property_group"].value_counts(normalize=True).mul(100).round(1)
        lines.append("- Property group share: " + ", ".join(f"{k} {v}%" for k, v in pg.items()))
        top = full["neighbourhood"].value_counts().head(5)
        avgp = g.groupby("neighbourhood")["price"].mean()
        lines.append("- Top 5 neighbourhoods by listings: " + ", ".join(
            f"{n} ({c:,}, avg ${avgp.get(n, np.nan):,.0f})" for n, c in top.items()))
        exp = g.groupby("neighbourhood").filter(lambda x: len(x) >= 20).groupby("neighbourhood")["price"].mean().nlargest(3)
        lines.append("- Most expensive neighbourhoods (min 20 listings): " + ", ".join(f"{n} ${p:,.0f}" for n, p in exp.items()))
        ht = full["host_type"].value_counts(normalize=True).mul(100).round(1)
        lines.append("- Host mix (listings): " + ", ".join(f"{k} {v}%" for k, v in ht.items()))
        hp = g.groupby("host_type").agg(avg_price=("price", "mean"), avg_reviews=("number_of_reviews", "mean"), avg_rating=("rating", "mean"))
        for k, r in hp.iterrows():
            lines.append(f"  - {k}: avg price ${r.avg_price:,.0f}, avg reviews {r.avg_reviews:.1f}, avg rating {r.avg_rating:.2f}")
        corr = g[["price", "number_of_reviews"]].corr().iloc[0, 1]
        lines.append(f"- Correlation price vs number_of_reviews: {corr:.2f}")
        lt = (full["stay_type"] == "Long-term (30+ nights)").mean() * 100
        lines.append(f"- Long-term (30+ night minimum) listings: {lt:.1f}%")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chicago", required=True)
    ap.add_argument("--neworleans", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.concat([load_city(a.chicago, "Chicago"), load_city(a.neworleans, "New Orleans")], ignore_index=True)
    clean = transform(raw)
    clean.to_csv(out / "airbnb_clean.csv", index=False, date_format="%Y-%m-%d")
    (out / "key_numbers.md").write_text(key_numbers(clean), encoding="utf-8")
    (out / "cleaning_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nWrote {out / 'airbnb_clean.csv'}, key_numbers.md, cleaning_log.txt")


if __name__ == "__main__":
    main()
