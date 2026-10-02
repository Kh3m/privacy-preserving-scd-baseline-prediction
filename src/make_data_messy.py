"""
Injects realistic data-quality problems into the clean synthetic dataset.

This mirrors the kinds of issues real clinical data from a hospital records
system tends to have, so we can practice writing a proper cleaning pipeline
before real patient data arrives.

Problems introduced:
1. Missing values (some fields, random rows) - like a nurse skipping a field
2. Inconsistent genotype spelling - "SS", "ss", "Hb-SS", "HBSS", " SS "
3. Inconsistent season spelling/case - "Dry", "DRY", "dry season"
4. Duplicate rows - same patient entered twice
5. Outlier/implausible values - e.g. negative age, temperature in Fahrenheit
   by mistake, WBC way out of any human range
6. Wrong dtypes - numeric fields stored as strings with units attached
   (e.g. "8.2 g/dl" instead of 8.2)
7. A few completely empty rows
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

IN_PATH = "data/raw/synthetic_sickle_cell_data.csv"
OUT_PATH = "data/raw/synthetic_sickle_cell_data_MESSY.csv"


def introduce_missing_values(df, cols, frac=0.03):
    df = df.copy()
    for col in cols:
        n_missing = int(len(df) * frac)
        idx = RNG.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan
    return df


def mess_up_genotype(df, frac=0.15):
    df = df.copy()
    variants = {
        "SS": ["SS", "ss", "Hb-SS", "HBSS", " SS ", "Hb SS"],
        "SC": ["SC", "sc", "Hb-SC", "HBSC", " SC "],
        "SBeta-thal": ["SBeta-thal", "S-beta thal", "Sbeta thal", "S/beta-thal", "sbeta-thal"],
    }
    n = int(len(df) * frac)
    idx = RNG.choice(df.index, size=n, replace=False)
    for i in idx:
        true_val = df.loc[i, "genotype"]
        df.loc[i, "genotype"] = RNG.choice(variants[true_val])
    return df


def mess_up_season(df, frac=0.15):
    df = df.copy()
    variants = {
        "dry": ["dry", "Dry", "DRY", "dry season", " dry"],
        "rainy": ["rainy", "Rainy", "RAINY", "rainy season", " rainy"],
    }
    n = int(len(df) * frac)
    idx = RNG.choice(df.index, size=n, replace=False)
    for i in idx:
        true_val = df.loc[i, "season"]
        df.loc[i, "season"] = RNG.choice(variants[true_val])
    return df


def add_duplicates(df, n_dupes=25):
    dupe_rows = df.sample(n=n_dupes, random_state=7)
    return pd.concat([df, dupe_rows], ignore_index=True)


def add_outliers(df, n_outliers=15):
    df = df.copy()
    idx = RNG.choice(df.index, size=n_outliers, replace=False)
    for i in idx:
        choice = RNG.integers(0, 4)
        if choice == 0:
            df.loc[i, "age"] = -5  # impossible negative age
        elif choice == 1:
            # temperature accidentally entered in Fahrenheit instead of Celsius
            df.loc[i, "temperature_c"] = round(df.loc[i, "temperature_c"] * 9 / 5 + 32, 2)
        elif choice == 2:
            df.loc[i, "wbc_10e9_l"] = 999.0  # obviously wrong entry
        else:
            df.loc[i, "spo2_pct"] = 150  # impossible SpO2 above 100%
    return df


def add_unit_strings(df, cols_units, frac=0.05):
    """Store some numeric fields as strings with units attached."""
    df = df.copy()
    for col, unit in cols_units.items():
        df[col] = df[col].astype(object)  # allow mixed str/float in this column
        n = int(len(df) * frac)
        idx = RNG.choice(df.index, size=n, replace=False)
        for i in idx:
            df.loc[i, col] = f"{df.loc[i, col]}{unit}"
    return df


def add_empty_rows(df, n_empty=5):
    empty_rows = pd.DataFrame(
        [{col: np.nan for col in df.columns} for _ in range(n_empty)]
    )
    return pd.concat([df, empty_rows], ignore_index=True)


def main():
    df = pd.read_csv(IN_PATH)

    df = introduce_missing_values(
        df, cols=["hbf_percent", "crp_mg_l", "hydration_score", "reticulocyte_pct"], frac=0.04
    )
    df = mess_up_genotype(df, frac=0.15)
    df = mess_up_season(df, frac=0.15)
    df = add_outliers(df, n_outliers=15)
    df = add_unit_strings(
        df,
        cols_units={"baseline_hb_g_dl": " g/dl", "wbc_10e9_l": " x10^9/L"},
        frac=0.05,
    )
    df = add_duplicates(df, n_dupes=25)
    df = add_empty_rows(df, n_empty=5)

    # shuffle again so injected problems aren't clustered at the end
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Messy dataset written -> {OUT_PATH}")
    print(f"Rows: {len(df)} (original had 2100)")
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nUnique genotype values (should be messy):")
    print(df["genotype"].unique())
    print("\nUnique season values (should be messy):")
    print(df["season"].unique())


if __name__ == "__main__":
    main()