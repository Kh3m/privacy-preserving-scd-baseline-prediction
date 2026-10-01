"""
Injects realistic data-quality problems into the clean synthetic
steady-state PCV baseline dataset, for cleaning practice.

Problems introduced:
1. Missing values across several fields
2. Inconsistent genotype spelling
3. Inconsistent environment/nutrition category spelling
4. Duplicate rows
5. Outlier/implausible values
6. Numeric fields contaminated with unit text
7. A few fully empty rows
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

IN_PATH = "data/raw/synthetic_scd_baseline_data.csv"
OUT_PATH = "data/raw/synthetic_scd_baseline_data_MESSY.csv"


def introduce_missing_values(df, cols, frac=0.04):
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
    non_null_idx = df.index[df["genotype"].notna()]
    n = int(len(df) * frac)
    idx = RNG.choice(non_null_idx, size=n, replace=False)
    for i in idx:
        true_val = df.loc[i, "genotype"]
        df.loc[i, "genotype"] = RNG.choice(variants[true_val])
    return df


def mess_up_category(df, col, variant_map, frac=0.15):
    df = df.copy()
    non_null_idx = df.index[df[col].notna()]
    n = int(len(df) * frac)
    idx = RNG.choice(non_null_idx, size=n, replace=False)
    for i in idx:
        true_val = df.loc[i, col]
        df.loc[i, col] = RNG.choice(variant_map[true_val])
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
            df.loc[i, "age"] = -3
        elif choice == 1:
            df.loc[i, "ldh_u_l"] = 9999.0
        elif choice == 2:
            df.loc[i, "steady_state_pcv_pct"] = 85.0  # impossible, PCV can't exceed 100 but this is way off physiologically
        else:
            df.loc[i, "reticulocyte_pct"] = -5.0
    return df


def add_unit_strings(df, cols_units, frac=0.05):
    df = df.copy()
    for col, unit in cols_units.items():
        df[col] = df[col].astype(object)
        n = int(len(df) * frac)
        idx = RNG.choice(df.index, size=n, replace=False)
        for i in idx:
            df.loc[i, col] = f"{df.loc[i, col]}{unit}"
    return df


def add_empty_rows(df, n_empty=5):
    empty_rows = pd.DataFrame([{col: np.nan for col in df.columns} for _ in range(n_empty)])
    return pd.concat([df, empty_rows], ignore_index=True)


def main():
    df = pd.read_csv(IN_PATH)

    df = introduce_missing_values(
        df, cols=["ldh_u_l", "reticulocyte_pct", "indirect_bilirubin_mg_dl", "nutrition_status", "hbf_percent"], frac=0.04
    )
    df = mess_up_genotype(df, frac=0.15)
    df = mess_up_category(
        df, "environment",
        {
            "urban_good_access": ["urban_good_access", "Urban Good Access", "URBAN_GOOD_ACCESS", " urban_good_access"],
            "urban_poor_access": ["urban_poor_access", "Urban Poor Access", " urban_poor_access"],
            "rural": ["rural", "Rural", "RURAL", " rural "],
        },
        frac=0.15,
    )
    df = mess_up_category(
        df, "nutrition_status",
        {
            "good": ["good", "Good", "GOOD"],
            "moderate": ["moderate", "Moderate", " moderate"],
            "poor": ["poor", "Poor", "POOR "],
        },
        frac=0.12,
    )
    df = add_outliers(df, n_outliers=15)
    df = add_unit_strings(
        df,
        cols_units={"ldh_u_l": " U/L", "indirect_bilirubin_mg_dl": " mg/dL"},
        frac=0.05,
    )
    df = add_duplicates(df, n_dupes=25)
    df = add_empty_rows(df, n_empty=5)

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Messy dataset written -> {OUT_PATH}")
    print(f"Rows: {len(df)} (original had 1500)")
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nUnique genotype values:", df["genotype"].unique())
    print("Unique environment values:", df["environment"].unique())
    print("Unique nutrition_status values:", df["nutrition_status"].unique())


if __name__ == "__main__":
    main()