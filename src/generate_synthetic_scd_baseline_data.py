"""
Synthetic dataset generator for predicting an individual sickle cell disease
(SCD) patient's steady-state PCV/Hb baseline.

This replaces the earlier pain-cause classification framing. The target is
now continuous: steady_state_pcv_pct (packed cell volume, at baseline,
outside of any acute crisis).

Features reflect the four factors named by the clinical collaborator, plus
one added factor:
- genotype (degree of disease severity)
- degree of haemolysis (lab markers: LDH, reticulocyte %, indirect bilirubin)
- environment (proxy: water/sanitation access, infection exposure)
- nutrition (nutritional status score)
- fetal haemoglobin (HbF) percent: added because genotype alone does not
  capture real within-genotype variation. Two patients labeled SS can have
  very different severity, and HbF is a routine, measurable lab value that
  partly explains why, higher HbF is protective, even within the same
  genotype.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1500

GENOTYPES = ["SS", "SC", "SBeta-thal"]
GENOTYPE_P = [0.70, 0.20, 0.10]  # SS most common and most severe

# Genotype-specific baseline PCV ranges (before other factors adjust it)
GENOTYPE_BASE_PCV = {
    "SS": 22.0,          # most severe, lowest typical baseline
    "SC": 30.0,          # milder
    "SBeta-thal": 26.0,  # intermediate
}

ENVIRONMENTS = ["urban_good_access", "urban_poor_access", "rural"]
ENV_P = [0.35, 0.30, 0.35]

# Environment effect on PCV: poor sanitation/water access -> more infections
# -> more haemolytic stress -> lower PCV
ENV_EFFECT = {
    "urban_good_access": 1.5,
    "urban_poor_access": -1.0,
    "rural": -2.0,
}

NUTRITION_LEVELS = ["good", "moderate", "poor"]
NUTRITION_P = [0.40, 0.40, 0.20]
NUTRITION_EFFECT = {
    "good": 1.5,
    "moderate": 0.0,
    "poor": -2.5,
}


def clip(x, lo, hi):
    return np.clip(x, lo, hi)


def generate_row():
    genotype = RNG.choice(GENOTYPES, p=GENOTYPE_P)
    environment = RNG.choice(ENVIRONMENTS, p=ENV_P)
    nutrition = RNG.choice(NUTRITION_LEVELS, p=NUTRITION_P)
    age = clip(RNG.normal(22, 11), 2, 65)

    # --- Haemolysis markers ---
    # Higher haemolysis = more red cell destruction = lower PCV.
    # SS genotype tends toward higher haemolysis markers than SC.
    haemolysis_severity = {
        "SS": RNG.normal(7, 1.8),
        "SC": RNG.normal(4, 1.5),
        "SBeta-thal": RNG.normal(5.5, 1.7),
    }[genotype]
    haemolysis_severity = clip(haemolysis_severity, 0, 10)  # composite 0-10 scale

    ldh_u_l = clip(300 + haemolysis_severity * 90 + RNG.normal(0, 80), 150, 1800)
    reticulocyte_pct = clip(3 + haemolysis_severity * 1.6 + RNG.normal(0, 2), 1, 25)
    indirect_bilirubin_mg_dl = clip(0.5 + haemolysis_severity * 0.6 + RNG.normal(0, 0.6), 0.2, 12)

    # --- Fetal haemoglobin (HbF) percent ---
    # Drawn mostly independently of genotype, this is exactly the point: two
    # patients with the same genotype can have very different HbF, which is
    # part of why they end up with different real-world severity. Higher
    # HbF is protective (less sickling), so it pulls PCV up.
    hbf_percent = clip(RNG.lognormal(mean=1.6, sigma=0.7), 0.5, 30)

    # --- Steady-state PCV, built from genotype baseline + adjustments ---
    base_pcv = GENOTYPE_BASE_PCV[genotype]
    pcv = (
        base_pcv
        - haemolysis_severity * 0.9          # more haemolysis -> lower PCV
        + ENV_EFFECT[environment]
        + NUTRITION_EFFECT[nutrition]
        + hbf_percent * 0.35                  # higher HbF -> protective, raises PCV
        + RNG.normal(0, 1.8)                  # individual variation / noise
    )
    pcv = clip(pcv, 12, 42)

    return {
        "age": round(age, 1),
        "genotype": genotype,
        "hbf_percent": round(hbf_percent, 2),
        "ldh_u_l": round(ldh_u_l, 1),
        "reticulocyte_pct": round(reticulocyte_pct, 2),
        "indirect_bilirubin_mg_dl": round(indirect_bilirubin_mg_dl, 2),
        "environment": environment,
        "nutrition_status": nutrition,
        "steady_state_pcv_pct": round(pcv, 1),
    }


def main():
    rows = [generate_row() for _ in range(N)]
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = "data/raw/synthetic_scd_baseline_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} records -> {out_path}")
    print("\nSteady-state PCV by genotype (mean):")
    print(df.groupby("genotype")["steady_state_pcv_pct"].mean().round(1))
    print("\nSample rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()