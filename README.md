# Privacy-Preserving SCD Baseline Prediction

Modeling what drives an individual sickle cell disease (SCD) patient's own
steady-state packed cell volume (PCV) and haemoglobin (Hb) baseline, without
ever exposing raw patient data during computation.

## Status: Early-stage research prototype

> **This is not a validated clinical tool.** The model is currently trained
> and tested only on synthetic data designed to mirror plausible clinical
> patterns. It has not been validated on real patient data and is not
> intended for diagnostic use. Real-data validation is planned in
> collaboration with a clinical partner.

## Why this project

SCD patients live with chronic haemolytic anaemia, but the degree of
anaemia each person experiences at steady state, that is, outside of an
acute crisis, varies considerably from patient to patient. Some remain
clinically stable with a PCV far below what would concern a doctor treating
anyone else; others need a higher baseline to stay well. Clinicians often
have only population-level reference ranges to judge against, which can
mean transfusions given to patients whose "low" reading was in fact normal
for them. That risk is real: blood is scarce and costly, even
well-screened blood carries residual risk, recurrent transfusion can
trigger antibody development that complicates future matching, and
transfusion reactions can be severe.

This project models what actually influences an individual patient's own
steady-state PCV/Hb, using degree of haemolysis, genotype, environment, and
nutrition as inputs, so a clinician can distinguish a patient's normal
baseline from genuine deterioration. Chronic pain frequency is a related
outcome of interest alongside this, not yet the primary target.

This project combines:
- **Baseline prediction**: a regression model estimating a patient's
  expected steady-state PCV/Hb from their own clinical and contextual
  factors
- **Homomorphic encryption (HE)**: patient feature data is encrypted before
  any inference is performed, so the computing system never sees plaintext
  patient data
- **Federated learning (FL)** *(planned)*: enabling multiple clinics to
  contribute to model training without sharing raw records
- **Differential privacy (DP)** *(planned)*: protecting against information
  leakage through shared model updates in the federated setting

## Repository structure

```
privacy-preserving-scd-baseline-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # data as generated, never edited in place
│   ├── processed/        # cleaned output
│   └── real/             # real patient data, local only (git-ignored)
├── notebooks/
│   ├── scd_baseline_data_cleaning.ipynb   # cleans the current PCV/Hb dataset
│   └── data_cleaning.ipynb                # legacy: earlier pain-cause dataset
├── docs/
│   └── data_cleaning_tutorial.md          # shareable data-cleaning tutorial
├── src/                  # reusable scripts (data generation, model, encryption, app)
└── models/               # trained model artifacts
```

### Data handling

All data in `data/raw/` and `data/processed/` is synthetic and safe to
commit. Any real patient data must be stored only in `data/real/`, which is
listed in `.gitignore` so it is never committed or pushed. Do not copy real
records into any other folder, and clear notebook outputs before committing
if a notebook has displayed real data.

## Setup

Using conda:
```
conda create -p venv python=3.13
conda activate ./venv
pip install -r requirements.txt
```

## How to run (current steps)

Run all commands from the project root.

1. Generate the synthetic dataset (writes
   `data/raw/synthetic_scd_baseline_data.csv`):
   ```
   python src/generate_synthetic_scd_baseline_data.py
   ```
2. (Optional, for practice or testing) generate a version with realistic
   data quality issues injected (writes
   `data/raw/synthetic_scd_baseline_data_MESSY.csv`):
   ```
   python src/make_scd_baseline_data_messy.py
   ```
3. Clean the data by running
   `notebooks/scd_baseline_data_cleaning.ipynb` top to bottom (writes
   `data/processed/synthetic_scd_baseline_data_CLEANED.csv`). Its file
   paths are relative to `notebooks/`, which is the kernel's default
   working directory in Jupyter and VS Code.

The notebook follows three phases: diagnose (shape, dtypes, per-column
profiling, duplicates), fix (drop empty and duplicate rows, strip unit text
from numeric fields, normalise genotype/environment/nutrition spellings,
null values outside clinically plausible ranges, drop rows with no valid
label, impute remaining feature gaps with genotype-group medians), and
verify (zero missing values, zero duplicates). Range limits are
deliberately wide and should be confirmed with a clinical partner before
use on real data.

### Legacy files

`notebooks/data_cleaning.ipynb`, `src/make_data_messy.py`, and the
`synthetic_sickle_cell_data*.csv` files belong to the earlier pain-cause
framing (labelled by `cause_label`). They are kept for reference only and
are not part of the current pipeline.

## Roadmap

- [x] Synthetic dataset generation for the earlier pain-cause framing
      (superseded, needs regeneration for the current target)
- [x] Synthetic "messy data" generator for data-cleaning practice
      (superseded, needs regeneration for the current target)
- [x] Data cleaning pipeline for the earlier pain-cause dataset
      (approach carries over to the new dataset)
- [x] Synthetic dataset regenerated for steady-state PCV/Hb prediction
- [x] Messy data generator for the steady-state PCV/Hb dataset
- [x] Cleaning notebook for the steady-state PCV/Hb dataset
- [ ] Baseline plaintext regression model
- [ ] Encrypted inference layer (homomorphic encryption via TenSEAL/CKKS)
- [ ] Federated learning extension across multiple sites
- [ ] Differential privacy on shared model updates
- [ ] Streamlit demo interface
- [ ] Real clinical data validation (pending clinical partnership)

## Related work

This project builds conceptually on prior work applying partial homomorphic
encryption to healthcare data analysis. The related-work landscape for this
specific framing, individual steady-state PCV/Hb prediction in SCD under a
privacy-preserving architecture, has not yet been re-checked since the
project's direction changed, and should be verified before this section
makes any novelty claims.

## Collaborators

- Abdul Kareem Adamu ([Khemshield](https://abdulkareem.khemshield.com)), research design,
  system architecture, implementation

## License

TBD