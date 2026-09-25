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
│   └── processed/        # cleaned output
├── notebooks/            # exploration and data cleaning work
├── src/                  # reusable scripts (data generation, model, encryption, app)
└── models/               # trained model artifacts
```

## Setup

Using conda:
```
conda create -p venv python=3.13
conda activate ./venv
pip install -r requirements.txt
```

## How to run (current steps)

1. Generate the synthetic dataset:
   ```
   python src/generate_synthetic_data.py
   ```
2. (Optional, for practice or testing) generate a version with realistic
   data quality issues injected:
   ```
   python src/make_data_messy.py
   ```
3. Clean the data (see `notebooks/` for the working cleaning notebook)

> **Note:** the synthetic dataset and cleaning pipeline below were built
> around the project's earlier framing (classifying the cause of a pain
> episode: VOC, infection, dehydration, AVN, ACS). They need to be
> regenerated around the current target, degree of haemolysis, genotype,
> environment, and nutrition as features, steady-state PCV/Hb as the label,
> before they're reusable for this direction.

## Roadmap

- [x] Synthetic dataset generation for the earlier pain-cause framing
      (superseded, needs regeneration for the current target)
- [x] Synthetic "messy data" generator for data-cleaning practice
      (superseded, needs regeneration for the current target)
- [x] Data cleaning pipeline (approach carries over; needs re-running on
      regenerated data)
- [ ] Synthetic dataset regenerated for steady-state PCV/Hb prediction
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