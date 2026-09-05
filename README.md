# Privacy-Preserving SCD Pain Diagnosis

Identifying the likely cause of a chronic pain episode in sickle cell disease
(SCD) patients — vaso-occlusive crisis, infection, dehydration, avascular
necrosis, acute chest syndrome, or other — without ever exposing raw patient
data during computation.

## Status: Early-stage research prototype

> **This is not a validated clinical tool.** The model is currently trained
> and tested only on synthetic data designed to mirror plausible clinical
> patterns. It has not been validated on real patient data and is not
> intended for diagnostic use. Real-data validation is planned in
> collaboration with a clinical partner.

## Why this project

Sickle cell chronic pain can stem from several overlapping causes, and
telling them apart at the point of care is difficult, especially in
resource-constrained clinical settings. Existing machine learning work in
this space largely focuses on predicting *whether* a crisis will occur, not
*what* is causing a current pain episode. Separately, patient health data is
sensitive, and any system touching it should be built so that raw data is
never exposed to the system performing the computation.

This project combines:
- **Cause classification** — a multiclass model distinguishing likely causes
  of a current pain episode
- **Homomorphic encryption (HE)** — patient feature data is encrypted before
  any inference is performed, so the computing system never sees plaintext
  patient data
- **Federated learning (FL)** *(planned)* — enabling multiple clinics to
  contribute to model training without sharing raw records
- **Differential privacy (DP)** *(planned)* — protecting against information
  leakage through shared model updates in the federated setting

As far as we've been able to determine, this specific combination has not
been applied to sickle cell disease before (see Related Work below).

## Repository structure

```
privacy-preserving-scd-pain-diagnosis/
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
conda create -n scd-pain python=3.11
conda activate scd-pain
pip install -r requirements.txt
```

## How to run (current steps)

1. Generate the synthetic dataset:
   ```
   python src/generate_synthetic_data.py
   ```
2. (Optional, for practice/testing) generate a version with realistic data
   quality issues injected:
   ```
   python src/make_data_messy.py
   ```
3. Clean the data (see `notebooks/` for the working cleaning notebook)

## Roadmap

- [x] Synthetic dataset generation (2,100 records, 6 cause classes)
- [x] Synthetic "messy data" generator for data-cleaning practice
- [ ] Data cleaning pipeline
- [ ] Baseline plaintext classification model
- [ ] Encrypted inference layer (homomorphic encryption via TenSEAL/CKKS)
- [ ] Federated learning extension across multiple sites
- [ ] Differential privacy on shared model updates
- [ ] Streamlit demo interface
- [ ] Real clinical data validation (pending clinical partnership)

## Related work

This project builds conceptually on prior work applying partial homomorphic
encryption to healthcare data analysis, and is distinct from the closest
related work we're aware of — a 2026 study applying federated learning
(without an encryption or differential privacy layer) to a sickle cell
classification task. This project differs by (1) adding a
privacy-preserving encryption and DP layer on top of FL, and (2) focusing on
cause differentiation for chronic pain rather than disease
presence/classification.

## Collaborators

- Abdul Kareem Adamu ([Khemshield](https://khemshield.com)) — research design,
  system architecture, implementation

## License

TBD