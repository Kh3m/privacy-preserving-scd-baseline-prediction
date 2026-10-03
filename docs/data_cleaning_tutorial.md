# Cleaning Tabular Data with pandas: A Practical Tutorial

By Abdul Kareem Adamu

A step-by-step guide to finding and fixing problems in a messy dataset
before it is used to train a model. Every example comes from a real
exercise: cleaning a synthetic sickle cell disease (SCD) dataset where the
goal is to predict a patient's steady-state packed cell volume (PCV).

You do not need to know anything about SCD to follow along. The same
steps apply to any table of data.

The examples deliberately include broken versions of the code alongside
the correct ones, so you can see exactly what each mistake looks like and
how to spot it in your own work.

> **About the numbers in this tutorial.** All counts and outputs shown
> (for example 1,530 rows, 76 contaminated values, 1,495 rows after
> cleaning) come from the current synthetic dataset, produced by
> `src/generate_synthetic_scd_baseline_data.py` (random seed 42) and
> `src/make_scd_baseline_data_messy.py` (random seed 7). If either script
> or its seed changes, your numbers will differ, but the steps and
> reasoning stay the same.

---

## Contents

1. The dataset
2. The three phases: Diagnose, Fix, Verify
3. Phase 1: Diagnose
4. Phase 2: Fix
5. Phase 3: Verify
6. The rules for imputation (filling missing values)
7. Common mistakes and how to avoid them
8. Checklist
9. Appendix: the full pipeline in one script

---

## 1. The dataset

Each row is one patient. The columns are:

| Column | Type | Meaning |
|---|---|---|
| `age` | number | Age in years |
| `genotype` | category | SS, SC, or SBeta-thal |
| `hbf_percent` | number | Fetal haemoglobin, % |
| `ldh_u_l` | number | Lactate dehydrogenase, U/L |
| `reticulocyte_pct` | number | Reticulocytes, % |
| `indirect_bilirubin_mg_dl` | number | Indirect bilirubin, mg/dL |
| `environment` | category | urban_good_access, urban_poor_access, rural |
| `nutrition_status` | category | good, moderate, poor |
| `steady_state_pcv_pct` | number | **The target** (what we want to predict) |

The messy version has 1,530 rows. Problems were deliberately injected:
missing values, inconsistent spelling, duplicate rows, impossible values,
numbers stored with unit text attached, and fully empty rows.

> **Key term: target (or label).** The column the model will learn to
> predict. Keep track of which column this is. Several cleaning
> decisions depend on it.

---

## 2. The three phases

Do not start fixing things as soon as you open the file. Work in three
phases:

| Phase | Goal | Rule |
|---|---|---|
| **Diagnose** | Find every problem | Do not change anything yet |
| **Fix** | Correct what you found | Work on a copy, never the original |
| **Verify** | Prove the fixes worked | Check numbers, not impressions |

Diagnosing first means you see the whole picture before deciding on a
fix. Verifying last means you catch fixes that silently did nothing.

---

## 3. Phase 1: Diagnose

### Step 1: Load and inspect

```python
import pandas as pd
import numpy as np

df = pd.read_csv("../data/raw/synthetic_scd_baseline_data_MESSY.csv")

df.head(10)
df.shape     # (1530, 9)
df.dtypes
```

Output of `df.dtypes`:

```
age                         float64
genotype                        str
hbf_percent                 float64
ldh_u_l                         str     <-- should be a number
reticulocyte_pct            float64
indirect_bilirubin_mg_dl        str     <-- should be a number
environment                     str
nutrition_status                str
steady_state_pcv_pct        float64
```

### Step 2: Read the dtype clues

A column that should hold numbers but has dtype `str` (or `object`)
usually contains some text. Find the rows that will not parse:

```python
def is_clean_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

bad = df[~df["ldh_u_l"].apply(is_clean_number)]
print(bad[["ldh_u_l"]].head(3))
print(f"{len(bad)} contaminated rows out of {len(df)}")
```

```
        ldh_u_l
68    833.8 U/L
97   1132.8 U/L
101  1039.7 U/L

76 contaminated rows out of 1530
```

The values are real readings with units stuck on the end. They can be
recovered, so they should not be thrown away.

A second clue: `age` is `float64` even though ages look like whole
numbers. A single missing value forces pandas to store a whole column as
float, so check for missing values:

```python
print(df["age"].isna().sum())          # 5
print((df["age"] % 1 == 0).all())      # False: some ages have decimals
```

### Step 3: Profile every column

One loop shows missing values, unique counts, ranges, and sample values
for every column:

```python
for col in df.columns:
    print(f"\n--- {col} ---")
    print(f"dtype: {df[col].dtype}")
    n_missing = df[col].isna().sum()
    print(f"missing: {n_missing} ({n_missing / len(df) * 100:.1f}%)")
    print(f"unique values: {df[col].nunique()}")
    if df[col].dtype in ["int64", "float64"]:
        print(f"min: {df[col].min()}, max: {df[col].max()}")
    else:
        print(f"sample values: {df[col].dropna().unique()[:8]}")
```

Selected output, with what each line tells you:

```
--- age ---
missing: 5 (0.3%)
min: -3.0, max: 58.2          <-- a negative age is impossible

--- genotype ---
unique values: 16             <-- there should only be 3
sample values: ['SS', 'SC', ' SS ', 'Hb SS', 'Hb-SS', 'SBeta-thal', 'ss', 'HBSS']

--- reticulocyte_pct ---
min: -5.0, max: 24.08         <-- a negative percentage is impossible

--- steady_state_pcv_pct ---
min: 12.0, max: 85.0          <-- 85% PCV is not believable (see Step 10)
```

### Step 4: Check for duplicate rows

Profiling columns one at a time never reveals a repeated row. Check
separately:

```python
print(df.duplicated().sum())    # 29
```

### What the diagnosis found

| Problem | Where |
|---|---|
| Fully empty rows | 5 rows |
| Duplicate rows | 29 rows |
| Unit text in numeric columns | `ldh_u_l` (76), `indirect_bilirubin_mg_dl` (75) |
| Inconsistent spelling | `genotype`, `environment`, `nutrition_status` |
| Impossible or implausible values | `age`, `reticulocyte_pct`, `ldh_u_l`, PCV |
| Missing values | Most columns, about 4% each |

Now, and only now, start fixing.

---

## 4. Phase 2: Fix

The order of the fix steps matters. Each step prepares the data for the
next one.

### Step 5: Work on a copy

```python
df_clean = df.copy()
```

The raw data stays untouched, so you can always compare before and after
or start again.

### Step 6: Drop fully empty rows

```python
df_clean = df_clean.dropna(how="all")
df_clean.shape    # (1525, 9)
```

`dropna` has three modes, and they do different things:

| Call | Drops a row when | Strength |
|---|---|---|
| `dropna()` | **any** column is missing | Most aggressive |
| `dropna(how="all")` | **every** column is missing | Least aggressive |
| `dropna(subset=["col"])` | the named column(s) are missing | Targeted |

Plain `dropna()` here would throw away hundreds of useful rows that are
missing just one value. Use `how="all"` for empty rows.

### Step 7: Drop duplicate rows

```python
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
df_clean.shape    # (1500, 9)
```

`reset_index(drop=True)` renumbers rows 0, 1, 2, ... so later lookups by
position behave as expected.

### Step 8: Fix type contamination

Pull the number out of each value and convert the column to numeric:

```python
for col in ["ldh_u_l", "indirect_bilirubin_mg_dl"]:
    df_clean[col] = df_clean[col].astype(str).str.extract(r"(-?\d+\.?\d*)")[0]
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
```

- `str.extract(r"(-?\d+\.?\d*)")` keeps the first number in the text, so
  `"833.8 U/L"` becomes `"833.8"`.
- `pd.to_numeric(..., errors="coerce")` converts to numbers. Anything
  that still cannot be parsed becomes `NaN` instead of raising an error.

**How to tell it worked:** count missing values before and after.

```python
df_clean["ldh_u_l"].isna().sum()   # before: 60
# ... run the fix ...
df_clean["ldh_u_l"].isna().sum()   # after: 60  (correct, nothing lost)
```

If the count jumps (for example from 60 to 135, see Section 7), the
extraction did not take effect and `coerce` turned real readings into
`NaN`.

### Step 9: Normalise categories

The `genotype` column had 16 spellings of 3 values. Build a mapping from
every variant to its clean label:

```python
print(df_clean["genotype"].dropna().unique())
# ['SS' 'SC' ' SS ' 'Hb SS' 'Hb-SS' 'SBeta-thal' 'ss' 'HBSS' 'sc'
#  'S-beta thal' 'HBSC' 'Hb-SC' 'sbeta-thal' ' SC ' 'S/beta-thal' 'Sbeta thal']

genotype_map = {}
for val in df_clean["genotype"].dropna().unique():
    v = str(val).strip().lower().replace("hb ", "").replace("hb-", "").replace("hb", "")
    if v == "ss":
        genotype_map[val] = "SS"
    elif v == "sc":
        genotype_map[val] = "SC"
    elif "beta" in v:
        genotype_map[val] = "SBeta-thal"

df_clean["genotype"] = df_clean["genotype"].map(genotype_map)
print(df_clean["genotype"].unique())    # ['SS' 'SC' 'SBeta-thal']
```

Print the mapping and read it before applying it. Any variant missing
from the map becomes `NaN` after `.map()`.

Simpler columns only need whitespace and case fixed:

```python
env_map = {v: str(v).strip().lower().replace(" ", "_")
           for v in df_clean["environment"].dropna().unique()}
df_clean["environment"] = df_clean["environment"].map(env_map)
# 'Urban Good Access', ' urban_good_access', 'URBAN_GOOD_ACCESS'
#   -> 'urban_good_access'
```

### Step 10: Null out impossible and implausible values

This is the step people find hardest, because it needs a decision:
**what range is believable for each column?** There are two kinds of
limit.

**1. Impossible values: limits by definition.** These need no domain
knowledge:

- Percentages cannot be below 0 or above 100.
- Counts, concentrations, and ages cannot be negative.

**2. Implausible values: limits from domain knowledge.** Some values are
technically possible but not believable. A PCV of 85% is under 100, but
healthy adults are around 36 to 50%, and SCD patients at steady state are
usually 15 to 35%. These limits come from textbooks, lab reference
sheets, published studies, or a domain expert.

**3. Let the data point to suspects.** Look at the distribution:

```python
df_clean.describe().T[["min", "25%", "50%", "75%", "max"]]
```

Red flags:

- **Sentinel values** such as `9999`, `-1`, `999`. Someone typed a code
  meaning "unknown". The same round number appearing several times is a
  strong sign: `df_clean["ldh_u_l"].value_counts().head()` shows `9999`
  four times.
- **A min or max far away from the 25% or 75% value.**
- **Unit mix-ups.** An Hb of 110 is probably g/L entered where g/dL was
  expected. Convert it rather than deleting it.

The ranges used in this exercise (deliberately wide; confirm with a
clinician before using on real data):

| Column | Typical in SCD | Limits used |
|---|---|---|
| `age` | children to adults | 0 to 100 |
| `steady_state_pcv_pct` | about 15 to 35% | 10 to 50 |
| `reticulocyte_pct` | about 5 to 20% | 0 to 40 |
| `ldh_u_l` | about 300 to 1,000+ | 50 to 3,000 |
| `indirect_bilirubin_mg_dl` | about 1 to 6 | 0 to 20 |
| `hbf_percent` | about 1 to 30% | 0 to 100 |

Keep limits wide. The aim is to remove values that cannot be true, not
unusual patients. A very sick patient with LDH 2,000 is real data and the
model needs to see it.

Apply all the limits in one place:

```python
RANGES = {
    "age": (0, 100),
    "steady_state_pcv_pct": (10, 50),
    "reticulocyte_pct": (0, 40),
    "ldh_u_l": (50, 3000),
    "indirect_bilirubin_mg_dl": (0, 20),
    "hbf_percent": (0, 100),
}

for col, (low, high) in RANGES.items():
    bad = ~df_clean[col].between(low, high) & df_clean[col].notna()
    print(f"{col}: {bad.sum()} out of range")
    df_clean.loc[bad, col] = np.nan
```

```
age: 3 out of range
steady_state_pcv_pct: 5 out of range
reticulocyte_pct: 3 out of range
ldh_u_l: 4 out of range
indirect_bilirubin_mg_dl: 0 out of range
hbf_percent: 0 out of range
```

Why set bad values to `NaN` instead of deleting the row? The rest of the
row is still good. A `NaN` feature can be filled in later (Step 12).

> **Note on the generator.** In this exercise the synthetic data came
> from a script that clips each value to a range. Do not copy those
> limits. Real data has no generator, so practise with domain ranges.

### Step 11: Drop rows with a missing target

```python
df_clean = df_clean.dropna(subset=["steady_state_pcv_pct"]).reset_index(drop=True)
df_clean.shape    # (1495, 9)
```

Features and the target are treated differently:

- A missing **feature** can be filled in (Step 12).
- A missing **target** must never be filled in. A guessed label teaches
  the model something you made up. Drop the row.

This must come **after** Step 10, because Step 10 turns impossible PCV
values into `NaN`, and **before** Step 12.

### Step 12: Fill in (impute) the remaining missing values

Numeric columns: use the median of a relevant group.

```python
numeric_cols = ["age", "hbf_percent", "ldh_u_l",
                "reticulocyte_pct", "indirect_bilirubin_mg_dl"]

for col in numeric_cols:
    df_clean[col] = (
        df_clean.groupby("genotype")[col]
        .transform(lambda s: s.fillna(s.median()))
    )
```

Categorical columns: use the most common value (the mode).

```python
for col in ["genotype", "environment", "nutrition_status"]:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
```

Why the median and not the mean? One extreme value can pull a mean a long
way. The median barely moves.

Choosing the column to group by is important enough to get its own
section: see Section 6.

---

## 5. Phase 3: Verify

### Step 13: Final checks

```python
print(f"Final shape: {df_clean.shape}")
print(f"Remaining missing values: {df_clean.isna().sum().sum()}")
print(f"Remaining duplicates: {df_clean.duplicated().sum()}")
print(df_clean.describe().T[["min", "max"]])
for col in ["genotype", "environment", "nutrition_status"]:
    print(col, df_clean[col].unique())
```

```
Final shape: (1495, 9)
Remaining missing values: 0
Remaining duplicates: 0

                            min      max
age                         2.0    58.20
hbf_percent                 0.5    30.00
ldh_u_l                   224.5  1351.90
reticulocyte_pct            1.0    24.08
indirect_bilirubin_mg_dl    0.2     8.05
steady_state_pcv_pct       12.0    39.90
```

Every check has an expected answer. If any number is not what you
expect, go back and find out why before saving.

### Step 14: Save to a separate file

```python
df_clean.to_csv("../data/processed/synthetic_scd_baseline_data_CLEANED.csv",
                index=False)
```

Never overwrite the raw file. `index=False` stops pandas writing the row
numbers as an extra column.

---

## 6. The rules for imputation (filling missing values)

### What the grouped fill actually does

```python
df_clean.groupby("genotype")[col].transform(lambda s: s.fillna(s.median()))
```

1. Split rows into groups by genotype (SS, SC, SBeta-thal).
2. Inside each group, fill missing values of `col` with **that group's**
   median.
3. Put the rows back in their original order.

A missing bilirubin value for an SS patient gets the SS median (4.7), not
the overall median. The idea is: fill the gap with what is typical for
patients **like this one**. The grouping column decides who counts as
"like this one."

### There is no single "always group by" column

Choose one each time, using these five rules.

**Rule 1: Never group by the target.**
If you fill features using the target, the answer leaks into the inputs.
The model looks better in testing than it will be in real use. This is
called **target leakage**.

**Rule 2: Use a categorical column with a few large groups.**
PCV is continuous, with about 230 distinct values. Grouping by it gave
most groups only one or two rows, so some rows had no other value to take
a median from. Genotype has 3 groups of 158 to 1,023 rows each.

**Rule 3: The grouping column must not be missing.**
`groupby` leaves out rows where the key is `NaN`, and `transform` returns
`NaN` for them. Fill the grouping column first, or pick one that is never
missing.

**Rule 4: The groups must actually differ.**
If the group medians are about the same, grouping adds nothing. Check:

```python
df_clean.groupby("genotype")[numeric_cols].median()
df_clean["genotype"].value_counts()
```

Medians from this dataset:

| Grouped by | Reticulocyte % | Bilirubin | Groups differ? |
|---|---|---|---|
| **genotype** | SS 14.0, SBeta 11.3, SC 9.3 | SS 4.7, SBeta 3.7, SC 2.9 | **Yes, clearly** |
| environment | about equal | 4.2 to 4.3 | No |
| nutrition_status | 12.6 to 13.2 | 4.2 to 4.4 | Barely |

This matches the medicine: SS is the most severe genotype, with the most
red cell breakdown, so its reticulocyte and bilirubin levels are highest.

**Rule 5: The column must be known at prediction time.**
For a new patient, you will know their genotype. You will not know their
PCV, because that is what you are predicting.

If no column passes all five rules, use the plain median:

```python
df_clean[col] = df_clean[col].fillna(df_clean[col].median())
```

### A worked example of leakage spreading

Leakage often spreads by copying. Suppose a related project predicts the
cause of a patient's pain, with a categorical target called
`cause_label`, and its cleaning notebook fills in values with
`groupby("cause_label")`. That code runs without errors because the
groups are large, but it still breaks Rules 1 and 5.

Now suppose the pattern is reused for this dataset as
`groupby("steady_state_pcv_pct")`. Here it also breaks Rules 2 and 3,
and the symptoms become visible: rows stranded alone in a group, and rows
with every numeric column wiped to `NaN`. The visible failure is useful,
because the quieter version of the same mistake would have gone
unnoticed.

The lesson: **before reusing a `groupby` from another notebook, ask "is
this column the thing I am trying to predict?"** If yes, do not group by
it.

### Going further: imputation and train/test splits

Even with a safe grouping column, the medians above are calculated from
the **whole** dataset. Once the data is split into training and test
sets, the test rows will have helped decide the fill values, which is a
quieter form of leakage.

At the modelling stage, put the imputation inside a scikit-learn
`Pipeline` so it learns medians from the training data only:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("regress", LinearRegression()),
])
model.fit(X_train, y_train)      # medians come from X_train only
```

---

## 7. Common mistakes and how to avoid them

These are the most common ways a cleaning notebook goes wrong. Each one
is shown with the exact symptom it produces on this dataset, so you can
recognise it in your own work.

### Mistake 1: Forgetting to assign the result

Most pandas operations return a **new** object. They do not change the
original.

```python
# Wrong: runs, prints a table, changes nothing
df_clean.dropna(subset=["steady_state_pcv_pct"]).reset_index(drop=True)

# Right
df_clean = df_clean.dropna(subset=["steady_state_pcv_pct"]).reset_index(drop=True)
```

The same bug can hide inside a loop, where it is much harder to see:

```python
# Wrong: the extracted numbers are computed and thrown away
for col in ["ldh_u_l", "indirect_bilirubin_mg_dl"]:
    df_clean[col].astype(str).str.extract(r"(-?\d+\.?\d*)")[0]
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
```

On this dataset, that version silently turns 75 real readings like
`"833.8 U/L"` into `NaN`, which are then replaced by guesses. Missing LDH
jumps from 60 to 135, and that jump is the symptom to look for.

**How to catch it:** after every fix, check a number that should have
changed (a shape, a missing count, a unique list).

### Mistake 2: Running notebook cells out of order

Re-running one cell after editing it uses whatever state `df_clean` is in
**now**, which may already be partly changed. For example, if you
correct the type fix above and re-run only that cell, the column has
already been converted and the lost values are gone. The cell looks
fixed, but the data is still damaged.

**How to avoid it:** after changing any cell, use **Kernel, Restart, Run
All**. Check that the execution numbers down the left side run in order.

### Mistake 3: Patching symptoms instead of causes

Grouping by PCV (breaking Rules 1 and 2) leaves one row with
`NaN` bilirubin after imputation. It is tempting to add a "fallback to
the overall median" cell to cover it. That hides the symptom and leaves
the cause in place. Once the grouping is fixed, the fallback is not
needed, and any diagnostic cell written to inspect the stranded row
crashes with `index 0 is out of bounds for axis 0 with size 0`, because
there is no longer a `NaN` row to look at.

**How to avoid it:** when a fix needs a second fix, stop and ask why the
first one did not work.

### Mistake 4: Fixing steps in the wrong order

| This step | must come before | because |
|---|---|---|
| Fix types (8) | Range checks (10) | You cannot compare text to a number |
| Range checks (10) | Drop missing target (11) | Range checks create new `NaN` targets |
| Drop missing target (11) | Imputation (12) | Never impute the target |
| Fill the grouping column | Grouped imputation (12) | Rule 3 |

### Mistake 5: Trusting a pattern because it ran without errors

"No error" does not mean "correct". The `cause_label` grouping in
Section 6 runs cleanly and is still leakage. Verify with numbers, and question patterns
you copy.

---

## 8. Checklist

**Diagnose**
- [ ] Shape, `head()`, `dtypes`
- [ ] Numeric columns stored as text?
- [ ] Per-column profile: missing %, unique values, min and max
- [ ] Duplicate rows
- [ ] Written list of every problem found

**Fix (in this order)**
- [ ] Work on a copy
- [ ] Drop fully empty rows (`how="all"`)
- [ ] Drop exact duplicates
- [ ] Extract numbers from contaminated text, convert to numeric
- [ ] Normalise category spellings; print the map before applying
- [ ] Null values outside definitional and domain ranges
- [ ] Drop rows with a missing or invalid target
- [ ] Fill categorical gaps with the mode
- [ ] Fill numeric gaps with a grouped median that passes all 5 rules
- [ ] Every operation assigned back (`df_clean = ...` or `df_clean[col] = ...`)

**Verify**
- [ ] Restart and Run All
- [ ] 0 missing values, 0 duplicates
- [ ] Every numeric column inside its range
- [ ] Every category column has only the expected labels
- [ ] Saved to a new file; raw file untouched

---

## 9. Appendix: the full pipeline in one script

```python
import numpy as np
import pandas as pd

RAW = "data/raw/synthetic_scd_baseline_data_MESSY.csv"
OUT = "data/processed/synthetic_scd_baseline_data_CLEANED.csv"
TARGET = "steady_state_pcv_pct"

RANGES = {
    "age": (0, 100),
    "steady_state_pcv_pct": (10, 50),
    "reticulocyte_pct": (0, 40),
    "ldh_u_l": (50, 3000),
    "indirect_bilirubin_mg_dl": (0, 20),
    "hbf_percent": (0, 100),
}
NUMERIC = ["age", "hbf_percent", "ldh_u_l",
           "reticulocyte_pct", "indirect_bilirubin_mg_dl"]
CATEGORICAL = ["genotype", "environment", "nutrition_status"]


def normalise_genotype(val):
    v = str(val).strip().lower().replace("hb ", "").replace("hb-", "").replace("hb", "")
    if v == "ss":
        return "SS"
    if v == "sc":
        return "SC"
    if "beta" in v:
        return "SBeta-thal"
    return np.nan


df = pd.read_csv(RAW)
df_clean = df.copy()

# Empty rows and duplicates
df_clean = df_clean.dropna(how="all")
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# Type contamination
for col in ["ldh_u_l", "indirect_bilirubin_mg_dl"]:
    df_clean[col] = df_clean[col].astype(str).str.extract(r"(-?\d+\.?\d*)")[0]
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Categories
df_clean["genotype"] = df_clean["genotype"].map(normalise_genotype, na_action="ignore")
for col in ["environment", "nutrition_status"]:
    df_clean[col] = df_clean[col].str.strip().str.lower().str.replace(" ", "_")

# Impossible / implausible values
for col, (low, high) in RANGES.items():
    bad = ~df_clean[col].between(low, high) & df_clean[col].notna()
    df_clean.loc[bad, col] = np.nan

# Target: drop, never impute
df_clean = df_clean.dropna(subset=[TARGET]).reset_index(drop=True)

# Categorical gaps first (genotype is the grouping key below)
for col in CATEGORICAL:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Numeric gaps: grouped median, never grouped by the target
for col in NUMERIC:
    df_clean[col] = (
        df_clean.groupby("genotype")[col]
        .transform(lambda s: s.fillna(s.median()))
    )

# Verify
assert df_clean.isna().sum().sum() == 0, "missing values remain"
assert df_clean.duplicated().sum() == 0, "duplicates remain"
for col, (low, high) in RANGES.items():
    assert df_clean[col].between(low, high).all(), f"{col} out of range"

df_clean.to_csv(OUT, index=False)
print(f"Saved {df_clean.shape[0]} rows to {OUT}")
```

The `assert` lines turn the verify phase into code. If anything is wrong,
the script stops with a message instead of saving bad data.
