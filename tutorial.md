
# DoE Toolkit (Streamlit) — Didactic Tutorial (GitHub README)

A simple **Design of Experiments (DoE)** web app to:
1) **Generate designs** (coded + real units)  
2) **Export a run sheet** for the lab  
3) **Upload results** (measured responses)  
4) Fit a **quadratic model** (main effects + interactions + curvature)  
5) Visualize **response surfaces** (2D contour + 3D surface)  
6) Run a **grid-search optimization** for best predicted conditions  

---

## Table of contents
- [1. Quick start](#1-quick-start)
- [2. Concepts (coded space vs real units)](#2-concepts-coded-space-vs-real-units)
- [3. App workflow (Step-by-step)](#3-app-workflow-step-by-step)
- [4. Design types (what they are, when to use, factor limits, run counts)](#4-design-types-what-they-are-when-to-use-factor-limits-run-counts)
  - [4.1 Box–Behnken](#41-boxbehnken)
  - [4.2 Central Composite (CCD)](#42-central-composite-ccd)
  - [4.3 2-Level Full Factorial (2^k)](#43-2-level-full-factorial-2k)
  - [4.4 3-Level Full Factorial (3^k)](#44-3-level-full-factorial-3k)
  - [4.5 2-Level Fractional Factorial (regular)](#45-2-level-fractional-factorial-regular)
  - [4.6 Latin Hypercube Sampling (LHS)](#46-latin-hypercube-sampling-lhs)
- [5. Response variables & scoring (multi-response “Results”)](#5-response-variables--scoring-multi-response-results)
- [6. Quadratic model (what is fitted)](#6-quadratic-model-what-is-fitted)
- [7. Visual diagnostics (what each plot means)](#7-visual-diagnostics-what-each-plot-means)
- [8. Optimization (grid search)](#8-optimization-grid-search)
- [9. CSV format (download + upload)](#9-csv-format-download--upload)
- [10. Troubleshooting (common errors)](#10-troubleshooting-common-errors)
- [11. Best practices (practical lab advice)](#11-best-practices-practical-lab-advice)

---

## 1. Quick start

### 1.1 Create and activate environment
```bash
conda create -n doe_toolkit python=3.11 -y
conda activate doe_toolkit
pip install streamlit plotly pandas numpy pillow
```

### 1.2 Run the app

```bash
streamlit run app.py
```

---

## 2. Concepts (coded space vs real units)

### Coded space

Each factor is internally converted to a **coded scale**:

* **Low**  = `-1`
* **Center** = `0`
* **High** = `+1`

Coded space is used because:

* it makes coefficients comparable across factors
* it is the standard for DoE modeling
* it simplifies surface/contour plots

### Real units

You define Low/Center/High in your **real experimental units** (°C, min, mL, %, etc.).

The app produces a **real-unit run sheet** for lab execution.

> Important: CCD includes axial points at ±sqrt(k), so coded values may go beyond ±1.

---

## 3. App workflow (Step-by-step)

### STEP 1 — Design

1. Choose **number of responses** (1–5)
2. Name each response and select a goal:

   * **Maximize** (higher is better)
   * **Minimize** (lower is better)
3. Choose a **design type**
4. Choose number of factors (**k**)
5. Define each factor:

   * name
   * low, center, high (real units)
6. Click **Generate Design**
7. Download the CSV run sheet (**separator = ;**)

### STEP 2 — Results (sidebar)

1. Upload the completed CSV (same file, now filled with measurements)
2. Set a weight for each response
3. App computes **Results** (a combined score)
4. You get:

   * preview table
   * histogram of Results
   * a quick response vs Results plot

### STEP 3 — Model & Visualize

1. Fit the **quadratic model**
2. Inspect:

   * R² (in-sample)
   * Observed vs Predicted
   * Residuals vs Predicted (±1σ, ±2σ bands)
   * coefficient magnitude plot
3. Explore:

   * 2D contour surfaces
   * 3D response surfaces
4. Run grid search to find **best predicted conditions** (coded + real)

---

## 4. Design types (what they are, when to use, factor limits, run counts)

This section details **every design available in the app**.

### General note on factor limits in this app

The UI slider for `k` is typically **2 to 6** to keep designs reasonable in size and keep the app responsive.

You *can* increase it in code, but some designs explode in run count.

---

## 4.1 Box–Behnken

### What it is

A classic **3-level response surface design** built from combinations where **two factors vary** at a time (±1) while the others stay at 0.

### Best for

* optimization when you expect curvature
* when you want a relatively small run count
* safe designs that avoid extreme corners (no all-high/all-low corners)

### Requirements / limits

* Requires **k ≥ 3**
* Uses coded levels: `-1, 0, +1`

### Run count (this implementation)

Runs are:

* 4 × C(k, 2) + 1 center point

So:

* k=3 → 4×3 + 1 = **13**
* k=4 → 4×6 + 1 = **25**
* k=5 → 4×10 + 1 = **41**
* k=6 → 4×15 + 1 = **61**

### When *not* to use

* if you must test extreme corners (all factors high/low together)
* if you need strong extrapolation behavior at boundaries

---

## 4.2 Central Composite (CCD)

### What it is

A response surface design combining:

* a **2-level factorial core** (±1)
* **axial (star) points** (±α)
* **center point** (0)

This app uses:

* α = sqrt(k) (a common “rotatable-ish” choice)

### Best for

* strong curvature modeling
* exploring beyond the basic ±1 range
* when you want a more “complete” quadratic estimation

### Requirements / limits

* Works for **k ≥ 2**
* Includes coded values beyond ±1 (axial points)

### Run count (this implementation)

Runs are:

* 2^k (factorial) + 2k (axial) + 1 (center)

Examples:

* k=2 → 4 + 4 + 1 = **9**
* k=3 → 8 + 6 + 1 = **15**
* k=4 → 16 + 8 + 1 = **25**
* k=5 → 32 + 10 + 1 = **43**
* k=6 → 64 + 12 + 1 = **77**

### Practical note

Axial points can represent conditions **outside** your defined Low/High range if interpreted literally.
In this app, axial coded values are mapped linearly beyond ±1.

---

## 4.3 2-Level Full Factorial (2^k)

### What it is

All combinations of factors at:

* `-1` (low)
* `+1` (high)

### Best for

* fast screening of main effects and interactions
* when curvature is not the primary goal
* simple “what matters?” studies

### Requirements / limits

* Works for **k ≥ 2**
* Only two levels per factor

### Run count

Runs are:

* 2^k

Examples:

* k=2 → **4**
* k=3 → **8**
* k=4 → **16**
* k=5 → **32**
* k=6 → **64**

### Limitation

Cannot estimate curvature (no 0 level), unless you add center points manually.

---

## 4.4 3-Level Full Factorial (3^k)

### What it is

All combinations of:

* `-1`, `0`, `+1`
  for each factor.

### Best for

* complete exploration when k is small
* situations where you want full symmetry and curvature information

### Requirements / limits

* Works for **k ≥ 2**
* Becomes huge quickly

### Run count

Runs are:

* 3^k

Examples:

* k=2 → **9**
* k=3 → **27**
* k=4 → **81**
* k=5 → **243**
* k=6 → **729** (usually too big for real lab work)

---

## 4.5 2-Level Fractional Factorial (regular)

### What it is

A reduced factorial design created by:

* selecting **base_k** independent factors (A, B, C…)
* defining remaining factors as **products** of base factors (generators)

This reduces runs but introduces **aliasing** (confounding).

### Best for

* screening many factors with limited runs
* early stage exploration to identify important factors

### Requirements / limits

* Total factors: `k`
* Base factors: `base_k` must satisfy:

  * **2 ≤ base_k ≤ k-1** (unless k=2, then it equals full factorial)
* Runs are:

  * 2^(base_k)

### Run count examples

If k=5 total factors:

* base_k=4 → runs=16 (half fraction of 32)
* base_k=3 → runs=8 (quarter fraction of 32)
* base_k=2 → runs=4 (very low resolution)

### Generators (how to write them)

Generators must use only base letters:

* `AB` means A×B
* `ABC` means A×B×C

**No spaces, order doesn’t matter.**

### Important warning about aliasing

In fractional designs, some effects can be confounded:

* main effects can be aliased with interactions (depending on resolution)

Use fractional designs mainly to **discover what matters**, then follow up with:

* CCD
* Box–Behnken
* full factorial + center points

---

## 4.6 Latin Hypercube Sampling (LHS)

### What it is

A **space-filling** sampling method (not a classic orthogonal DoE).

Each factor range is divided into bins, and sampling ensures uniform coverage per factor.

### Best for

* many factors (k ≥ 5)
* expensive experiments
* building datasets for ML
* broad exploration

### Requirements / limits

* Choose:

  * number of runs: `n_runs`
  * random seed for reproducibility
* Coded values are approximately in `[-1, +1]`

### Choosing n_runs (rule of thumb)

* Minimum: ~2×k
* Better: ~4×k
* Very good: 6–8×k

Examples:

* k=5 → 10 (minimum), 20 (better), 30–40 (very good)

### Limitation

LHS does not guarantee:

* orthogonality
* clean effect estimation
* balanced interactions

It guarantees:

* coverage of the factor space

---

## 5. Response variables & scoring (multi-response “Results”)

You can define **1 to 5** responses.

For each response:

* choose **Maximize** or **Minimize**
* choose a **weight**

### How the app computes Results

For each response:

1. Convert to numeric (non-numeric becomes NaN)
2. Standardize via z-score:

  * z = (x - mean) / std

3. Weighted sum:

   * maximize: `+ weight*z`
   * minimize: `- weight*z`

Finally:

* `Results = sum(all weighted z-scores)`

### Interpretation

* Higher `Results` means “better” considering all objectives.

---

## 6. Quadratic model (what is fitted)

The app fits a quadratic regression in **coded space**:

$$
y = b_0 + \sum b_i x_i + \sum b_{ij} x_i x_j + \sum b_{ii} x_i^2
$$

Where:

* `b0` = intercept
* `bi` = main effects
* `bij` = pairwise interactions
* `bii` = curvature terms

### How it is fit

* Ordinary least squares (NumPy `lstsq`)
* In-sample R² is reported

---

## 7. Visual diagnostics (what each plot means)

### R² (in-sample)

* Higher = model explains more variance
* But it can be optimistic (no CV yet)

### Observed vs Predicted

Good:

* points near diagonal y=x
  Bad:
* curved pattern → missing structure
* clusters → batch effect / drift

### Residuals vs Predicted (±1σ and ±2σ)

Good:

* random scatter around 0
  Bad:
* funnel shape → heteroscedasticity
* U-shape → wrong model form
* many points beyond ±2σ → outliers or poor model

### Coefficient magnitudes

Shows the largest terms by absolute coefficient.
Useful to identify:

* dominant factors
* dominant interactions
* strong curvature

---

## 8. Optimization (grid search)

The app runs a brute-force grid search in **coded space**:

* you choose search min/max (e.g., -1..+1 or -2..+2)
* you choose step size

It returns:

* best predicted `Results` (coded)
* converted best point in **real units**

### Practical interpretation

* optimum at boundary → expand range and re-run DoE
* optimum inside space → good candidate for validation
* always validate experimentally

---

## 9. CSV format (download + upload)

### Downloaded file (Step 1)

* Separator: `;`
* Columns include:

  * `Experiment#`
  * factor columns (real units)
  * response columns (empty)
  * `Results` (empty)

### Upload file (Step 2)

You upload the same CSV after filling response measurements.

Requirements:

* Response columns must exist exactly as named in Step 1
* Values should be numeric (text will be coerced to NaN)

---

## 10. Troubleshooting (common errors)

### “score is not defined”

Fix:

* make sure `score = 0.0` is defined before summing.

### Plotly Express error about missing DataFrame

This happens when you pass `None` to `px.*` functions.
Fix:

* only plot if `results_df` exists and is not empty.

### “colA is not defined”

This happens when using:

```python
with colA:
```

outside the scope where `colA, colB = st.columns(...)` is defined.
Fix:

* define `colA, colB` inside the same block where you use them, or avoid referencing them in the “no results yet” branch.

---

## 11. Best practices (practical lab advice)

* Start with **screening** if you have many factors:

  * Fractional factorial or LHS
* Then optimize with:

  * Box–Behnken or CCD
* Always include **replicates / center points** for noise estimation (future improvement)
* If your system drifts (instrument/time/batch):

  * randomize run order
  * include QC runs
* Validate optimum conditions experimentally (model ≠ truth)

---

## License / Citation

If you use this app for academic work, consider citing:

* the GitHub repository
* your lab/institution (IPPN-UFRJ / LAABio / LabMAS / LabCrom, etc.)

---
