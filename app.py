# ==========================================================
# DoE for Chromatography (NumPy-only designs)
# + Quadratic model + visualization + response surfaces
# Works with Python 3.12+
# ==========================================================

import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# MUST be the first Streamlit call
st.set_page_config(page_title="DoE Chromatography", layout="wide")

# -----------------------------
# LOGOS — AFTER page config
# -----------------------------
from pathlib import Path
from PIL import Image

STATIC_DIR = Path(__file__).parent / "static"

laabio_logo = STATIC_DIR / "LAABio.png"
doe_logo = STATIC_DIR / "logo_DoE.png"

col_left, col_center, col_right = st.columns([1.2, 2, 1.2])

# Left logo (LAABio)
if laabio_logo.exists():
    try:
        with col_left:
            st.image(Image.open(laabio_logo), width=150)
    except Exception:
        pass

# Center logo (Main DoE branding)
if doe_logo.exists():
    try:
        with col_center:
            st.image(Image.open(doe_logo), width=150)
    except Exception:
        pass

# Right column intentionally empty for symmetry

st.title("Design of Experiments (DoE) for Chromatography")

st.markdown("""
This tool allows you to:

1. Generate experimental designs
2. Run experiments
3. Upload results
4. Fit a **quadratic** model (main + interactions + curvature)
5. Visualize response surfaces (2D contour + 3D surface)
6. Suggest optimal conditions
""")

# ==========================================================
# DESIGN GENERATORS (Pure NumPy)
# ==========================================================

def full_factorial_2level(k: int) -> np.ndarray:
    return np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)

def box_behnken(k: int) -> np.ndarray:
    if k < 3:
        raise ValueError("Box-Behnken requires at least 3 factors.")
    design = []
    for i in range(k):
        for j in range(i + 1, k):
            for pair in itertools.product([-1, 1], repeat=2):
                row = [0] * k
                row[i] = pair[0]
                row[j] = pair[1]
                design.append(row)
    design.append([0] * k)  # center point
    return np.array(design, dtype=float)

def central_composite(k: int) -> np.ndarray:
    factorial = full_factorial_2level(k)
    axial = []
    alpha = float(np.sqrt(k))  # rotatable-ish default
    for i in range(k):
        row_pos = [0] * k
        row_neg = [0] * k
        row_pos[i] = alpha
        row_neg[i] = -alpha
        axial.append(row_pos)
        axial.append(row_neg)
    center = np.zeros((1, k), dtype=float)
    return np.vstack([factorial, np.array(axial, dtype=float), center])


def full_factorial_3level(k: int) -> np.ndarray:
    """
    3-level full factorial in coded space: {-1, 0, +1}
    Runs = 3^k (can explode quickly).
    """
    levels = [-1.0, 0.0, 1.0]
    return np.array(list(itertools.product(levels, repeat=k)), dtype=float)


def lhs_design(n_runs: int, k: int, seed: int = 123) -> np.ndarray:
    """
    Latin Hypercube Sampling in coded space approximately in [-1, +1].
    Good for low-resolution exploration when k is large.
    """
    rng = np.random.default_rng(seed)
    # Stratify [0,1] into n bins; sample one from each bin per factor
    H = np.zeros((n_runs, k), dtype=float)
    for j in range(k):
        cut = np.linspace(0, 1, n_runs + 1)
        u = rng.random(n_runs)
        pts = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(pts)
        H[:, j] = pts
    # map [0,1] -> [-1,1]
    return 2.0 * H - 1.0


def _parse_generator(gen: str, base_names: list[str]) -> list[int]:
    """
    gen like 'ABC' means product of columns A*B*C.
    """
    g = gen.strip().upper().replace(" ", "")
    if g == "":
        raise ValueError("Empty generator.")
    idx = []
    for ch in g:
        if ch not in base_names:
            raise ValueError(f"Generator '{gen}' uses unknown base factor '{ch}'. Base={base_names}")
        idx.append(base_names.index(ch))
    return idx


def fractional_factorial_2level_regular(
    base_k: int,
    generators: list[str],
) -> np.ndarray:
    """
    Regular 2-level fractional factorial.

    You build a 2^(base_k) full factorial in base factors A,B,C,...
    Then add derived factors defined by generators, e.g.:
      generators=["AB", "ACD"] means:
        D = A*B
        E = A*C*D   (BUT note: 'D' is not base; so keep generators only in base letters A..)
    For simplicity: generators must use ONLY base letters A.. (no derived letters).

    Total factors = base_k + len(generators)
    Runs = 2^(base_k)
    """
    if base_k < 2:
        raise ValueError("base_k must be >= 2")

    base_names = [chr(ord("A") + i) for i in range(base_k)]
    base = full_factorial_2level(base_k)  # shape: (2^base_k, base_k)

    derived_cols = []
    for gen in generators:
        idx = _parse_generator(gen, base_names)
        col = np.prod(base[:, idx], axis=1).reshape(-1, 1)
        derived_cols.append(col)

    if len(derived_cols) > 0:
        return np.hstack([base] + derived_cols).astype(float)

    return base.astype(float)

# ==========================================================
# MODEL HELPERS
# ==========================================================

def build_quadratic_matrix(df_coded: pd.DataFrame, factor_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Quadratic model basis:
      1
      x_i
      x_i*x_j  (i<j)
      x_i^2
    """
    n = len(df_coded)
    X_parts = [np.ones((n, 1), dtype=float)]
    names = ["Intercept"]

    # linear
    for c in factor_cols:
        X_parts.append(df_coded[[c]].to_numpy(dtype=float))
        names.append(c)

    # interactions
    for i in range(len(factor_cols)):
        for j in range(i + 1, len(factor_cols)):
            xi = df_coded[factor_cols[i]].to_numpy(dtype=float)
            xj = df_coded[factor_cols[j]].to_numpy(dtype=float)
            X_parts.append((xi * xj).reshape(-1, 1))
            names.append(f"{factor_cols[i]}*{factor_cols[j]}")

    # squares
    for c in factor_cols:
        xi = df_coded[c].to_numpy(dtype=float)
        X_parts.append((xi ** 2).reshape(-1, 1))
        names.append(f"{c}^2")

    X = np.hstack(X_parts)
    return X, names

def fit_lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return coef.flatten(), yhat.flatten()

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def predict_quadratic(point: dict, coef: np.ndarray, factor_cols: list[str]) -> float:
    """
    point: dict factor->coded float
    coef: fitted coefficients aligned with build_quadratic_matrix()
    """
    x = np.array([point[c] for c in factor_cols], dtype=float)
    row = [1.0]                       # intercept
    row.extend(list(x))               # linear
    for i in range(len(x)):           # interactions
        for j in range(i + 1, len(x)):
            row.append(x[i] * x[j])
    for i in range(len(x)):           # squares
        row.append(x[i] ** 2)
    return float(np.dot(np.array(row, dtype=float), coef))

def coded_to_real_value(coded_val: float, spec: dict) -> float:
    """
    Map coded value to real value by linear interpolation using low/center/high.
    Handles coded values beyond [-1,1] (e.g., CCD axial points).
    """
    low, center, high = float(spec["low"]), float(spec["center"]), float(spec["high"])
    if coded_val == 0:
        return center
    # piecewise linear: [-1..0] and [0..+1], extend linearly beyond
    if coded_val < 0:
        # between low (-1) and center (0)
        return center + coded_val * (center - low)
    else:
        # between center (0) and high (+1)
        return center + coded_val * (high - center)

# ----------------------------------------------------------
# Session state initialization (prevents KeyError)
# ----------------------------------------------------------
defaults = {
    "coded": None,
    "design": None,
    "factor_specs": [],   # <-- IMPORTANT: list, not None
    "results_df": None,
}
for k0, v0 in defaults.items():
    if k0 not in st.session_state:
        st.session_state[k0] = v0

# ==========================================================
# UI TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(["STEP 1 — Design", 
    "STEP 2 — Results", 
    "STEP 3 — Model & Visualize"])

# ----------------------------------------------------------
# STEP 1 — DESIGN
# ----------------------------------------------------------
with tab1:
    st.header("STEP 1 — Create Experimental Design")

    st.info(
        """
**Pick a design**, define factor names and LOW/CENTER/HIGH values, then generate and download the table.

Tips:
- If you are not sure: **Box–Behnken**
- If you want axial points and better curvature estimation: **Central Composite**
- If you want fast screening with fewer runs: **Fractional factorial** or **LHS**
"""
    )

    design_type = st.selectbox(
        "Choose Design Type",
        [
            "Box-Behnken (recommended)",
            "Central Composite",
            "2-Level Full Factorial (2^k)",
            "3-Level Full Factorial (3^k)",
            "2-Level Fractional Factorial (regular)",
            "Latin Hypercube (low-resolution)",
        ],
        index=0,
    )

    k = st.slider("Number of Factors", 2, 6, 3)

    # ---------------------------------------
    # Extra settings (only used by some designs)
    # ---------------------------------------
    base_k = None
    frac_generators: list[str] = []
    lhs_runs = None
    lhs_seed = 123

    # ---------------------------------------
    # Fractional factorial settings
    # ---------------------------------------
    if design_type.startswith("2-Level Fractional"):
        st.markdown("### Fractional Factorial — Settings")

        st.info(
            """
A fractional factorial reduces the number of experiments by **aliasing** (confounding) some effects.

How it works:
- Choose **base factors** (independent: A, B, C, ...)
- Define the remaining factors as **products** of base factors (generators)

This can shrink the design dramatically.

Example:
k = 5 total factors, base_k = 3  →  runs = 2³ = 8 (instead of 2⁵ = 32)
"""
        )

        # --- Base factor selection (safe version)

        if k <= 2:
            # Only possible option is full factorial
            base_k = k
            st.info("With only 2 factors, fractional design is identical to full factorial (2² = 4 runs).")
        else:
            base_k = st.slider(
                "Number of base factors (independent A, B, C, ...)",
                min_value=2,
                max_value=k - 1,   # IMPORTANT FIX
                value=min(3, k - 1),
                key="frac_base_k",
            )

        st.success(
            f"Runs (fractional) = **{2**base_k}**   |   Runs (full 2-level) = {2**k}"
        )

        derived_k = k - base_k

        if derived_k == 0:
            st.info("No derived factors (k = base_k) → this becomes a full 2-level factorial.")
        else:
            st.markdown("#### Generators for derived factors")
            st.caption(
                """
Each derived factor must be defined as a product of base factors.

Examples (valid generators):
- AB   → A × B
- AC   → A × C
- ABC  → A × B × C

Rules:
- Use only the base letters (A..)
- No spaces
- Order does not matter (AB = BA)
"""
            )

            frac_generators = []
            for gi in range(derived_k):
                frac_generators.append(
                    st.text_input(
                        f"Generator for derived factor #{gi+1}",
                        value="AB",
                        key=f"frac_gen_{gi}",
                    ).strip().upper()
                )

            st.warning(
                """
⚠️ Interpretation warning (aliasing):
- In lower-resolution fractionals, some **main effects** can be confounded with **2-factor interactions**.
- Use fractional designs mainly for **screening** (finding what matters), not final optimization.
"""
            )

    # ---------------------------------------
    # LHS settings
    # ---------------------------------------
    if design_type.startswith("Latin Hypercube"):
        st.markdown("### Latin Hypercube Sampling (LHS) — Settings")

        st.info(
            """
LHS is a **space-filling** design.

It is great when you have many factors and need **fewer runs** than factorial designs.
It is NOT a classic “effect-estimation” design like factorials.

Good for:
- many factors (k ≥ 5)
- expensive experiments
- broad exploration / response surfaces
- ML training data

Not ideal for:
- clean main-effect / interaction interpretation
"""
        )

        lhs_runs = st.slider(
            "Number of runs (LHS)",
            min_value=max(8, 2 * k),
            max_value=200,
            value=4 * k,
            key="lhs_runs",
        )

        st.caption(
            f"""
Rule of thumb:
- Minimum: 2 × k = {2*k}
- Better: 4 × k = {4*k}
- Very good coverage: 6–8 × k = {6*k}–{8*k}
"""
        )

        lhs_seed = st.number_input(
            "Random seed (reproducibility)",
            value=123,
            step=1,
            key="lhs_seed",
        )

        st.warning(
            """
⚠️ LHS does NOT guarantee:
- orthogonality
- balanced interaction estimation

It guarantees:
- uniform coverage of each factor range
"""
        )

    # ---------------------------------------
    # Factor definitions (names + low/center/high)
    # ---------------------------------------
    factor_specs = []
    for i in range(k):
        with st.expander(f"Factor {i+1}", expanded=True):
            name = st.text_input("Factor Name", value=f"Factor_{i+1}", key=f"name_{i}")
            low = st.number_input("Low (-1)", value=5.0, key=f"low_{i}")
            center = st.number_input("Center (0)", value=15.0, key=f"center_{i}")
            high = st.number_input("High (+1)", value=25.0, key=f"high_{i}")

            factor_specs.append({"name": name, "low": low, "center": center, "high": high})

            st.caption("Coded values: -1=Low, 0=Center, +1=High (CCD may include ±sqrt(k)).")

    # ---------------------------------------
    # Generate design
    # ---------------------------------------
    if st.button("Generate Design", type="primary"):
        # NOTE: you will plug the new generators here:
        # - full_factorial_3level(k)
        # - fractional_factorial_2level(k, base_k, generators)
        # - lhs_design(k, lhs_runs, seed=lhs_seed)
        if design_type.startswith("Box"):
            coded = box_behnken(k)
        elif design_type.startswith("Central"):
            coded = central_composite(k)
        elif design_type.startswith("3-Level Full"):
            coded = full_factorial_3level(k)  # <-- you will implement this
        elif design_type.startswith("2-Level Fractional"):
            coded = fractional_factorial_2level(k, base_k, frac_generators)  # <-- implement this
        elif design_type.startswith("Latin Hypercube"):
            coded = lhs_design(k, lhs_runs, seed=lhs_seed)  # <-- implement this
        else:
            coded = full_factorial_2level(k)

        factor_names = [f["name"] for f in factor_specs]
        coded_df = pd.DataFrame(coded, columns=factor_names)

        # Real-valued table for the lab
        real_df = coded_df.copy()
        for f in factor_specs:
            real_df[f["name"]] = real_df[f["name"]].apply(lambda v: coded_to_real_value(float(v), f))

        real_df.insert(0, "Experiment#", np.arange(1, len(real_df) + 1))
        real_df["Number of Peaks"] = ""
        real_df["Run Time"] = ""
        real_df["Results"] = ""

        st.session_state["coded"] = coded_df
        st.session_state["design"] = real_df
        st.session_state["factor_specs"] = factor_specs

        st.success(f"Design created with {len(real_df)} runs.")

    # ==========================================================
    # DESIGN VISUALIZATION
    # ==========================================================
    st.divider()
    st.subheader("Design Visualization")

    coded_df = st.session_state.get("coded")
    real_df = st.session_state.get("design")
    factor_specs_state = st.session_state.get("factor_specs", [])

    if coded_df is None or real_df is None or len(factor_specs_state) == 0:
        st.info("Generate a design first (click **Generate Design**) to see the design plots.")
    else:
        factor_cols = [f["name"] for f in factor_specs_state]

        space = st.radio(
            "Plot design in:",
            ["Coded Space (-1 to +1)", "Real Units"],
            horizontal=True,
            key="design_space",
        )

        plot_df = coded_df.copy() if space.startswith("Coded") else real_df[factor_cols].copy()

        col1, col2 = st.columns(2)
        with col1:
            fx = st.selectbox("X axis", factor_cols, key="design_fx")
        with col2:
            fy = st.selectbox("Y axis", factor_cols, index=1 if len(factor_cols) > 1 else 0, key="design_fy")

        # (optional) Add a simple 2D plot — useful even with 3+ factors
        #st.markdown("### 2D Projection")
        #fig2d = px.scatter(plot_df, x=fx, y=fy, text=np.arange(1, len(plot_df) + 1),
        #                   title=f"Design Points — {fx} vs {fy}")
        #fig2d.update_traces(marker=dict(size=10))
        #fig2d.update_layout(height=520)
        #st.plotly_chart(fig2d, use_container_width=True)

        if len(factor_cols) >= 3:
            st.markdown("### 3D Projection")
            fz = st.selectbox("Z axis", factor_cols, index=2, key="design_fz")
            fig3d = px.scatter_3d(plot_df, x=fx, y=fy, z=fz, text=np.arange(1, len(plot_df) + 1),
                                  title=f"3D Design — {fx}, {fy}, {fz}")
            fig3d.update_traces(marker=dict(size=6))
            fig3d.update_layout(height=650)
            st.plotly_chart(fig3d, use_container_width=True)

    # Experimental Table
    design_df = st.session_state.get("design")

    if design_df is not None:
        st.subheader("Experimental Table (Real units for the lab)")
        st.dataframe(design_df, use_container_width=True, height=380)

        st.download_button(
            "Download Experimental Table (CSV ;)",
            design_df.to_csv(index=False, sep=";"),
            file_name="Experimental_Table.csv",
            mime="text/csv",
        )
    else:
        st.info("Generate a design to see the experimental table.")

# ----------------------------------------------------------
# STEP 2 — RESULTS
# ----------------------------------------------------------
with tab2:
    st.header("STEP 2 — Upload Completed CSV & Compute Results")

    # Guard: you must have generated a design first
    if st.session_state.get("design") is None:
        st.info("Generate the design in STEP 1 first.")
        st.stop()

    # Upload
    uploaded = st.file_uploader("Upload Completed CSV (separator ';')", type=["csv"])

    if uploaded is None:
        st.info("Upload the CSV you downloaded from STEP 1 (filled with Number of Peaks and Run Time).")
        st.stop()

    df = pd.read_csv(uploaded, sep=";")

    # Basic validation
    required = ["Number of Peaks", "Run Time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    st.subheader("Preview uploaded data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Define how to compute **Results**")
    st.caption("Default: more peaks = better; shorter run time = better.")

    c1, c2, c3 = st.columns(3)
    with c1:
        w_peaks = st.number_input("Weight: Peaks (positive)", value=1.0)
    with c2:
        w_time = st.number_input("Weight: Run Time (negative)", value=1.0)
    with c3:
        overwrite = st.checkbox("Overwrite Results column", value=True)

    # Compute results
    peaks = pd.to_numeric(df["Number of Peaks"], errors="coerce")
    runtime = pd.to_numeric(df["Run Time"], errors="coerce")

    if peaks.isna().any() or runtime.isna().any():
        st.warning(
            "Some values in Number of Peaks or Run Time are missing/non-numeric. "
            "Those rows will be ignored later when fitting the model."
        )

    peaks_std = float(peaks.std()) if float(peaks.std()) != 0.0 else 1.0
    time_std = float(runtime.std()) if float(runtime.std()) != 0.0 else 1.0

    peaks_z = (peaks - peaks.mean()) / peaks_std
    time_z = (runtime - runtime.mean()) / time_std

    results = w_peaks * peaks_z - w_time * time_z

    if overwrite or ("Results" not in df.columns):
        df["Results"] = results

    st.session_state["results_df"] = df
    st.success("Results computed and stored.")

    st.subheader("Quick plots (raw)")
    colA, colB = st.columns(2)

    with colA:
        fig = px.histogram(df, x="Results", nbins=20, title="Results distribution")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.scatter(
            df,
            x="Run Time",
            y="Number of Peaks",
            color="Results",
            title="Peaks vs Run Time (colored by Results)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# STEP 3 — MODEL & VISUALIZE
# ----------------------------------------------------------
with tab3:
    st.header("STEP 3 — Fit Quadratic Model & Visualize Surfaces")

    # --- pull state safely
    df_state = st.session_state.get("results_df")
    coded_df = st.session_state.get("coded")
    factor_specs = st.session_state.get("factor_specs", [])

    # --- guard conditions (avoid None crashes)
    if df_state is None:
        st.info("Upload results in STEP 2 first.")
        st.stop()

    if coded_df is None or len(factor_specs) == 0:
        st.error("Missing design metadata. Please regenerate design in STEP 1 and re-upload results.")
        st.stop()

    # --- local copies
    df = df_state.copy()
    coded_df = coded_df.copy()

    factor_cols = [f["name"] for f in factor_specs]

    # align row count
    if len(df) != len(coded_df):
        st.warning("Uploaded table row count differs from the coded design. Fitting will use matching first N rows.")
        n = min(len(df), len(coded_df))
        df = df.iloc[:n].reset_index(drop=True)
        coded_df = coded_df.iloc[:n].reset_index(drop=True)

    y = pd.to_numeric(df["Results"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)

    if ok.sum() < max(8, 2 * len(factor_cols)):
        st.error("Not enough valid Results values to fit a quadratic model.")
        st.stop()

    df_fit = df.loc[ok].reset_index(drop=True)
    coded_fit = coded_df.loc[ok].reset_index(drop=True)

    X, term_names = build_quadratic_matrix(coded_fit, factor_cols)
    coef, yhat = fit_lstsq(X, df_fit["Results"].to_numpy(dtype=float))
    r2v = r2_score(df_fit["Results"].to_numpy(dtype=float), yhat)

    st.subheader("Model quality")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² (in-sample)", f"{r2v:.3f}")
    with c2:
        st.caption("If R² is low: either the response is noisy or your factor ranges are too narrow/too wide.")

    st.subheader("Observed vs Predicted")
    ovp = pd.DataFrame({"Observed": df_fit["Results"].to_numpy(dtype=float), "Predicted": yhat})
    fig = px.scatter(ovp, x="Observed", y="Predicted", trendline="ols", title="Observed vs Predicted")
    st.plotly_chart(fig, use_container_width=True)
    st.info("""
    ### How to interpret **Observed vs Predicted**

    **Good model:**
    - Points close to the diagonal (y = x).
    - No systematic deviation.

    **Bad model:**
    - Wide scatter → low predictive power.
    - Curved pattern → missing terms or wrong factor range.
    - Two clouds / clusters → unmodeled grouping effect (batch/day/instrument drift).
    """)  

    st.subheader("Residuals (with ±1σ and ±2σ bands)")

    resid = (ovp["Observed"] - ovp["Predicted"]).to_numpy(dtype=float)
    pred = ovp["Predicted"].to_numpy(dtype=float)

    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    # Sort by predicted for nicer band drawing
    order = np.argsort(pred)
    pred_s = pred[order]
    resid_s = resid[order]

    fig_res = go.Figure()

    # --- shaded band: ±2σ
    fig_res.add_trace(
        go.Scatter(
            x=np.concatenate([pred_s, pred_s[::-1]]),
            y=np.concatenate([2*sigma*np.ones_like(pred_s), (-2*sigma)*np.ones_like(pred_s)[::-1]]),
            fill="toself",
            name="±2σ band",
            hoverinfo="skip",
            line=dict(width=0),
            opacity=0.18,
            showlegend=True,
        )
    )

    # --- shaded band: ±1σ
    fig_res.add_trace(
        go.Scatter(
            x=np.concatenate([pred_s, pred_s[::-1]]),
            y=np.concatenate([1*sigma*np.ones_like(pred_s), (-1*sigma)*np.ones_like(pred_s)[::-1]]),
            fill="toself",
            name="±1σ band",
            hoverinfo="skip",
            line=dict(width=0),
            opacity=0.28,
            showlegend=True,
        )
    )

    # --- reference lines: 0, ±1σ, ±2σ
    for yline, nm in [(0.0, "0"), (sigma, "+1σ"), (-sigma, "-1σ"), (2*sigma, "+2σ"), (-2*sigma, "-2σ")]:
        fig_res.add_trace(
            go.Scatter(
                x=[pred_s.min(), pred_s.max()],
                y=[yline, yline],
                mode="lines",
                name=nm,
                hoverinfo="skip",
                line=dict(dash="dash", width=1),
                showlegend=(nm in ["0", "+1σ", "+2σ"]),  # avoid legend clutter
            )
        )

    # --- residual points
    fig_res.add_trace(
        go.Scatter(
            x=pred,
            y=resid,
            mode="markers",
            name="Residuals",
            marker=dict(size=8),
        )
    )

    fig_res.update_layout(
        title="Residuals vs Predicted (shaded ±1σ and ±2σ)",
        xaxis_title="Predicted",
        yaxis_title="Residual",
        height=520,
    )

    st.plotly_chart(fig_res, use_container_width=True)
    st.info("""
    ### How to interpret **Residuals vs Predicted** (with ±1σ and ±2σ bands)

    Residual = Observed − Predicted.

    **Good model:**
    - Points randomly scattered around 0 (no pattern).
    - Most points inside **±1σ**.
    - Almost all points inside **±2σ**.

    **Red flags:**
    - Curved “U-shape” → missing curvature / wrong quadratic structure.
    - Funnel shape (wider spread at high predicted) → non-constant variance.
    - Trend (residuals drift up/down) → missing interaction or wrong mapping.
    - Many points outside **±2σ** → outliers, bad runs, or model not capturing reality.

    Rule of thumb: residual plot should look like random noise, not a drawing.
    """)

    st.subheader("Coefficient magnitudes (Pareto-like view)")
    coef_df = pd.DataFrame({"Term": term_names, "Coefficient": coef})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).reset_index(drop=True)
    fig = px.bar(coef_df.head(20), x="Term", y="Abs", title="Top coefficient magnitudes (|coef|)")
    st.plotly_chart(fig, use_container_width=True)


    st.divider()
    st.subheader("Response surfaces (2D contour + 3D surface)")

    st.caption("""
    Choose two factors to plot.  
    All other factors will be fixed at a chosen coded value (default 0 = center).
    """)

    colA, colB, colC = st.columns([1.2, 1.2, 1.6])
    with colA:
        fx = st.selectbox("X factor", factor_cols, index=0)
    with colB:
        fy = st.selectbox("Y factor", factor_cols, index=1 if len(factor_cols) > 1 else 0)
    with colC:
        space = st.radio("Plot axis units", ["Coded (-1..+1)", "Real units"], horizontal=True)

    fixed = {}
    for f in factor_cols:
        if f not in (fx, fy):
            fixed[f] = st.slider(f"Fix {f} (coded)", -1.5, 1.5, 0.0, 0.05)

    grid_n = st.slider("Surface grid resolution", 25, 90, 55)
    gx = np.linspace(-1.5, 1.5, grid_n)
    gy = np.linspace(-1.5, 1.5, grid_n)

    Z = np.zeros((grid_n, grid_n), dtype=float)
    for i, xv in enumerate(gx):
        for j, yv in enumerate(gy):
            point = {f: fixed.get(f, 0.0) for f in factor_cols}
            point[fx] = float(xv)
            point[fy] = float(yv)
            Z[j, i] = predict_quadratic(point, coef, factor_cols)

    # axis mapping
    if space == "Real units":
        spec_x = next(s for s in factor_specs if s["name"] == fx)
        spec_y = next(s for s in factor_specs if s["name"] == fy)
        gx_plot = np.array([coded_to_real_value(v, spec_x) for v in gx], dtype=float)
        gy_plot = np.array([coded_to_real_value(v, spec_y) for v in gy], dtype=float)
        x_label = fx
        y_label = fy
    else:
        gx_plot = gx
        gy_plot = gy
        x_label = f"{fx} (coded)"
        y_label = f"{fy} (coded)"

    # 2D contour
    fig2d = go.Figure(
        data=go.Contour(
            x=gx_plot,
            y=gy_plot,
            z=Z,
            contours_coloring="heatmap",
            showscale=True,
        )
    )
    fig2d.update_layout(
        title=f"2D Contour — Predicted Results — {fx} vs {fy}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=560
    )
    st.plotly_chart(fig2d, use_container_width=True)
    st.info("""
    ### How to interpret the contour map:

    Each color = predicted response value  

    • Circular contours → weak interaction  
    • Elliptical contours → interaction present  
    • Tilted ellipses → strong interaction  
    • Sharp curvature → strong quadratic effects  

    The peak region shows where optimum conditions lie.

    Flat map → design range may be too small.
    """)                

    # 3D surface
    fig3d = go.Figure(
        data=[go.Surface(x=gx_plot, y=gy_plot, z=Z)]
    )
    fig3d.update_layout(
        title=f"3D Surface — Predicted Results — {fx} vs {fy}",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title="Predicted Results"
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.info("""
    ### How to interpret the 3D surface:

    • Dome shape → maximum inside design space  
    • Valley → minimum inside space  
    • Ridge → trade-off region  
    • Flat surface → factor not influential  

    Smooth surface → stable system  
    Sharp spikes → unstable region or extrapolation  

    Always verify that the optimum is inside experimental space.
    """)               

    st.divider()
    st.subheader("Optimization (grid search in coded space)")

    st.caption("We search in coded space. If your CCD uses ±sqrt(k), you can expand the search range here.")
    st.info("""
    ### How to interpret optimization:

    The optimizer searches in coded space and finds the highest predicted value.

    Important:
    • If optimum is near boundary → expand factor range and re-run DoE  
    • If optimum is inside → design likely captured real maximum  
    • Always validate predicted optimum experimentally  

    Models guide decisions — experiments confirm them.
    """)
    search_min, search_max = st.slider("Coded search range", -2.0, 2.0, (-1.0, 1.0), 0.1)
    step = st.slider("Grid step (smaller = slower)", 0.02, 0.3, 0.08, 0.01)

    if st.button("Find best conditions", type="primary"):
        grid = np.arange(search_min, search_max + 1e-9, step)
        best_val = -np.inf
        best_point = None

        for point_tuple in itertools.product(grid, repeat=len(factor_cols)):
            point = {factor_cols[i]: float(point_tuple[i]) for i in range(len(factor_cols))}
            val = predict_quadratic(point, coef, factor_cols)
            if val > best_val:
                best_val = val
                best_point = point

        st.success(f"Best predicted Results: {best_val:.4f}")
        st.write("Best coded point:", best_point)

        # convert to real
        real_best = {}
        for spec in factor_specs:
            cval = best_point[spec["name"]]
            real_best[spec["name"]] = coded_to_real_value(cval, spec)

        st.write("Best real conditions:", real_best)




