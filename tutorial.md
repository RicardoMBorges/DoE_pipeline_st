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
