# Transformer Circuit Studio

![License: DACR](https://img.shields.io/badge/License-DACR-blueviolet)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-app-brightgreen)
![PRs](https://img.shields.io/badge/PRs-welcome-orange)
![Last Commit](https://img.shields.io/github/last-commit/emcdo411/transformer-circuit-studio)
![Issues](https://img.shields.io/github/issues/emcdo411/transformer-circuit-studio)

A hands-on sandbox for **mechanistic interpretability**. Train a tiny transformer on CPU, predict arithmetic expressions, and visualize attention patterns in a **modern Streamlit UI**. Clear, extensible, and ideal for learning or reverse-engineering transformer internals.

---

## Table of Contents
- [What is this](#what-is-this)
- [Features](#features)
- [Workflow](#workflow)
- [Folder structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Quick start (Windows PowerShell)](#quick-start-windows-powershell)
- [Quick start (Mac or Linux)](#quick-start-mac-or-linux)
- [Step-by-step guide](#step-by-step-guide)
- [Run the app](#run-the-app)
- [Commit and push to GitHub](#commit-and-push-to-github)
- [Optional: add PyTorch CPU](#optional-add-pytorch-cpu)
- [Optional: deploy to Hugging Face Spaces](#optional-deploy-to-hugging-face-spaces)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## What is this
**Transformer Circuit Studio** is a minimal, free, open-source project for exploring transformer internals through **mechanistic interpretability**.

It includes:
- A tiny transformer trained on CPU for two tasks:
  - **Addition** (e.g., `12+7=` → `19`)
  - **Precalc Eval** (e.g., `(3+2)^2-4/2=` → `23`) using integer division (`/` → `//`)
- A **Streamlit UI** to input expressions, predict results, and visualize **attention heatmaps** (with optional interpretability hooks).
- A clear, modular codebase for students, engineers, and researchers to extend.

**Expected outcomes**
- Train models to predict arithmetic results accurately.
- Visualize how attention distributes over digits and operators.
- Begin reverse-engineering the model’s behavior from interpretable signals.

---

## Features
- **Free & Local:** CPU-only; PyTorch + Streamlit.
- **Two tasks:** Addition and Precalc Eval (`+ - * / ^ ( )`, integer results).
- **Modern UI:** Input box, predicted vs ground truth metrics, attention heatmaps.
- **Extensible:** Swap in your model, tasks, and interpretability tools.
- **Small checkpoints:** Saved under `models/checkpoints/` as `.pt` files.

---

## Workflow
```mermaid
flowchart TD
  %% -------- Styles (dark, professional) --------
  classDef startEnd fill:#0B1220,stroke:#7B5CFF,stroke-width:2px,color:#EAEFF7;
  classDef action   fill:#121826,stroke:#2B395B,stroke-width:1.2px,color:#EAEFF7;
  classDef decision fill:#172235,stroke:#00D6D6,stroke-width:2px,color:#EAEFF7;
  classDef output   fill:#0F1A2B,stroke:#5A3DFF,stroke-width:1.5px,color:#FFFFFF;
  classDef artifact fill:#0C1C20,stroke:#4DB6AC,stroke-width:1.5px,color:#DFF7F7;
  classDef optional fill:#1A1F2E,stroke:#9AA4AF,stroke-width:1px,color:#CFD6DF;

  %% -------- Setup lane --------
  subgraph L0[Setup]
    direction TB
    A([Start]):::startEnd
    B[Clone repository]:::action
    C[Create virtual env .venv]:::action
    D[Install dependencies]:::action
    A --> B --> C --> D
  end

  %% -------- Train lane --------
  subgraph L1[Train (CPU)]
    direction TB
    E{Choose task}:::decision
    F[Train Addition model]:::action
    G[Train Precalc model]:::action
    H[[Save checkpoint: addition_tiny.pt]]:::artifact
    I[[Save checkpoint: precalc_tiny.pt]]:::artifact
    E -->|Addition| F --> H
    E -->|Precalc| G --> I
  end

  %% Merge
  J([Checkpoint ready]):::output
  D --> E
  H --> J
  I --> J

  %% -------- App lane --------
  subgraph L2[Streamlit App]
    direction TB
    K[Run app: streamlit run app/app.py]:::action
    L{Select mode}:::decision
    M[Input expression (end with =)]:::action
    N[Predict result]:::output
    K --> L --> M --> N
  end

  %% -------- Interpret lane --------
  subgraph L3[Interpretability]
    direction TB
    O[Attention heatmaps]:::output
    P[Logit lens and confidence]:::output
    Q{Meets expectations?}:::decision
    R[Analyze model behavior]:::action
    N --> O
    N --> P
    O --> R
    P --> R
    R --> Q
  end

  %% -------- Iterate & Ship lane --------
  subgraph L4[Iterate and Ship]
    direction TB
    S[Extend: new tasks or tools]:::action
    T[Commit and push to GitHub]:::action
    U[Optional: deploy to HF Spaces]:::optional
    Q -- Yes --> T
    Q -- No  --> S --> T
    T --> U
  end

  %% Final link
  J --> K

  %% Link styling
  linkStyle default stroke:#6A7BAA,stroke-width:1.2px;
````
---
## Folder structure

```text
transformer-circuit-studio/
├─ app/
│  ├─ app.py                 # Streamlit UI and prediction
│  ├─ assets/
│  │  └─ styles.css          # Custom CSS for modern theme
│  └─ components/            # UI components (extend here)
├─ models/
│  ├─ tiny_transformer.py    # Tiny transformer (with attention hooks)
│  ├─ train_addition.py      # Training script (stub for addition)
│  └─ checkpoints/           # .pt files (git-ignored)
├─ tasks/                    # Datasets / evaluation (extend)
├─ interpret/                # Patching, ablations (extend)
├─ notebooks/                # Explorations (extend)
├─ requirements.txt          # Python dependencies
├─ .gitignore                # Git ignore rules
├─ README.md                 # This file
└─ LICENSE                   # DACR license
```

---

## Prerequisites

* **Git**
* **Python 3.9+**
* **Windows PowerShell** (on Windows) or a shell (Mac/Linux)
* GitHub repo: `https://github.com/emcdo411/transformer-circuit-studio`
* Internet for dependency install and pushes

> Optional UI theming: add `.streamlit/config.toml` and `app/assets/styles.css` if you want the dark, modern look (see Troubleshooting for a quick snippet).

---

## Quick start (Windows PowerShell)

```powershell
# 1) Clone
cd "C:\Users\Veteran\Documents"
git clone https://github.com/emcdo411/transformer-circuit-studio.git
cd .\transformer-circuit-studio

# 2) Virtual env
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1

# 3) Install deps
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4) Run app
python -m streamlit run .\app\app.py
```

## Quick start (Mac or Linux)

```bash
# 1) Clone
cd ~
git clone https://github.com/emcdo411/transformer-circuit-studio.git
cd transformer-circuit-studio

# 2) Virtual env
python3 -m venv .venv
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4) Run app
python -m streamlit run app/app.py
```

---

## Step-by-step guide

1. **Clone** the repo (see Quick start).
2. **Create & activate** a virtual environment.
3. **Install dependencies** with `pip install -r requirements.txt` and PyTorch CPU.
4. **Train models (optional now):**
   Launch the app (`python -m streamlit run app/app.py`) and use the sidebar:

   * **Train Addition** → saves `models/checkpoints/addition_tiny.pt`
   * **Train Precalc** → saves `models/checkpoints/precalc_tiny.pt`
5. **Predict:** choose **Mode** (Auto/Addition/Precalc), enter an expression, click **Predict**.
6. **Visualize:** inspect attention heatmaps; extend interpretability under `interpret/`.

---

## Run the app

```bash
# Windows
python -m streamlit run .\app\app.py

# Mac/Linux
python -m streamlit run app/app.py
```

Open the printed URL (usually `http://localhost:8501`) and try:

* Addition: `12+7=`
* Precalc: `(3+2)^2-4/2=`

---

## Commit and push to GitHub

```bash
git add -A
git commit -m "feat: tiny transformer + Streamlit UI + checkpoints"
git branch -M main
git remote add origin https://github.com/emcdo411/transformer-circuit-studio.git  # if not set
git push -u origin main
```

On Windows, you can also run:

```powershell
.\push.ps1
```

---

## Optional: add PyTorch CPU

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then train via the Streamlit sidebar.

---

## Optional: deploy to Hugging Face Spaces

1. Create a new Space (**Streamlit**).
2. Push your repo or connect GitHub.
3. Ensure `requirements.txt` includes:

   * `streamlit`, `plotly`, `pandas`, `einops`, `scikit-learn`, and `torch` (CPU)
4. App entry: `app/app.py`
5. Keep checkpoints small (or generate on the fly).

---

## Troubleshooting

**Modern theme not applied**

* Add `.streamlit/config.toml`:

  ```toml
  [theme]
  base = "dark"
  primaryColor = "#7b5cff"
  backgroundColor = "#0c0f14"
  secondaryBackgroundColor = "#121826"
  textColor = "#e6e6e6"
  ```
* Add `app/assets/styles.css` and load via `app.py` (already wired in the provided code).

**PowerShell: scripts disabled**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

**Streamlit not found**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Model size mismatch / bad checkpoint**

```powershell
Remove-Item models\checkpoints\*.pt -ErrorAction Ignore
# retrain in app sidebar
```

**Proxy errors (pip/Git)**

```powershell
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction Ignore
git config --global --unset http.proxy
git config --global --unset https.proxy
# netsh winhttp reset proxy  # (Admin shell)
```

**CRLF warnings on Windows**

```powershell
git config core.autocrlf true
```

---

## Roadmap

* More interpretability hooks in `tiny_transformer.py` (residual stream, probes).
* Add algorithmic tasks (Dyck-1, copying, sorting) in `tasks/`.
* Activation patching and head ablations in `interpret/`.
* Export charts (PNG/GIF) for reports and posts.
* Faster training configs and slightly larger models.

---

## License

**DACR** — see `LICENSE`.

```
 # Transformer Circuit Studio

![License: DACR](https://img.shields.io/badge/License-DACR-blueviolet)
Starter scaffold. Folders and files are populated so they appear on GitHub.
Replace these stubs with your real code and docs when ready.

- app/ - Streamlit UI
- models/ - tiny models and training
- tasks/, interpret/, notebooks/ - future extensions

