# Transformer Circuit Studio

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-app-brightgreen)
![PRs](https://img.shields.io/badge/PRs-welcome-orange)
![Last Commit](https://img.shields.io/github/last-commit/emcdo411/transformer-circuit-studio)
![Issues](https://img.shields.io/github/issues/emcdo411/transformer-circuit-studio)

A tiny, free, hands-on sandbox for mechanistic interpretability. Train a miniature transformer on CPU, then visualize attention and build the UI in Streamlit. Designed to be clear, simple, and easy to extend.

---

## Table of Contents
- [What is this](#what-is-this)
- [Features](#features)
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
A minimal project to explore transformer internals on a zero-cost stack:
- Streamlit UI for fast, simple visualization
- CPU-friendly training of a toy model
- Clear scaffolding so you can extend to real interpretability (attention maps, patching, ablations)

## Features
- Free to run locally (no GPU required)
- Small, readable code stubs you can replace as you grow
- Modern README with step-by-step instructions and deployment hints

## Folder structure
```

transformer-circuit-studio/
\|-- app/
\|   |-- app.py                 # Streamlit entry point (UI)
\|   |-- components/            # UI helpers (add later)
\|   `-- assets/                # images or GIFs for docs |-- models/ |   |-- tiny_transformer.py    # tiny model stub (replace later) |   |-- train_addition.py      # CPU-friendly training script (stub) |   `-- checkpoints/           # saved .pt files (ignored by git)
\|-- tasks/                     # datasets and eval (add later)
\|-- interpret/                 # attention, patching, ablations (add later)
\|-- notebooks/                 # exploratory work (add later)
\|-- requirements.txt
\|-- .gitignore
\|-- README.md
\`-- LICENSE

````

---

## Prerequisites
- Git installed
- Python 3.9+ installed
- Windows PowerShell (on Windows) or a shell (Mac/Linux)
- A GitHub repo: `https://github.com/emcdo411/transformer-circuit-studio`

---

## Quick start (Windows PowerShell)
```powershell
# 1) clone
cd "C:\Users\Veteran\Documents"
git clone https://github.com/emcdo411/transformer-circuit-studio.git
cd .\transformer-circuit-studio

# 2) create venv (PowerShell)
py -m venv .venv

# 3) allow activation for this session only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1

# 4) install base deps (no PyTorch yet)
pip install -r requirements.txt

# 5) run the app (will show a placeholder)
python -m streamlit run .\app\app.py
````

## Quick start (Mac or Linux)

```bash
# 1) clone
cd ~
git clone https://github.com/emcdo411/transformer-circuit-studio.git
cd transformer-circuit-studio

# 2) venv
python3 -m venv .venv
source .venv/bin/activate

# 3) install deps
pip install -r requirements.txt

# 4) run app
python -m streamlit run app/app.py
```

---

## Step-by-step guide

1. Clone the repo

   * Windows: see Quick start (PowerShell)
   * Mac/Linux: see Quick start (Mac or Linux)

2. Create and activate a virtual environment

   * Windows: `py -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
   * Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`

3. Install base requirements

   * `pip install -r requirements.txt`
   * These are small packages (Streamlit, Plotly, sklearn, einops)

4. (Optional) Install PyTorch CPU wheels

   * See [Optional: add PyTorch CPU](#optional-add-pytorch-cpu)

5. Train a tiny checkpoint (optional for now)

   * `python .\models\train_addition.py` (Windows)
   * `python models/train_addition.py` (Mac/Linux)
   * This writes `models/checkpoints/addition_tiny.pt`

6. Run the Streamlit app

   * `python -m streamlit run app/app.py`

7. Edit and extend

   * Replace `models/tiny_transformer.py` with a tiny transformer you like
   * Add real attention hooks in `interpret/`
   * Add tasks in `tasks/` and visual components in `app/components/`

---

## Run the app

```bash
# Windows
python -m streamlit run .\app\app.py

# Mac/Linux
python -m streamlit run app/app.py
```

Open the local URL printed by Streamlit (usually [http://localhost:8501](http://localhost:8501)).

---

## Commit and push to GitHub

```bash
git add -A
git commit -m "init: scaffold + basic README"
git branch -M main
git remote add origin https://github.com/emcdo411/transformer-circuit-studio.git  # if not set
git push -u origin main
```

On Windows you can also run the helper script if present:

```powershell
.\push.ps1
```

---

## Optional: add PyTorch CPU

If you want a tiny model and training loop to run fully locally:

**Windows PowerShell**

```powershell
# inside the venv
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# train a tiny checkpoint
python .\models\train_addition.py
```

**Mac/Linux**

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
python models/train_addition.py
```

---

## Optional: deploy to Hugging Face Spaces (free)

1. Create a new Space: type = Streamlit
2. Push your repo files to the Space or connect your GitHub
3. Ensure `requirements.txt` includes:

   * `streamlit`, `plotly`, `einops`, `scikit-learn`
   * Add `torch` if you plan to run CPU models in the Space
4. App entry file: `app/app.py`

Tip: keep checkpoints under 10 MB or generate them on the fly.

---

## Troubleshooting

**PowerShell says scripts are disabled**
Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

**Streamlit not found**
Make sure you are using the venv:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Pip or Git cannot reach the internet (proxy errors)**
Clear proxy env vars:

```powershell
Remove-Item Env:HTTP_PROXY  -ErrorAction Ignore
Remove-Item Env:HTTPS_PROXY -ErrorAction Ignore
Remove-Item Env:ALL_PROXY   -ErrorAction Ignore
```

Unset Git proxy:

```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

Check WinHTTP proxy (Admin shell if you need to reset):

```powershell
netsh winhttp show proxy
# netsh winhttp reset proxy
```

**CRLF warnings on Windows**
Harmless. To quiet them:

```powershell
git config core.autocrlf true
```

---

## Roadmap

* Implement real TinyTransformer in `models/tiny_transformer.py`
* Add attention hooks and heatmaps in `interpret/`
* Add activation patching and head ablations
* Add more tasks (Dyck-1, algorithmic sequences)
* Export images/GIFs for LinkedIn posts

---

## License

MIT. See `LICENSE`.

```

If you’d like, I can drop this into your `README.md` automatically and add badges that point to specific branches or a Hugging Face Space once it’s live.
::contentReference[oaicite:0]{index=0}
```
 # Transformer Circuit Studio

![License: DACR](https://img.shields.io/badge/License-DACR-blueviolet)
Starter scaffold. Folders and files are populated so they appear on GitHub.
Replace these stubs with your real code and docs when ready.

- app/ - Streamlit UI
- models/ - tiny models and training
- tasks/, interpret/, notebooks/ - future extensions

