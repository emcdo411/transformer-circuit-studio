Transformer Circuit Studio

A hands-on sandbox for mechanistic interpretability. Train a miniature transformer on CPU, predict arithmetic expressions, and visualize attention patterns using a modern Streamlit UI. Designed to be clear, extensible, and ideal for learning or reverse engineering transformer internals.

Table of Contents

What is this
Features
Workflow
Folder structure
Prerequisites
Quick start (Windows PowerShell)
Quick start (Mac or Linux)
Step-by-step guide
Run the app
Commit and push to GitHub
Optional: add PyTorch CPU
Optional: deploy to Hugging Face Spaces
Troubleshooting
Roadmap
License


What is this
Transformer Circuit Studio is a minimal, free, and open-source project for exploring transformer model internals through mechanistic interpretability. It includes:

A tiny transformer model trained on CPU for two tasks: basic addition (e.g., 12+7=19) and precalc evaluation (e.g., (3+2)^2-4/2=23).
A Streamlit UI for inputting arithmetic expressions, predicting results, and visualizing attention heatmaps, logit lens, and prediction confidence.
Clear, modular code for researchers, students, or engineers to extend with new tasks, models, or interpretability tools.
Expected outcomes:
Train models to predict arithmetic results accurately.
Visualize how the transformer attends to input tokens (e.g., digits, operators) to understand its reasoning.
Reverse engineer the model’s decision-making via attention patterns and intermediate predictions.



This project is ideal for learning transformer mechanics, experimenting with interpretability, or reverse engineering model behavior in a lightweight, CPU-friendly environment.
Features

Free and Local: Runs on CPU with no GPU required, using PyTorch and Streamlit.
Two Tasks:
Addition: Predicts sums of two numbers (e.g., 12+7=19).
Precalc Eval: Evaluates expressions with +, -, *, /, ^, %, and parentheses (e.g., (3+2)^2-4/2=23), using integer division.


Streamlit UI:
Input expressions and view predicted vs. ground truth results.
Visualize attention heatmaps for each layer and head.
Display logit lens (predictions per layer) and confidence probabilities.


Extensible Codebase: Modular structure with clear stubs for adding tasks, models, or interpretability tools.
Mechanistic Interpretability: Tools to analyze attention patterns and intermediate representations.
Small Checkpoints: Trained models saved as .pt files (typically <10 MB) in models/checkpoints/.

Workflow
Below is the workflow for training, predicting, and visualizing results in Transformer Circuit Studio.
graph TD
    A[Start] --> B[Clone Repository]
    B --> C[Set Up Virtual Environment]
    C --> D[Install Dependencies]
    D --> E{Choose Task}
    E -->|Addition| F[Train Addition Model]
    E -->|Precalc Eval| G[Train Precalc Model]
    F --> H[Save Checkpoint: addition_tiny.pt]
    G --> I[Save Checkpoint: precalc_tiny.pt]
    H --> J[Run Streamlit App]
    I --> J
    J --> K[Select Mode: Auto, Addition, or Precalc]
    K --> L[Input Expression]
    L --> M[Predict Result]
    M --> N[Visualize Outputs]
    N --> O[Attention Heatmaps]
    N --> P[Logit Lens]
    N --> Q[Prediction Confidence]
    O --> R[Analyze Model Behavior]
    P --> R
    Q --> R
    R --> S[Extend: Add Tasks or Interpretability Tools]
    S --> T[Commit and Push to GitHub]
    T --> U[Optional: Deploy to Hugging Face Spaces]

Workflow Steps:

Setup: Clone the repo, create a virtual environment, and install dependencies.
Training: Train a tiny transformer for addition or precalc evaluation, saving checkpoints.
Prediction: Run the Streamlit app, select a mode, input an expression (e.g., 12+7= or (3+2)^2-4/2=), and predict the result.
Visualization: View attention heatmaps, logit lens, and prediction confidence to understand model behavior.
Extension: Add new tasks, models, or interpretability tools and share via GitHub or Hugging Face Spaces.

Folder structure
transformer-circuit-studio/
├── app/
│   ├── app.py                 # Streamlit entry point for UI and predictions
│   ├── assets/
│   │   └── styles.css         # Custom CSS for Streamlit UI
│   └── components/            # UI components (add later for custom widgets)
├── models/
│   ├── tiny_transformer.py    # Tiny transformer model with attention hooks
│   ├── train_addition.py      # Training script for addition task (stub)
│   └── checkpoints/           # Saved model checkpoints (.pt files, git-ignored)
├── tasks/                     # Datasets and evaluation logic (add later)
├── interpret/                 # Attention analysis, patching, ablations (add later)
├── notebooks/                 # Exploratory Jupyter notebooks (add later)
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
└── LICENSE                   # MIT license

Prerequisites

Git: For cloning the repository.
Python 3.9+: For running the app and training models.
Windows PowerShell (Windows) or shell (Mac/Linux): For setup and execution.
GitHub repo: https://github.com/emcdo411/transformer-circuit-studio.
Optional: Internet access for dependency installation and GitHub pushes.

Quick start (Windows PowerShell)
# 1) Clone the repository
cd "C:\Users\Veteran\Documents"
git clone https://github.com/emcdo411/transformer-circuit-studio.git
cd .\transformer-circuit-studio

# 2) Create and activate virtual environment
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4) Run the Streamlit app
python -m streamlit run .\app\app.py

Quick start (Mac or Linux)
# 1) Clone the repository
cd ~
git clone https://github.com/emcdo411/transformer-circuit-studio.git
cd transformer-circuit-studio

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4) Run the Streamlit app
python -m streamlit run app/app.py

Step-by-step guide

Clone the Repository:

Windows: See Quick start (Windows PowerShell).
Mac/Linux: See Quick start (Mac or Linux).


Create and Activate Virtual Environment:

Windows: py -m venv .venv then .\.venv\Scripts\Activate.ps1.
Mac/Linux: python3 -m venv .venv then source .venv/bin/activate.


Install Dependencies:

Run pip install -r requirements.txt for Streamlit, Plotly, pandas, and other dependencies.
Install PyTorch CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu.


Train Models (Optional):

Run the Streamlit app: python -m streamlit run app/app.py.
In the app’s sidebar, use “Train Addition” or “Train Precalc” to generate models/checkpoints/addition_tiny.pt or models/checkpoints/precalc_tiny.pt.
Checkpoints are saved automatically after training.


Run the Streamlit App:

Execute python -m streamlit run app/app.py.
Open http://localhost:8501 in your browser.
Select a mode (Auto, Addition, or Precalc Eval), input an expression, and click “Predict” to see results and visualizations.


Analyze and Extend:

Use attention heatmaps to study how the model processes tokens.
Modify models/tiny_transformer.py to experiment with model architecture.
Add new tasks in tasks/ or interpretability tools in interpret/.



Run the app
# Windows
python -m streamlit run .\app\app.py

# Mac/Linux
python -m streamlit run app/app.py

Open the local URL printed by Streamlit (usually http://localhost:8501). Expected outcomes:

Input expressions like 12+7= or (3+2)^2-4/2=.
View predicted results, ground truth, attention heatmaps, logit lens, and confidence plots.
Train models via the sidebar to generate or update checkpoints.

Commit and push to GitHub
git add -A
git commit -m "Update: Added addition and precalc tasks with Streamlit UI"
git branch -M main
git remote add origin https://github.com/emcdo411/transformer-circuit-studio.git  # if not set
git push -u origin main

On Windows, you can also run a helper script if present:
.\push.ps1

Optional: add PyTorch CPU
To train and run models locally without a GPU:
# Windows (inside venv)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Mac/Linux (inside venv)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

After installing, train models via the Streamlit app’s sidebar.
Optional: deploy to Hugging Face Spaces (free)

Create a new Space on Hugging Face (type: Streamlit).
Push your repo files to the Space or connect your GitHub repository.
Update requirements.txt to include:
streamlit, plotly, pandas, torch (CPU version).


Set the app entry file to app/app.py.
Ensure checkpoints (addition_tiny.pt, precalc_tiny.pt) are included or generated on the fly (keep under 10 MB).

Troubleshooting
PowerShell scripts disabled:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

Streamlit not found:Ensure the virtual environment is activated:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Size mismatch error in model loading:

Delete incompatible checkpoints:Remove-Item -Path models\checkpoints\*.pt -ErrorAction Ignore


Retrain models using the Streamlit app’s “Train Addition” or “Train Precalc” buttons.

Proxy errors with pip or Git:Clear proxy environment variables:
Remove-Item Env:HTTP_PROXY  -ErrorAction Ignore
Remove-Item Env:HTTPS_PROXY -ErrorAction Ignore
Remove-Item Env:ALL_PROXY   -ErrorAction Ignore

Unset Git proxy:
git config --global --unset http.proxy
git config --global --unset https.proxy

Check WinHTTP proxy:
netsh winhttp show proxy
# netsh winhttp reset proxy  # Run in Admin shell if needed

CRLF warnings on Windows:Set Git to handle line endings:
git config core.autocrlf true

Roadmap

Enhance tiny_transformer.py with more interpretability hooks (e.g., residual stream analysis).
Add tasks like Dyck-1 or algorithmic sequences in tasks/.
Implement activation patching and head ablations in interpret/.
Support exporting attention visualizations as images/GIFs for sharing.
Optimize training for faster convergence or larger models.

License
DACR. See LICENSE.
```
 # Transformer Circuit Studio

![License: DACR](https://img.shields.io/badge/License-DACR-blueviolet)
Starter scaffold. Folders and files are populated so they appear on GitHub.
Replace these stubs with your real code and docs when ready.

- app/ - Streamlit UI
- models/ - tiny models and training
- tasks/, interpret/, notebooks/ - future extensions

