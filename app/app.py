# app/app.py  (MARKER: AUTO_MODE, CHECKPOINT_STATUS)
import re, time, random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import streamlit as st

from models.tiny_transformer import TinyTransformer

# -------- Page --------
st.set_page_config(page_title="Transformer Circuit Studio — Auto Addition/Precalc", layout="wide")
st.title("Transformer Circuit Studio — Auto Addition/Precalc")

BASE_DIR = Path(__file__).resolve().parents[1]

TASKS = {
    "Addition": {
        "ckpt": BASE_DIR / "models" / "checkpoints" / "addition_tiny.pt",
        "vocab": list("0123456789+ ="),
        "offset": 0,
        "num_classes": 40,
        "max_len": 16,
        "model_kwargs": dict(d_model=64, nhead=2, num_layers=2, dim_ff=128),
        "example": "12+7=",
        "hint": "Format like 12+7="
    },
    "Precalc Eval": {
        "ckpt": BASE_DIR / "models" / "checkpoints" / "precalc_tiny.pt",
        "vocab": list("0123456789+-*/^() ="),
        "offset": 200,  # map [-200,200] -> [0..400]
        "num_classes": 401,
        "max_len": 32,
        "model_kwargs": dict(d_model=96, nhead=3, num_layers=2, dim_ff=192),
        "example": "(3+2)^2-4/2=",
        "hint": "Use digits and + - * / ^ ( ), end with '='. Integer division only."
    }
}

# -------- Helpers --------
def encode(s: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def is_addition_expr(expr: str) -> bool:
    return re.match(r"^\s*\d+\+\d+=\s*$", expr) is not None

def parse_add(expr: str) -> Optional[int]:
    m = re.match(r"^\s*(\d+)\+(\d+)=\s*$", expr)
    return None if not m else int(m.group(1)) + int(m.group(2))

def precalc_truth(expr: str) -> Optional[int]:
    try:
        s = expr.strip().rstrip("=")
        s = s.replace("^", "**").replace("/", "//")
        if re.search(r"[^0-9\+\-\*\(\)\s/]", s):
            return None
        val = eval(s, {"__builtins__": None}, {})
        if isinstance(val, int) and -200 <= val <= 200:
            return int(val)
    except Exception:
        return None
    return None

# -------- Training data (optional quick trainer uses these) --------
def make_batch_addition(bs=256, max_n=19, stoi=None, device="cpu"):
    xs, ys = [], []
    for _ in range(bs):
        a = random.randint(0, max_n); b = random.randint(0, max_n)
        s = f"{a}+{b}="
        xs.append(encode(s, stoi)); ys.append(a + b)
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=stoi[" "])
    y = torch.tensor(ys, dtype=torch.long)
    return x.to(device), y.to(device)

def _rand_int(lo=0, hi=12): return random.randint(lo, hi)
def _make_divisible_pair():
    b = random.randint(1, 12); k = random.randint(1, 12)
    return b * k, b  # a/b is integer

def _expr(depth=0, max_depth=2):
    if depth >= max_depth:
        if random.random() < 0.3:
            v = _rand_int(0, 12); return f"-{v}", -v
        v = _rand_int(0, 12); return f"{v}", v
    op = random.choice(["+", "-", "*", "/", "^"])
    if op == "/":
        a, b = _make_divisible_pair(); return f"({a}/{b})", a // b
    s1, v1 = _expr(depth+1, max_depth); s2, v2 = _expr(depth+1, max_depth)
    if op == "+": return f"({s1}+{s2})", v1 + v2
    if op == "-": return f"({s1}-{s2})", v1 - v2
    if op == "*": return f"({s1}*{s2})", v1 * v2
    base = max(-6, min(6, v1)); exp = random.choice([2, 3])
    return f"(({s1})^{exp})", int(base ** exp)

def make_sample_precalc(max_depth=2, offset=200, max_len=32):
    for _ in range(1000):
        s, v = _expr(0, max_depth)
        if -offset <= v <= offset:
            expr = s + "="
            if len(expr) <= max_len: return expr, v
    return "0=", 0

def make_batch_precalc(bs=256, stoi=None, device="cpu", offset=200, max_len=32):
    xs, ys = [], []
    for _ in range(bs):
        expr, val = make_sample_precalc(max_depth=random.choice([2,2,3]), offset=offset, max_len=max_len)
        xs.append(encode(expr, stoi)); ys.append(val + offset)
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=stoi[" "])
    y = torch.tensor(ys, dtype=torch.long)
    return x.to(device), y.to(device)

# -------- Model loading / training --------
@st.cache_resource
def load_model(task_name: str) -> TinyTransformer:
    cfg = TASKS[task_name]
    model = TinyTransformer(
        vocab_size=len(cfg["vocab"]),
        num_classes=cfg["num_classes"],
        max_len=cfg["max_len"],
        **cfg["model_kwargs"]
    )
    ckpt = cfg["ckpt"]
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        st.success(f"[{task_name}] Loaded checkpoint: {ckpt}")
    else:
        st.info(f"[{task_name}] No checkpoint found at: {ckpt}")
    model.eval()
    return model

def train_quick(task_name: str, steps=800, lr=3e-4):
    cfg = TASKS[task_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(
        vocab_size=len(cfg["vocab"]),
        num_classes=cfg["num_classes"],
        max_len=cfg["max_len"],
        **cfg["model_kwargs"]
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    ckpt = cfg["ckpt"]; ckpt.parent.mkdir(parents=True, exist_ok=True)
    stoi = {c:i for i,c in enumerate(cfg["vocab"])}
    st.write(f"Training {task_name} on {device.upper()} for {steps} steps ...")
    pb = st.progress(0); status = st.empty()
    model.train()
    for step in range(1, steps+1):
        if task_name == "Addition":
            x, y = make_batch_addition(bs=256, max_n=19, stoi=stoi, device=device)
        else:
            x, y = make_batch_precalc(bs=256, stoi=stoi, device=device,
                                      offset=cfg["offset"], max_len=cfg["max_len"])
        logits = model(x); loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            status.write(f"step {step}/{steps}  loss={loss.item():.3f}  acc={acc:.3f}")
        pb.progress(step/steps)
    torch.save(model.state_dict(), ckpt)
    st.success(f"Saved: {ckpt}")
    time.sleep(0.5)
    st.cache_resource.clear()
    st.rerun()

# -------- Sidebar (MARKER: CHECKPOINT_STATUS) --------
with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", ["Auto", "Addition", "Precalc Eval"])  # MARKER: mode = st.selectbox

    st.subheader("Checkpoint status")
    for name in ("Addition", "Precalc Eval"):
        p = TASKS[name]["ckpt"]
        if p.exists():
            kb = int(p.stat().st_size / 1024)
            st.write(f"- {name}: OK {p.name}  ({kb} KB, {time.ctime(p.stat().st_mtime)})")
        else:
            st.write(f"- {name}: MISSING {p.name}")

    st.subheader("Train checkpoint (CPU)")
    steps = st.slider("Steps", 200, 4000, 1200, 100)
    colA, colB = st.columns(2)
    if colA.button("Train Addition"):
        train_quick("Addition", steps=steps)
    if colB.button("Train Precalc"):
        train_quick("Precalc Eval", steps=steps)

st.caption(f"Active file: {__file__}")

# -------- Prediction --------
def resolve_task(expr: str) -> str:
    if mode == "Auto":
        return "Addition" if is_addition_expr(expr) else "Precalc Eval"
    return mode

def predict_value(task_name: str, expr: str):
    cfg = TASKS[task_name]
    stoi = {c:i for i,c in enumerate(cfg["vocab"])}
    x = encode(expr, stoi).unsqueeze(0)
    model = load_model(task_name)
    logits = model(x)
    if cfg["num_classes"] == 401:
        pred = int(torch.argmax(logits, dim=-1).item()) - cfg["offset"]
    else:
        pred = int(torch.argmax(logits, dim=-1).item())
    return pred

col_left, col_right = st.columns([2,1])
default_expr = TASKS["Addition"]["example"] if mode == "Addition" else TASKS["Precalc Eval"]["example"]
expr = col_right.text_input("Input expression", value=default_expr,
                            help="Ends with '='. Auto mode picks Addition if it matches 'd+d=', else Precalc.")
if col_right.button("Predict"):
    chosen = resolve_task(expr)
    st.info(f"Active task: {chosen}")
    truth = parse_add(expr) if chosen == "Addition" else precalc_truth(expr)
    if truth is None:
        if chosen == "Addition":
            st.error("Invalid Addition input. Example: 12+7=")
        else:
            st.error("Invalid Precalc input. Use digits and + - * / ^ ( ), end with '='. Integer result required.")
    else:
        pred = predict_value(chosen, expr)
        a, b = col_left.columns(2)
        a.metric("Predicted", pred)
        b.metric("Ground truth", truth)
