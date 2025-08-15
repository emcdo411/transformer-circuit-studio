# app/app.py  (MARKER: MODERN_UI, AUTO_MODE, CHECKPOINT_STATUS)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re, time, random, math
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

try:
    from models.tiny_transformer import TinyTransformer
except ImportError:
    st.error("Could not import TinyTransformer. Ensure 'models/tiny_transformer.py' exists.")
    st.stop()

# ---------- Page ----------
st.set_page_config(page_title="Transformer Circuit Studio — Auto Addition/Precalc", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

# ===== Styling helpers =====
def load_css():
    css_path = BASE_DIR / "app" / "assets" / "styles.css"
    default_css = """
    .hero { text-align: center; padding: 2rem; }
    .hero h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.2rem; color: #888; }
    .pill { background: #333; padding: 0.3rem 0.8rem; border-radius: 1rem; margin-right: 0.5rem; }
    .panel { background: #1f1f1f; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    """
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)
        st.warning(f"CSS file not found at {css_path}. Using default styles.")

def hero(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="hero">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def status_bar(mode_text: str, ckpt_info: str):
    st.markdown(f"""
    <div style="margin-top:10px;margin-bottom:6px;">
      <span class="pill">Mode: {mode_text}</span>
      <span class="pill">Checkpoint: {ckpt_info}</span>
    </div>
    """, unsafe_allow_html=True)

def style_plot(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        title=title or fig.layout.title.text,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=14),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(255,255,255,.15)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,.15)", zeroline=False),
    )
    return fig

# Load CSS + hero header
load_css()
hero(
    "Transformer Circuit Studio",
    "Modern Streamlit UI • addition + precalc • interpretability tools"
)

# ---------- Tasks ----------
TASKS = {
    "Addition": {
        "ckpt": BASE_DIR / "models" / "checkpoints" / "addition_tiny.pt",
        "vocab": list("0123456789+ ="),
        "offset": 0,
        "num_classes": 200,
        "max_len": 16,
        "model_kwargs": dict(d_model=64, nhead=2, num_layers=2, dim_ff=128),
        "example": "12+7=",
        "hint": "Format like 12+7="
    },
    "Precalc Eval": {
        "ckpt": BASE_DIR / "models" / "checkpoints" / "precalc_tiny.pt",
        "vocab": list("0123456789+-*/^%() ="),
        "offset": 200,  # map [-200,200] -> [0..400]
        "num_classes": 401,
        "max_len": 32,
        "model_kwargs": dict(d_model=96, nhead=3, num_layers=2, dim_ff=192),
        "example": "(3+2)^2-4/2=",
        "hint": "Use digits and + - * / ^ % ( ), end with '='. Integer division only."
    }
}

# ---------- Helpers ----------
def encode(s: str, stoi: dict) -> torch.Tensor:
    try:
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    except KeyError as e:
        st.error(f"Invalid character in expression: {e}")
        return torch.tensor([], dtype=torch.long)

def is_addition_expr(expr: str) -> bool:
    return re.match(r"^\s*\d+\+\d+=\s*$", expr) is not None

def parse_add(expr: str) -> Optional[int]:
    m = re.match(r"^\s*(\d+)\+(\d+)=\s*$", expr)
    return None if not m else int(m.group(1)) + int(m.group(2))

def precalc_truth(expr: str) -> Optional[int]:
    """Integer evaluator for + - * / ^ % and parentheses. Normalizes ^ -> ** and / -> //."""
    try:
        s = expr.strip().rstrip("=")
        s = s.replace("^", "**").replace("/", "//")
        if re.search(r"[^0-9+\*\(\)\s/%^-]", s):
            return None
        val = eval(s, {"__builtins__": None}, {})
        if isinstance(val, int) and -200 <= val <= 200:
            return int(val)
    except Exception as e:
        st.error(f"Error evaluating expression: {e}")
        return None
    return None

# ---------- Training batches ----------
def make_batch_addition(bs=256, max_n=99, stoi=None, device="cpu"):
    xs, ys = [], []
    for _ in range(bs):
        a = random.randint(0, max_n); b = random.randint(0, max_n)
        s = f"{a}+{b}="
        xs.append(encode(s, stoi)); ys.append(a + b)
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=stoi[" "])
    y = torch.tensor(ys, dtype=torch.long)
    return x.to(device), y.to(device)

def _rand_int(lo=0, hi=20): return random.randint(lo, hi)
def _make_divisible_pair():
    b = random.randint(1, 12); k = random.randint(1, 12)
    return b * k, b  # a/b is integer

def _expr(depth=0, max_depth=2):
    if depth >= max_depth:
        if random.random() < 0.3:
            v = _rand_int(0, 12); return f"-{v}", -v
        v = _rand_int(0, 12); return f"{v}", v
    op = random.choice(["+", "-", "*", "/", "^", "%"])
    s1, v1 = _expr(depth+1, max_depth); s2, v2 = _expr(depth+1, max_depth)
    if op in ["/", "%"]:
        count = 0
        while v2 == 0 and count < 10:
            s2, v2 = _expr(depth+1, max_depth)
            count += 1
        if v2 == 0:
            return f"{v1}", v1  # fallback
        if op == "/":
            return f"({s1}/{s2})", v1 // v2
        else:
            return f"({s1}%{s2})", v1 % v2
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
        expr, val = make_sample_precalc(max_depth=random.choice([2,3,4]), offset=offset, max_len=max_len)
        xs.append(encode(expr, stoi)); ys.append(val + offset)
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=stoi[" "])
    y = torch.tensor(ys, dtype=torch.long)
    return x.to(device), y.to(device)

# ---------- Model loading / training ----------
@st.cache_resource
def load_model(task_name: str) -> TinyTransformer:
    cfg = TASKS[task_name]
    try:
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
    except Exception as e:
        st.error(f"Error loading model for {task_name}: {e}")
        return None

def train_quick(task_name: str, steps=800, lr=3e-4, bs=256):
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
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    model.train()
    for step in range(1, steps+1):
        if task_name == "Addition":
            x, y = make_batch_addition(bs=bs, max_n=99, stoi=stoi, device=device)
        else:
            x, y = make_batch_precalc(bs=bs, stoi=stoi, device=device,
                                      offset=cfg["offset"], max_len=cfg["max_len"])
        logits = model(x); loss = lossf(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            model.eval()
            if task_name == "Addition":
                val_x, val_y = make_batch_addition(bs=128, max_n=99, stoi=stoi, device=device)
            else:
                val_x, val_y = make_batch_precalc(bs=128, stoi=stoi, device=device,
                                                  offset=cfg["offset"], max_len=cfg["max_len"])
            with torch.no_grad():
                vlogits = model(val_x)
                vloss = lossf(vlogits, val_y).item()
                vacc = (vlogits.argmax(-1) == val_y).float().mean().item()
            model.train()
            status.write(f"step {step}/{steps}  loss={loss.item():.3f}  acc={acc:.3f}  val_loss={vloss:.3f}  val_acc={vacc:.3f}")
            train_losses.append(loss.item())
            train_accs.append(acc)
            val_losses.append(vloss)
            val_accs.append(vacc)
        pb.progress(step/steps)
    torch.save(model.state_dict(), ckpt)
    st.success(f"Saved: {ckpt}")

    # Plot training curves
    steps_list = list(range(100, steps + 1, 100))
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=steps_list, y=train_losses, mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=steps_list, y=val_losses, mode='lines', name='Val Loss'))
    st.plotly_chart(style_plot(fig_loss, "Loss Curve"))

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=steps_list, y=train_accs, mode='lines', name='Train Acc'))
    fig_acc.add_trace(go.Scatter(x=steps_list, y=val_accs, mode='lines', name='Val Acc'))
    st.plotly_chart(style_plot(fig_acc, "Accuracy Curve"))

    time.sleep(0.5)
    st.cache_resource.clear()
    st.rerun()

# ---------- Manual forward for attention ----------
def embed_inputs(model: TinyTransformer, x: torch.Tensor) -> torch.Tensor:
    B, S = x.size()
    pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    return model.token_emb(x) + model.pos_emb(pos)

@torch.no_grad()
def forward_collect(model: TinyTransformer, x: torch.Tensor, return_attn=True)\
        -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    h = embed_inputs(model, x)
    hs = [h]
    attns = []
    for layer in model.layers:
        attn_out, attn_w = layer.attn(h, h, h, need_weights=return_attn, average_attn_weights=False)
        h1 = layer.ln1(h + attn_out)
        ff_out = layer.ff(h1)
        h = layer.ln2(h1 + ff_out)
        hs.append(h)
        if return_attn: attns.append(attn_w)
    return hs, attns

@torch.no_grad()
def forward_from(model: TinyTransformer, h: torch.Tensor, start_layer: int) -> torch.Tensor:
    for li in range(start_layer, len(model.layers)):
        layer = model.layers[li]
        attn_out, _ = layer.attn(h, h, h, need_weights=False)
        h1 = layer.ln1(h + attn_out)
        ff_out = layer.ff(h1)
        h = layer.ln2(h1 + ff_out)
    h_last = model.norm(h[:, -1, :])
    return model.head(h_last)

# ---------- Sidebar (mode + status + training) ----------
with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", ["Auto", "Addition", "Precalc Eval"])  # MARKER: mode = st.selectbox

    st.subheader("Checkpoint status")
    status_lines = []
    for name in ("Addition", "Precalc Eval"):
        p = TASKS[name]["ckpt"]
        if p.exists():
            kb = int(p.stat().st_size / 1024)
            line = f"- {name}: OK {p.name} ({kb} KB, {time.ctime(p.stat().st_mtime)})"
        else:
            line = f"- {name}: MISSING {p.name}"
        st.write(line)
        status_lines.append(line)

    st.subheader("Train checkpoint (CPU)")
    steps = st.slider("Steps", 200, 4000, 1200, 100)
    bs = st.slider("Batch size", 64, 512, 256, 64)
    lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=3e-4, format="%e")
    colA, colB = st.columns(2)
    if colA.button("Train Addition"):
        train_quick("Addition", steps=steps, lr=lr, bs=bs)
    if colB.button("Train Precalc"):
        train_quick("Precalc Eval", steps=steps, lr=lr, bs=bs)

# ---------- Prediction UI ----------
def resolve_task(expr: str) -> str:
    if mode == "Auto":
        return "Addition" if is_addition_expr(expr) else "Precalc Eval"
    return mode

def predict_value_and_attn(task_name: str, expr: str):
    cfg = TASKS[task_name]
    stoi = {c:i for i,c in enumerate(cfg["vocab"])}
    x = encode(expr, stoi)
    if x.numel() == 0:  # Check for empty tensor due to invalid input
        return None, None, None, None, None, None, None
    x = x.unsqueeze(0)
    model = load_model(task_name)
    if model is None:
        return None, None, None, None, None, None, None
    hs, attns = forward_collect(model, x, return_attn=True)
    logits = forward_from(model, hs[0], 0)
    offset = cfg.get("offset", 0)
    pred = int(torch.argmax(logits, dim=-1).item()) - offset
    preds_per_layer = []
    for h in hs:
        h_last = model.norm(h[:, -1, :])
        logits_i = model.head(h_last)
        class_i = logits_i.argmax(-1).item()
        pred_i = class_i - offset
        preds_per_layer.append(pred_i)
    return pred, attns, model, stoi, cfg, preds_per_layer, logits

# Card wrapper helpers
def panel_open():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
def panel_close():
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "input_expr" not in st.session_state:
    st.session_state["input_expr"] = TASKS["Addition"]["example"]

# Top status pills (mode + which ckpt the page will use)
_ckpt_info = []
for name in ("Addition", "Precalc Eval"):
    p = TASKS[name]["ckpt"]
    _ckpt_info.append(f"{name}:{p.name if p.exists() else 'missing'}")
status_bar(mode, " | ".join(_ckpt_info))

tab1, = st.tabs(["Predict + Attention"])
with tab1:
    panel_open()
    c1, c2 = st.columns([2,1])
    default_expr = TASKS["Addition"]["example"] if mode == "Addition" else TASKS["Precalc Eval"]["example"]
    expr = c2.text_input("Input expression", value=st.session_state["input_expr"], key="input_expr",
                         help="Ends with '='. Auto mode picks Addition if it matches 'd+d=', else Precalc.")
    col_pred, col_rand = c2.columns(2)
    if col_pred.button("Predict"):
        chosen = resolve_task(expr)
        st.info(f"Active task: {chosen}")
        truth = parse_add(expr) if chosen == "Addition" else precalc_truth(expr)
        if truth is None:
            if chosen == "Addition":
                st.error("Invalid Addition input. Example: 12+7=")
            else:
                st.error("Invalid Precalc input. Use digits and + - * / ^ % ( ), end with '='. Integer result required.")
        else:
            pred, attns, model, stoi, cfg, preds_per_layer, logits = predict_value_and_attn(chosen, expr)
            if pred is None:
                st.error(f"Prediction failed for {chosen}. Ensure model and input are valid.")
            else:
                m1, m2 = c1.columns(2)
                m1.metric("Predicted", pred)
                m2.metric("Ground truth", truth)

                # Logit Lens
                st.subheader("Logit Lens")
                df_lens = pd.DataFrame({"Layer": range(len(preds_per_layer)), "Predicted Value": preds_per_layer})
                st.table(df_lens)

                # Prediction Probabilities
                st.subheader("Prediction Confidence")
                offset = cfg.get("offset", 0)
                min_val = -offset
                max_val = cfg["num_classes"] - 1 - offset
                probs = torch.softmax(logits, -1)[0].cpu().numpy()
                if cfg["num_classes"] < 50:
                    xr = list(range(min_val, max_val + 1))
                    y = probs
                else:
                    span = 20
                    low = max(min_val, pred - span)
                    high = min(max_val, pred + span)
                    idx_low = low + offset
                    idx_high = high + offset + 1
                    xr = list(range(low, high + 1))
                    y = probs[idx_low:idx_high]
                df_prob = pd.DataFrame({"Value": xr, "Probability": y})
                df_prob["Type"] = "Other"
                df_prob.loc[df_prob["Value"] == pred, "Type"] = "Predicted"
                if pred != truth:
                    df_prob.loc[df_prob["Value"] == truth, "Type"] = "Ground Truth"
                fig_prob = px.bar(df_prob, x="Value", y="Probability", color="Type",
                                  color_discrete_map={"Predicted": "lime", "Ground Truth": "red", "Other": "royalblue"})
                st.plotly_chart(style_plot(fig_prob, "Prediction Probabilities"), use_container_width=True)

                # Attention heatmaps
                if cfg["ckpt"].exists() and attns and attns[0] is not None:
                    st.subheader("Attention Heatmaps")
                    tokens = list(expr)
                    for li, A in enumerate(attns):
                        A = A[0] if A.dim() == 4 else A  # ensure (heads, S, S)
                        for h in range(A.shape[0]):
                            st.caption(f"Layer {li}  •  Head {h}")
                            fig = px.imshow(
                                A[h].cpu().numpy(),
                                x=tokens, y=tokens, origin="upper",
                                labels=dict(x="Key tokens", y="Query tokens", color="Attn"),
                                title=f"Layer {li} - Head {h}"
                            )
                            fig = style_plot(fig)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No checkpoint for {chosen}. Use the sidebar to train, then try again.")
    if col_rand.button("Random"):
        if mode == "Addition":
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            new_expr = f"{a}+{b}="
        elif mode == "Precalc Eval":
            cfg_pre = TASKS["Precalc Eval"]
            new_expr, _ = make_sample_precalc(max_depth=random.choice([2, 3, 4]), offset=cfg_pre["offset"], max_len=cfg_pre["max_len"])
        else:  # Auto
            if random.random() < 0.5:
                a = random.randint(0, 99)
                b = random.randint(0, 99)
                new_expr = f"{a}+{b}="
            else:
                cfg_pre = TASKS["Precalc Eval"]
                new_expr, _ = make_sample_precalc(max_depth=random.choice([2, 3, 4]), offset=cfg_pre["offset"], max_len=cfg_pre["max_len"])
        st.session_state["input_expr"] = new_expr
        st.rerun()
    panel_close()

st.caption(f"Active file: {__file__}")
