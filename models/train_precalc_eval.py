# models/train_precalc_eval.py
import random, math, sys
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim

# Ensure repo root on sys.path so "from models..." works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tiny_transformer import TinyTransformer

# --- Precalc vocab (char-level)
VOCAB = list("0123456789+-*/^() =")
STOI  = {c:i for i,c in enumerate(VOCAB)}
OFFSET = 200          # map y in [-200,200] to class y+OFFSET
NUM_CLASSES = 401     # -200..200 inclusive
MAX_LEN = 32

def encode(s: str):
    return torch.tensor([STOI[c] for c in s], dtype=torch.long)

# ---- Expression generator that guarantees integer results & bounded range
def _rand_int(lo=0, hi=12):
    return random.randint(lo, hi)

def _make_divisible_pair():
    b = random.randint(1, 12)
    k = random.randint(1, 12)
    a = b * k
    return a, b  # a/b is integer

def _expr(depth=0, max_depth=2):
    # returns (string, value) with integer value
    if depth >= max_depth:
        # leaf: maybe unary minus
        if random.random() < 0.3:
            v = _rand_int(0, 12)
            return f"-{v}", -v
        v = _rand_int(0, 12)
        return f"{v}", v

    # choose op
    op = random.choice(["+", "-", "*", "/", "^"])

    if op == "/":
        a, b = _make_divisible_pair()
        sa, sb = str(a), str(b)
        return f"({sa}/{sb})", a // b

    # build children
    s1, v1 = _expr(depth+1, max_depth)
    s2, v2 = _expr(depth+1, max_depth)

    if op == "+":
        val = v1 + v2
        return f"({s1}+{s2})", val
    elif op == "-":
        val = v1 - v2
        return f"({s1}-{s2})", val
    elif op == "*":
        val = v1 * v2
        return f"({s1}*{s2})", val
    elif op == "^":
        # keep magnitudes small
        base = max(-6, min(6, v1))
        exp  = random.choice([2, 3])
        val = int(base ** exp)
        # string uses '^' (we'll interpret it as exponent)
        return f"(({s1})^{exp})", val

def make_sample(max_depth=2):
    # loop until we get an integer in range and short length
    for _ in range(1000):
        s, v = _expr(0, max_depth)
        if -OFFSET <= v <= OFFSET:
            expr = s + "="
            if len(expr) <= MAX_LEN:
                return expr, v
    # fallback
    return "0=", 0

def make_batch(bs=256, device="cpu"):
    xs, ys = [], []
    for _ in range(bs):
        expr, val = make_sample(max_depth=random.choice([2,2,3]))
        xs.append(encode(expr))
        ys.append(val + OFFSET)  # to class id
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=STOI[" "])
    y = torch.tensor(ys, dtype=torch.long)
    return x.to(device), y.to(device)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(
        vocab_size=len(VOCAB),
        num_classes=NUM_CLASSES,
        max_len=MAX_LEN,
        d_model=96, nhead=3, num_layers=2, dim_ff=192
    ).to(device)

    opt   = optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()

    ckpt = ROOT / "models" / "checkpoints" / "precalc_tiny.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    steps = 2000
    model.train()
    for step in range(1, steps+1):
        x, y = make_batch(bs=256, device=device)
        logits = model(x)
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"step {step:4d} loss={loss.item():.3f} acc={acc:.3f}")

    torch.save(model.state_dict(), ckpt)
    print(f"Saved: {ckpt}")
