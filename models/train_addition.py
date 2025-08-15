import os, random
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from models.tiny_transformer import TinyTransformer

VOCAB = list("0123456789+ =")
STOI  = {c:i for i,c in enumerate(VOCAB)}

def encode(s: str):
    return torch.tensor([STOI[c] for c in s], dtype=torch.long)

def make_batch(bs=256, max_n=19, device="cpu"):
    xs, ys = [], []
    for _ in range(bs):
        a, b = random.randint(0, max_n), random.randint(0, max_n)
        s = f"{a}+{b}="
        xs.append(encode(s))
        ys.append(a + b)
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=STOI[" "])
    y = torch.tensor(ys, dtype=torch.long)
    return x.to(device), y.to(device)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(vocab_size=len(VOCAB)).to(device)
    opt   = optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()

    base = Path(__file__).resolve().parents[1]
    ckpt_path = base / "models" / "checkpoints" / "addition_tiny.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    steps = 1200
    model.train()
    for step in range(1, steps+1):
        x, y = make_batch(bs=256, device=device)
        logits = model(x)
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"step {step:4d}  loss={loss.item():.3f}  acc={acc:.3f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")
