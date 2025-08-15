import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=2, dim_ff=128, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Linear(dim_ff, d_model))
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x, need_weights=False):
        attn_out, attn_w = self.attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x, attn_w

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=14, d_model=64, nhead=2, num_layers=2, dim_ff=128, max_len=16, num_classes=40):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.layers    = nn.ModuleList([TransformerBlock(d_model, nhead, dim_ff) for _ in range(num_layers)])
        self.norm      = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, num_classes)

    def forward(self, x, return_attn=False):
        B, S = x.size()
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        h = self.token_emb(x) + self.pos_emb(pos)
        attns = []
        for layer in self.layers:
            h, attn_w = layer(h, need_weights=return_attn)
            if return_attn:
                attns.append(attn_w)
        h_last = self.norm(h[:, -1, :])
        logits = self.head(h_last)
        return (logits, attns) if return_attn else logits
