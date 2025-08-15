# models/tiny_transformer.py
import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, need_weights=False):
        attn_out, attn_w = self.attn(x, x, x, need_weights=need_weights)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x, attn_w

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, max_len, d_model, nhead, num_layers, dim_ff, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(self, x):
        h = self.embed_inputs(x)
        for layer in self.layers:
            h, _ = layer(h, need_weights=False)
        h_last = self.norm(h[:, -1, :])
        return self.head(h_last)

    def embed_inputs(self, x):
        B, S = x.size()
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        return self.token_emb(x) + self.pos_emb(pos)

    @torch.no_grad()
    def forward_collect(self, x, return_attn=True):
        h = self.embed_inputs(x)
        hs = [h]
        attns = []
        for layer in self.layers:
            h, attn_w = layer(h, need_weights=return_attn)
            hs.append(h)
            if return_attn:
                attns.append(attn_w)
        return hs, attns
