"""
AI651 PA1 - Task 2: compact Autoformer for the 168-step leaderboard forecast.

Architecture follows Wu et al. (2021), Autoformer, Sec. 3.1-3.2, adapted from the
reference implementation https://github.com/thuml/Autoformer (models/Autoformer.py,
layers/Autoformer_EncDec.py, layers/AutoCorrelation.py). Changes from the reference:
  * covariates enter through a linear "mark" embedding (no calendar is available);
  * Auto-Correlation selects top-k delays per example (as in Task 1) and centres q/k;
  * delay 0 is excluded from the top-k candidates.

Usage on Kaggle (GPU):
  !python task2_autoformer.py --smoke                 # 1-minute pipeline check
  !python task2_autoformer.py --modes none exog --seeds 0 1 2
Outputs (in --out): runs.csv, summary.csv, forecast_<mode>_seed<k>.txt
"""
import argparse, glob, math, os, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------------------- arguments
p = argparse.ArgumentParser()
p.add_argument("--data", default=None, help="folder with the three CSVs (auto-found on Kaggle)")
p.add_argument("--out", default="task2_out")
p.add_argument("--modes", nargs="+", default=["none", "exog"])   # ablation of the optional file
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--seq_len", type=int, default=336)
p.add_argument("--label_len", type=int, default=168)
p.add_argument("--pred_len", type=int, default=168)
p.add_argument("--d_model", type=int, default=16)
p.add_argument("--n_heads", type=int, default=4)
p.add_argument("--d_ff", type=int, default=32)
p.add_argument("--e_layers", type=int, default=1)
p.add_argument("--d_layers", type=int, default=1)
p.add_argument("--moving_avg", type=int, default=25)
p.add_argument("--factor", type=float, default=1.0)       # k = factor * ln(L)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--max_epochs", type=int, default=10)
p.add_argument("--patience", type=int, default=2)
p.add_argument("--train_stride", type=int, default=1)
p.add_argument("--val_weeks", type=int, default=12)       # chronological validation length
p.add_argument("--val_stride", type=int, default=24)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
if args.smoke:
    args.modes, args.seeds, args.max_epochs, args.train_stride = ["none", "exog"], [0], 1, 16

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)

# ----------------------------------------------------------------- data
def find(name):
    if args.data:
        return os.path.join(args.data, name)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"**/{name}", recursive=True)
    assert hits, f"{name} not found - attach the dataset or pass --data"
    return hits[0]

train_df = pd.read_csv(find("student_train.csv"))
ext_df = pd.read_csv(find("optional_external_data.csv"))
y = train_df["value"].to_numpy(np.float32)
N, H = len(y), args.pred_len
assert len(ext_df) == N + H and (ext_df["time_idx"].to_numpy() == np.arange(1, N + H + 1)).all()

val_start = N - 168 * args.val_weeks                   # first validation target index
mu, sd = y[:val_start].mean(), y[:val_start].std()     # target scaling: training part only
z = (y - mu) / sd

cont = [f"feature_{c}" for c in "ABCDEF"]
binr = [f"feature_{c}" for c in "GHIJ"]
X = ext_df[cont].to_numpy(np.float32)
X = (X - X[:val_start].mean(0)) / (X[:val_start].std(0) + 1e-6)   # training part only
X = np.concatenate([X, ext_df[binr].to_numpy(np.float32)], 1)    # [N+H, 10]
t = np.arange(N + H)
phase = np.stack([np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24)], 1).astype(np.float32)

def marks_for(mode):
    """Covariates for every time step. 'none' = index phase only; 'exog' = phase + optional file."""
    m = phase if mode == "none" else np.concatenate([phase, X], 1)
    return torch.tensor(m, device=device)

Z = torch.tensor(np.concatenate([z, np.zeros(H, np.float32)]), device=device)  # padded, never read past N in training
L, LL = args.seq_len, args.label_len

def batch(origins, M):
    """origins = index of first forecast step. Returns encoder/decoder inputs and target."""
    o = torch.as_tensor(origins, device=device)[:, None]
    enc_idx = o - L + torch.arange(L, device=device)
    dec_idx = o - LL + torch.arange(LL + H, device=device)
    tgt_idx = o + torch.arange(H, device=device)
    return Z[enc_idx][..., None], M[enc_idx], M[dec_idx], Z[tgt_idx]

train_origins = np.arange(L, val_start - H + 1, args.train_stride)
val_origins = np.arange(val_start, N - H + 1, args.val_stride)
block_origins = np.arange(val_start, N - H + 1, 168)  # non-overlapping weekly blocks

# ----------------------------------------------------------------- Autoformer
class MovingAvg(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k = k
    def forward(self, x):                                   # [B,L,C]
        q = (self.k - 1) // 2
        x = torch.cat([x[:, :1].repeat(1, q, 1), x, x[:, -1:].repeat(1, q, 1)], 1)
        return F.avg_pool1d(x.transpose(1, 2), self.k, stride=1).transpose(1, 2)

class SeriesDecomp(nn.Module):
    def __init__(self, k):
        super().__init__(); self.ma = MovingAvg(k)
    def forward(self, x):
        trend = self.ma(x)
        return x - trend, trend                             # (seasonal, trend)

class AutoCorrelation(nn.Module):
    """Period-based dependencies (FFT) + time-delay aggregation, Autoformer Sec. 3.2."""
    def __init__(self, factor):
        super().__init__(); self.factor = factor
    def forward(self, q, k, v):                             # q [B,L,h,e], k/v [B,S,h,e]
        B, Lq, h, e = q.shape
        S = k.shape[1]
        if Lq > S:                                          # pad / truncate keys to query length
            pad = torch.zeros(B, Lq - S, h, e, device=q.device)
            k, v = torch.cat([k, pad], 1), torch.cat([v, pad], 1)
        else:
            k, v = k[:, :Lq], v[:, :Lq]
        q, k, v = (a.permute(0, 2, 3, 1) for a in (q, k, v))  # [B,h,e,L]
        q = q - q.mean(-1, keepdim=True); kc = k - k.mean(-1, keepdim=True)
        corr = torch.fft.irfft(torch.fft.rfft(q, dim=-1) * torch.fft.rfft(kc, dim=-1).conj(),
                               n=Lq, dim=-1).mean((1, 2))    # [B,L], R(tau)
        no_zero = torch.zeros_like(corr, dtype=torch.bool); no_zero[:, 0] = True
        corr = corr.masked_fill(no_zero, float("-inf"))      # exclude delay 0 (out-of-place)
        top_k = max(1, int(self.factor * math.log(Lq)))
        score, delay = corr.topk(top_k, dim=-1)              # [B,K]
        w = score.softmax(-1)
        tt = torch.arange(Lq, device=q.device)
        out = torch.zeros_like(v)
        for j in range(top_k):                               # z_t = sum_j w_j v_{(t - tau_j) mod L}
            idx = ((tt - delay[:, j:j + 1]) % Lq)[:, None, None, :].expand_as(v)
            out = out + w[:, j, None, None, None] * v.gather(-1, idx)
        return out.permute(0, 3, 1, 2)                       # [B,L,h,e]

class AutoCorrelationLayer(nn.Module):
    def __init__(self, d, heads, factor):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.inner = AutoCorrelation(factor)
    def forward(self, x, kv):
        B, Lq, d = x.shape
        S = kv.shape[1]
        out = self.inner(self.q(x).view(B, Lq, self.h, -1), self.k(kv).view(B, S, self.h, -1),
                         self.v(kv).view(B, S, self.h, -1))
        return self.o(out.reshape(B, Lq, d))

class LayerNormS(nn.Module):
    """Autoformer's 'my_Layernorm': LayerNorm, then remove the time mean (seasonal part)."""
    def __init__(self, d):
        super().__init__(); self.ln = nn.LayerNorm(d)
    def forward(self, x):
        x = self.ln(x)
        return x - x.mean(1, keepdim=True)

def ffn(d, d_ff, drop):
    return nn.Sequential(nn.Conv1d(d, d_ff, 1, bias=False), nn.GELU(), nn.Dropout(drop),
                         nn.Conv1d(d_ff, d, 1, bias=False))

class EncoderLayer(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.attn = AutoCorrelationLayer(a.d_model, a.n_heads, a.factor)
        self.ff = ffn(a.d_model, a.d_ff, a.dropout)
        self.dec1, self.dec2 = SeriesDecomp(a.moving_avg), SeriesDecomp(a.moving_avg)
        self.drop = nn.Dropout(a.dropout)
    def forward(self, x):
        x, _ = self.dec1(x + self.drop(self.attn(x, x)))            # progressive decomposition
        y = self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, _ = self.dec2(x + y)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, a, c_out=1):
        super().__init__()
        self.self_attn = AutoCorrelationLayer(a.d_model, a.n_heads, a.factor)
        self.cross_attn = AutoCorrelationLayer(a.d_model, a.n_heads, a.factor)
        self.ff = ffn(a.d_model, a.d_ff, a.dropout)
        self.dec1, self.dec2, self.dec3 = (SeriesDecomp(a.moving_avg) for _ in range(3))
        self.drop = nn.Dropout(a.dropout)
        self.trend_proj = nn.Conv1d(a.d_model, c_out, 3, padding=1, padding_mode="circular", bias=False)
    def forward(self, x, enc):
        x, t1 = self.dec1(x + self.drop(self.self_attn(x, x)))
        x, t2 = self.dec2(x + self.drop(self.cross_attn(x, enc)))
        y = self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, t3 = self.dec3(x + y)
        trend = self.trend_proj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
        return x, trend                                              # trend accumulates outside

class Embedding(nn.Module):
    """Value (token) embedding + linear covariate embedding; no positional embedding (as in Autoformer)."""
    def __init__(self, mark_dim, d, drop):
        super().__init__()
        self.tok = nn.Conv1d(1, d, 3, padding=1, padding_mode="circular", bias=False)
        self.mark = nn.Linear(mark_dim, d, bias=False)
        self.drop = nn.Dropout(drop)
    def forward(self, x, m):
        return self.drop(self.tok(x.transpose(1, 2)).transpose(1, 2) + self.mark(m))

class Autoformer(nn.Module):
    def __init__(self, a, mark_dim):
        super().__init__()
        self.a = a
        self.decomp = SeriesDecomp(a.moving_avg)
        self.enc_emb, self.dec_emb = Embedding(mark_dim, a.d_model, a.dropout), Embedding(mark_dim, a.d_model, a.dropout)
        self.enc = nn.ModuleList([EncoderLayer(a) for _ in range(a.e_layers)])
        self.dec = nn.ModuleList([DecoderLayer(a) for _ in range(a.d_layers)])
        self.enc_norm, self.dec_norm = LayerNormS(a.d_model), LayerNormS(a.d_model)
        self.proj = nn.Linear(a.d_model, 1)
    def forward(self, x_enc, m_enc, m_dec):
        a = self.a
        mean = x_enc.mean(1, keepdim=True).repeat(1, a.pred_len, 1)
        seasonal, trend = self.decomp(x_enc)
        trend = torch.cat([trend[:, -a.label_len:], mean], 1)                       # trend init
        seasonal = torch.cat([seasonal[:, -a.label_len:],
                              torch.zeros_like(mean)], 1)                           # seasonal init
        e = self.enc_emb(x_enc, m_enc)
        for layer in self.enc:
            e = layer(e)
        e = self.enc_norm(e)
        d = self.dec_emb(seasonal, m_dec)
        for layer in self.dec:
            d, t_part = layer(d, e)
            trend = trend + t_part
        seasonal_out = self.proj(self.dec_norm(d))
        return (trend + seasonal_out)[:, -a.pred_len:, 0]                          # [B,H]

# ----------------------------------------------------------------- metrics
def to_units(zhat):
    return np.clip(zhat * sd + mu, 0, None)                 # target is non-negative

def metrics(pred, true):
    err = pred - true
    denom = np.abs(true) + np.abs(pred)
    smape = np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0).mean() * 100
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()), smape=float(smape))

@torch.no_grad()
def predict(model, origins, M):
    model.eval(); out = []
    for i in range(0, len(origins), 256):
        xe, me, md, _ = batch(origins[i:i + 256], M)
        out.append(model(xe, me, md).cpu().numpy())
    return to_units(np.concatenate(out))

def truth(origins):
    return y[np.asarray(origins)[:, None] + np.arange(H)]

# naive reference: mean of the last 168 observed values, held flat
naive = np.repeat(np.array([y[o - 168:o].mean() for o in val_origins])[:, None], H, 1)
print("naive mean-of-last-week, validation:", metrics(naive, truth(val_origins)))

# ----------------------------------------------------------------- runs
rows = []
for mode in args.modes:
    M = marks_for(mode)
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Autoformer(args, M.shape[1]).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, best_state, best_epoch, bad, epochs_run = float("inf"), None, 0, 0, 0
        start = time.time()
        for epoch in range(1, args.max_epochs + 1):
            model.train(); epochs_run = epoch
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.5 ** (epoch - 1)          # halve every epoch (reference "type1")
            perm = np.random.permutation(train_origins)
            for i in range(0, len(perm), args.batch):
                xe, me, md, tgt = batch(perm[i:i + args.batch], M)
                loss = F.mse_loss(model(xe, me, md), tgt)
                opt.zero_grad(); loss.backward(); opt.step()
            v = metrics(predict(model, val_origins, M), truth(val_origins))
            print(f"[{mode} seed{seed}] epoch {epoch}: val RMSE {v['rmse']:.2f}  MAE {v['mae']:.2f}  sMAPE {v['smape']:.1f}")
            if v["rmse"] < best:
                best, best_epoch, bad = v["rmse"], epoch, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        v = metrics(predict(model, val_origins, M), truth(val_origins))
        blocks = [metrics(predict(model, [o], M), truth([o]))["rmse"] for o in block_origins]
        # final 168-step forecast: encoder sees the last seq_len observations, decoder marks cover the hidden horizon
        fc = predict(model, [N], M)[0]
        assert len(fc) == 168 and np.isfinite(fc).all()
        fname = os.path.join(args.out, f"forecast_{mode}_seed{seed}.txt")
        with open(fname, "w") as f:
            f.write(",".join(f"{x:.4f}" for x in fc))
        rows.append(dict(mode=mode, seed=seed, params=n_params, epochs_run=epochs_run, best_epoch=best_epoch,
                         val_rmse=v["rmse"], val_mae=v["mae"], val_smape=v["smape"],
                         block_rmse_mean=float(np.mean(blocks)), block_rmse_std=float(np.std(blocks)),
                         seconds=round(time.time() - start, 1), forecast_file=fname))
        print(rows[-1])

runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
summary = runs.groupby("mode")[["val_rmse", "val_mae", "val_smape", "block_rmse_mean"]].agg(["mean", "std"]).round(2)
summary.to_csv(os.path.join(args.out, "summary.csv"))
print("\nPer-run results:\n", runs.drop(columns="forecast_file").round(2).to_string(index=False))
print("\nAblation (mean +- std over seeds):\n", summary.to_string())
