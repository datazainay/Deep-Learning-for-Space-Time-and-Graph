"""
AI651 PA1 - Task 2 (v2): compact Autoformer for the 168-step leaderboard forecast.

Architecture: Wu et al. (2021), Autoformer, Sec. 3.1-3.2, adapted from the reference
implementation https://github.com/thuml/Autoformer (models/Autoformer.py,
layers/Autoformer_EncDec.py, layers/AutoCorrelation.py). Changes from the reference:
  * covariates enter through a linear "mark" embedding (no calendar is available);
  * Auto-Correlation selects top-k delays per example and centres q/k; delay 0 excluded.

New in v2 (all optional, chosen on the chronological validation split):
  --delta_feats   add window-relative covariates f(t) - f(origin-1) for the six continuous
                  features: "how much has the weather changed since the forecast origin"
  --val_weeks     longer validation for steadier model selection
  win_rmse        mean over validation windows of the 168-step RMSE: the expected value of
                  a single leaderboard score (pooled RMSE over-weights a few spiky weeks)
  val_*.npy       saved validation predictions, so blend_task2.py can combine runs

Usage on Kaggle (GPU):
  !python task2_autoformer_v2.py --smoke
  !python task2_autoformer_v2.py --modes exog --seeds 0 1 2 --d_model 32 --d_ff 64 --e_layers 2 \
      --lr_step 3 --max_epochs 20 --patience 3 --delta_feats --out runs_D
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
p.add_argument("--out", default="task2_out_v2")
p.add_argument("--modes", nargs="+", default=["exog"])     # "none" = no optional file
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--seq_len", type=int, default=336)
p.add_argument("--label_len", type=int, default=168)
p.add_argument("--pred_len", type=int, default=168)
p.add_argument("--d_model", type=int, default=32)
p.add_argument("--n_heads", type=int, default=4)
p.add_argument("--d_ff", type=int, default=64)
p.add_argument("--e_layers", type=int, default=2)
p.add_argument("--d_layers", type=int, default=1)
p.add_argument("--moving_avg", type=int, default=25)
p.add_argument("--factor", type=float, default=1.0)       # k = factor * ln(L)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight_decay", type=float, default=0.0)
p.add_argument("--lr_step", type=int, default=3)          # halve the LR every lr_step epochs
p.add_argument("--batch", type=int, default=64)
p.add_argument("--max_epochs", type=int, default=20)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--train_stride", type=int, default=1)
p.add_argument("--val_weeks", type=int, default=12)       # chronological validation length
p.add_argument("--val_stride", type=int, default=24)
p.add_argument("--target", choices=["raw", "log1p"], default="raw")
p.add_argument("--delta_feats", action="store_true")
p.add_argument("--select", choices=["val_rmse", "win_rmse"], default="win_rmse",
               help="metric used for early stopping")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
if args.smoke:
    args.seeds, args.max_epochs, args.train_stride = [0], 1, 16

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
y_model = np.log1p(y) if args.target == "log1p" else y
mu, sd = y_model[:val_start].mean(), y_model[:val_start].std()     # training part only
z = (y_model - mu) / sd

cont = [f"feature_{c}" for c in "ABCDEF"]
binr = [f"feature_{c}" for c in "GHIJ"]
X = ext_df[cont].to_numpy(np.float32)
X = (X - X[:val_start].mean(0)) / (X[:val_start].std(0) + 1e-6)   # training part only
XC = torch.tensor(X, device=device)                               # continuous, for deltas
X = np.concatenate([X, ext_df[binr].to_numpy(np.float32)], 1)    # [N+H, 10]
t = np.arange(N + H)
phase = np.stack([np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24)], 1).astype(np.float32)

def marks_for(mode):
    """Covariates for every time step. 'none' = index phase only; 'exog' = phase + optional file."""
    m = phase if mode == "none" else np.concatenate([phase, X], 1)
    return torch.tensor(m, device=device)

Z = torch.tensor(np.concatenate([z, np.zeros(H, np.float32)]), device=device)
L, LL = args.seq_len, args.label_len

def batch(origins, M, delta):
    """origins = index of first forecast step. Returns encoder/decoder inputs and target."""
    o = torch.as_tensor(np.asarray(origins), device=device)[:, None]
    enc_idx = o - L + torch.arange(L, device=device)
    dec_idx = o - LL + torch.arange(LL + H, device=device)
    tgt_idx = o + torch.arange(H, device=device)
    me, md = M[enc_idx], M[dec_idx]
    if delta:                                             # change since the last observed step
        base = XC[o[:, 0] - 1][:, None, :]
        me = torch.cat([me, XC[enc_idx] - base], -1)
        md = torch.cat([md, XC[dec_idx] - base], -1)
    return Z[enc_idx][..., None], me, md, Z[tgt_idx]

train_origins = np.arange(L, val_start - H + 1, args.train_stride)
val_origins = np.arange(val_start, N - H + 1, args.val_stride)

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
    u = zhat * sd + mu
    if args.target == "log1p":
        u = np.expm1(u)
    return np.clip(u, 0, None)                              # target is non-negative

def metrics(pred, true):
    """pred/true [n_windows, H]. win_rmse = mean of per-window RMSE (one leaderboard-like score each)."""
    err = pred - true
    denom = np.abs(true) + np.abs(pred)
    smape = np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0).mean() * 100
    return dict(val_rmse=float(np.sqrt((err ** 2).mean())), win_rmse=float(np.sqrt((err ** 2).mean(1)).mean()),
                win_rmse_std=float(np.sqrt((err ** 2).mean(1)).std()),
                mae=float(np.abs(err).mean()), smape=float(smape))

@torch.no_grad()
def predict(model, origins, M, delta):
    model.eval(); out = []
    for i in range(0, len(origins), 256):
        xe, me, md, _ = batch(origins[i:i + 256], M, delta)
        out.append(model(xe, me, md).cpu().numpy())
    return to_units(np.concatenate(out))

def truth(origins):
    return y[np.asarray(origins)[:, None] + np.arange(H)]

np.save(os.path.join(args.out, "val_truth.npy"), truth(val_origins))
naive = np.repeat(np.array([y[o - 168:o].mean() for o in val_origins])[:, None], H, 1)
print("naive mean-of-last-week, validation:", {k: round(v, 2) for k, v in metrics(naive, truth(val_origins)).items()})

# ----------------------------------------------------------------- runs
rows, store = [], {}
for mode in args.modes:
    M = marks_for(mode)
    delta = args.delta_feats and mode == "exog"
    mark_dim = M.shape[1] + (XC.shape[1] if delta else 0)
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Autoformer(args, mark_dim).to(device)
        n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, best_state, best_epoch, bad, epochs_run = float("inf"), None, 0, 0, 0
        start = time.time()
        for epoch in range(1, args.max_epochs + 1):
            model.train(); epochs_run = epoch
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.5 ** ((epoch - 1) // args.lr_step)
            perm = np.random.permutation(train_origins)
            for i in range(0, len(perm), args.batch):
                xe, me, md, tgt = batch(perm[i:i + args.batch], M, delta)
                loss = F.mse_loss(model(xe, me, md), tgt)
                opt.zero_grad(); loss.backward(); opt.step()
            v = metrics(predict(model, val_origins, M, delta), truth(val_origins))
            print(f"[{mode} seed{seed}] epoch {epoch}: val RMSE {v['val_rmse']:.2f}  win RMSE {v['win_rmse']:.2f}"
                  f"  MAE {v['mae']:.2f}  sMAPE {v['smape']:.1f}")
            if v[args.select] < best:
                best, best_epoch, bad = v[args.select], epoch, 0
                best_state = {k: q.detach().clone() for k, q in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        val_pred = predict(model, val_origins, M, delta)
        v = metrics(val_pred, truth(val_origins))
        fc = predict(model, [N], M, delta)[0]                # the hidden 168 steps
        assert len(fc) == 168 and np.isfinite(fc).all()
        tag = f"{mode}_seed{seed}"
        np.save(os.path.join(args.out, f"val_{tag}.npy"), val_pred)
        fname = os.path.join(args.out, f"forecast_{tag}.txt")
        with open(fname, "w") as f:
            f.write(",".join(f"{x:.4f}" for x in fc))
        store.setdefault(mode, []).append((val_pred, fc, n_params, epochs_run))
        rows.append(dict(mode=mode, seed=seed, params=n_params, epochs_run=epochs_run, best_epoch=best_epoch,
                         **{k: round(val, 3) for k, val in v.items()}, zeros_in_forecast=int((fc == 0).sum()),
                         seconds=round(time.time() - start, 1), forecast_file=fname))
        print(rows[-1])

# seed ensemble per mode: average the forecasts; P and E are summed over members
ens = []
for mode, runs_m in store.items():
    if len(runs_m) < 2:
        continue
    vp = np.mean([r[0] for r in runs_m], 0); fc = np.mean([r[1] for r in runs_m], 0)
    m = metrics(vp, truth(val_origins))
    fname = os.path.join(args.out, f"forecast_{mode}_ensemble.txt")
    with open(fname, "w") as f:
        f.write(",".join(f"{x:.4f}" for x in fc))
    np.save(os.path.join(args.out, f"val_{mode}_ensemble.npy"), vp)
    ens.append(dict(mode=mode, members=len(runs_m), params=sum(r[2] for r in runs_m),
                    epochs_run=sum(r[3] for r in runs_m), **{k: round(val, 3) for k, val in m.items()},
                    zeros_in_forecast=int((fc == 0).sum()), forecast_file=fname))

runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
cols = ["val_rmse", "win_rmse", "mae", "smape"]
summary = runs.groupby("mode")[cols].agg(["mean", "std"]).round(2)
summary.to_csv(os.path.join(args.out, "summary.csv"))
print("\nPer-run results:\n", runs.drop(columns="forecast_file").round(2).to_string(index=False))
print("\nMean +- std over seeds:\n", summary.to_string())
if ens:
    pd.DataFrame(ens).to_csv(os.path.join(args.out, "ensembles.csv"), index=False)
    print("\nSeed ensembles:\n", pd.DataFrame(ens).drop(columns="forecast_file").round(2).to_string(index=False))
