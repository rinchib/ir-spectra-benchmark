import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG, FLOOR
from metrics import SELECTION_CM, sis_metric


def _to_dev_tensor(X, device, dtype=torch.float32):
    return torch.as_tensor(X, dtype=dtype).to(device)

def make_loader(Xdev, Ydev, batch, shuffle):
    ds = TensorDataset(Xdev, Ydev)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=shuffle)

def sid_loss_torch(pred, target, eps=FLOOR):
    p = pred.clamp_min(0) + eps; q = target.clamp_min(0) + eps
    p = p / p.sum(dim=1, keepdim=True); q = q / q.sum(dim=1, keepdim=True)
    return (p * torch.log(p / q) + q * torch.log(q / p)).sum(dim=1).mean()

LOSS_NAME = CONFIG.get("loss", "sid")
def get_loss():  return sid_loss_torch if LOSS_NAME == "sid" else nn.MSELoss()

class _Exp(nn.Module):
    def forward(self, x): return torch.exp(x.clamp(max=11))
_ACTS = {"sigmoid": nn.Sigmoid, "softplus": nn.Softplus, "exp": _Exp}
def out_activation():  return _ACTS[CONFIG.get("output_activation", "softplus")]()
print("training loss:", LOSS_NAME, "| output activation:", CONFIG.get("output_activation", "softplus"))

def _make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)
def _autocast_cuda():
    try:
        return torch.autocast(device_type="cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()

@torch.no_grad()
def _val_selection_sis(model, Xva_dev, Yva_np):
    model.eval()
    vp = model(Xva_dev).float().cpu().numpy()
    return sis_metric(Yva_np, np.clip(vp, 0.0, None), broaden_cm=SELECTION_CM)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, out_dim), out_activation()]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def _train_loop(model, Xtr, Ytr, device, params, n_epochs, Xva=None, Yva=None,
                patience=None, trial=None):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params.get("weight_decay", 0.0))
    lossf = get_loss()
    use_amp = bool(CONFIG["use_amp"]) and str(device).startswith("cuda")
    scaler  = _make_grad_scaler() if use_amp else None
    Xtr_d = _to_dev_tensor(Xtr, device, torch.float32 if Xtr.dtype.kind == "f" else torch.long)
    Ytr_d = _to_dev_tensor(Ytr, device)
    tl = make_loader(Xtr_d, Ytr_d, params["batch"], True)
    Xva_d = _to_dev_tensor(Xva, device, torch.float32 if (Xva is not None and Xva.dtype.kind=='f') else torch.long) if Xva is not None else None
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(n_epochs):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with _autocast_cuda():
                    loss = lossf(model(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = lossf(model(xb), yb); loss.backward(); opt.step()
        if Xva_d is not None:
            vsis = _val_selection_sis(model, Xva_d, Yva)
            if vsis > best + 1e-6:
                best, best_ep, bad = vsis, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if trial is not None:
                trial.report(vsis, ep)
                if trial.should_prune(): raise optuna.TrialPruned()
            if patience and bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, (best if Xva_d is not None else None), (best_ep if Xva_d is not None else n_epochs)

def predict_torch(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(_to_dev_tensor(X, device,
                     torch.float32 if X.dtype.kind == "f" else torch.long)).float().cpu().numpy()

def build_vocab(smiles_list):
    chars = sorted(set("".join(smiles_list)))
    return {c: i + 1 for i, c in enumerate(chars)}

def encode_smiles(smiles_list, stoi, max_len):
    M = np.zeros((len(smiles_list), max_len), dtype=np.int64)
    for i, s in enumerate(smiles_list):
        for j, c in enumerate(s[:max_len]):
            M[i, j] = stoi.get(c, 0)
    return M

class SmilesCNN(nn.Module):
    def __init__(self, vocab, out_dim, emb=64, ch=128, k=7, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, ch, k, padding=k // 2), nn.ReLU(), nn.BatchNorm1d(ch),
            nn.Conv1d(ch, ch, k, padding=k // 2), nn.ReLU(), nn.BatchNorm1d(ch),
            nn.AdaptiveMaxPool1d(1))
        self.head = nn.Sequential(nn.Linear(ch, 512), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(512, out_dim), out_activation())
    def forward(self, x):
        z = self.emb(x.long()).transpose(1, 2)
        return self.head(self.conv(z).squeeze(-1))

class SmilesTransformer(nn.Module):
    def __init__(self, vocab, out_dim, max_len, d_model=128, nhead=4, n_layers=3,
                 ff_dim=256, emb=None, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        emb = d_model if emb is None else emb
        self.tok = nn.Embedding(vocab + 1, emb, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.in_proj = nn.Identity() if emb == d_model else nn.Linear(emb, d_model)
        self.in_drop = nn.Dropout(dropout)
        try:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                                   dropout=dropout, activation="gelu",
                                                   batch_first=True, norm_first=True)
        except TypeError:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(ff_dim, out_dim), out_activation())
        self.max_len = max_len
    def forward(self, x):
        x = x.long()
        pad_mask = (x == 0)
        L = x.size(1)
        pos_ids = torch.arange(L, device=x.device).clamp_max(self.max_len - 1)
        h = self.in_proj(self.tok(x)) + self.pos(pos_ids)[None, :, :]
        h = self.in_drop(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
