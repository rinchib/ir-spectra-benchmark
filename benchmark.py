import json
import os
import random
import time

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from config import CFG_HASH, CONFIG, N_GPU, RUN_SEED, SEED
from data import ids, morgan, N, SMILES, WAVENUMBERS, Y
from splits import grouped_kfold, inner_split, make_groups, RESOLUTION, single_split, single_split_strat
from metrics import _sumnorm, SELECTION_CM, sis_metric
from nets import _train_loop, build_vocab, encode_smiles, get_loss, MLP, out_activation, predict_torch, SmilesCNN, SmilesTransformer


def fit_tree(Xtr, Ytr, params, pca_k):
    pca_k = int(max(2, min(pca_k, Ytr.shape[0] - 1, Ytr.shape[1])))
    pca = PCA(n_components=pca_k, random_state=SEED).fit(Ytr)
    Ztr = pca.transform(Ytr)
    m = RandomForestRegressor(n_jobs=-1, random_state=RUN_SEED, **params)
    m.fit(Xtr, Ztr)
    return (m, pca)

def predict_tree(bundle, X):
    m, pca = bundle
    return np.clip(pca.inverse_transform(m.predict(X)), 0.0, None)

def _explicit_h_smiles(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, allHsExplicit=True) if m is not None else s
SMILES_H = [_explicit_h_smiles(s) for s in SMILES]
print("explicit-H SMILES ready (allHsExplicit), e.g.", SMILES[0], "->", SMILES_H[0])

optuna.logging.set_verbosity(optuna.logging.WARNING)

ME, PA = CONFIG["max_epochs"], CONFIG["patience"]
CKPT   = CONFIG["ckpt_dir"]
CURRENT_SCHEME = None
FIXED_PARAMS = {}

def _save_weights(kind, model, base):
    try:
        if kind == "torch":  torch.save(model.state_dict(), str(base) + ".pt")
        elif kind == "tree": joblib.dump(model, str(base) + ".joblib")
    except Exception as e:
        print("  [warn] weight save failed:", str(e)[:90], flush=True)

def gpu_for(n):  return f"cuda:{n % N_GPU}" if N_GPU else "cpu"

def run_study(objective, n_trials, n_jobs, name="study"):
    storage = os.environ.get("IR_OPTUNA_STORAGE")
    study = optuna.create_study(
        study_name=(name if storage else None), storage=storage, load_if_exists=bool(storage),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15))
    t0 = time.time()
    def progress(st, tr):
        val = "pruned" if tr.value is None else f"{tr.value:.4f}"
        try:    best = f"{st.best_value:.4f}"
        except ValueError: best = "n/a"
        print(f"  [{name}] trial {tr.number + 1}/{n_trials}  val={val}  best={best}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{name}] starting {n_trials} trials on {'cpu' if not N_GPU else str(N_GPU)+' GPU(s)'} ...", flush=True)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[progress])
    print(f"[{name}] done — best val={study.best_value:.4f}", flush=True)
    return study

def _study_best(study):
    r = dict(study.best_params)
    be = study.best_trial.user_attrs.get("best_ep", None)
    if be is not None: r["_best_ep"] = int(be)
    return r

def _key(name, fold):  return f"{name}_{CURRENT_SCHEME}_{CFG_HASH}_f{fold}"
def save_json(obj, name): (CKPT / name).write_text(json.dumps(obj, indent=2))
def load_json(name):
    p = CKPT / name; return json.loads(p.read_text()) if p.exists() else None

def tuned(name, run_fn):
    if name in FIXED_PARAMS:
        return FIXED_PARAMS[name]
    saved = load_json(f"best_{name}.json")
    if CONFIG["use_checkpoints"] and saved is not None:
        print(f"[{name}] loaded HPO checkpoint -> skipping search", flush=True); return saved
    result = run_fn(); save_json(result, f"best_{name}.json"); return result

def _bits_param(trial):
    return trial.suggest_categorical("bits", CONFIG["morgan_sizes"]) if CONFIG["morgan_search"] else 2048

def cv_eval_mlp(itr, iva, te, fold, groups):
    def obj(trial):
        bits = _bits_param(trial)
        nl = trial.suggest_int("n_layers", 1, 3)
        w  = trial.suggest_int("width", 128, 2048, log=True)
        params = {"hidden": [w]*nl,
                  "dropout":      trial.suggest_float("dropout", 0.0, 0.6),
                  "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
                  "batch":        trial.suggest_categorical("batch", [32, 64, 128, 256])}
        dev = gpu_for(trial.number)
        m = MLP(morgan[bits].shape[1], Y.shape[1], params["hidden"], params["dropout"])
        m, _, _bep = _train_loop(m, morgan[bits][itr], Y[itr], dev, params, ME,
                                 morgan[bits][iva], Y[iva], patience=PA, trial=trial)
        trial.set_user_attr("best_ep", int(_bep))
        return sis_metric(Y[iva], predict_torch(m, morgan[bits][iva], dev), broaden_cm=SELECTION_CM)
    best = tuned(_key("mlp", fold),
                 lambda: _study_best(run_study(obj, CONFIG["mlp_trials"], max(1, N_GPU), f"MLP-{CURRENT_SCHEME}-f{fold}")))
    bits = best.get("bits", 2048)
    p = {"hidden": [best["width"]]*best["n_layers"], "dropout": best["dropout"],
         "lr": best["lr"], "weight_decay": best["weight_decay"], "batch": best["batch"]}
    dev = gpu_for(fold)
    best_ep = best.get("_best_ep", None)
    if best_ep is None:
        m0 = MLP(morgan[bits].shape[1], Y.shape[1], p["hidden"], p["dropout"])
        _, _, best_ep = _train_loop(m0, morgan[bits][itr], Y[itr], dev, p, ME,
                                    morgan[bits][iva], Y[iva], patience=PA)
    otr = np.sort(np.concatenate([itr, iva]))
    m = MLP(morgan[bits].shape[1], Y.shape[1], p["hidden"], p["dropout"])
    m, _, _ = _train_loop(m, morgan[bits][otr], Y[otr], dev, p, max(1, best_ep + 1))
    return {"pred": predict_torch(m, morgan[bits][te], dev), "model": m, "kind": "torch",
            "predict_all": lambda idx: predict_torch(m, morgan[bits][idx], dev)}

def cv_eval_rf(itr, iva, te, fold, groups):
    def obj(trial):
        bits  = _bits_param(trial)
        pca_k = trial.suggest_int("pca_k", CONFIG["pca_min"], CONFIG["pca_max"])
        params = {"n_estimators":     trial.suggest_int("n_estimators", 200, 700),
                  "max_depth":        trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
                  "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                  "max_features":     trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5])}
        b = fit_tree(morgan[bits][itr], Y[itr], params, pca_k)
        return sis_metric(Y[iva], predict_tree(b, morgan[bits][iva]), broaden_cm=SELECTION_CM)
    best = tuned(_key("rf", fold),
                 lambda: run_study(obj, CONFIG["rf_trials"], 1, f"RF-{CURRENT_SCHEME}-f{fold}").best_params)
    bits = best.get("bits", 2048); pk = best["pca_k"]
    pr = {k: v for k, v in best.items() if k not in ("pca_k", "bits", "_best_ep")}
    otr = np.sort(np.concatenate([itr, iva]))
    b = fit_tree(morgan[bits][otr], Y[otr], pr, pk)
    return {"pred": predict_tree(b, morgan[bits][te]), "model": b, "kind": "tree",
            "predict_all": lambda idx: predict_tree(b, morgan[bits][idx])}

def cv_eval_cnn(itr, iva, te, fold, groups, smiles_src=None, key="cnn", label="CNN"):
    SRC = SMILES if smiles_src is None else smiles_src
    otr = np.sort(np.concatenate([itr, iva]))
    stoi = build_vocab([SRC[i] for i in otr])
    max_len = int(np.percentile([len(SRC[i]) for i in otr], 99))
    Xs = encode_smiles(SRC, stoi, max_len)
    def obj(trial):
        arch = {"emb": trial.suggest_int("emb", 16, 256, log=True),
                "ch":  trial.suggest_int("ch", 32, 512, log=True),
                "k":   trial.suggest_categorical("k", [3, 5, 7]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5)}
        tp = {"lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
              "batch": trial.suggest_categorical("batch", [32, 64, 128]),
              "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)}
        dev = gpu_for(trial.number)
        m = SmilesCNN(len(stoi), Y.shape[1], arch["emb"], arch["ch"], arch["k"], arch["dropout"])
        m, _, _bep = _train_loop(m, Xs[itr], Y[itr], dev, tp, ME, Xs[iva], Y[iva], patience=PA, trial=trial)
        trial.set_user_attr("best_ep", int(_bep))
        return sis_metric(Y[iva], predict_torch(m, Xs[iva], dev), broaden_cm=SELECTION_CM)
    best = tuned(_key(key, fold),
                 lambda: _study_best(run_study(obj, CONFIG["cnn_trials"], max(1, N_GPU), f"{label}-{CURRENT_SCHEME}-f{fold}")))
    ct = {"lr": best["lr"], "batch": best["batch"], "weight_decay": best["weight_decay"]}
    dev = gpu_for(fold)
    best_ep = best.get("_best_ep", None)
    if best_ep is None:
        m0 = SmilesCNN(len(stoi), Y.shape[1], best["emb"], best["ch"], best["k"], best["dropout"])
        _, _, best_ep = _train_loop(m0, Xs[itr], Y[itr], dev, ct, ME, Xs[iva], Y[iva], patience=PA)
    m = SmilesCNN(len(stoi), Y.shape[1], best["emb"], best["ch"], best["k"], best["dropout"])
    m, _, _ = _train_loop(m, Xs[otr], Y[otr], dev, ct, max(1, best_ep + 1))
    return {"pred": predict_torch(m, Xs[te], dev), "model": m, "kind": "torch",
            "predict_all": lambda idx: predict_torch(m, Xs[idx], dev)}

def cv_eval_smiles_transformer(itr, iva, te, fold, groups, smiles_src=None, key="smilestransformer", label="SMILEST"):
    SRC = SMILES if smiles_src is None else smiles_src
    otr = np.sort(np.concatenate([itr, iva]))
    stoi = build_vocab([SRC[i] for i in otr])
    max_len = max(1, int(np.percentile([len(SRC[i]) for i in otr], 99)))
    Xs = encode_smiles(SRC, stoi, max_len)
    def obj(trial):
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        d_mult = trial.suggest_int("d_mult", 16, 32)
        arch = {"nhead": nhead, "d_mult": d_mult,
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "ff_dim":   trial.suggest_categorical("ff_dim", [128, 256, 512]),
                "emb":      trial.suggest_int("emb", 32, 256, log=True),
                "dropout":  trial.suggest_float("dropout", 0.0, 0.3)}
        tp = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              "batch": trial.suggest_categorical("batch", [32, 64, 128]),
              "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)}
        dev = gpu_for(trial.number)
        m = SmilesTransformer(len(stoi), Y.shape[1], max_len, d_model=nhead*d_mult, nhead=nhead,
                              n_layers=arch["n_layers"], ff_dim=arch["ff_dim"], emb=arch["emb"], dropout=arch["dropout"])
        m, _, _bep = _train_loop(m, Xs[itr], Y[itr], dev, tp, ME, Xs[iva], Y[iva], patience=PA, trial=trial)
        trial.set_user_attr("best_ep", int(_bep))
        return sis_metric(Y[iva], predict_torch(m, Xs[iva], dev), broaden_cm=SELECTION_CM)
    best = tuned(_key(key, fold),
                 lambda: _study_best(run_study(obj, CONFIG["smilestransformer_trials"], max(1, N_GPU),
                                               f"{label}-{CURRENT_SCHEME}-f{fold}")))
    d_model = int(best["nhead"]) * int(best["d_mult"])
    def _build():
        return SmilesTransformer(len(stoi), Y.shape[1], max_len, d_model=d_model, nhead=best["nhead"],
                                 n_layers=best["n_layers"], ff_dim=best["ff_dim"], emb=best["emb"], dropout=best["dropout"])
    ct = {"lr": best["lr"], "batch": best["batch"], "weight_decay": best["weight_decay"]}
    dev = gpu_for(fold)
    best_ep = best.get("_best_ep", None)
    if best_ep is None:
        m0 = _build(); _, _, best_ep = _train_loop(m0, Xs[itr], Y[itr], dev, ct, ME, Xs[iva], Y[iva], patience=PA)
    m = _build()
    m, _, _ = _train_loop(m, Xs[otr], Y[otr], dev, ct, max(1, best_ep + 1))
    return {"pred": predict_torch(m, Xs[te], dev), "model": m, "kind": "torch",
            "predict_all": lambda idx: predict_torch(m, Xs[idx], dev)}

def cv_eval_cnn_h(itr, iva, te, fold, groups):
    return cv_eval_cnn(itr, iva, te, fold, groups, smiles_src=SMILES_H, key="cnnh", label="CNNH")
def cv_eval_smiles_transformer_h(itr, iva, te, fold, groups):
    return cv_eval_smiles_transformer(itr, iva, te, fold, groups, smiles_src=SMILES_H,
                                      key="smilestransformerh", label="SMILESTH")

_BV2048 = None
def _bitvects():
    global _BV2048
    if _BV2048 is None:
        _BV2048 = []
        for row in morgan[2048]:
            bv = DataStructs.ExplicitBitVect(2048)
            on = np.nonzero(row)[0]
            if len(on): bv.SetBitsFromList([int(x) for x in on])
            _BV2048.append(bv)
    return _BV2048

def cv_eval_mean(itr, iva, te, fold, groups):
    otr = np.concatenate([itr, iva])
    mean_spec = _sumnorm(Y[otr]).mean(0, keepdims=True)
    return np.repeat(mean_spec, len(te), axis=0).astype(np.float32)

def cv_eval_nn(itr, iva, te, fold, groups):
    bvs = _bitvects(); otr = np.concatenate([itr, iva])
    otr_bv = [bvs[i] for i in otr]
    pred = np.empty((len(te), Y.shape[1]), dtype=np.float32)
    for r, i in enumerate(te):
        sims = DataStructs.BulkTanimotoSimilarity(bvs[i], otr_bv)
        pred[r] = Y[otr[int(np.argmax(sims))]]
    return pred

try:
    import dgl
    from dgl.nn.pytorch import GINConv
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
    from dgllife.model import MPNNPredictor, AttentiveFPPredictor
    GRAPH_OK = True
except Exception as e:
    GRAPH_OK = False
    print("DGL / DGL-LifeSci unavailable -> graph models will be SKIPPED:", str(e)[:140])

if GRAPH_OK:
    _ATOM_F = CanonicalAtomFeaturizer(atom_data_field='h')
    _BOND_F = CanonicalBondFeaturizer(bond_data_field='e', self_loop=True)
    NODE_FDIM = _ATOM_F.feat_size('h')
    EDGE_FDIM = _BOND_F.feat_size('e')

    def _featurize_all_graphs():
        cache = CONFIG["cache_dir"] / ("graphs_%d.bin" % N)
        if cache.exists():
            gs, _ = dgl.load_graphs(str(cache)); return gs
        t0 = time.time(); gs = []
        for s in SMILES:
            g = smiles_to_bigraph(s, add_self_loop=True, node_featurizer=_ATOM_F, edge_featurizer=_BOND_F)
            if g is None:
                g = dgl.graph(([0], [0]), num_nodes=1)
                g.ndata['h'] = torch.zeros((1, NODE_FDIM)); g.edata['e'] = torch.zeros((1, EDGE_FDIM))
            gs.append(g)
        dgl.save_graphs(str(cache), gs)
        print("  featurized %d graphs in %.0fs" % (len(gs), time.time() - t0))
        return gs

    GRAPHS = _featurize_all_graphs()
    print("graph featurization ready:", len(GRAPHS), "graphs | node_fdim=", NODE_FDIM, "| edge_fdim=", EDGE_FDIM)

if GRAPH_OK:
    import torch.utils.data as _tud

    class GINNet(nn.Module):
        def __init__(self, in_dim, out_dim, hidden, n_layers, dropout):
            super().__init__()
            self.lin0 = nn.Linear(in_dim, hidden)
            self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
            for _ in range(n_layers):
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINConv(mlp, 'sum', learn_eps=True)); self.bns.append(nn.BatchNorm1d(hidden))
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hidden, out_dim), out_activation())
        def forward(self, g, h, e=None):
            h = self.lin0(h)
            for conv, bn in zip(self.convs, self.bns):
                h = self.drop(torch.relu(bn(conv(g, h))))
            g.ndata['_h'] = h
            return self.head(dgl.mean_nodes(g, '_h'))

    class MPNNNet(nn.Module):
        def __init__(self, node_in, edge_in, out_dim, node_out, edge_hidden, n_mp):
            super().__init__()
            self.net = MPNNPredictor(node_in_feats=node_in, edge_in_feats=edge_in, node_out_feats=node_out,
                                     edge_hidden_feats=edge_hidden, n_tasks=out_dim, num_step_message_passing=n_mp)
            self.act = out_activation()
        def forward(self, g, h, e): return self.act(self.net(g, h, e))

    class AttentiveFPNet(nn.Module):
        def __init__(self, node_in, edge_in, out_dim, n_layers, n_timesteps, graph_feat, dropout):
            super().__init__()
            self.net = AttentiveFPPredictor(node_feat_size=node_in, edge_feat_size=edge_in, num_layers=n_layers,
                                            num_timesteps=n_timesteps, graph_feat_size=graph_feat,
                                            n_tasks=out_dim, dropout=dropout)
            self.act = out_activation()
        def forward(self, g, h, e): return self.act(self.net(g, h, e))

    _GRAPH_NEEDS_EDGE = {"mpnn", "attentivefp"}
    def _make_graph_model(kind, p):
        if kind == "gin":  return GINNet(NODE_FDIM, Y.shape[1], p["hidden"], p["n_layers"], p["dropout"])
        if kind == "mpnn": return MPNNNet(NODE_FDIM, EDGE_FDIM, Y.shape[1], p["node_out"], p["edge_hidden"], p["n_mp"])
        return AttentiveFPNet(NODE_FDIM, EDGE_FDIM, Y.shape[1], p["n_layers"], p["n_timesteps"], p["graph_feat"], p["dropout"])

    class _GraphDS(_tud.Dataset):
        def __init__(self, idx): self.idx = np.asarray(idx)
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = int(self.idx[i]); g = GRAPHS[j]
            src, dst = g.edges()
            ng = dgl.graph((src, dst), num_nodes=g.num_nodes())
            ng.ndata['h'] = g.ndata['h']
            if 'e' in g.edata: ng.edata['e'] = g.edata['e']
            return ng, torch.as_tensor(Y[j], dtype=torch.float32)
    def _gcollate(samples):
        gs, ys = zip(*samples); return dgl.batch(gs), torch.stack(ys)
    def _graph_loader(idx, batch, shuffle):
        return _tud.DataLoader(_GraphDS(idx), batch_size=batch, shuffle=shuffle, drop_last=shuffle, collate_fn=_gcollate)

    def _forward_graph(model, bg, kind):
        h = bg.ndata['h']
        return model(bg, h, bg.edata['e']) if kind in _GRAPH_NEEDS_EDGE else model(bg, h)

    @torch.no_grad()
    def _set_cuda(dev):
        if str(dev).startswith("cuda"): torch.cuda.set_device(torch.device(dev))

    @torch.no_grad()
    def predict_graph(model, idx, dev, kind):
        _set_cuda(dev); model.eval(); outs = []
        for bg, yb in _graph_loader(idx, 256, False):
            bg = bg.to(dev); outs.append(_forward_graph(model, bg, kind).float().cpu().numpy())
        return np.concatenate(outs)

    def train_graph(model, kind, itr, dev, params, n_epochs, iva=None, patience=None, trial=None):
        _set_cuda(dev)
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 0.0))
        lossf = get_loss(); tl = _graph_loader(itr, params["batch"], True)
        best, best_state, best_ep, bad = -1.0, None, 0, 0
        for ep in range(n_epochs):
            model.train()
            for bg, yb in tl:
                bg = bg.to(dev); yb = yb.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = lossf(_forward_graph(model, bg, kind), yb); loss.backward(); opt.step()
            if iva is not None:
                vsis = sis_metric(Y[iva], np.clip(predict_graph(model, iva, dev, kind), 0.0, None), broaden_cm=SELECTION_CM)
                if vsis > best + 1e-6:
                    best, best_ep, bad = vsis, ep, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else: bad += 1
                if trial is not None:
                    trial.report(vsis, ep)
                    if trial.should_prune(): raise optuna.TrialPruned()
                if patience and bad >= patience: break
        if best_state is not None: model.load_state_dict(best_state)
        return model, (best if iva is not None else None), (best_ep if iva is not None else n_epochs)

    def _search_gin(t):
        return {"hidden": t.suggest_int("hidden", 64, 300, log=True), "n_layers": t.suggest_int("n_layers", 2, 5),
                "dropout": t.suggest_float("dropout", 0.0, 0.5), "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True),
                "batch": t.suggest_categorical("batch", [32, 64, 128]), "weight_decay": t.suggest_float("weight_decay", 1e-7, 1e-2, log=True)}
    def _search_mpnn(t):
        return {"node_out": t.suggest_int("node_out", 64, 300, log=True), "edge_hidden": t.suggest_int("edge_hidden", 32, 128, log=True),
                "n_mp": t.suggest_int("n_mp", 2, 6), "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True),
                "batch": t.suggest_categorical("batch", [32, 64, 128]), "weight_decay": t.suggest_float("weight_decay", 1e-7, 1e-2, log=True)}
    def _search_afp(t):
        return {"n_layers": t.suggest_int("n_layers", 2, 4), "n_timesteps": t.suggest_int("n_timesteps", 1, 3),
                "graph_feat": t.suggest_int("graph_feat", 64, 300, log=True), "dropout": t.suggest_float("dropout", 0.0, 0.5),
                "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True), "batch": t.suggest_categorical("batch", [32, 64, 128]),
                "weight_decay": t.suggest_float("weight_decay", 1e-7, 1e-2, log=True)}
    _GRAPH_SEARCH = {"mpnn": _search_mpnn, "attentivefp": _search_afp, "gin": _search_gin}

    def _graph_dev(n):
        if CONFIG.get("graph_parallel"): return gpu_for(n)
        return f"cuda:{CONFIG.get('graph_gpu', 0)}" if N_GPU else "cpu"

    def _cv_eval_graph(kind, name, itr, iva, te, fold):
        search = _GRAPH_SEARCH[kind]
        def obj(trial):
            p = search(trial); dev = _graph_dev(trial.number)
            m = _make_graph_model(kind, p)
            m, _, _bep = train_graph(m, kind, itr, dev, p, ME, iva, patience=PA, trial=trial)
            trial.set_user_attr("best_ep", int(_bep))
            return sis_metric(Y[iva], predict_graph(m, iva, dev, kind), broaden_cm=SELECTION_CM)
        gjobs = max(1, N_GPU) if CONFIG.get("graph_parallel") else 1
        best = tuned(_key(name, fold),
                     lambda: _study_best(run_study(obj, CONFIG[name + "_trials"], gjobs,
                                       name.upper() + "-" + str(CURRENT_SCHEME) + "-f" + str(fold))))
        dev = _graph_dev(fold)
        best_ep = best.get("_best_ep", None)
        if best_ep is None:
            m0 = _make_graph_model(kind, best)
            _, _, best_ep = train_graph(m0, kind, itr, dev, best, ME, iva, patience=PA)
        otr = np.sort(np.concatenate([itr, iva]))
        m = _make_graph_model(kind, best)
        m, _, _ = train_graph(m, kind, otr, dev, best, max(1, best_ep + 1))
        return {"pred": predict_graph(m, te, dev, kind), "model": m, "kind": "torch",
                "predict_all": lambda idx: predict_graph(m, idx, dev, kind)}

    def cv_eval_mpnn(itr, iva, te, fold, groups):        return _cv_eval_graph("mpnn", "mpnn", itr, iva, te, fold)
    def cv_eval_attentivefp(itr, iva, te, fold, groups): return _cv_eval_graph("attentivefp", "attentivefp", itr, iva, te, fold)
    def cv_eval_gin(itr, iva, te, fold, groups):         return _cv_eval_graph("gin", "gin", itr, iva, te, fold)
    print("graph models ready: MPNN, AttentiveFP, GIN")

try:
    from mordred import Calculator as _MordredCalc, descriptors as _mordred_desc
    MORDRED_OK = True
except Exception as e:
    MORDRED_OK = False
    print("mordred unavailable -> Mordred descriptor row will be SKIPPED:", str(e)[:140])

MORDRED = None
if MORDRED_OK:
    def _featurize_mordred():
        global MORDRED
        maxnan = float(CONFIG.get("mordred_max_nan", 0.2))
        cache = CONFIG["cache_dir"] / f"mordred2D_{N}_nan{int(round(maxnan*100))}.npy"
        if cache.exists():
            MORDRED = np.load(cache); return MORDRED
        mols = [Chem.MolFromSmiles(s) for s in SMILES]
        calc = _MordredCalc(_mordred_desc, ignore_3D=True)
        nproc = max(1, (os.cpu_count() or 2) - 1)
        t0 = time.time()
        try:
            df = calc.pandas(mols, nproc=nproc)
        except TypeError:
            df = calc.pandas(mols)
        X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        keep = np.isnan(X).mean(0) < maxnan
        X = X[:, keep]
        tmp = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        X = X[:, np.nanstd(tmp, axis=0) > 1e-12]
        MORDRED = X.astype(np.float32)
        np.save(cache, MORDRED)
        print(f"  Mordred 2D descriptors {MORDRED.shape} (usable cols after prune) in {time.time()-t0:.0f}s", flush=True)
        return MORDRED
    MORDRED = _featurize_mordred()
    if MORDRED is None or MORDRED.shape[1] == 0:
        MORDRED_OK = False
        print("  Mordred produced no usable columns -> row SKIPPED")
    else:
        print("Mordred descriptor row ready:", MORDRED.shape)

if MORDRED_OK:
    def _mordred_prep(train_rows):
        Xt = MORDRED[train_rows]
        med = np.nanmedian(Xt, axis=0); med = np.where(np.isnan(med), 0.0, med)
        Xt = np.where(np.isnan(Xt), med, Xt)
        mu = Xt.mean(0); sd = Xt.std(0) + 1e-8
        full = np.where(np.isnan(MORDRED), med, MORDRED)
        return ((full - mu) / sd).astype(np.float32)

    def cv_eval_mordred(itr, iva, te, fold, groups):
        Xs = _mordred_prep(itr)
        def obj(trial):
            nl = trial.suggest_int("n_layers", 1, 3)
            w  = trial.suggest_int("width", 128, 2048, log=True)
            params = {"hidden": [w]*nl,
                      "dropout":      trial.suggest_float("dropout", 0.0, 0.6),
                      "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                      "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
                      "batch":        trial.suggest_categorical("batch", [32, 64, 128, 256])}
            dev = gpu_for(trial.number)
            m = MLP(Xs.shape[1], Y.shape[1], params["hidden"], params["dropout"])
            m, _, _bep = _train_loop(m, Xs[itr], Y[itr], dev, params, ME, Xs[iva], Y[iva], patience=PA, trial=trial)
            trial.set_user_attr("best_ep", int(_bep))
            return sis_metric(Y[iva], predict_torch(m, Xs[iva], dev), broaden_cm=SELECTION_CM)
        best = tuned(_key("mordred", fold),
                     lambda: _study_best(run_study(obj, CONFIG["mordred_trials"], max(1, N_GPU),
                                                   f"Mordred-{CURRENT_SCHEME}-f{fold}")))
        p = {"hidden": [best["width"]]*best["n_layers"], "dropout": best["dropout"],
             "lr": best["lr"], "weight_decay": best["weight_decay"], "batch": best["batch"]}
        dev = gpu_for(fold)
        otr = np.sort(np.concatenate([itr, iva]))
        Xf = _mordred_prep(otr)
        best_ep = best.get("_best_ep", None)
        if best_ep is None:
            m0 = MLP(Xs.shape[1], Y.shape[1], p["hidden"], p["dropout"])
            _, _, best_ep = _train_loop(m0, Xs[itr], Y[itr], dev, p, ME, Xs[iva], Y[iva], patience=PA)
        m = MLP(Xf.shape[1], Y.shape[1], p["hidden"], p["dropout"])
        m, _, _ = _train_loop(m, Xf[otr], Y[otr], dev, p, max(1, best_ep + 1))
        return {"pred": predict_torch(m, Xf[te], dev), "model": m, "kind": "torch",
                "predict_all": lambda idx: predict_torch(m, Xf[idx], dev)}

    def cv_eval_mordred_rf(itr, iva, te, fold, groups):
        Xs = _mordred_prep(itr)
        def obj(trial):
            pca_k = trial.suggest_int("pca_k", CONFIG["pca_min"], CONFIG["pca_max"])
            params = {"n_estimators":     trial.suggest_int("n_estimators", 200, 700),
                      "max_depth":        trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
                      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                      "max_features":     trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5])}
            b = fit_tree(Xs[itr], Y[itr], params, pca_k)
            return sis_metric(Y[iva], predict_tree(b, Xs[iva]), broaden_cm=SELECTION_CM)
        best = tuned(_key("mordredrf", fold),
                     lambda: run_study(obj, CONFIG["mordred_rf_trials"], 1, f"MordredRF-{CURRENT_SCHEME}-f{fold}").best_params)
        pk = best["pca_k"]; pr = {k: v for k, v in best.items() if k not in ("pca_k", "bits", "_best_ep")}
        otr = np.sort(np.concatenate([itr, iva]))
        Xf = _mordred_prep(otr)
        b = fit_tree(Xf[otr], Y[otr], pr, pk)
        return {"pred": predict_tree(b, Xf[te]), "model": b, "kind": "tree",
                "predict_all": lambda idx: predict_tree(b, Xf[idx])}
    print("Mordred models ready (MLP head + RandomForest head on standardized descriptors)")

MODELS = {"MeanSpectrum":  cv_eval_mean,
          "1NN-Tanimoto":  cv_eval_nn,
          "MLP":           cv_eval_mlp,
          "RandomForest":  cv_eval_rf,
          "SMILES-CNN":    cv_eval_cnn,
          "SMILES-Transformer": cv_eval_smiles_transformer,
          "SMILES-CNN-H":  cv_eval_cnn_h,
          "SMILES-Transformer-H": cv_eval_smiles_transformer_h}
if GRAPH_OK:
    MODELS.update({"MPNN":        cv_eval_mpnn,
                   "AttentiveFP": cv_eval_attentivefp,
                   "GIN":         cv_eval_gin})
if MORDRED_OK:
    MODELS["Mordred-MLP"] = cv_eval_mordred
    MODELS["Mordred-RF"]  = cv_eval_mordred_rf
print("models:", list(MODELS))

HPO_KEY = {"MLP": "mlp", "RandomForest": "rf", "SMILES-CNN": "cnn",
           "SMILES-Transformer": "smilestransformer", "SMILES-CNN-H": "cnnh",
           "SMILES-Transformer-H": "smilestransformerh", "Mordred-MLP": "mordred", "Mordred-RF": "mordredrf",
           "MPNN": "mpnn", "AttentiveFP": "attentivefp", "GIN": "gin"}

if CONFIG.get("save_splits", True):
    (CKPT / "compound_ids.json").write_text(json.dumps(ids))
    np.save(CKPT / "wavenumbers.npy", WAVENUMBERS)

def run_model(mname, fn, a, b, c, fold, groups, tag, light=False):
    pf = CKPT / f"oof_{mname}_{CURRENT_SCHEME}_{CFG_HASH}_{tag}.npy"
    if CONFIG["use_checkpoints"] and pf.exists():
        print(f"  [{mname} {tag}] loaded prediction checkpoint", flush=True)
        return np.load(pf)
    out = fn(a, b, c, fold, groups)
    if isinstance(out, dict):
        pred = out["pred"]
        if CONFIG.get("save_all_preds", True) and not light:
            np.save(CKPT / f"predall_{mname}_{CURRENT_SCHEME}_{CFG_HASH}_{tag}.npy",
                    out["predict_all"](np.arange(N)))
        if CONFIG.get("save_models", True) and not light:
            _save_weights(out["kind"], out["model"],
                          CKPT / f"model_{mname}_{CURRENT_SCHEME}_{CFG_HASH}_{tag}")
    else:
        pred = out
    np.save(pf, pred)
    return pred

def _tune_split(scheme):
    groups = make_groups(scheme)
    if CONFIG.get("stratify_resolution") and scheme == "random":
        return single_split_strat(groups, SEED, CONFIG["val_frac"], CONFIG["test_frac"], RESOLUTION)
    return single_split(groups, SEED, CONFIG["val_frac"], CONFIG["test_frac"])

for scheme in CONFIG["split_schemes"]:
    CURRENT_SCHEME = scheme
    groups = make_groups(scheme)
    print(f"\n########## SCHEME = {scheme} | mode = {CONFIG['eval_mode']} | {CFG_HASH} ##########", flush=True)

    if CONFIG["eval_mode"] == "single":
        tr, va, te = single_split(groups, SEED, CONFIG["val_frac"], CONFIG["test_frac"])
        print(f"  train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)
        if CONFIG.get("save_splits", True):
            np.savez(CKPT / f"split_{scheme}_{CFG_HASH}.npz", train=tr, val=va, test=te)
        for mname, fn in MODELS.items():
            run_model(mname, fn, tr, va, te, 0, groups, "single")

    elif CONFIG["eval_mode"] == "tune_eval":
        ttr, tva, tte = _tune_split(scheme)
        print(f"  [tune] train {len(ttr)}  val {len(tva)}  test {len(tte)}  (HPO split; then "
              f"{CONFIG['n_eval_folds']}-fold x {CONFIG.get('n_eval_repeats',1)} eval)", flush=True)
        if CONFIG.get("save_splits", True):
            np.savez(CKPT / f"split_{scheme}_{CFG_HASH}_tune.npz", train=ttr, val=tva, test=tte)
        FIXED_PARAMS.clear()
        tuned_params = {}
        for mname, fn in MODELS.items():
            hk = HPO_KEY.get(mname)
            if hk is None: continue
            run_model(mname, fn, ttr, tva, tte, 0, groups, "tune")
            tuned_params[mname] = load_json(f"best_{_key(hk, 0)}.json")
            print(f"  [tune {mname}] hyperparameters ready", flush=True)
        K = CONFIG["n_eval_folds"]; R = int(CONFIG.get("n_eval_repeats", 1))
        OUTER = grouped_kfold(groups, K, SEED)
        if CONFIG.get("save_splits", True):
            np.savez(CKPT / f"split_{scheme}_{CFG_HASH}.npz",
                     **{f"fold{f}_test": ote for f, (otr, ote) in enumerate(OUTER)})
        for r in range(R):
            for f, (otr, ote) in enumerate(OUTER):
                RUN_SEED = SEED + 1000*r + f
                random.seed(RUN_SEED); np.random.seed(RUN_SEED)
                torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
                FIXED_PARAMS.clear()
                for dn, hk in HPO_KEY.items():
                    if tuned_params.get(dn) is not None:
                        FIXED_PARAMS[_key(hk, f)] = tuned_params[dn]
                itr, iva = inner_split(otr, groups, f)
                print(f"\n===== [{scheme}] EVAL repeat {r+1}/{R} fold {f+1}/{K} | "
                      f"itr {len(itr)} iva {len(iva)} test {len(ote)} | RUN_SEED={RUN_SEED} =====", flush=True)
                for mname, fn in MODELS.items():
                    run_model(mname, fn, itr, iva, ote, f, groups, f"r{r}f{f}", light=True)
        FIXED_PARAMS.clear(); RUN_SEED = SEED

    else:
        OUTER = grouped_kfold(groups, CONFIG["n_outer_folds"], SEED)
        if CONFIG.get("save_splits", True):
            np.savez(CKPT / f"split_{scheme}_{CFG_HASH}.npz",
                     **{f"fold{f}_test": ote for f, (otr, ote) in enumerate(OUTER)})
        for fold, (otr, ote) in enumerate(OUTER):
            itr, iva = inner_split(otr, groups, fold)
            print(f"\n===== [{scheme}] FOLD {fold+1}/{len(OUTER)} | "
                  f"itr {len(itr)} iva {len(iva)} test {len(ote)} =====", flush=True)
            for mname, fn in MODELS.items():
                run_model(mname, fn, itr, iva, ote, fold, groups, f"f{fold}")

print("\nEvaluation complete for schemes:", CONFIG["split_schemes"])
