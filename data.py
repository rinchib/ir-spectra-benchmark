import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from rdkit import Chem, RDLogger

from config import CONFIG


def load_features(fname, n_meta):
    src   = CONFIG["data_dir"] / fname
    stem  = Path(fname).stem
    npy   = CONFIG["cache_dir"] / f"{stem}.npy"
    pk    = CONFIG["cache_dir"] / f"{stem}.meta.pkl"
    cols  = CONFIG["cache_dir"] / f"{stem}.cols.json"
    if npy.exists() and pk.exists():
        return pd.read_pickle(pk), np.load(npy), json.loads(cols.read_text())
    t0 = time.time()
    wb = load_workbook(src, read_only=True, data_only=True); ws = wb.active
    it = ws.iter_rows(values_only=True)
    header = list(next(it)); meta_names, feat_names = header[:n_meta], header[n_meta:]
    meta_rows, feats = [], []
    for r in it:
        meta_rows.append(r[:n_meta]); feats.append(r[n_meta:])
    wb.close()
    meta = pd.DataFrame(meta_rows, columns=meta_names)
    X = np.asarray(feats, dtype=np.float32); feat = [str(c) for c in feat_names]
    np.save(npy, X); meta.to_pickle(pk); cols.write_text(json.dumps(feat))
    print(f"  loaded+cached {fname} {X.shape} in {time.time()-t0:.0f}s")
    return meta, X, feat

def load_smiles(fname):
    src = CONFIG["data_dir"] / fname
    pk  = CONFIG["cache_dir"] / (Path(fname).stem + ".pkl")
    if pk.exists():
        return pd.read_pickle(pk)
    df = pd.read_excel(src); df.to_pickle(pk); return df

RDLogger.DisableLog('rdApp.*')

meta_t, Y, wn_cols = load_features(CONFIG["target_file"], CONFIG["n_meta_target"])
WAVENUMBERS = np.array([float(c) for c in wn_cols])
ids = meta_t["NIST ID"].astype(str).tolist()
print("raw targets Y:", Y.shape, "| wavenumbers:", WAVENUMBERS[0], "...", WAVENUMBERS[-1])

morgan, morgan_ids = {}, {}
for n in CONFIG["morgan_sizes"]:
    m_meta, Xn, _ = load_features(f"MorganFP_{n}.xlsx", CONFIG["n_meta_morgan"])
    morgan[n] = Xn.astype(np.float32)
    morgan_ids[n] = m_meta["NIST ID"].astype(str).tolist()

smiles_df = load_smiles(CONFIG["smiles_file"]).set_index("NIST ID")
smiles_df.index = smiles_df.index.astype(str)
SMILES = smiles_df.loc[ids, "SMILES"].tolist()

for n in CONFIG["morgan_sizes"]:
    assert morgan_ids[n] == ids, f"MorganFP_{n} NIST-ID order does not match the spectra file!"
assert len(SMILES) == Y.shape[0]
print("alignment OK: spectra / MorganFP / SMILES share identical NIST-ID order.")

if CONFIG["clip_negative"]:
    nneg = int((Y < 0).sum())
    Y = np.clip(Y, 0.0, None)
    print(f"clipped {nneg} negative absorbance values to 0 ({100*nneg/Y.size:.4f}% of points)")

def _canon(s):
    m = Chem.MolFromSmiles(s);  return Chem.MolToSmiles(m) if m is not None else None
canon = [_canon(s) for s in SMILES]

if CONFIG["chemprop_atom_filter"]:
    _ALLOWED = {"C", "H", "O", "N", "Si", "P", "S", "F", "Cl", "Br", "I"}
    keep, n_salt, n_atom = [], 0, 0
    for i, s in enumerate(SMILES):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if "." in s or len(Chem.GetMolFrags(m)) > 1:
            n_salt += 1; continue
        if set(a.GetSymbol() for a in m.GetAtoms()) - _ALLOWED:
            n_atom += 1; continue
        keep.append(i)
    if len(keep) < len(SMILES):
        print(f"Chemprop-IR atom/salt filter: dropped {len(SMILES)-len(keep)} "
              f"({n_salt} mixtures/salts, {n_atom} non-allowed atoms incl. '*' wildcard)")
        keep = np.array(keep, dtype=int)
        Y = Y[keep]; SMILES = [SMILES[i] for i in keep]
        ids = [ids[i] for i in keep]; canon = [canon[i] for i in keep]
        for n in CONFIG["morgan_sizes"]: morgan[n] = morgan[n][keep]

if CONFIG["dedup_canonical"]:
    seen, keep = set(), []
    for i, c in enumerate(canon):
        key = c if c is not None else f"__row{i}"
        if key in seen:  continue
        seen.add(key);  keep.append(i)
    keep = np.array(keep, dtype=int)
    if len(keep) < Y.shape[0]:
        print(f"de-duplicated {Y.shape[0]-len(keep)} exact-duplicate molecules "
              f"(kept {len(keep)} unique canonical SMILES)")
    Y      = Y[keep]
    SMILES = [SMILES[i] for i in keep]
    ids    = [ids[i]    for i in keep]
    canon  = [canon[i]  for i in keep]
    for n in CONFIG["morgan_sizes"]:
        morgan[n] = morgan[n][keep]

if CONFIG.get("dedup_spectra", False):
    thr = float(CONFIG.get("spectra_dup_cosine", 0.999))
    Xn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    S = (Xn @ Xn.T).astype(np.float32)
    nrow = Y.shape[0]; ar = np.arange(nrow); keepm = np.ones(nrow, dtype=bool); ndrop = 0
    examples = []
    for i in range(nrow):
        if not keepm[i]:
            continue
        dup = (ar > i) & keepm & (S[i] >= thr)
        if dup.any():
            if len(examples) < 4:
                j0 = int(np.where(dup)[0][0]); examples.append((ids[i], ids[j0], float(S[i, j0])))
            ndrop += int(dup.sum()); keepm[dup] = False
    del S, Xn
    if ndrop:
        keep = np.where(keepm)[0]
        print(f"spectral dedup: dropped {ndrop} near-duplicate spectra (cosine >= {thr}; kept {len(keep)})")
        for a, b, c in examples:
            print(f"    e.g. kept {a} / dropped {b}  (cosine {c:.5f})")
        Y = Y[keep]; SMILES = [SMILES[i] for i in keep]
        ids = [ids[i] for i in keep]; canon = [canon[i] for i in keep]
        for n in CONFIG["morgan_sizes"]:
            morgan[n] = morgan[n][keep]

if CONFIG.get("target_norm", "sum") == "sum":
    Y = (Y / np.clip(Y.sum(1, keepdims=True), 1e-12, None)).astype(np.float32)
    print("targets sum-normalized -> each spectrum is a distribution (sum=1)")

N = len(ids)
assert all(morgan[n].shape[0] == N for n in CONFIG["morgan_sizes"]) and len(SMILES) == N
print(f"All inputs aligned on {N} compounds.  SMILES e.g. {SMILES[0]}")
