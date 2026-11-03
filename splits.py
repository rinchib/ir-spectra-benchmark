import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from config import CONFIG, SEED
from data import ids, meta_t, N, SMILES


def _ecfp_bits(smiles_list, nbits=2048, radius=2):
    bvs = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        bvs.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits) if m is not None
                   else DataStructs.ExplicitBitVect(nbits))
    return bvs

def butina_groups(smiles_list, cutoff):
    cache = CONFIG["cache_dir"] / f"butina_{N}_{str(cutoff).replace('.','p')}.npy"
    if cache.exists():
        return np.load(cache)
    bvs = _ecfp_bits(smiles_list)
    n = len(bvs); dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    grp = np.empty(n, dtype=int)
    for ci, c in enumerate(clusters):
        for idx in c: grp[idx] = ci
    np.save(cache, grp)
    print(f"  Butina: {n} molecules -> {len(clusters)} clusters (cutoff={cutoff})")
    return grp

def make_groups(scheme):
    if scheme == "random":
        return np.arange(N)
    if scheme == "cluster":
        return butina_groups(SMILES, CONFIG["cluster_cutoff"])
    raise ValueError(scheme)

def grouped_kfold(groups, k, seed):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    load = np.zeros(k); fold_of = np.empty(len(uniq), dtype=int)
    for gi in order[np.argsort(-sizes[order], kind="stable")]:
        f = int(np.argmin(load)); fold_of[gi] = f; load[f] += sizes[gi]
    row_fold = fold_of[inv]
    folds = []
    allidx = np.arange(len(groups))
    for f in range(k):
        te = allidx[row_fold == f]; tr = allidx[row_fold != f]
        folds.append((np.sort(tr), np.sort(te)))
    return folds

def inner_split(otr, groups, fold):
    g = groups[otr]; uniq = np.unique(g)
    rng = np.random.default_rng(SEED + 1 + fold); rng.shuffle(uniq)
    target = CONFIG["inner_val_frac"] * len(otr); val_groups = set(); cnt = 0
    for gg in uniq:
        if cnt >= target: break
        val_groups.add(gg); cnt += int((g == gg).sum())
    vmask = np.isin(g, list(val_groups))
    return np.sort(otr[~vmask]), np.sort(otr[vmask])

def single_split(groups, seed, val_frac, test_frac):
    groups = np.asarray(groups); uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv); rng = np.random.default_rng(seed); order = rng.permutation(len(uniq))
    n_te, n_va = test_frac * len(groups), val_frac * len(groups)
    part = np.full(len(uniq), 0)
    cte = cva = 0
    for gi in order:
        if cte < n_te:      part[gi] = 2; cte += sizes[gi]
        elif cva < n_va:    part[gi] = 1; cva += sizes[gi]
    rp = part[inv]; allidx = np.arange(len(groups))
    return np.sort(allidx[rp == 0]), np.sort(allidx[rp == 1]), np.sort(allidx[rp == 2])

def single_split_strat(groups, seed, val_frac, test_frac, strata):
    strata = np.asarray(strata); rng = np.random.default_rng(seed)
    part = np.zeros(len(strata), dtype=int)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]; rng.shuffle(idx)
        n_te = int(round(test_frac * len(idx))); n_va = int(round(val_frac * len(idx)))
        part[idx[:n_te]] = 2; part[idx[n_te:n_te + n_va]] = 1
    allidx = np.arange(len(strata))
    return np.sort(allidx[part == 0]), np.sort(allidx[part == 1]), np.sort(allidx[part == 2])

_res_by_id = dict(zip(meta_t["NIST ID"].astype(str), meta_t["Up or downsampled?"].astype(str)))
RESOLUTION = np.array([_res_by_id.get(i, "?") for i in ids])

print("eval_mode:", CONFIG["eval_mode"])
for sch in CONFIG["split_schemes"]:
    gr = make_groups(sch)
    if CONFIG["eval_mode"] in ("single", "tune_eval"):
        tr, va, te = single_split(gr, SEED, CONFIG["val_frac"], CONFIG["test_frac"])
        tail = ("  -> then %d-fold eval" % CONFIG["n_eval_folds"]) if CONFIG["eval_mode"] == "tune_eval" else ""
        print(f"scheme={sch:8s}: {len(np.unique(gr))} groups -> tune train {len(tr)}  val {len(va)}  test {len(te)}{tail}")
    else:
        print(f"scheme={sch:8s}: {len(np.unique(gr))} groups over {N} compounds "
              f"(test ~{N//CONFIG['n_outer_folds']} per fold)")
