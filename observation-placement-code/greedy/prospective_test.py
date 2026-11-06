"""Run the prospective test frozen in formal_analysis/prospective/PREDICTIONS.md. Independent of the C++ engine."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
OUT = Path(__file__).resolve().parents[1] / 'formal_analysis' / 'prospective'
SEED, GRAPHS = 2260000, 10
SIZES, FAMS = [30, 70, 150, 400], {'BA': 0, 'ER': 1}


def h(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def build(fam, n, seed):
    g = nx.gnp_random_graph(n, 4 / (n - 1), seed=seed) if fam == 'ER' else nx.barabasi_albert_graph(n, 2, seed=seed)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    return nx.convert_node_labels_to_integers(g, ordering='sorted')


def dist(rng, e, n, limit):
    w = rng.exponential(size=len(e))
    return dijkstra(csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)), directed=True, limit=limit)


def find_tau(rng, e, n, reps):
    grid = np.linspace(.01, 1., 397); tot = np.zeros(len(grid))
    for _ in range(reps):
        d = dist(rng, e, n, 1.)
        v = np.sort(d[np.isfinite(d)])
        tot += np.searchsorted(v, grid, side='right') / n
    return float(np.interp(2., tot / reps, grid))


def run(job):
    fam, N, k, Rp, R, label = job['fam'], job['N'], job['k'], job['Rp'], job['R'], job['label']
    path = job['out'] / f'{label}.csv'
    if path.exists(): return label
    seed = SEED + 1000 * FAMS[fam] + 10000 * SIZES.index(N) + k if job['seed'] is None else job['seed']
    g = build(fam, N, seed); n = g.number_of_nodes(); e = np.array(g.edges()); deg = np.bincount(e.ravel(), minlength=n)
    rng = np.random.default_rng(seed + 7)
    tau = find_tau(rng, e, n, job['Rt'])
    P = np.zeros((n, n))
    for _ in range(Rp): P += dist(rng, e, n, tau) <= tau
    P /= Rp; P = (P + P.T) / 2; off = P.copy(); np.fill_diagonal(off, 0.)
    S = off.sum(1); q = (1 + S) / n; hs = h(off).sum(1)
    Iv = h(q) - hs / n; F = (1 + S) * (np.log(n / (1 + S)) + 1) - hs
    Lp = np.zeros((n, n))
    for v in range(n):
        Lp[v] = 2 * off[v] - np.maximum(off[v], (off * off[v][:, None]).max(0))
    Lp = (Lp + Lp.T) / 2; np.fill_diagonal(Lp, 0.)
    sets = []
    for frac in [.05, .1, .2]:
        m = max(2, round(n * frac))
        sets.append(('T_I', frac, m, np.argsort(-Iv)[:m]))
        M = [int(rng.integers(n))]; cost = Lp[M[0]].copy(); cost[M[0]] = np.inf
        while len(M) < m:
            v = int(np.argmin(cost)); M.append(v); cost += Lp[v]; cost[M] = np.inf
        sets.append(('T_L', frac, m, np.array(M)))
        sets.append(('R', frac, m, rng.choice(n, m, replace=False)))
    salt = np.random.default_rng(seed + 11).integers(1, 2 ** 62, size=n, dtype=np.int64) * 2 + 1
    code = np.zeros((len(sets), 2 * R, n), np.int64)
    for r in range(2 * R):
        b = dist(rng, e, n, tau) <= tau
        for i, (*_, M) in enumerate(sets):
            code[i, r] = (b[:, M].astype(np.int64) * salt[M]).sum(1)
    rows = []
    src = np.arange(n)
    for i, (name, frac, m, M) in enumerate(sets):
        tr, te = code[i, :R].ravel(), code[i, R:].ravel(); st = np.tile(src, R)
        k1, inv = np.unique(tr, return_inverse=True)
        c = pd.DataFrame(dict(y=inv, s=st)).value_counts().reset_index(name='c').sort_values(['y', 'c', 's'], ascending=[True, False, True]).drop_duplicates('y').sort_values('y')
        rule = c.s.values
        pos = np.searchsorted(k1, te); pos[pos == len(k1)] = 0
        seen = k1[pos] == te
        a = (np.sum(seen & (rule[pos] == st)) + np.sum(~seen) / n) / (R * n)
        allc = np.r_[tr, te]; ku, inva = np.unique(allc, return_inverse=True)
        st2 = np.tile(src, 2 * R); joint = np.bincount(inva.astype(np.int64) * n + st2) / (2 * R * n)
        py = np.bincount(inva) / (2 * R * n); joint = joint[joint > 0]; py = py[py > 0]
        info = np.log(n) - (-(joint * np.log(joint)).sum() + (py * np.log(py)).sum())
        A = off[np.ix_(M, M)]; mask = np.ones(n, bool); mask[M] = False; T = off[:, M][mask]
        best = (T[:, :, None] * T[:, None, :]).max(0)
        L = (2 * A - np.maximum(A, best)).sum() / 2
        rows.append(dict(graph=label, family=fam, N=N, n=n, tau=tau, frac=frac, m=m, set=name, a=a, info=info, unseen=float((~seen).mean()), patterns=len(ku), L=L, C=A.sum() / 2,
                         deg=float(deg[M].mean()), internal=int(np.isin(e, M).all(1).sum()), corr_I_deg=pd.Series(Iv).rank().corr(pd.Series(deg).rank()), corr_F_I=float(np.corrcoef(n * Iv, F)[0, 1])))
    pd.DataFrame(rows).to_csv(path, index=False)
    return label


def main(mode):
    if mode == 'smoke':
        out = OUT / 'smoke'; out.mkdir(exist_ok=True)
        for f in out.glob('*'): f.unlink()
        t = time.time()
        print(run(dict(fam='BA', N=30, k=0, Rp=64, R=64, Rt=32, label='smoke', seed=31337, out=out)), time.time() - t)
        print(pd.read_csv(out / 'smoke.csv')[['set', 'm', 'unseen', 'patterns']].to_string()); return
    out = OUT / 'results'; out.mkdir(exist_ok=True)
    for N, wk in zip(SIZES, [8, 8, 8, 4]):
        jobs = [dict(fam=f, N=N, k=k, Rp=4096, R=1 << 14, Rt=2048, label=f'{f}_{N}_{k}', seed=None, out=out) for f in FAMS for k in range(GRAPHS)]
        t = time.time()
        with ProcessPoolExecutor(wk) as p:
            for l in p.map(run, jobs): pass
        print(N, 'done', round(time.time() - t), 's', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
