"""Independent greedy-MI selection (SciPy, plug-in, separate seeds), compared with the engine's order.

Purpose: the jump of BA D_g between N=50 and N=100 at nu=1 depends on which nodes greedy-MI selects,
which formal_independent.py does not retest.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
RUN = Path(__file__).resolve().parents[1] / 'medium_scale_run' / 'results'
OUT = Path(__file__).resolve().parents[1] / 'formal_analysis' / 'independent_greedy.csv'
REPS, STEPS = 2048, 5


def draw(a):
    edges, n, tau, reps, seed = a
    rng = np.random.default_rng(seed)
    x, y = edges[:, 0], edges[:, 1]
    out = np.zeros((reps, n, n), bool)
    for r in range(reps):
        w = rng.exponential(size=len(edges))
        out[r] = dijkstra(csr_matrix((np.r_[w, w], (np.r_[x, y], np.r_[y, x])), shape=(n, n)), directed=True, limit=tau) <= tau
    return out


def mi(code, n, K):
    reps = code.shape[0]
    sc = (code + K * np.arange(n)[None, :]).ravel()
    joint = np.bincount(sc, minlength=K * n).reshape(n, K) / reps
    py = joint.sum(0) / n
    h = -(py[py > 0] * np.log(py[py > 0])).sum()
    hc = -(joint[joint > 0] * np.log(joint[joint > 0])).sum() / n
    return h - hc


def main():
    rows = []
    for label in sys.argv[1:]:
        meta = json.load(open(RUN / f'{label}.json')); d = pd.read_csv(RUN / f'{label}.csv.gz', dtype={'set': str})
        n, edges = meta['n'], np.array(meta['edge_list']); deg = np.bincount(edges.ravel(), minlength=n)
        tau = next(s['tau'] for s in meta['stages'] if s['kind'] == 'nu')
        with ProcessPoolExecutor(8) as p:
            X = np.concatenate(list(p.map(draw, [(edges, n, tau, REPS, 7700000 + i + 1000 * n + 10 * sum(map(ord, label))) for i in range(8)])))
        eng = d[(d.method == 'greedy_info') & (d.stage_kind == 'nu')].sort_values('m').iloc[-1].set.split()[:STEPS]
        eng = list(map(int, eng))
        code = np.zeros(X.shape[:2], np.int64); chosen = []
        for s in range(STEPS):
            K = 2 ** (s + 1)
            vals = [(-1. if v in chosen else mi(code * 2 + X[:, :, v], n, K)) for v in range(n)]
            v = int(np.argmax(vals)); chosen.append(v); code = code * 2 + X[:, :, v]
            rows.append(dict(graph=label, n=n, step=s + 1, tau=tau, independent=v, deg_ind=int(deg[v]), engine=eng[s], deg_eng=int(deg[eng[s]]), mi_ind_choice=vals[v], mi_ind_of_engine=vals[eng[s]] if eng[s] not in chosen[:-1] else np.nan))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == '__main__':
    main()
