"""First-passage probabilities p_uv(tau) on the formal graphs, by SciPy, with separate seeds.

Post-hoc analysis (not in the frozen plan). Only pairwise marginals are stored.
"""
import glob, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
ROOT = Path(__file__).resolve().parents[1]
RUN, OUT = ROOT / 'medium_scale_run' / 'results', ROOT / 'formal_analysis' / 'single_node'
REPS = 4096


def work(path):
    meta = json.load(open(path)); label = meta['graph']
    out = OUT / f'{label}.npz'
    if out.exists(): return label
    n, e = meta['n'], np.array(meta['edge_list'])
    stages = [s for s in meta['stages'] if (s['kind'] == 'nu') or (s['kind'] == 'rho' and s['target'] == .1)]
    rng = np.random.default_rng(9100000 + meta['seed'])
    taus = np.array([s['tau'] for s in stages]); tm = taus.max()
    counts = np.zeros((len(taus), n, n))
    for _ in range(REPS):
        w = rng.exponential(size=len(e))
        d = dijkstra(csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)), directed=True, limit=tm)
        for i, t in enumerate(taus): counts[i] += d <= t
    p = counts / REPS; p = (p + p.transpose(0, 2, 1)) / 2
    np.savez_compressed(out, p=p.astype(np.float32), taus=taus, kinds=[s['kind'] for s in stages], targets=[s['target'] for s in stages], edges=e, n=n)
    return label


if __name__ == '__main__':
    files = sorted(glob.glob(f'{RUN}/*.json'), key=lambda f: json.load(open(f))['n'])
    with ProcessPoolExecutor(int(sys.argv[1])) as pool:
        for i, l in enumerate(pool.map(work, files)):
            if i % 20 == 0: print(i, l, flush=True)
    print('done', flush=True)
