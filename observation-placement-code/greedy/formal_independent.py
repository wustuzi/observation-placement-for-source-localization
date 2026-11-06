"""Formal-run check: recompute formal values with code that shares nothing with the C++ engine.

Reads the selected sets from a pilot run, draws SI first-passage samples with SciPy,
and rebuilds the MAP rule, held-out accuracy, and plug-in information from scratch.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'medium_scale_run' / 'results'
R = 1 << 15
CASES = ['BA_200_3', 'ER_200_3', 'REG_200_3', 'WS_200_3', 'BA_100_3', 'ER_100_3']


def draw(args):
    edges, n, tau, nodes, reps, seed = args
    rng = np.random.default_rng(seed)
    out = []
    a, b = edges[:, 0], edges[:, 1]
    for _ in range(reps):
        w = rng.exponential(size=len(edges))
        g = csr_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
        out.append(dijkstra(g, directed=True, limit=tau)[:, nodes] <= tau)
    return np.packbits(np.stack(out), axis=2)


def key_counts(bits, n):
    """Return {report: per-source counts} for bits shaped (reps, n, bytes)."""
    table = {}
    reps = bits.shape[0]
    for u in range(n):
        keys, counts = np.unique(bits[:, u], axis=0, return_counts=True)
        for k, c in zip(keys, counts):
            table.setdefault(k.tobytes(), {})[u] = int(c)
    return table, reps


def evaluate(train, test, n):
    rule = {y: max(c.items(), key=lambda kv: kv[1]) for y, c in train[0].items()}
    hit, unseen = 0, 0
    reps = test[1]
    for y, c in test[0].items():
        if y in rule:
            hit += c.get(rule[y][0], 0)
        else:
            hit += sum(c.values()) / n
            unseen += sum(c.values())
    info = 0.
    for y, c in train[0].items():
        py = sum(c.values()) / (n * reps)
        info += sum(v / (n * reps) * np.log2(v / reps / py) for v in c.values())
    return hit / (n * reps), unseen / (n * reps), info, len(train[0])


def main(workers):
    rows = []
    for ci, label in enumerate(CASES):
        d = pd.read_csv(RUN / f'{label}.csv.gz', dtype={'set': str})
        meta = json.load(open(RUN / f'{label}.json'))
        n, edges = meta['n'], np.array(meta['edge_list'])
        rn = meta['requested_n']
        keep = ((d.stage_kind == 'rho') & (d.stage_target == .1) & d.m.isin([rn // 20, rn // 10])) | ((d.stage_kind == 'nu') & (d.m == rn // 10))
        d = d[keep & d.method.isin(['greedy_acc', 'greedy_info'])]
        for j, (_, r) in enumerate(d.iterrows()):
            nodes = np.array(list(map(int, r.set.split())))
            chunk = R // workers
            with ProcessPoolExecutor(workers) as pool:
                parts = list(pool.map(draw, [(edges, n, r.tau, nodes, chunk, 5000000 + 100000 * ci + 1000 * j + i) for i in range(2 * workers)]))
            train = key_counts(np.concatenate(parts[:workers]), n)
            test = key_counts(np.concatenate(parts[workers:]), n)
            a, unseen, info, patterns = evaluate(train, test, n)
            rows.append(dict(graph=label, stage=f"{r.stage_kind}{r.stage_target}", method=r.method, m=int(r.m), a_engine=r.a_test, a_scipy=a, unseen_engine=r.unseen, unseen_scipy=unseen,
                             info_engine_nats=r['info'], info_scipy_plugin_nats=info * np.log(2), patterns_engine=r.patterns, patterns_scipy=patterns))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(HERE.parent / 'formal_analysis' / 'independent_check.csv', index=False)


if __name__ == '__main__':
    main(int(sys.argv[1]))
