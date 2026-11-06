"""Post-hoc: refined pair loss, competing predictors, fixed-condition correlations, nu=1 focus.

Two observed nodes v,w: reports {v},{w} are named v,w; report {v,w} goes to the likeliest source. With a third
source u taken as independent on its two paths, N A_pair = 2 - 2p_vw + max(p_vw, max_u p_uv p_uw), so
L_vw = 2 p_vw - max(p_vw, max_u p_uv p_uw) replaces p_vw. The independence is an approximation.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RUN, SN = ROOT / 'medium_scale_run' / 'results', ROOT / 'formal_analysis' / 'single_node'
rows = []
for f in sorted(glob.glob(f'{SN}/*.npz')):
    label = Path(f).stem; z = np.load(f); n = int(z['n']); e = z['edges']; deg = np.bincount(e.ravel(), minlength=n)
    d = pd.read_csv(RUN / f'{label}.csv.gz', dtype={'set': str})
    for i, (kind, tgt) in enumerate(zip(z['kinds'], z['targets'])):
        kind, tgt = str(kind), float(tgt)
        off = z['p'][i].astype(float).copy(); np.fill_diagonal(off, 0.)
        s = d[(d.stage_kind == kind) & (d.stage_target == tgt) & (d.method != 'greedy_plugin') & (d.method != 'greedy_jack')]
        for r in s.itertuples():
            M = np.array(list(map(int, r.set.split())))
            P = off[np.ix_(M, M)]
            third = off[:, M]                     # (n, m)
            mask = np.ones(n, bool); mask[M] = False
            T = third[mask]
            best = (T[:, :, None] * T[:, None, :]).max(0) if T.size else np.zeros_like(P)
            L = 2 * P - np.maximum(P, best)
            rows.append(dict(graph=label, family=label.split('_')[0], N=int(label.split('_')[1]), n=n, kind=kind, target=tgt, m=r.m, method=r.method, draw=r.draw, a=r.a_test, unseen=r.unseen,
                             C=P.sum() / 2, L=L.sum() / 2, internal=int(np.isin(e, M).all(1).sum()), degsum=deg[M].sum()))
X = pd.DataFrame(rows); X.to_csv(SN / 'overlap.csv.gz', index=False)
pd.set_option('display.width', 200, 'display.max_rows', 300)
X['Na'] = X.n * X.a; X['res'] = X.Na - (X.m + 1 - X.L); X = X[X.unseen <= .05]
keys = ['graph', 'target', 'm']
def dm(z, c): return z[c] - z.groupby(keys)[c].transform('mean')
out = []
for (kind, N), z in X.groupby(['kind', 'N']):
    row = dict(kind=kind, N=N, rows=len(z), res_L=z.res.mean(), rms_L=np.sqrt((z.res ** 2).mean()))
    y = dm(z, 'Na')
    for c in ['L', 'C', 'internal', 'degsum']:
        v = dm(z, c); row['r_' + c] = np.corrcoef(y, v)[0, 1]; row['sl_' + c] = (y * v).sum() / (v * v).sum()
    out.append(row)
print(pd.DataFrame(out).round(3).to_string(index=False))
A, I = X[X.method == 'greedy_acc'].set_index(['graph', 'kind', 'target', 'm']), X[X.method == 'greedy_info'].set_index(['graph', 'kind', 'target', 'm'])
j = A.join(I, lsuffix='_a', rsuffix='_i', how='inner').reset_index()
j['ND'] = j.Na_a - j.Na_i
for c in ['L', 'C', 'internal', 'degsum']: j['d' + c] = j[c + '_i'] - j[c + '_a']
print('\nFixed condition: across graphs at same N, family, stage, m; mean corr over conditions')
rs = []
for k, z in j.groupby(['kind', 'N_a', 'family_a', 'm']):
    if len(z) < 8 or z.ND.std() == 0: continue
    r = dict(zip(['kind', 'N', 'family', 'm'], k)) | {('r_' + c): z.ND.corr(z['d' + c]) for c in ['L', 'C', 'internal', 'degsum']} | dict(ND=z.ND.mean(), dL=z.dL.mean(), dC=z.dC.mean())
    rs.append(r)
R = pd.DataFrame(rs); R.to_csv(SN / 'fixed_condition.csv', index=False)
print(R.groupby(['kind', 'family'])[['r_L', 'r_C', 'r_internal', 'r_degsum', 'ND', 'dL', 'dC']].mean().round(3).to_string())
print('\nnu=1 BA ER: ND vs dL per N,m'); print(R[(R.kind == 'nu') & R.family.isin(['BA', 'ER']) & (R.N >= 100)][['N', 'family', 'm', 'ND', 'dL', 'dC', 'r_L', 'r_internal']].round(3).to_string(index=False))
Z = j[(j.kind == 'nu')]; print('\nnu=1 overall pooled ND on dL: slope', round((Z.ND * Z.dL).sum() / (Z.dL ** 2).sum(), 3), 'r', round(Z.ND.corr(Z.dL), 3), 'rows', len(Z))
Z = j[(j.kind == 'rho')]; print('rho=.1 overall pooled ND on dL: slope', round((Z.ND * Z.dL).sum() / (Z.dL ** 2).sum(), 3), 'r', round(Z.ND.corr(Z.dL), 3), 'rows', len(Z))
