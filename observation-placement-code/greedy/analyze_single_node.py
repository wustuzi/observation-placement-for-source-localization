"""Post-hoc mechanism test. Hypotheses written before looking at output:

H1  N I_v ~ (1+S)[log(N/(1+S))+1] - sum_{u!=v} h(p_uv), S=sum_{u!=v} p_uv. Information rewards reach once N grows:
    adding a source with first-passage probability p helps iff p > e*q (first order). So the best single node
    has an interior reach that grows with N, explaining the degree shift of G_I.
H2  N A_1(M) ~ m+1 - C_M, C_M = sum_{v<w in M} p_vw: observed pairs whose balls overlap produce shared reports,
    and only one of the two sources can be named. C_M reduces to tau*e_M as tau->0.
H3  N*D_g ~ C(G_I)-C(G_A).
H4  N I_M ~ sum N I_v - log2 * C_M at low overlap: same overlap costs information only log 2 per unit.
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RUN, SN = ROOT / 'medium_scale_run' / 'results', ROOT / 'formal_analysis' / 'single_node'


def h(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def spearman(a, b):
    r = pd.Series(a).rank().corr(pd.Series(b).rank())
    return r


def main():
    pd.set_option('display.width', 220, 'display.max_rows', 300, 'display.max_columns', 40)
    first, rows = [], []
    for f in sorted(glob.glob(f'{SN}/*.npz')):
        label = Path(f).stem
        z = np.load(f); P = z['p'].astype(float); n = int(z['n']); e = z['edges']
        deg = np.bincount(e.ravel(), minlength=n)
        d = pd.read_csv(RUN / f'{label}.csv.gz', dtype={'set': str})
        for i, (kind, tgt, tau) in enumerate(zip(z['kinds'], z['targets'], z['taus'])):
            kind, tgt = str(kind), float(tgt)
            p = P[i].copy(); np.fill_diagonal(p, 1.)
            off = p.copy(); np.fill_diagonal(off, 0.)
            S = off.sum(1); q = p.mean(1)
            Iv = h(q) - h(off).sum(1) / n   # h(1)=0 for the diagonal
            F = (1 + S) * (np.log(n / (1 + S)) + 1) - h(off).sum(1)
            s = d[(d.stage_kind == kind) & (d.stage_target == tgt)]
            if s.empty: continue
            gi = s[s.method == 'greedy_info'].sort_values('m').iloc[-1].set.split()
            ga = s[s.method == 'greedy_acc'].sort_values('m').iloc[-1].set.split()
            top = int(np.argmax(Iv)); topF = int(np.argmax(F))
            first.append(dict(graph=label, family=label.split('_')[0], N=int(label.split('_')[1]), n=n, kind=kind, target=tgt, tau=tau, first_info=int(gi[0]), argmax_I=top, argmax_F=topF,
                              deg_first_info=deg[int(gi[0])], deg_argmax_I=deg[top], deg_argmax_F=deg[topF], deg_first_acc=deg[int(ga[0])], deg_max=deg.max(), deg_mean=deg.mean(),
                              corr_I_deg=spearman(Iv, deg), corr_F_I=np.corrcoef(n * Iv, F)[0, 1], maxerr_F=np.abs(n * Iv - F).max(),
                              S_top=S[top], rank_first_in_I=int((Iv > Iv[int(gi[0])]).sum()) + 1))
            sub = s[s.method.isin(['greedy_info', 'greedy_acc', 'degree', 'coverage', 'kcenter', 'random'])]
            for r in sub.itertuples():
                M = np.array(list(map(int, r.set.split())))
                C = off[np.ix_(M, M)].sum() / 2
                rows.append(dict(graph=label, family=label.split('_')[0], N=int(label.split('_')[1]), n=n, kind=kind, target=tgt, m=r.m, method=r.method, draw=r.draw, a=r.a_test, unseen=r.unseen, info=r.info,
                                 C=C, sumI=n * Iv[M].sum(), sumS=S[M].sum(), internal=int(np.isin(e, M).all(1).sum())))
    F = pd.DataFrame(first); F.to_csv(SN / 'first_node.csv', index=False)
    X = pd.DataFrame(rows); X.to_csv(SN / 'sets.csv.gz', index=False)
    print('H1 first node'); print(F.groupby(['family', 'N', 'kind']).agg(deg_first_info=('deg_first_info', 'mean'), deg_argmax_I=('deg_argmax_I', 'mean'), deg_argmax_F=('deg_argmax_F', 'mean'), deg_first_acc=('deg_first_acc', 'mean'), deg_max=('deg_max', 'mean'),
          corr_I_deg=('corr_I_deg', 'mean'), match=('first_info', lambda v: float((v == F.loc[v.index, 'argmax_I']).mean())), rank=('rank_first_in_I', 'mean'), corr_F_I=('corr_F_I', 'min'), err=('maxerr_F', 'max')).round(3).to_string())
    X['Na'] = X.n * X.a; X['pred'] = X.m + 1 - X.C; X['res'] = X.Na - X.pred
    G = X[X.unseen <= .05]
    print('\nH2 N*A vs m+1-C, unseen<=5%')
    def stats(z):
        w = z.Na - z.groupby(['graph', 'target', 'm']).Na.transform('mean'); v = z.pred - z.groupby(['graph', 'target', 'm']).pred.transform('mean')
        slope = (w * v).sum() / (v * v).sum() if (v * v).sum() > 0 else np.nan
        return pd.Series(dict(rows=len(z), mean_res=z.res.mean(), rms_res=np.sqrt((z.res ** 2).mean()), within_r=np.corrcoef(w, v)[0, 1] if v.std() > 0 else np.nan, within_slope=slope))
    print(G.groupby(['kind', 'N', 'family']).apply(stats, include_groups=False).round(3).to_string())
    print(G[G.method != 'random'].groupby(['kind', 'N', 'method']).res.mean().unstack().round(3).to_string())
    a, b = G[G.method == 'greedy_acc'].set_index(['graph', 'kind', 'target', 'm']), G[G.method == 'greedy_info'].set_index(['graph', 'kind', 'target', 'm'])
    j = a.join(b, lsuffix='_a', rsuffix='_i', how='inner')
    j['ND'] = j.Na_a - j.Na_i; j['dC'] = j.C_i - j.C_a; j['dInt'] = j.internal_i - j.internal_a
    j = j.reset_index()
    j['frac'] = j.m / j.N_a
    print('\nH3 N*D_g vs dC')
    def s3(z):
        sl = (z.ND * z.dC).sum() / (z.dC ** 2).sum() if (z.dC ** 2).sum() > 0 else np.nan
        return pd.Series(dict(rows=len(z), ND=z.ND.mean(), dC=z.dC.mean(), dInt=z.dInt.mean(), r=z.ND.corr(z.dC), slope0=sl, resid=(z.ND - z.dC).mean()))
    print(j.groupby(['kind', 'N_a', 'family_a']).apply(s3, include_groups=False).round(3).to_string())
    print(j.groupby(['kind', 'N_a']).apply(s3, include_groups=False).round(3).to_string())
    print('\nH4 N I_M = sumI - c*C, by OLS within graph-stage-m on sets with few reports')
    Y = G.copy(); Y['NI'] = Y.n * Y['info']
    Y = Y[Y.m <= Y.N * .1]
    out = []
    for (kind, N), z in Y.groupby(['kind', 'N']):
        keys = ['graph', 'target', 'm']
        A = np.c_[z.sumI - z.groupby(keys).sumI.transform('mean'), z.C - z.groupby(keys).C.transform('mean')]
        y = z.NI - z.groupby(keys).NI.transform('mean')
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ c) ** 2).sum() / (y ** 2).sum()
        out.append(dict(kind=kind, N=N, rows=len(z), coef_sumI=c[0], coef_C=c[1], r2=r2))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    # Decompose the info preference: how much of G_I's info lead comes from reach vs overlap.
    j2 = j[j.frac <= .1 + 1e-9].copy(); j2['NIgap'] = j2.n_i * (j2.info_i - j2.info_a); j2['dSum'] = j2.sumI_i - j2.sumI_a
    print(j2.groupby(['kind', 'N_a', 'family_a'])[['NIgap', 'dSum', 'dC', 'ND']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
