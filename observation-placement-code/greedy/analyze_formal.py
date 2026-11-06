"""Frozen analysis (EXPERIMENT_DESIGN Sec. 7, 14, 15) of the formal middle-plan run.

Reads only medium_scale_run/results. Writes tables to formal_analysis/.
Nothing here edits the paper.
"""
from pathlib import Path
import glob, json, sys
import numpy as np, pandas as pd, networkx as nx

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'medium_scale_run' / 'results'
OUT = HERE.parent / 'formal_analysis'
OUT.mkdir(exist_ok=True)
BATCH = [f'b{i}' for i in range(8)]
KEY = ['graph', 'family', 'requested_n', 'n', 'stage_kind', 'stage_target', 'm']
BOOT = 20000
EXP = {20: [2, 4, 6, 10], 50: [3, 5, 10, 15, 25], 100: [5, 10, 20, 30, 50], 200: [10, 20, 40]}
THR = {'<=20%': (.53, .27, .13), '30-50%': (2.63, 1.32, .66)}


def load():
    d = pd.concat(pd.read_csv(p, dtype={'set': str}) for p in sorted(glob.glob(f'{RUN}/*.csv.gz')))
    meta = {json.load(open(p))['graph']: json.load(open(p)) for p in sorted(glob.glob(f'{RUN}/*.json'))}
    return d, meta


def integrity(d, meta):
    rep = {}
    rep['graphs'] = d.graph.nunique()
    rep['rows'] = len(d)
    rep['graphs_per_n_family'] = sorted(set(d.groupby(['requested_n', 'family']).graph.nunique()))
    rep['engines'] = sorted({m['engine'] for m in meta.values()})
    rep['seeds_unique'] = len({m['seed'] for m in meta.values()}) == len(meta)
    rep['min_seed'] = min(m['seed'] for m in meta.values())
    rep['max_seed'] = max(m['seed'] for m in meta.values())
    num = d.select_dtypes('number')
    rep['nan_a_test'] = int(d.a_test.isna().sum())
    rep['a_test_out_of_range'] = int(((d.a_test < 0) | (d.a_test > 1)).sum())
    rep['a_test_above_plugin'] = int((d.a_test > d.a_plugin + 2e-3).sum())
    miss = []
    for g, z in d.groupby('graph'):
        rn = int(z.requested_n.iloc[0])
        for st, s in z.groupby(['stage_kind', 'stage_target']):
            for meth in ['greedy_info', 'greedy_acc', 'degree', 'coverage', 'kcenter']:
                ms = sorted(s[s.method == meth].m)
                if ms != EXP[rn]:
                    miss.append((g, st, meth, ms))
            r = s[s.method == 'random'].groupby('m').draw.nunique()
            if sorted(r.index) != EXP[rn] or (r != 32).any():
                miss.append((g, st, 'random', r.to_dict()))
    rep['missing_cells'] = miss[:10]
    rep['n_missing'] = len(miss)
    rep['stage_counts'] = {str(k): int(v) for k, v in d[d.method == 'greedy_info'].groupby('requested_n').apply(lambda z: z.groupby('graph').apply(lambda y: y[['stage_kind', 'stage_target']].drop_duplicates().shape[0], include_groups=False).value_counts().to_dict(), include_groups=False).items()} if False else None
    st = d[d.method == 'greedy_info'].drop_duplicates(['graph', 'stage_kind', 'stage_target'])
    rep['stages'] = st.groupby(['requested_n', 'stage_kind', 'stage_target']).graph.nunique().reset_index().to_dict('records')
    rep['merged'] = {g: [s for s in m['stages'] if s['merged']] for g, m in meta.items() if any(s['merged'] for s in m['stages'])}
    rep['merged'] = {int(k): int(sum(1 for g in rep['merged'] if meta[g]['requested_n'] == k)) for k in EXP}
    rep['n_actual'] = d.groupby(['requested_n', 'family']).n.agg(['min', 'max']).reset_index().to_dict('records')
    return rep


def pairs(d, a='greedy_acc', b='greedy_info'):
    x, y = d[d.method == a].set_index(KEY), d[d.method == b].set_index(KEY)
    w = pd.DataFrame(index=x.index)
    y = y.loc[x.index]
    w['A_acc'], w['A_info'] = x.a_test, y.a_test
    w['Dg'] = x.a_test - y.a_test
    w['rel'] = 100 * w.Dg / x.a_test
    w['same_set'] = [set(s.split()) == set(t.split()) for s, t in zip(x.set, y.set)]
    w['unseen'] = np.maximum(x.unseen, y.unseen)
    w['width'] = np.maximum(x.a_plugin - x.a_test, y.a_plugin - y.a_test)
    # Directional bracket: held-out A of the accuracy set minus plug-in A of the information set.
    w['Dg_cons'] = x.a_test - y.a_plugin
    w['mc_se'] = (x[BATCH].values - y[BATCH].values).std(axis=1, ddof=1) / np.sqrt(8)
    w['info_gap'] = y['info'] - x['info']
    w['patterns_acc'], w['patterns_info'] = x.patterns, y.patterns
    for c in ['top1', 'top3', 'top5', 'mean_rank', 'err_dist', 'eff_sources']:
        w[c + '_acc'], w[c + '_info'] = x[c], y[c]
    w = w.reset_index()
    w['frac'] = w.m / w.requested_n
    w['group'] = np.where(w.frac <= .2 + 1e-9, '<=20%', '30-50%')
    w['early'] = (w.stage_kind == 'nu') | (w.stage_target <= .2)
    return w


def cell_table(w, by=('requested_n', 'family', 'stage_kind', 'stage_target', 'm')):
    by = list(by)
    g = w.groupby(by)
    o = g.agg(graphs=('graph', 'nunique'), A_acc=('A_acc', 'mean'), A_info=('A_info', 'mean'), Dg=('Dg', 'mean'), rel=('rel', 'mean'),
              Dg_sd=('Dg', 'std'), rel_sd=('rel', 'std'), neg=('Dg', lambda v: float((v < 0).mean())), same=('same_set', 'mean'),
              unseen_max=('unseen', 'max'), width_mean=('width', 'mean'), Dg_cons=('Dg_cons', 'mean'), mc_se=('mc_se', 'mean'),
              info_gap=('info_gap', 'mean')).reset_index()
    o['frac_pass'] = g.apply(lambda z: float(((z.unseen <= .05) & (z.width < z.Dg.abs())).mean()), include_groups=False).values
    o['unseen_ok'] = o.unseen_max <= .05
    o['estimable'] = o.unseen_ok & (o.width_mean < o.Dg.abs())
    return o


def strat_boot(gm, value='v', fam='family', rng=None, B=BOOT):
    """Family-stratified graph bootstrap of the equal-weight mean of family means."""
    rng = rng or np.random.default_rng(20261104)
    means = np.zeros(B)
    fams = sorted(gm[fam].unique())
    for f in fams:
        v = gm.loc[gm[fam] == f, value].values
        means += v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    means /= len(fams)
    point = gm.groupby(fam)[value].mean().mean()
    return point, *np.percentile(means, [2.5, 97.5])


def classify(lo, hi, grp):
    b, ex, ne = THR[grp]
    return 'exists' if lo > ex else 'negligible' if hi < ne else 'uncertain'


def confirm(w, cells):
    e = w[w.early].merge(cells[['requested_n', 'family', 'stage_kind', 'stage_target', 'm', 'estimable', 'unseen_ok']], on=['requested_n', 'family', 'stage_kind', 'stage_target', 'm'])
    rows = []
    for (n, grp), z in e.groupby(['requested_n', 'group']):
        for name, sel in [('all_cells', z), ('unseen_ok', z[z.unseen_ok]), ('frozen_estimable', z[z.estimable])]:
            if sel.empty:
                rows.append(dict(N=n, group=grp, variant=name, cells=0)); continue
            gm = sel.groupby(['family', 'graph']).rel.mean().rename('v').reset_index()
            p, lo, hi = strat_boot(gm)
            gd = sel.groupby(['family', 'graph']).Dg.mean().rename('v').reset_index()
            pd_, lod, hid = strat_boot(gd)
            cc = sel[['family', 'stage_kind', 'stage_target', 'm']].drop_duplicates()
            rows.append(dict(N=n, group=grp, variant=name, cells=len(cc), families=sel.family.nunique(), rel=p, lo=lo, hi=hi, verdict=classify(lo, hi, grp),
                             Dg_pp=100 * pd_, Dg_lo=100 * lod, Dg_hi=100 * hid))
    return pd.DataFrame(rows)


def by_family(w):
    rows = []
    e = w[w.early]
    for (n, grp, fam), z in e.groupby(['requested_n', 'group', 'family']):
        gm = z.groupby('graph').rel.mean().values
        rng = np.random.default_rng(int(n) * 7 + len(fam))
        bs = gm[rng.integers(0, len(gm), (BOOT, len(gm)))].mean(axis=1)
        rows.append(dict(N=n, group=grp, family=fam, rel=gm.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), neg_graphs=int((gm < 0).sum()), same=z.same_set.mean()))
    return pd.DataFrame(rows)


def baselines(d):
    r = d[d.method == 'random'].groupby(KEY).a_test.mean().rename('A_rand')
    t = d[d.method != 'random'].pivot_table(index=KEY, columns='method', values='a_test')
    t = t.join(r).reset_index()
    t['Rg'] = (t.greedy_info - t.A_rand) / (t.greedy_acc - t.A_rand)
    t['info_below_rand'] = t.greedy_info < t.A_rand
    return t


def set_structure(d, meta):
    rows = []
    gs = {g: nx.Graph([tuple(e) for e in m['edge_list']]) for g, m in meta.items()}
    for r in d[d.method.isin(['greedy_info', 'greedy_acc', 'degree'])].itertuples():
        G = gs[r.graph]
        s = [int(v) for v in r.set.split()]
        deg = np.mean([G.degree(v) for v in s])
        S = set(s)
        e = G.subgraph(S).number_of_edges()
        c = sum(1 for u in S for v in G[u] if v not in S)
        rows.append(dict(graph=r.graph, family=r.family, requested_n=r.requested_n, stage_kind=r.stage_kind, stage_target=r.stage_target, m=r.m, method=r.method,
                         mean_deg=deg, graph_deg=2 * G.number_of_edges() / G.number_of_nodes(), internal=e, boundary=c))
    return pd.DataFrame(rows)


def main():
    pd.set_option('display.width', 240, 'display.max_rows', 500, 'display.max_columns', 60)
    d, meta = load()
    rep = integrity(d, meta); print(json.dumps(rep, indent=1, default=str))
    w = pairs(d)
    cells = cell_table(w)
    cells.to_csv(OUT / 'cells_family.csv', index=False)
    conf = confirm(w, cells); conf.to_csv(OUT / 'confirmatory.csv', index=False)
    print('\nCONFIRMATORY'); print(conf.round(3).to_string(index=False))
    fam = by_family(w); fam.to_csv(OUT / 'early_by_family.csv', index=False)
    print('\nEARLY BY FAMILY (all cells)'); print(fam.round(3).to_string(index=False))
    print('\nCELLS'); print(cells.drop(columns=['Dg_sd', 'rel_sd']).round(4).to_string(index=False))
    # Pooled over families, per (N, stage, m).
    rows = []
    for k, z in w.groupby(['requested_n', 'stage_kind', 'stage_target', 'm']):
        gm = z.groupby(['family', 'graph']).rel.mean().rename('v').reset_index()
        p, lo, hi = strat_boot(gm, B=5000)
        rows.append(dict(zip(['N', 'kind', 'stage', 'm'], k)) | dict(rel=p, lo=lo, hi=hi, Dg_pp=100 * z.Dg.mean(), cons_pp=100 * z.Dg_cons.mean(), unseen_max=z.unseen.max(), width=z.width.mean()))
    pooled = pd.DataFrame(rows); pooled.to_csv(OUT / 'pooled_cells.csv', index=False)
    print('\nPOOLED'); print(pooled.round(3).to_string(index=False))
    b = baselines(d); b.to_csv(OUT / 'baselines.csv', index=False)
    bb = b.groupby(['requested_n', 'family', 'stage_kind', 'stage_target', 'm'])[['greedy_acc', 'greedy_info', 'A_rand', 'degree', 'coverage', 'kcenter', 'info_below_rand']].mean().reset_index()
    bb.to_csv(OUT / 'baselines_family.csv', index=False)
    print('\nBASELINES'); print(bb[bb.stage_kind.eq('nu') | bb.stage_target.le(.2)].round(4).to_string(index=False))
    # Secondary metrics.
    sec = w.copy()
    for c in ['top1', 'top3', 'top5']:
        sec['d_' + c] = 100 * (sec[c + '_acc'] - sec[c + '_info'])
    sec['d_rank'] = sec.mean_rank_info - sec.mean_rank_acc
    sec['d_err'] = sec.err_dist_info - sec.err_dist_acc
    s2 = sec[sec.early].groupby(['requested_n', 'family', 'group'])[['d_top1', 'd_top3', 'd_top5', 'd_rank', 'd_err']].mean().reset_index()
    s2.to_csv(OUT / 'secondary_early.csv', index=False)
    print('\nSECONDARY positive = accuracy set better'); print(s2.round(3).to_string(index=False))
    # Plug-in versus jackknife at N=50.
    rows = []
    for meth in ['greedy_plugin', 'greedy_jack']:
        t = pairs(d[d.requested_n == 50], 'greedy_acc', meth)
        s = pairs(d[d.requested_n == 50], 'greedy_info', meth)
        rows.append(t[t.early].groupby(['family', 'group']).agg(rel=('rel', 'mean')).rename(columns={'rel': meth}))
        rows.append(s[s.early].groupby(['family', 'group']).agg(x=('same_set', 'mean')).rename(columns={'x': meth + '_sameMM'}))
    sens = pd.concat(rows, axis=1).join(w[(w.requested_n == 50) & w.early].groupby(['family', 'group']).rel.mean().rename('MM'))
    sens.to_csv(OUT / 'sensitivity_n50.csv')
    print('\nSENS N50'); print(sens.round(3).to_string())
    st = set_structure(d, meta); st.to_csv(OUT / 'set_structure.csv', index=False)
    s3 = st[(st.stage_kind == 'nu') | (st.stage_target <= .2)].groupby(['requested_n', 'family', 'method'])[['mean_deg', 'graph_deg', 'internal', 'boundary']].mean().round(2).reset_index()
    print('\nSTRUCTURE'); print(s3.to_string(index=False))
    # Same-graph trend, descriptive, fixed fraction and stage.
    fix = w[w.early & w.frac.round(3).isin([.1, .2])].groupby(['requested_n', 'frac', 'stage_kind', 'stage_target', 'family']).rel.mean().unstack('family').round(2)
    print('\nTREND'); print(fix.to_string())
    (OUT / 'integrity.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
