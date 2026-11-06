"""Score PREDICTIONS.md exactly as written."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
D = Path(__file__).resolve().parents[1] / 'formal_analysis' / 'prospective'
pd.set_option('display.width', 200, 'display.max_rows', 200)
X = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f'{D}/results/*.csv')))
X['Na'] = X.n * X.a; X['res'] = X.Na - (X.m + 1 - X.L)
print('graphs', X.graph.nunique(), 'cells', len(X), 'excluded unseen>5%', int((X.unseen > .05).sum()))
print(X[X.unseen > .05].groupby(['N', 'family', 'set', 'm']).size().to_string() if (X.unseen > .05).any() else 'none')
G = X[X.unseen <= .05]; out = {}
pr = {('BA', 30): (-.41, 'neg'), ('BA', 70): (.33, 'pos'), ('BA', 150): (.76, None), ('BA', 400): (.58, 'min'), ('ER', 30): (-.49, 'neg'), ('ER', 70): (.42, 'pos'), ('ER', 150): (.95, None), ('ER', 400): (.70, 'min')}
c = X.drop_duplicates('graph').groupby(['family', 'N']).corr_I_deg.mean()
print('\nP1'); rows = []
for k, (v, t) in pr.items():
    o = c[k]
    ok = (o >= v) if t == 'min' else (abs(o - v) <= .25 and (t is None or (o < 0 if t == 'neg' else o > 0)))
    rows.append(dict(family=k[0], N=k[1], predicted=v, type=t, observed=round(o, 3), passed=bool(ok)))
print(pd.DataFrame(rows).to_string(index=False)); out['P1'] = all(r['passed'] for r in rows)
m = X.drop_duplicates('graph').groupby('N').corr_F_I.min(); print('\nP2', m.round(4).to_dict())
out['P2'] = bool((m[[70, 150, 400]] >= .95).all() and m[30] >= .90)
a, b = G[G.set == 'T_L'].set_index(['graph', 'frac']), G[G.set == 'T_I'].set_index(['graph', 'frac'])
j = a.join(b, lsuffix='_L', rsuffix='_I', how='inner').reset_index()
j['ND'] = j.Na_L - j.Na_I; j['dL'] = j.L_I - j.L_L; j['dC'] = j.C_I - j.C_L; j['dI'] = j.n_I * (j.info_I - j.info_L)
q = j[(j.N_L >= 70) & (j.dC >= .2)]
p3a, p3b = (q.ND > 0).mean(), (q.dI > 0).mean(); print('\nP3 cases', len(q), 'A', round(p3a, 3), 'I', round(p3b, 3)); out['P3'] = bool(p3a >= .9 and p3b >= .9)
print(q.groupby(['family_L', 'N_L']).agg(cases=('ND', 'size'), A=('ND', lambda v: (v > 0).mean()), I=('dI', lambda v: (v > 0).mean())).round(3).to_string())
def sl(z): return (z.ND * z.dL).sum() / (z.dL ** 2).sum()
r, s = j.ND.corr(j.dL), sl(j); se, sb = sl(j[j.family_L == 'ER']), sl(j[j.family_L == 'BA'])
print('\nP4 rows', len(j), 'r', round(r, 3), 'slope', round(s, 3), 'ER', round(se, 3), 'BA', round(sb, 3)); out['P4'] = bool(r >= .9 and .45 <= s <= .75 and se > sb)
print(j.groupby(['family_L', 'N_L', 'frac']).agg(ND=('ND', 'mean'), dL=('dL', 'mean'), dI=('dI', 'mean')).round(2).to_string())
lv = G.groupby(['N', 'family']).res.mean(); print('\nP5'); print(lv.round(3).to_string()); out['P5'] = bool(lv.between(-1, 1.5).all())
ba = X[(X.family == 'BA') & (X.N >= 150) & (X.frac == .1) & (X.unseen <= .05)].pivot(index='graph', columns='set', values='Na')
f = (ba.T_I < ba.R).mean(); print('\nP6 graphs', len(ba), 'fraction', f, ba.mean().round(2).to_dict()); out['P6'] = bool(f >= .8)
print('\nDegrees'); print(G.groupby(['family', 'N', 'set'])[['deg', 'internal', 'L', 'Na']].mean().round(2).to_string())
print('\nRESULT', out); (D / 'score.json').write_text(json.dumps(out, indent=1) + '\n')
