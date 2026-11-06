"""Fig. 5 and Table IV of the medium-scale insert. Reads saved results only."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
import analyze_formal as af, numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parents[3] / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.2, 'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7})
FAM = [('BA', '#2166ac', 'o'), ('ER', '#e08214', 's'), ('WS', '#7b3294', '^'), ('REG', '#1b7f5c', 'D')]
d, meta = af.load(); w = af.pairs(d); c = af.cell_table(w)
K = ['requested_n', 'family', 'stage_kind', 'stage_target', 'm']
st = af.set_structure(d, meta)


def series(fam, kind, tgt):
    rows = []
    for n in [20, 50, 100, 200]:
        m = n // 10
        z = w[(w.requested_n == n) & (w.family == fam) & (w.stage_kind == kind) & (w.stage_target == tgt) & (w.m == m)]
        if z.graph.nunique() < 10: continue
        v = z.groupby('graph').rel.mean().values
        rng = np.random.default_rng(n + len(fam))
        b = v[rng.integers(0, len(v), (20000, len(v)))].mean(axis=1)
        e = bool(c[(c.requested_n == n) & (c.family == fam) & (c.stage_kind == kind) & (c.stage_target == tgt) & (c.m == m)].estimable.iloc[0])
        rows.append((n, v.mean(), *np.percentile(b, [2.5, 97.5]), e))
    return rows


fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), constrained_layout=True)
for ax, (kind, tgt, title) in zip(axs[:2], [('nu', 1., r'(a) $\nu=1$, $m/N=10\%$'), ('rho', .1, r'(b) $\rho=0.1$, $m/N=10\%$')]):
    for fam, col, mk in FAM:
        r = series(fam, kind, tgt)
        x = np.array([q[0] for q in r], float); y = np.array([q[1] for q in r]); lo = np.array([q[2] for q in r]); hi = np.array([q[3] for q in r])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=col, lw=1.2, elinewidth=.8, capsize=2, zorder=2)
        for q in r:
            ax.plot(q[0], q[1], mk, ms=4.5, mfc=col if q[4] else 'white', mec=col, mew=1, zorder=3)
    ax.axhline(0, color='.6', lw=.6, zorder=1); ax.set_xscale('log'); ax.set_xticks([20, 50, 100, 200]); ax.set_xticklabels(['20', '50', '100', '200']); ax.minorticks_off(); ax.set_xlim(17, 235)
    ax.set_xlabel('$N$'); ax.set_ylabel(r'$D_g/A_1(G_A)$ (%)'); ax.set_title(title, loc='left', fontweight='bold', pad=4)
ax = axs[2]
s = st[(st.stage_kind == 'rho') & (st.stage_target == .1) & (st.m == st.requested_n // 10) & (st.requested_n >= 50)]
for fam, col, mk in FAM:
    t = s[s.family == fam].groupby(['requested_n', 'method'])[['mean_deg', 'graph_deg']].mean().reset_index()
    for meth, ls, fc in [('greedy_info', '-', col), ('greedy_acc', '--', 'white')]:
        q = t[t.method == meth]
        ax.plot(q.requested_n, q.mean_deg / q.graph_deg, ls, color=col, marker=mk, ms=4.5, mfc=fc, mec=col, mew=1)
ax.axhline(1, color='.6', lw=.6, zorder=1); ax.set_xscale('log'); ax.set_xticks([50, 100, 200]); ax.set_xticklabels(['50', '100', '200']); ax.minorticks_off()
ax.set_xlabel('$N$'); ax.set_ylabel(r'$\bar k_{\rm sel}/\bar k$'); ax.set_title(r'(c) $\rho=0.1$, $m/N=10\%$', loc='left', fontweight='bold', pad=4)
h = [plt.Line2D([], [], color=col, marker=mk, ms=4.5, lw=1.2, label=f) for f, col, mk in FAM]
axs[0].legend(handles=h, frameon=False, loc='upper left', handlelength=1.6, labelspacing=.25)
h2 = [plt.Line2D([], [], color='.25', ls='-', marker='o', ms=4.5, label='$G_I$'), plt.Line2D([], [], color='.25', ls='--', marker='o', mfc='white', ms=4.5, label='$G_A$')]
ax.legend(handles=h2, frameon=False, loc='upper left', handlelength=2.2, labelspacing=.25)
fig.savefig(OUT / 'fig5.pdf', metadata={'CreationDate': None, 'ModDate': None}); fig.savefig(OUT / 'fig5.png', dpi=600)
for a in fig.findobj(lambda a: a.get_alpha() is not None and a.get_alpha() < 1): a.set_rasterized(True)
fig.savefig(OUT / 'fig5.eps', dpi=300)

# Table: estimable early cells.
e = w[w.early].merge(c[K + ['estimable']], on=K)
rows = []
conf = pd.read_csv(af.OUT / 'confirmatory.csv')
for (n, g), z in e.groupby(['requested_n', 'group']):
    r = conf[(conf.N == n) & (conf.group == g)].set_index('variant')
    y = z[z.estimable]
    fam = y.groupby(['family', 'graph']).rel.mean().groupby('family').mean()
    rows.append(dict(N=int(n), group=g, cells=int(r.loc['frozen_estimable', 'cells']), total=int(r.loc['all_cells', 'cells']),
                     rel=r.loc['frozen_estimable', 'rel'], lo=r.loc['frozen_estimable', 'lo'], hi=r.loc['frozen_estimable', 'hi'], verdict=r.loc['frozen_estimable', 'verdict'],
                     all_rel=r.loc['all_cells', 'rel'], all_lo=r.loc['all_cells', 'lo'], all_hi=r.loc['all_cells', 'hi'], all_verdict=r.loc['all_cells', 'verdict'], **fam.to_dict()))
T = pd.DataFrame(rows).sort_values(['group', 'N'], ascending=[False, True]); T.to_csv(af.OUT / 'table_iv.csv', index=False)
pd.set_option('display.width', 200); print(T.round(2).to_string(index=False))
# Numbers quoted in the text.
b = af.baselines(d); q = b[(b.requested_n == 200) & (b.family == 'BA') & (b.stage_kind == 'rho') & (b.m == 10)]
print(q[['greedy_acc', 'greedy_info', 'A_rand', 'kcenter', 'degree', 'coverage']].mean().round(4).to_dict(), int(q.info_below_rand.sum()))
print(st[(st.requested_n == 200) & (st.family == 'BA') & (st.stage_kind == 'rho') & (st.m == 10)].groupby('method')[['mean_deg', 'graph_deg', 'internal']].mean().round(2))
print('tau range', {n: (round(min(s['tau'] for m in meta.values() if m['requested_n'] == n for s in m['stages'] if s['kind'] == 'nu'), 3), round(max(s['tau'] for m in meta.values() if m['requested_n'] == n for s in m['stages'] if s['kind'] == 'nu'), 3)) for n in [20, 50, 100, 200]})
