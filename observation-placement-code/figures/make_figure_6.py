"""Fig. 6 of the medium-scale insert. Reads saved post-hoc tables only."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
R = Path(__file__).resolve().parents[1]; SN = R / 'formal_analysis' / 'single_node'; OUT = R.parents[1] / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2, 'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7})
FAM = [('BA', '#2166ac', 'o'), ('ER', '#e08214', 's'), ('WS', '#7b3294', '^'), ('REG', '#1b7f5c', 'D')]
F = pd.read_csv(SN / 'first_node.csv')
fig, ax = plt.subplots(1, 3, figsize=(7, 2.45), constrained_layout=True)
for fam, c, mk in FAM[:3]:
    z = F[(F.family == fam) & (F.kind == 'nu')].groupby('N').corr_I_deg.mean()
    ax[0].plot(z.index, z.values, '-', color=c, marker=mk, ms=4.5, label=fam)
    y = F[(F.family == fam) & (F.kind == 'nu')].groupby('N')[['deg_first_info', 'deg_first_acc']].mean()
    ax[1].plot(y.index, y.deg_first_info, '-', color=c, marker=mk, ms=4.5)
    ax[1].plot(y.index, y.deg_first_acc, '--', color=c, marker=mk, ms=4.5, mfc='white', mec=c)
ax[0].axhline(0, color='.6', lw=.6); ax[0].set_ylabel(r'Spearman corr. of $I_v$ and degree'); ax[0].legend(frameon=False, loc='lower right', handlelength=1.6, labelspacing=.25)
ax[1].set_ylabel('degree of first selected node')
ax[1].legend(handles=[plt.Line2D([], [], color='.25', marker='o', ms=4.5, label='$G_I$'), plt.Line2D([], [], color='.25', ls='--', marker='o', mfc='white', ms=4.5, label='$G_A$')], frameon=False, loc='upper left', handlelength=2.2)
for a, t in zip(ax[:2], ['(a)', '(b)']):
    a.set_xscale('log'); a.set_xticks([20, 50, 100, 200]); a.set_xticklabels(['20', '50', '100', '200']); a.minorticks_off(); a.set_xlabel('$N$'); a.set_title(t + r' $\nu=1$', loc='left', fontweight='bold', pad=4)
X = pd.read_csv(SN / 'overlap.csv.gz'); X = X[(X.kind == 'nu') & (X.unseen <= .05)]; X['Na'] = X.n * X.a
A, I = X[X.method == 'greedy_acc'].set_index(['graph', 'm']), X[X.method == 'greedy_info'].set_index(['graph', 'm'])
j = A.join(I, lsuffix='_a', rsuffix='_i', how='inner'); j['ND'] = j.Na_a - j.Na_i; j['dL'] = j.L_i - j.L_a
for fam, c, mk in FAM:
    z = j[j.family_a == fam]; ax[2].plot(z.dL, z.ND, mk, ms=3, mfc='none', mec=c, mew=.6, label=fam)
lim = [min(j.dL.min(), j.ND.min()) - .2, max(j.dL.max(), j.ND.max()) + .2]; ax[2].plot(lim, lim, color='.6', lw=.7, zorder=1)
ax[2].set_xlabel(r'$L_{G_I}-L_{G_A}$'); ax[2].set_ylabel(r'$N D_g$'); ax[2].set_title(r'(c) $\nu=1$', loc='left', fontweight='bold', pad=4); ax[2].legend(frameon=False, loc='upper left', handletextpad=.1, labelspacing=.25)
fig.savefig(OUT / 'fig6.pdf', metadata={'CreationDate': None, 'ModDate': None}); fig.savefig(OUT / 'fig6.png', dpi=600); fig.savefig(OUT / 'fig6.eps', dpi=300)
print(len(j), j.ND.corr(j.dL))
