"""Redraw the four reorganized-manuscript figures from saved results only.

No channel, simulation, root search, or new observation time is evaluated. Each
figure is drawn directly from saved tables, so the PDFs contain no cropped-away
panels. Differences are shown as cluster minus spread: the saved accuracy
columns are spread minus cluster, so they are negated here. Outputs go to
figures_reorg/; figures_main/ is left untouched.

Run with a Python environment that has numpy, pandas, networkx, and matplotlib.
"""
from pathlib import Path
import json, math, os, sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
SOURCE = STUDY / 'revision/large_network_revision'
THEORY = SOURCE / 'audit/local_reruns/source_snapshot/revision/decision_spectrum/theory'
DESIGN = SOURCE / 'audit/local_reruns/source_snapshot/revision/design'
RELATIVE = SOURCE / 'local_reproduction/relative'
OUT = STUDY / 'figures'
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, GRAY = '#2166ac', '#d6604d', '#1b7f5c', '#bcbcbc'
SIZE_MARKERS = ['o', 's', '^', 'D', 'v']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 8,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.title_fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
    'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT / f'{name}.png', dpi=600)
    # PostScript has no transparency, so translucent shading is rasterized there.
    for artist in fig.findobj(lambda a: a.get_alpha() is not None and a.get_alpha() < 1):
        artist.set_rasterized(True)
    fig.savefig(OUT / f'{name}.eps', dpi=300)
    plt.close(fig)


def panel(ax, letter):
    ax.set_title(f'({letter})', loc='left', fontweight='bold', pad=4)


def cycle_schematic(ax, y0, observed, label):
    angle = 2 * np.pi * np.arange(10) / 10
    x, y = .5 + .34 * np.cos(angle), y0 + .17 * np.sin(angle)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color='.55', lw=.8, zorder=1)
    chosen = np.isin(np.arange(10), sorted(observed))
    ax.scatter(x[~chosen], y[~chosen], s=20, zorder=2, linewidths=0, c=GRAY)
    ax.scatter(x[chosen], y[chosen], s=26, zorder=2, linewidths=0, c=ORANGE, marker='s')
    ax.text(.5, y0 - .205, label, ha='center', va='top')


def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.45), layout='constrained',
                           gridspec_kw={'width_ratios': [.62, 1, 1]})
    a = ax[0]
    a.set(xlim=(0, 1), ylim=(0, 1))
    a.axis('off')
    panel(a, 'a')
    cycle_schematic(a, .76, {0, 1}, r'adjacent: $c_M=2,\ e_M=1$')
    cycle_schematic(a, .27, {0, 5}, r'separated: $c_M=4,\ e_M=0$')

    a = ax[1]
    summary = pd.read_csv(DESIGN / 'early_time_summary.csv')
    for name, label, color, marker, limit in [
            ('cycle10_m2', r'$C_{10},\ m=2$', BLUE, 'o', .1),
            ('petersen10_m2', r'Petersen, $m=2$', ORANGE, 's', .1),
            ('cycle12_m3', r'$C_{12},\ m=3$', GREEN, '^', 1 / 6)]:
        z = summary[summary.graph == name].sort_values('time')
        a.semilogx(z.time, z.top1_regret / z.time, marker=marker, ms=3, color=color, label=label)
        a.axhline(limit, color=color, ls='--', lw=.7)
    a.set(xlabel=r'Elapsed time $\tau$', ylabel=r'$D/\tau$')
    a.legend(frameon=False, loc='center left', bbox_to_anchor=(.03, .5))
    panel(a, 'b')

    a = ax[2]
    curve = pd.read_csv(THEORY / 'figure_boundary_curve.csv')
    points = pd.read_csv(THEORY / 'crossover_boundary.csv').set_index('L').alpha_c
    a.fill_between(curve.L, 1, curve.alpha_c, color=BLUE, alpha=.08, lw=0)
    a.fill_between(curve.L, curve.alpha_c, 5, color=ORANGE, alpha=.08, lw=0)
    a.plot(curve.L, curve.alpha_c, color='.15', lw=1.4)
    for L, color in [(8, BLUE), (9, GREEN)]:
        a.plot(L, points[L], 'o', ms=4, mfc=color, mec='white', mew=.5)
    a.annotate(r'$L=8,9$', xy=(8.5, 2.2), xytext=(16, 2.95), fontsize=8,
               arrowprops={'arrowstyle': '-', 'lw': .7, 'color': '.4'})
    a.text(65, 3.5, r'$e_M=e_{\min}$', color=ORANGE, fontsize=9)
    a.text(12, 1.22, r'$e_M=e_{\max}$', color=BLUE, fontsize=9)
    a.text(145, 1.72, r'$\alpha_c(L)$', color='.15')
    a.set_xscale('log')
    a.set(xlim=(3, 1000), ylim=(1, 5), xlabel=r'Unobserved nodes $L=N-m$',
          ylabel=r'Order $\alpha$')
    a.set_xticks([3, 10, 100, 1000], labels=['3', '10', r'$10^2$', r'$10^3$'])
    panel(a, 'c')
    save(fig, 'fig1')


def fig2():
    summary = pd.read_csv(RELATIVE / 'by_rho_budget.csv')
    rhos = ['[0,0.20]', '(0.20,0.40]', '(0.40,0.60]', '(0.60,0.80]', '(0.80,0.95]', '(0.95,1]']
    budgets = ['(0,0.25]', '(0.25,0.50]', '(0.50,0.75]', '(0.75,1)']
    xlabels = ['0–.20', '.20–.40', '.40–.60', '.60–.80', '.80–.95', '.95–1']
    ylabels = [r'$0<m/N\leq1/4$', r'$1/4<m/N\leq1/2$', r'$1/2<m/N\leq3/4$', r'$3/4<m/N<1$']

    def table(column):
        return summary.pivot(index='budget_band', columns='rho_stage', values=column).reindex(
            index=budgets, columns=rhos).to_numpy()

    fig = plt.figure(figsize=(7.1, 6.3))
    top = fig.add_gridspec(1, 2, left=.135, right=.915, top=.955, bottom=.565, wspace=.36)
    bottom = fig.add_gridspec(1, 3, left=.035, right=.875, top=.40, bottom=.11, wspace=.36,
                              width_ratios=[1.13, 1, 1])

    a = fig.add_subplot(top[0])
    largest, median = 100 * table('relative_regret_max'), 100 * table('relative_regret_median')
    image = a.imshow(median, origin='lower', aspect='auto', vmin=0, vmax=1.5, cmap='YlOrRd')
    for iy in range(4):
        for ix in range(6):
            color = 'white' if median[iy, ix] > 1. else '#222222'
            a.text(ix, iy, f'{median[iy, ix]:.2f}\n[{largest[iy, ix]:.1f}]', ha='center',
                   va='center', color=color, fontsize=8, linespacing=1.05)
    bar = fig.colorbar(image, ax=a, fraction=.04, pad=.02)
    bar.set_label(r'Median $100D/A_1^*$', fontsize=8)
    a.set_yticks(range(4), labels=ylabels)
    a.set_title(r'$\bf{(a)}$ Relative loss (%): median [largest]', loc='left', pad=4)

    b = fig.add_subplot(top[1])
    lowest = 100 * table('min_retention_5pct')
    middle = 100 * table('median_retention_5pct')
    counts = table('eligible_5pct')
    image = b.imshow(lowest, origin='lower', aspect='auto',
                     norm=TwoSlopeNorm(vmin=-110, vcenter=0, vmax=100), cmap='RdBu')
    for iy in range(4):
        for ix in range(6):
            if np.isnan(lowest[iy, ix]):
                b.text(ix, iy, '—', ha='center', va='center', color='#777777')
                continue
            color = 'white' if abs(lowest[iy, ix]) > 65 else '#222222'
            b.text(ix, iy, f'{lowest[iy, ix]:.0f}\n{middle[iy, ix]:.0f}\n[{int(counts[iy, ix])}]',
                   ha='center', va='center', color=color, fontsize=8, linespacing=1.0)
    bar = fig.colorbar(image, ax=b, fraction=.04, pad=.02)
    bar.set_label(r'Lowest $100R_{\rm gain}$', fontsize=8)
    b.set_yticks(range(4), labels=[])
    b.set_title(r'$\bf{(b)}$ $100R_{\rm gain}$: lowest, median [count]', loc='left', pad=4)
    for axis in (a, b):
        axis.set_xticks(range(6), labels=xlabels, rotation=45, ha='right')
        axis.set_xlabel(r'Expected infected fraction $\rho$')
        axis.set_xticks(np.arange(-.5, 6, 1), minor=True)
        axis.set_yticks(np.arange(-.5, 4, 1), minor=True)
        axis.grid(which='minor', color='white', linewidth=.7)
        axis.tick_params(which='minor', length=0)
        axis.spines[['top', 'right']].set_visible(False)

    z = np.load(RELATIVE / 'exception_posterior.npz')
    designs = pd.read_csv(RELATIVE / 'exception_designs.csv')
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(z['edges'].tolist())
    pos = {2: (-1., 0.), 5: (-.55, .6), 9: (-.55, -.6), 0: (-.15, 1.), 3: (.15, 0.), 1: (.6, .75),
           4: (1., .7), 6: (.65, -.7), 7: (1.25, -.5), 8: (1.4, .25)}
    info, accuracy = set(z['MI_sensors']), set(z['Accuracy_sensors'])
    c = fig.add_subplot(bottom[0])
    nx.draw_networkx_edges(graph, pos, ax=c, width=.8, edge_color='#8a8a8a')
    groups = [('Information only', BLUE, 's', info - accuracy), ('Accuracy only', ORANGE, 'D', accuracy - info),
              ('Both', GRAY, 'o', info & accuracy)]
    for _, color, marker, nodes in groups:
        nx.draw_networkx_nodes(graph, pos, ax=c, nodelist=sorted(nodes), node_size=150, node_color=color,
                               node_shape=marker, edgecolors='white', linewidths=.7)
    nx.draw_networkx_labels(graph, pos, ax=c, font_size=8)
    c.set_axis_off()
    c.margins(.1)
    c.set_title('(c)', loc='left', fontweight='bold', pad=4)
    c.set_title('Observation sets', loc='center', pad=4)
    handles = [Line2D([], [], color='none', marker=marker, markerfacecolor=color,
                      markeredgecolor='none', markersize=6, label=label)
               for label, color, marker, _ in groups]
    c.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.02), ncol=2, frameon=False,
             handletextpad=.2, columnspacing=.6)
    for spec, design, letter, title in [(bottom[1], 'MI', 'd', 'Information choice'),
                                        (bottom[2], 'Accuracy', 'e', 'Accuracy choice')]:
        d = fig.add_subplot(spec)
        matrix = z[design + '_confusion']
        image = d.imshow(matrix, vmin=0, vmax=1, cmap='Blues', origin='upper')
        value = float(designs[designs.design == design].accuracy.iloc[0])
        d.set_title(f'({letter})', loc='left', fontweight='bold', pad=4)
        d.set_title(title + '\n' + rf'$A_1={value:.3f}$', loc='center', pad=4)
        d.set_xlabel('MAP estimate')
        d.set_ylabel('True source')
        d.set_xticks(range(10))
        d.set_yticks(range(10))
        d.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=1.3))
    cax = fig.add_axes([.895, .14, .012, .23])
    bar = fig.colorbar(image, cax=cax)
    bar.set_ticks([0, .5, 1])
    bar.set_label('Conditional probability', fontsize=8)
    save(fig, 'fig2')


def fig3():
    rows = pd.read_csv(SOURCE / 'cycles/contrasts.csv')
    scaling = pd.read_csv(SOURCE / 'cycles/scaled_time.csv')
    roots = pd.read_csv(SOURCE / 'cycles/crossovers.csv')
    sizes = [100, 200, 500, 1000, 2000]
    colors = plt.cm.viridis(np.linspace(.12, .85, len(sizes)))
    fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.1), layout='constrained')
    for n, color, marker in zip(sizes, colors, SIZE_MARKERS):
        selected = rows[(rows.n == n) & (rows.m == 2)]
        assert selected.tau.tolist() == [.1, .5, 1., 2., 4., 8., 12., 20.]
        ax[0, 0].plot(selected.tau, selected.delta_MI_cluster_minus_spread, '-', marker=marker, ms=3, color=color, label=str(n))
        ax[0, 1].plot(selected.tau, -selected.N_delta_A1, '-', marker=marker, ms=3, color=color)
        early = scaling[(scaling.n == n) & (scaling.m == 2) & (scaling.N_tau <= 4)]
        ax[1, 0].plot(early.N_tau, early.N2_delta_MI, color=color)
    ax[0, 0].set(xscale='log', xlabel=r'Elapsed time $\tau$', ylabel=r'$\Delta I$ (nats)')
    ax[0, 0].legend(title=r'$N$', ncol=2, loc='lower left', frameon=False)
    ax[0, 1].set(xscale='log', xlabel=r'Elapsed time $\tau$', ylabel=r'$N\,\Delta A$')
    ax[1, 0].axhline(0, color='.5', lw=.7)
    x = np.linspace(.01, 4, 300)
    ax[1, 0].plot(x, 2 * x * (1 - np.log(2 * x)), 'k--', lw=.9, label='Short-time prediction')
    ax[1, 0].axvspan(0, math.e / 2, color=ORANGE, alpha=.1, lw=0)
    ax[1, 0].text(.08, -2.3, 'conflict:\n' + r'$\Delta I>0,\ \Delta A<0$', color='#8f3b2c')
    ax[1, 0].set(xlabel=r'$N\tau$', ylabel=r'$N^2\,\Delta I$', xlim=(0, 4.2))
    ax[1, 0].legend(frameon=False, loc='lower left')
    selected = roots[roots.n.isin(sizes) & (roots.m == 2)]
    ax[1, 1].plot(1 / selected.n, selected.N_tau_c, 'o-', color='#315980', ms=4, label='Channel crossing')
    inverse = np.linspace(0, .0105, 300)
    ax[1, 1].plot(inverse, (math.e / 2) / (1 - 2 * inverse), color='#b75b28', ls='--', lw=1,
                  label=r'Leading term $(e/2)N/(N-2)$')
    ax[1, 1].axhline(math.e / 2, color='.35', ls=':', lw=1, label=r'$e/2$ limit')
    ax[1, 1].set(xlabel=r'$1/N$', ylabel=r'$N\tau_c$', xlim=(0, .0105))
    ax[1, 1].legend(frameon=False, loc='upper left', handlelength=1.6)
    for k, a in enumerate(ax.flat):
        panel(a, chr(97 + k))
    save(fig, 'fig3')


def fig4():
    original = pd.read_csv(SOURCE / 'random_mc/contrasts.csv')
    matched = pd.read_csv(SOURCE / 'random_mc/degree_matched/contrasts.csv')
    sizes, times = [100, 250, 500, 1000, 2000], [.1, .5, 1., 2., 4.]
    colors = plt.cm.viridis(np.linspace(.05, .88, 5))
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 5.8), sharex=True, layout='constrained')
    ticks = [[[-.02, 0, .02, .04, .06], [-.04, -.02, 0, .02, .04], [-.25, -.20, -.15, -.10, -.05, 0]],
             [[-.05, -.04, -.03, -.02, -.01, 0, .01], [-.05, -.04, -.03, -.02, -.01, 0], [-.25, -.20, -.15, -.10, -.05, 0]],
             [[-1.4, -1.2, -1., -.8, -.6, -.4, -.2, 0]] * 3]
    configurations = [(original, 'delta_I_jackknife', 1), (matched, 'delta_I_jackknife', 1),
                      (original, 'scaled_delta_A', -1)]
    for row, (data, key, sign) in enumerate(configurations):
        for col, family in enumerate(['ER', 'BA', 'WS']):
            a = axes[row, col]
            for n, color, marker in zip(sizes, colors, SIZE_MARKERS):
                selected = data[(data.family == family) & (data.requested_n == n)]
                expected = 2 if row == 1 and family == 'BA' and n == 250 else 3
                groups = [selected[selected.tau == t][key].to_numpy() * sign for t in times]
                assert all(len(g) == expected for g in groups)
                values = np.array(groups)
                a.plot(times, values.mean(1), '-', marker=marker, ms=3, lw=1, color=color, label=str(n))
                a.fill_between(times, values.min(1), values.max(1), color=color, alpha=.12, lw=0)
            a.axhline(0, color='.4', ls=':', lw=.8)
            a.set_xscale('log')
            a.set_xticks(times, labels=['0.1', '0.5', '1', '2', '4'])
            limits = a.get_ylim()
            a.set_yticks(ticks[row][col])
            a.set_ylim(limits)
            panel(a, chr(97 + row * 3 + col))
            if row == 0:
                a.set_title(family, loc='center', fontsize=10, pad=4)
            if row == 2:
                a.set_xlabel(r'Elapsed time $\tau$')
    axes[0, 0].set_ylabel('Original pairs\n' + r'$\Delta I$ (nats)')
    axes[1, 0].set_ylabel('Degree-matched pairs\n' + r'$\Delta I$ (nats)')
    axes[2, 0].set_ylabel('Original pairs\n' + r'$N\,\Delta A$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Requested $N$', frameon=False, ncol=5, loc='outside upper center',
               columnspacing=1.2, handlelength=1.7)
    save(fig, 'fig4')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print(json.dumps({'output': str(OUT), 'files': sorted(p.name for p in OUT.iterdir())}))
