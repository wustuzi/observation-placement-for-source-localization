"""Independent small-graph audit for the focused revision (no manuscript imports).

Run from project root:
  PYTHONPATH=source_study/revision/partial_observation/dependencies \
    OPENBLAS_NUM_THREADS=1 python3 source_study/revision/focused_revision/theory/verify_retained_theory.py

This is a verification artifact, not the source of the manuscript's ensemble data.
The connected graph atlas exhausts all unlabeled simple graphs through N=7.
Exact finite-state checks exhaust N<=6, all masks, at four graph-scaled ages.
"""
from pathlib import Path
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.special import xlogy

OUT = Path(__file__).resolve().parent


def channel_generator(g):
    """Row-generator convention; one bit per infected vertex."""
    n = len(g)
    rr, cc, vv = [], [], []
    for state in range(1 << n):
        total = 0
        for v in range(n):
            if state & (1 << v):
                continue
            rate = sum(bool(state & (1 << u)) for u in g.neighbors(v))
            if rate:
                rr.append(state)
                cc.append(state | (1 << v))
                vv.append(rate)
                total += rate
        rr.append(state)
        cc.append(state)
        vv.append(-total)
    return csr_matrix((vv, (rr, cc)), shape=(1 << n, 1 << n), dtype=float)


def project(p, mask):
    # Keeping original vertex bit labels avoids assumptions about local ordering.
    labels = np.arange(p.shape[1]) & mask
    return np.array([np.bincount(labels, weights=row, minlength=p.shape[1]) for row in p])


def accuracy(p):
    return p.max(axis=0).sum() / len(p)


def information(p):
    marginal = p.mean(axis=0)
    return (xlogy(p, p) - xlogy(p, marginal)).sum() / len(p)


def geometry(g, mask):
    sensors = {v for v in g if mask & (1 << v)}
    external = set(g) - sensors
    e = sum(u in sensors and v in sensors for u, v in g.edges())
    c = sum((u in sensors) != (v in sensors) for u, v in g.edges())
    a = min(sum(v in sensors for v in g[u]) for u in external)
    return len(sensors), e, c, a


def run():
    started = time.time()
    atlas = [g for g in nx.graph_atlas_g() if len(g) >= 2 and nx.is_connected(g)]
    combinatorial_masks = 0
    endpoint_masks = 0
    class_a_spread = []
    records = []
    first_order_channels = 0
    numerical_masks = 0
    full_expansions = 0
    max_euler_error = 0.0
    max_stochastic_error = 0.0
    max_bound_fraction = 0.0
    max_class_interval_violation = 0.0
    max_residual_difference_violation = 0.0
    for graph_id, g in enumerate(atlas):
        n = len(g)
        edge_count = g.number_of_edges()
        delta = max(dict(g.degree()).values())
        geometry_rows = [(mask, *geometry(g, mask)) for mask in range(1, (1 << n) - 1)]
        combinatorial_masks += len(geometry_rows)
        by_budget = {}
        for m in range(1, n):
            rows = np.array([row for row in geometry_rows if row[1] == m], dtype=int)
            masks, _, e, c, a = rows.T
            assert int(e.sum()) * n * (n - 1) == edge_count * m * (m - 1) * len(masks)
            assert int(c.sum()) * n * (n - 1) == edge_count * 2 * m * (n - m) * len(masks)
            degree_volumes = np.array([sum(g.degree(v) for v in g if mask & (1 << v)) for mask in masks])
            assert np.array_equal(c, degree_volumes - 2 * e)
            if m == n - 1:
                assert np.all(e + a == edge_count)
                endpoint_masks += len(masks)
            c_min = c.min()
            e_min_cut = e[c == c_min].min()
            leading = (c == c_min) & (e == e_min_cut)
            l = e + a
            if n > 2 and m == 1 and edge_count < n * (n - 1) // 2:
                assert np.all(l[leading] == 0)
                assert l.min() == 0
            if n > 2 and m * (delta + 1) < n:
                assert np.all(a == 0)
            if np.ptp(a[leading]):
                class_a_spread.append(dict(atlas_id=graph_id, n=n, m=m,
                    edges=list(g.edges()), leading_masks=masks[leading].tolist(),
                    leading_a=a[leading].tolist(), c_min=int(c_min), e_min_cut=int(e_min_cut)))
            by_budget[m] = (masks, e, c, a, leading)
        if n > 6:
            continue
        q = channel_generator(g)
        initial = np.zeros((1 << n, n))
        initial[2 ** np.arange(n), np.arange(n)] = 1
        tau_limit = min(1 / edge_count, 1 / (delta + 1))
        for fraction in [0.001, 0.01, 0.1, 1.0]:
            tau = fraction * tau_limit
            exact = expm_multiply(q.T * tau, initial, traceA=q.diagonal().sum() * tau).T
            euler = (initial + tau * q.T @ initial).T
            assert exact.min() >= -1e-14 and euler.min() >= -1e-14
            stochastic_error = max(np.abs(exact.sum(axis=1) - 1).max(), np.abs(euler.sum(axis=1) - 1).max())
            max_stochastic_error = max(max_stochastic_error, float(stochastic_error))
            assert stochastic_error < 2e-12
            s = edge_count * tau
            bound = -s * np.expm1(-s)
            full_linear = 1 - edge_count * tau / n
            full_a = accuracy(exact)
            assert abs(accuracy(euler) - full_linear) < 2e-13
            assert abs(full_a - full_linear) <= bound + 2e-13
            full_expansions += 1
            for m, (masks, e, c, a, leading) in by_budget.items():
                exact_a, exact_i, euler_a = [], [], []
                for mask in masks:
                    observed = project(exact, mask)
                    exact_a.append(accuracy(observed))
                    exact_i.append(information(observed))
                    euler_a.append(accuracy(project(euler, mask)))
                exact_a = np.array(exact_a)
                exact_i = np.array(exact_i)
                euler_a = np.array(euler_a)
                numerical_masks += len(masks)
                first_order_channels += len(masks)
                l = e + a
                linear_a = (m + 1) / n - l * tau / n
                linear_i = np.log(n) - (n - m) / n * np.log(n - m) \
                    - c / n * tau * np.log(1 / tau) \
                    + tau / n * (c * (np.log(n - m) - 1) - 2 * e * np.log(2))
                euler_error = np.abs(euler_a - linear_a).max()
                max_euler_error = max(max_euler_error, float(euler_error))
                assert euler_error < 2e-13
                ra = float(np.abs(exact_a - linear_a).max())
                ri = float(np.abs(exact_i - linear_i).max())
                assert ra <= bound + 2e-13
                max_bound_fraction = max(max_bound_fraction, ra / bound)
                # Tolerance treats only roundoff-level information gaps as exact ties.
                information_ties = exact_i >= exact_i.max() - 5e-14
                chosen_i = np.flatnonzero(information_ties)[np.argmax(exact_a[information_ties])]
                chosen_a = int(np.argmax(exact_a))
                objective_loss = float(exact_a[chosen_a] - exact_a[chosen_i])
                budget_loss = full_a - exact_a[chosen_a]
                total_loss = full_a - exact_a[chosen_i]
                assert budget_loss >= -2e-13 and objective_loss >= -2e-13
                assert abs(total_loss - budget_loss - objective_loss) < 2e-13
                # Measured residual certificate: valid at every age without asymptotic class selection.
                actual_coefficient = tau / n * (l[chosen_i] - l[chosen_a])
                residual_difference_violation = abs(objective_loss - actual_coefficient) - 2 * ra
                assert residual_difference_violation <= 2e-13
                max_residual_difference_violation = max(max_residual_difference_violation, float(residual_difference_violation))
                class_hit = bool(leading[chosen_i])
                predicted_lower = tau / n * (l[leading].min() - l.min())
                predicted_upper = tau / n * (l[leading].max() - l.min())
                if class_hit:
                    interval_violation = max(predicted_lower - 2 * ra - objective_loss,
                        objective_loss - predicted_upper - 2 * ra)
                    assert interval_violation <= 2e-13
                    max_class_interval_violation = max(max_class_interval_violation, float(interval_violation))
                budget_linear = (n - m - 1) / n - tau / n * (edge_count - l.min())
                # Difference between a full linear expansion and the maximum partial expansion.
                assert abs(budget_loss - budget_linear) <= abs(full_a - full_linear) + ra + 2e-13
                records.append(dict(atlas_id=graph_id, n=n, edges=edge_count, delta=delta,
                    m=m, tau=tau, tau_limit=tau_limit, fraction_of_limit=fraction,
                    max_accuracy_residual=ra, max_information_residual=ri,
                    uniformization_bound=bound, full_accuracy=full_a,
                    full_linear_residual=abs(full_a - full_linear),
                    max_accuracy_residual_over_tau2=ra / tau**2,
                    max_information_residual_over_tau2log=ri / (tau**2 * abs(np.log(tau))),
                    mi_in_lexicographic_class=class_hit,
                    all_mi_ties_in_lexicographic_class=bool(np.all(leading[information_ties])),
                    objective_loss=objective_loss, budget_loss=budget_loss, total_loss=total_loss,
                    geometry_regret_lower=predicted_lower, geometry_regret_upper=predicted_upper,
                    actual_mask_linear_regret=actual_coefficient,
                    budget_linear_prediction=budget_linear))
        if graph_id % 30 == 0:
            print(f'checked {graph_id + 1} atlas graphs', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'retained_theory_checks.csv', index=False)
    summary = dict(
        connected_atlas_graphs_through_7=len(atlas),
        combinatorial_masks=combinatorial_masks, endpoint_masks=endpoint_masks,
        exact_graphs_through_6=sum(len(g) <= 6 for g in atlas),
        exact_graph_time_budget_conditions=len(frame), exact_mask_time_conditions=numerical_masks,
        full_expansions=full_expansions, first_order_channels=first_order_channels,
        largest_probability_sum_error=max_stochastic_error,
        largest_first_order_channel_accuracy_error=max_euler_error,
        largest_accuracy_residual_fraction_of_bound=max_bound_fraction,
        largest_class_interval_violation=max_class_interval_violation,
        largest_actual_mask_residual_difference_violation=max_residual_difference_violation,
        class_spread_examples=class_a_spread[:10], class_spread_count=len(class_a_spread),
        class_hit_by_time_fraction=frame.groupby('fraction_of_limit').mi_in_lexicographic_class.mean().to_dict(),
        singleton_objective_loss_over_tau2_max=float((frame.loc[frame.m == 1, 'objective_loss'] / frame.loc[frame.m == 1, 'tau']**2).max()),
        nearfull_budget_loss_over_tau2_max=float((frame.loc[frame.m == frame.n - 1, 'budget_loss'] / frame.loc[frame.m == frame.n - 1, 'tau']**2).max()),
        nearfull_objective_loss_over_tau2_max=float((frame.loc[frame.m == frame.n - 1, 'objective_loss'] / frame.loc[frame.m == frame.n - 1, 'tau']**2).max()),
        runtime_seconds=time.time() - started,
        software=dict(numpy=np.__version__, networkx=nx.__version__),
        assertions='All passed. Numerics verify examples; the accompanying derivations provide the proofs.',
    )
    (OUT / 'retained_theory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
