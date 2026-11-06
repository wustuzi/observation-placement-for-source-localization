"""Independent exact-CTMC and 90-digit verification of the decision spectrum.

PYTHONPATH=source_study/revision/partial_observation/dependencies \
OPENBLAS_NUM_THREADS=1 python3 source_study/revision/decision_spectrum/theory/verify_spectrum.py

No manuscript experiment module is imported. Decimal uniformization is
implemented independently from the sparse matrix-exponential calculation.
"""
from pathlib import Path
import decimal
from decimal import Decimal as D
from itertools import combinations
import json
import math
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.special import xlogy, lambertw

OUT = Path(__file__).resolve().parent
decimal.getcontext().prec = 90


def generator(g):
    n = len(g)
    rows, cols, vals = [], [], []
    for state in range(1 << n):
        escape = 0
        for v in range(n):
            if state & (1 << v):
                continue
            rate = sum(bool(state & (1 << u)) for u in g[v])
            if rate:
                rows.append(state | (1 << v))
                cols.append(state)
                vals.append(rate)
                escape += rate
        rows.append(state)
        cols.append(state)
        vals.append(-escape)
    return csr_matrix((vals, (rows, cols)), shape=(1 << n, 1 << n), dtype=float)


def channel(g, tau, sources=None):
    n = len(g)
    sources = list(range(n)) if sources is None else list(sources)
    initial = np.zeros((1 << n, len(sources)))
    initial[2 ** np.array(sources), np.arange(len(sources))] = 1
    q = generator(g)
    p = expm_multiply(tau * q, initial, traceA=tau * q.diagonal().sum()).T
    assert p.min() >= -1e-14 and np.max(abs(p.sum(1) - 1)) < 1e-12
    return p


def project(p, sensors):
    mask = sum(1 << v for v in sensors)
    labels = np.arange(p.shape[1]) & mask
    return np.array([np.bincount(labels, weights=row, minlength=p.shape[1]) for row in p])


def score(p, alpha):
    if alpha == 1:
        return (xlogy(p, p) - xlogy(p, p.mean(0))).sum() / len(p)
    maxima = p.max(0)
    if np.isinf(alpha):
        return maxima.sum() / len(p)
    good = maxima > 0
    columns = (p[:, good] / maxima[good]) ** alpha
    return float((maxima[good] * columns.sum(0) ** (1 / alpha)).sum() / len(p))


def geometry(g, sensors):
    sensors = set(sensors)
    outside = set(g) - sensors
    e = sum(u in sensors and v in sensors for u, v in g.edges())
    c = sum((u in sensors) != (v in sensors) for u, v in g.edges())
    a = min(sum(v in sensors for v in g[u]) for u in outside)
    return e, c, a


def crossing(l):
    assert l > 2
    return 1 / brentq(lambda x: 2 ** x + 2 * l ** (x - 1) - 2, 0, 1, xtol=1e-14)


def decimal_channel(g, tau):
    """High-precision Poisson uniformization; omitted weight < 1e-85."""
    n = len(g)
    size = 1 << n
    rate = D(g.number_of_edges())
    transitions = []
    for state in range(size):
        row = []
        escape = 0
        for v in range(n):
            if state & (1 << v):
                continue
            r = sum(bool(state & (1 << u)) for u in g[v])
            if r:
                row.append((state | (1 << v), D(r) / rate))
                escape += r
        row.append((state, 1 - D(escape) / rate))
        transitions.append(row)
    result = []
    s = rate * tau
    for source in range(n):
        current = [D(0)] * size
        current[1 << source] = D(1)
        weight = (-s).exp()
        total_weight = weight
        accumulated = [weight * value for value in current]
        for step in range(1, 300):
            following = [D(0)] * size
            for state, value in enumerate(current):
                if not value:
                    continue
                for destination, probability in transitions[state]:
                    following[destination] += value * probability
            current = following
            weight *= s / step
            total_weight += weight
            for state, value in enumerate(current):
                accumulated[state] += weight * value
            # For s<1, the remaining weights form a decreasing sequence,
            # bounded above by the first omitted weight/(1-s/(step+2)).
            tail_bound = weight * s / (step + 1) / (1 - s / (step + 2))
            if tail_bound < D('1e-85'):
                break
        else:
            raise AssertionError('uniformization did not converge')
        assert abs(sum(accumulated) - 1) < D('2e-85')
        result.append(accumulated)
    return result


def decimal_project(p, sensors):
    mask = sum(1 << v for v in sensors)
    result = []
    for row in p:
        projected = [D(0)] * len(row)
        for state, value in enumerate(row):
            projected[state & mask] += value
        result.append(projected)
    return result


def decimal_s(p, alpha):
    answer = D(0)
    for values in zip(*p):
        maximum = max(values)
        if not maximum:
            continue
        if alpha is None:
            answer += maximum
            continue
        power_sum = D(0)
        for value in values:
            if value:
                exponent = alpha * (value / maximum).ln()
                if exponent > -1000:  # smaller terms are below 90-digit accuracy
                    power_sum += exponent.exp()
        answer += maximum * (power_sum.ln() / alpha).exp()
    return answer / len(p)


def decimal_information(p):
    n = D(len(p))
    answer = D(0)
    for column in zip(*p):
        marginal = sum(column) / n
        for value in column:
            if value:
                answer += value / n * (value.ln() - marginal.ln())
    return answer


def run():
    started = time.time()
    rows, optima, fractional = [], [], []
    graphs = [
        ('cycle8', nx.cycle_graph(8), 2),
        ('cycle10', nx.cycle_graph(10), 2),
        ('cycle14', nx.cycle_graph(14), 2),
        ('path6', nx.path_graph(6), 2),
        ('star6', nx.star_graph(5), 2),
        ('triangle_pendant4', nx.Graph([(0, 3), (1, 2), (1, 3), (2, 3)]), 2),
    ]
    max_scaled_error = 0.0
    for name, g, m in graphs:
        n = len(g)
        l = n - m
        masks = list(combinations(range(n), m))
        geom = np.array([geometry(g, mask) for mask in masks])
        alphas = [1.1, 1.5, 2., 3., 10., np.inf]
        ac = crossing(l) if l > 2 else np.nan
        if name.startswith('cycle'):
            alphas += [(1 + ac) / 2, ac, ac + .5]
        for tau in [.00001, .0001, .001, .01]:
            full = channel(g, tau)
            projected = [project(full, mask) for mask in masks]
            for alpha in alphas:
                values = np.array([score(p, alpha) for p in projected])
                if np.isinf(alpha):
                    k = m + 1
                    f = geom[:, 0] + geom[:, 2]
                    exponent = 2
                else:
                    k = m + l ** (1 / alpha)
                    f = (2 - 2 ** (1 / alpha)) * geom[:, 0] + l ** (1 / alpha - 1) * geom[:, 1]
                    exponent = min(2, alpha)
                residual = n * values - k + f * tau
                scaled = abs(residual).max() / tau ** exponent
                max_scaled_error = max(max_scaled_error, float(scaled))
                rows.append(dict(graph=name, n=n, m=m, tau=tau, alpha=alpha,
                    max_NS_linear_residual=float(abs(residual).max()), remainder_power=exponent,
                    max_scaled_residual=float(scaled), alpha_c=ac))
                if 1 < alpha < 2:
                    remainder = residual - geom[:, 1] / alpha * tau ** alpha
                    fractional.append(dict(graph=name, n=n, tau=tau, alpha=alpha,
                        max_fractional_corrected_residual=float(abs(remainder).max()),
                        max_corrected_over_tau2=float(abs(remainder).max() / tau**2)))
                if name.startswith('cycle') and np.isfinite(alpha) and abs(alpha - ac) > 1e-6:
                    chosen = np.flatnonzero(values >= values.max() - 1e-13)
                    target_e = geom[:, 0].max() if alpha < ac else geom[:, 0].min()
                    correct = bool(np.all(geom[chosen, 0] == target_e))
                    if tau <= .0001:
                        assert correct
                    optima.append(dict(graph=name, n=n, m=m, tau=tau, alpha=alpha,
                        alpha_c=ac, target_e=int(target_e), selected_e_min=int(geom[chosen, 0].min()),
                        selected_e_max=int(geom[chosen, 0].max()), containment=correct))
        print(name, 'done', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'finite_alpha_coefficients.csv', index=False)
    pd.DataFrame(optima).to_csv(OUT / 'regular_crossover_checks.csv', index=False)
    pd.DataFrame(fractional).to_csv(OUT / 'fractional_remainders.csv', index=False)

    # High-precision tests isolate genuinely nonanalytic corrections from float noise.
    g = nx.Graph([(0, 3), (1, 2), (1, 3), (2, 3)])
    m, n, l = 2, 4, 2
    sensors = [1, 2]
    e, c, a = geometry(g, sensors)  # e=1,c=2,a=0; external degrees are 0 and 2.
    decimal_rows, near_one, near_infinity, coarse_tie = [], [], [], []
    decimal_crosscheck_error = 0.0
    for tau_string in ['.0001', '.00000001', '.0000000000000001']:
        tau = D(tau_string)
        decimal_full = decimal_channel(g, tau)
        double_full = channel(g, float(tau))
        crosscheck = float(abs(double_full - np.array(decimal_full, dtype=float)).max())
        assert crosscheck < 1e-14
        decimal_crosscheck_error = max(decimal_crosscheck_error, crosscheck)
        p = decimal_project(decimal_full, sensors)
        information_gap = decimal_information(decimal_project(decimal_full, [0, 3])) - decimal_information(p)
        tie_coefficient = information_gap / tau**2
        assert abs(tie_coefficient - D(2).ln()/4) < 10*tau*abs(tau.ln())
        coarse_tie.append(dict(tau=str(tau), information_gap=str(information_gap),
            measured_second_order_coefficient=str(tie_coefficient),
            exact_coefficient=str(D(2).ln()/4)))
        for alpha_string in ['1.1', '1.5', '2', '3', '1000']:
            alpha = D(alpha_string)
            inv = 1 / alpha
            k = D(m) + (D(l).ln() * inv).exp()
            f = (2 - (D(2).ln() * inv).exp()) * e + (D(l).ln() * (inv - 1)).exp() * c
            actual = decimal_s(p, alpha)
            residual = D(n) * actual - k + f * tau
            corrected = residual - D(c) / alpha * (alpha * tau.ln()).exp() if alpha < 2 else residual
            assert abs(corrected / tau**2) < 1000
            decimal_rows.append(dict(tau=str(tau), alpha=str(alpha),
                finite_slope=str(f), measured_slope=str((k - n * actual) / tau),
                corrected_residual_over_tau2=str(corrected / tau**2)))
        # alpha=1+z/log(1/tau): normalize by the Shannon singular scale.
        for z in [D('.1'), D('1'), D('3')]:
            r = -tau.ln()
            alpha = 1 + z / r
            inv = 1 / alpha
            k = D(m) + (D(l).ln() * inv).exp()
            information_drop = alpha / (alpha - 1) * ((D(n) * decimal_s(p, alpha) / k).ln())
            prediction = -D(c) / n * (1 - (-z).exp()) / z
            near_one.append(dict(tau=str(tau), z=str(z), alpha=str(alpha),
                normalized_information_change=str(information_drop / (tau * r)),
                limiting_prediction=str(prediction)))
        for z in [D('.01'), D('1'), D('10'), D('100')]:
            alpha = z / tau
            k = D(m) + (D(l).ln() / alpha).exp()
            measured = (k - n * decimal_s(p, alpha)) / tau
            softmin = -(((1 + (-2 * z).exp()) / 2).ln()) / z
            assert abs(measured - D(e) - softmin) < 10*tau
            near_infinity.append(dict(tau=str(tau), z=str(z), alpha=str(alpha),
                measured_NS_slope=str(measured), limiting_prediction=str(D(e) + softmin),
                minimum_coverage=a, mean_coverage=c / l))
        infinite_slope = (D(m + 1) - n * decimal_s(p, None)) / tau
        assert abs(infinite_slope - (e + a)) < 10 * tau
    pd.DataFrame(decimal_rows).to_csv(OUT / 'high_precision_coefficients.csv', index=False)
    pd.DataFrame(near_one).to_csv(OUT / 'shannon_boundary_layer.csv', index=False)
    pd.DataFrame(near_infinity).to_csv(OUT / 'map_boundary_layer.csv', index=False)
    pd.DataFrame(coarse_tie).to_csv(OUT / 'coarse_tie_high_precision.csv', index=False)
    for z in ['0.1', '1', '3']:
        selection = [row for row in near_one if row['z'] == z]
        errors = [abs(D(row['normalized_information_change']) - D(row['limiting_prediction'])) for row in selection]
        assert errors[-1] < D('.3') * errors[0]

    # Fixed disjoint pool: three balanced pairs versus a singleton substitution.
    source_pool = range(6)
    g = nx.path_graph(6)
    g.add_edges_from([(0, 6), (1, 6), (2, 7), (3, 7), (4, 8), (5, 8), (0, 9)])
    allowed = [6, 7, 8, 9]
    pool_rows, pool_spectrum = [], []
    for tau in [.00001, .0001, .001, .01]:
        full = channel(g, tau, source_pool)
        for sensors in combinations(allowed, 3):
            p = project(full, sensors)
            bs = [len(set(g[v]) & set(source_pool)) for v in sensors]
            coverage_min = min(sum(v in sensors for v in g[u]) for u in source_pool)
            information_coefficient = sum(b / 6 * math.log(6 / b) for b in bs if b)
            accuracy_coefficient = (sum(b > 0 for b in bs) - coverage_min) / 6
            mi, acc = score(p, 1), score(p, np.inf)
            pool_rows.append(dict(tau=tau, sensors=' '.join(map(str, sensors)),
                MI=mi, A1=acc, MI_coefficient=information_coefficient,
                accuracy_coefficient=accuracy_coefficient,
                MI_remainder_scaled=(mi - information_coefficient * tau) / (tau**2 * abs(math.log(tau))),
                A_remainder_scaled=(acc - 1/6 - accuracy_coefficient * tau) / tau**2))
            for alpha in [1.5, 2., 3., 10.]:
                baseline = 6**(1/alpha-1)
                coefficient = (sum(b**(1/alpha) for b in bs) - sum(bs)*6**(1/alpha-1))/6
                error = (score(p,alpha)-baseline-coefficient*tau)/tau**2
                assert abs(error) < 100
                pool_spectrum.append(dict(tau=tau, sensors=' '.join(map(str,sensors)),
                    alpha=alpha, S_coefficient=coefficient, residual_over_tau2=error))
        latest = pool_rows[-4:]
        assert max(latest, key=lambda row: row['MI'])['sensors'] == '6 7 8'
        assert max(latest, key=lambda row: row['A1'])['sensors'] != '6 7 8'
    pd.DataFrame(pool_rows).to_csv(OUT / 'disjoint_pool_conflict.csv', index=False)
    pd.DataFrame(pool_spectrum).to_csv(OUT / 'disjoint_pool_spectrum.csv', index=False)

    star_rows = []
    star = nx.star_graph(7)
    for tau in [1e-5, 1e-3]:
        full = channel(star,tau)
        for budget in range(1,7):
            masks = list(combinations(range(8),budget))
            values = [(score(project(full,mask),1), score(project(full,mask),np.inf)) for mask in masks]
            mi = np.array([row[0] for row in values])
            acc = np.array([row[1] for row in values])
            mi_masks = [mask for mask, value in zip(masks,mi) if value >= mi.max()-1e-13]
            accuracy_masks = [mask for mask, value in zip(masks,acc) if value >= acc.max()-1e-13]
            mi_center = budget > 8-budget
            assert all((0 in mask)==mi_center for mask in mi_masks)
            assert all(0 not in mask for mask in accuracy_masks)
            star_rows.append(dict(tau=tau,m=budget,L=8-budget,
                Shannon_prefers_center=mi_center,accuracy_prefers_leaves=True,
                accuracy_regret=float(acc.max()-max(value for mask,value in zip(masks,acc) if mask in mi_masks))))
    pd.DataFrame(star_rows).to_csv(OUT/'star_threshold_checks.csv',index=False)

    boundary = []
    for l in [3, 6, 8, 9, 12, 13, 98, 10**3, 10**6, 10**12, 10**30, 10**100]:
        ac = crossing(l)
        approximation = float(lambertw(math.log(l) / math.log(2)).real / math.log(l))
        boundary.append(dict(L=str(l), alpha_c=ac, alpha_c_minus_one=ac-1,
            large_L_equivalent=approximation, ratio=(ac-1)/approximation))
    pd.DataFrame(boundary).to_csv(OUT / 'crossover_boundary.csv', index=False)
    summary = dict(graphs=[name for name, _, _ in graphs], finite_alpha_conditions=len(rows),
        regular_class_checks=len(optima), earliest_regular_checks_passed=True,
        fractional_conditions=len(fractional), high_precision_decimal_digits=90,
        decimal_uniformization_tail_bound='1e-85',
        decimal_conditions=len(decimal_rows), shannon_boundary_layer_cases=len(near_one),
        map_boundary_layer_cases=len(near_infinity), disjoint_pool_conditions=len(pool_rows),
        decimal_vs_double_max_channel_error=decimal_crosscheck_error,
        coarse_tie_high_precision_cases=len(coarse_tie),disjoint_finite_alpha_conditions=len(pool_spectrum),
        star_threshold_conditions=len(star_rows),
        disjoint_conflict_all_ages=True, largest_scaled_NS_residual=max_scaled_error,
        runtime_seconds=time.time()-started,
        assertions='All passed; numerical verification supplements the analytic proofs.')
    (OUT / 'verification_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
