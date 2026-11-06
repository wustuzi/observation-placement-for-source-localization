# Observation placement for source localization

Code for the article *Task-dependent observation placement for source localization across spreading scales* (Qiao Ke, submitted to Physical Review E).

The article compares two criteria for choosing which nodes to observe when locating the source of continuous-time susceptible–infected (SI) spreading on a known network: source mutual information and Bayes-optimal identification accuracy.

This repository contains the key code for every result in the paper. The numerical data (graph inputs, exact channels, Monte Carlo report counts and result tables) are available from the author upon reasonable request.

## Contents

| Folder | Paper content | Main files |
|---|---|---|
| `theory/` | Sec. III: exact numerical checks of the early-time expansions on small graphs and of the Arimoto crossover order | `verify_retained_theory.py`, `verify_spectrum.py`, `verify_cycle_fine_boundary.py`, `early_time.py` |
| `small_networks/` | Sec. IV: exact channels on 96 small graphs, accuracy loss and $R_{\rm gain}$ (Fig. 2), placement rules (Table III), independent 300-graph cohort | `run_exact.py` (full-state SI channel), `run_geometry.py`, `analyze_relative.py`, `analyze_baselines_topk.py`, `run_cohort.py`, `analyze_cohort.py` |
| `cycles/` | Sec. V.A, Fig. 3: deterministic Poisson-front channels on cycles, crossover location, high-precision sign checks | `run_cycles.py`, `check_high_precision.py` |
| `heterogeneous/` | Sec. V.B, Fig. 4: first-passage Monte Carlo channels on ER, BA and WS networks, degree-matched control, report-frequency decomposition | `fpp_mc.cpp` (C++ backend), `run.py`, `statistics.py`, `summarize.py`, `degree_matched.py`, `validate.py`, `analyze_coverage.py` |
| `greedy/` | Sec. V.C, Table IV, Appendix F: greedy designs at $N=20$–$200$, mechanism checks, and the new-graph test | `medium_engine.cpp` (C++ engine), `run_medium_scale.py`, `analyze_formal.py`, `pair_matrix.py`, `analyze_single_node.py`, `analyze_overlap.py`, `prospective_test.py`, `score_prospective.py`, independent checks `formal_independent.py`, `formal_greedy_check.py` |
| `figures/` | Figs. 1–6 drawn from saved results | `make_figures_1_4.py`, `make_figure_5.py`, `make_figure_6.py` |

`greedy/PREDICTIONS.md` records the predictions and pass criteria of the new-graph test, written before the test graphs were generated; `PREDICTIONS.sha256` is its checksum, and `score.json` and `score_output.txt` give the scored outcome.

## Requirements

Python 3 with the packages in `requirements.txt`. The two C++ programs need a C++17 compiler (`clang++ -O3 -std=c++17` was used); the Python drivers compile them.

## Notes

- The scripts are provided as they were used. Their input and output paths refer to the layout of the original project, so running them requires the data and adjusting those paths.
- Small-graph and cycle results are deterministic probability calculations. Results on heterogeneous networks and in the greedy comparison are Monte Carlo estimates; the estimators and uncertainty calculations are described in Appendices E and F of the paper.

## Contact

Qiao Ke, International Research Center for Complexity Sciences, Hangzhou International Innovation Institute, Beihang University. qiaoke.cn@gmail.com
