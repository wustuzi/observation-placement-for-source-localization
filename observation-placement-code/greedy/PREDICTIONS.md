# Prospective test of the single-node and overlap formulas

Written before any new graph was generated. Do not edit after the run starts; report failures as failures.

## Setup
- New graphs: BA and ER, requested N = 30, 70, 150, 400 (none used before), 10 graphs per family and size; seed base 2260000.
- Stage nu = 1 only, tau inverted from 2048 realizations. p_uv from 4096 realizations.
- Budgets: m/N = 5%, 10%, 20% (rounded, at least 2).
- Sets built from pair quantities only, with no simulation of joint reports:
  - T_I: the m nodes with the largest single-node information I_v.
  - T_L: greedy minimum of the added overlap loss L (formula in analyze_overlap.py), random first node.
  - R: one uniformly random set (control).
- Evaluation on new samples: 2^14 realizations for the MAP rule, 2^14 held out; plug-in information on all 2^15. Cells with unseen > 5% are excluded and counted.
- Interpolations in log N come from the formal run.

## Predictions
**P1. Rank correlation of I_v and degree (family mean), pass if within +-0.25 and sign as stated.**

| | N=30 | 70 | 150 | 400 |
|---|---|---|---|---|
| BA | -0.41 (negative) | 0.33 (positive) | 0.76 | at least 0.58 |
| ER | -0.49 (negative) | 0.42 (positive) | 0.95 | at least 0.70 |

**P2. Expansion.** Per-graph correlation of N I_v with the expansion has minimum at least 0.95 for N >= 70 and at least 0.90 for N = 30.

**P3. Direction.** Where T_I has overlap larger than T_L by at least 0.2 at N >= 70:
- N A(T_L) > N A(T_I) in at least 90% of graph-budget cases;
- I(T_I) > I(T_L) in at least 90% of graph-budget cases.

**P4. Size.** Over T_L versus T_I pairs at all sizes: N D against the difference in sum L has r >= 0.90 and through-origin slope in [0.45, 0.75]; the ER slope exceeds the BA slope.

**P5. Level.** Mean of N A - (m + 1 - sum L) over all three set types, per size and family, lies in [-1.0, 1.5].

**P6. Below random.** BA, N >= 150, m/N = 10%: N A(T_I) < N A(R) on at least 80% of graphs.

## Not predicted
Nothing here tests rho = 0.1, where the formulas are not supported. No pass/fail threshold will change after the run.
