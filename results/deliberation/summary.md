# Deliberation Phase 2 — Benchmark Summary

## Scenario A — Difficulty vs cycles

- total scenarios: 30
- achieved:  30
- converged: 0
- suspended: 0
- accuracy (Achieved + Converged): 1.000
- avg final score: 0.936
- pearson(difficulty_idx, cycles_used): 0.824

## Scenario B — Candidate source comparison

| source | accuracy | avg cycles | avg score |
|--------|----------|------------|-----------|
| attraction_only | 1.000 | 1.55 | 0.922 |
| vocabulary_only | 1.000 | 2.00 | 0.041 |
| combined | 1.000 | 1.55 | 0.922 |

## Verdict: **STRONG**

Criteria:
- STRONG: pearson(diff,cycles) > 0.6 AND accuracy > 0.80 AND suspended_rate < 0.20
- MODERATE: pearson(diff,cycles) > 0.3 AND accuracy > 0.60
- WEAK: otherwise

Measured:
- accuracy = 1.000
- pearson(diff,cycles) = 0.824
- suspended rate = 0.000
