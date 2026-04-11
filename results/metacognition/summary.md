# Metacognition Phase 3 — Benchmark Summary

## Scenario A — Parameter adaptation
- pearson(difficulty, alpha_max): -1.000
- pearson(difficulty, gamma_adjustment): 1.000
- alpha_max gap (easy - hard): 0.373

## Scenario B — Self-model maturation
- predict(A) < predict(B) after 100 steps: true
- prediction error first 30 steps: 0.168
- prediction error last 30 steps:  0.197

## Scenario C — Overloaded signal
- Overloaded count: 10/10

## Verdict: **STRONG**

Criteria:
- STRONG: r_alpha < −0.5 AND r_gamma > 0.5 AND predict(A)<predict(B) AND overloaded ≥ 7/10
- MODERATE: r_alpha < −0.3 AND predict(A)<predict(B) AND overloaded ≥ 5/10
- WEAK: otherwise
