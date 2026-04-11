# Volition Phase 1 — Benchmark Summary

## γ sweep

| γ | last aligned sim | last attractor sim | avg resistance (attractor) |
|---|------------------|--------------------|----------------------------|
| 0.00 | 1.000 | -0.021 | 0.947 |
| 0.20 | 1.000 | 0.774 | 0.973 |
| 0.40 | 1.000 | 0.907 | 0.978 |
| 0.60 | 1.000 | 0.972 | 0.981 |
| 0.80 | 1.000 | 0.996 | 0.983 |

## Candidate-selection
- correct: 10/10
- accuracy: 1.00

## Verdict: **STRONG**

Criteria:
- STRONG: γ=0.6 attractor_end > 0.5 AND γ=0.0 attractor_end < 0.3 AND correct ≥ 7
- MODERATE: gap between low/high γ > 0.1 AND correct ≥ 7
- WEAK: otherwise

Measured:
- γ=0.0 attractor_end = -0.021
- γ=0.6 attractor_end = 0.972
- γ=0.8 attractor_end = 0.996
- gap (γ=0.8 − γ=0.0) = 1.016
