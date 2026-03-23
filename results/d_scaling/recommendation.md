# Raccomandazione: Dimensionalità Default

## Raccomandazione: D=8192 per composizionalità

Basandosi sullo sweep D=1024→8192:

- **D=8192** porta la composizionalità da 33.3% a **41.7% Top-1** e **100% Top-5**
- Tutte le altre metriche restano invariate (distribuzionale, memoria, anafora, consistenza)
- Performance accettabile (~6.5ms/step, ~1.8ms/encode)

## Trade-off

| Aspetto | D=1024 | D=8192 | Note |
|---------|--------|--------|------|
| Comp Top-1 | 33.3% | 41.7% | +8.4pp |
| Comp Top-5 | 66.7% | 100% | +33pp |
| Sim sinonimi | 0.83 | 0.82 | invariato |
| Step time | 3.7ms | 6.5ms | 1.8x più lento |
| RAM per vettore | 4KB | 32KB | 8x |

## Implementazione suggerita

1. Usare D=8192 come default nel SemanticFieldConfig
2. Mantenere D=1024 disponibile per test rapidi e distributional learning
3. I test composizionali dovrebbero girare a D=8192
4. Il corpus e i test distribuzionali possono restare a D=1024 (nessun beneficio)
