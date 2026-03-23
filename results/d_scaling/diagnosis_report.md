# Diagnosi: Transizione Non Monotona della Composizionalità

## Scoperta Fondamentale

**Non esiste nessuna transizione di fase.** La "transizione" D=1024:33% → D=2048:0% → D=8192:42% era un **artefatto del singolo seed 1042**.

## Evidenza

### Multi-seed averaging (10 seed per D)

| D | Top-1 (media±std) | Top-3 | Top-5 |
|---|-------------------|-------|-------|
| 1024 | 12.5% ± 15.5% | 30.8% | 58.3% |
| 2048 | 10.8% ± 12.4% | 31.7% | 65.0% |
| 4096 | 15.0% ± 11.7% | 40.8% | 57.5% |
| 6144 | 7.5% ± 15.1% | 40.0% | 67.5% |
| 8192 | 15.0% ± 17.4% | 38.3% | 60.0% |

La composizionalità Top-1 è **piatta intorno al 10-15%** con deviazione standard che supera la media. La varianza è dominata dal seed, non da D.

### Seed sensitivity (10 seed a D=2048)

Top-1 a D=2048 varia da 0% (seed 1042) a **67%** (seed 1044, 1047). La stessa varianza esiste a tutti i D.

### Binary vs Real search (pure HDC, no dynamics)

Binary e real danno risultati identici (±0%). La conversione to_binary() non è responsabile.

## Causa Radice

Il test di composizionalità usa **solo 8 vettori** (4 ruoli + 4 valori). Dopo unbinding, la similarità con il valore corretto è ~0.01 — appena sopra il rumore. Con 8 candidati tutti a ~0.01 di similarità, il nearest è quasi casuale.

Il 33% a D=1024 con seed=1042 era una **coincidenza statistica**, non una proprietà di D=1024.

## Implicazione

Per migliorare la composizionalità serve:
1. Un vocabolario di test MOLTO più grande (50-100 valori) per separare segnale da rumore
2. Record con più role-value pairs (bundle di più binding) per un segnale più forte
3. Mantenere la ricerca in spazio reale e aumentare D per abbassare il noise floor

## Correzione al Report Precedente

Il report `results/d_scaling/report.md` che descriveva una "transizione di fase" è **errato**. La composizionalità non dipende significativamente da D con il setup di test attuale (8 vettori). Aumentare D non risolve il problema — serve un test design migliore.
