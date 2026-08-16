## Table 2: Architecture Ablation (exit=40%)

| Variant | AUC | F1 | Acc | ΔAUC | Params |
|---------|-----|----|----|------|--------|
| **Transformer (full)** | 0.996 | 0.944 | 96.1% | ref | 803,585 |
| No CLS token (mean pool) | 0.998 | 0.967 | 97.7% | +0.002 | 803,457 |
| No positional encoding | 0.998 | 0.968 | 97.9% | +0.003 | 803,585 |
| No context features | 0.996 | 0.956 | 97.0% | +0.001 | 803,585 |
| LSTM | 0.993 | 0.939 | 95.8% | -0.003 | 274,305 |