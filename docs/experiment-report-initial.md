# Venom-GNN experiment report

All numbers come from experiments that were actually run. Test-node results on a
clean held-out test set, mean +/- std over seeds. Synthetic data: simulated
experiments, not evidence of real-world security performance.

## F1-score (ROC-AUC)

| attack | strategy | ratio | gcn | random_forest | rf_neighborhood |
|---|---|---|---|---|---|
| none | - | 0% | 0.52 +/- 0.12 (0.86) | 0.84 +/- 0.05 (0.98) | 0.78 +/- 0.12 (0.98) |
| feature | camouflage | 5% | 0.56 +/- 0.11 (0.86) | 0.87 +/- 0.03 (0.98) | 0.80 +/- 0.08 (0.98) |
| feature | camouflage | 10% | 0.56 +/- 0.12 (0.87) | 0.80 +/- 0.06 (0.98) | 0.84 +/- 0.09 (0.99) |
| feature | camouflage | 20% | 0.49 +/- 0.08 (0.87) | 0.80 +/- 0.08 (0.98) | 0.80 +/- 0.11 (0.98) |
| feature | camouflage | 30% | 0.55 +/- 0.12 (0.88) | 0.80 +/- 0.06 (0.98) | 0.83 +/- 0.09 (0.98) |
| label | - | 5% | 0.60 +/- 0.08 (0.87) | 0.79 +/- 0.06 (0.96) | 0.81 +/- 0.06 (0.96) |
| label | - | 10% | 0.52 +/- 0.14 (0.87) | 0.66 +/- 0.10 (0.96) | 0.73 +/- 0.10 (0.97) |
| label | - | 20% | 0.48 +/- 0.17 (0.82) | 0.65 +/- 0.14 (0.91) | 0.57 +/- 0.16 (0.90) |
| label | - | 30% | 0.39 +/- 0.08 (0.74) | 0.44 +/- 0.20 (0.80) | 0.50 +/- 0.22 (0.83) |
| combined | camouflage | 10% | 0.52 +/- 0.15 (0.88) | 0.72 +/- 0.13 (0.97) | 0.68 +/- 0.05 (0.97) |
| combined | camouflage | 20% | 0.43 +/- 0.07 (0.84) | 0.54 +/- 0.14 (0.92) | 0.48 +/- 0.07 (0.90) |
| combined | camouflage | 30% | 0.41 +/- 0.08 (0.77) | 0.40 +/- 0.11 (0.78) | 0.46 +/- 0.21 (0.78) |

## F1 drop relative to the clean condition (positive = worse)

| attack | strategy | ratio | model | F1 drop | ROC-AUC drop |
|---|---|---|---|---|---|
| feature | camouflage | 5% | gcn | -0.043 | -0.002 |
| feature | camouflage | 5% | random_forest | -0.034 | +0.001 |
| feature | camouflage | 5% | rf_neighborhood | -0.020 | -0.002 |
| feature | camouflage | 10% | gcn | -0.047 | -0.012 |
| feature | camouflage | 10% | random_forest | +0.033 | -0.003 |
| feature | camouflage | 10% | rf_neighborhood | -0.066 | -0.008 |
| feature | camouflage | 20% | gcn | +0.025 | -0.008 |
| feature | camouflage | 20% | random_forest | +0.036 | -0.001 |
| feature | camouflage | 20% | rf_neighborhood | -0.027 | -0.002 |
| feature | camouflage | 30% | gcn | -0.036 | -0.021 |
| feature | camouflage | 30% | random_forest | +0.037 | +0.002 |
| feature | camouflage | 30% | rf_neighborhood | -0.056 | +0.002 |
| label | - | 5% | gcn | -0.087 | -0.010 |
| label | - | 5% | random_forest | +0.044 | +0.017 |
| label | - | 5% | rf_neighborhood | -0.039 | +0.016 |
| label | - | 10% | gcn | -0.002 | -0.007 |
| label | - | 10% | random_forest | +0.172 | +0.020 |
| label | - | 10% | rf_neighborhood | +0.047 | +0.008 |
| label | - | 20% | gcn | +0.036 | +0.036 |
| label | - | 20% | random_forest | +0.186 | +0.073 |
| label | - | 20% | rf_neighborhood | +0.208 | +0.079 |
| label | - | 30% | gcn | +0.126 | +0.122 |
| label | - | 30% | random_forest | +0.399 | +0.183 |
| label | - | 30% | rf_neighborhood | +0.276 | +0.150 |
| combined | camouflage | 10% | gcn | +0.002 | -0.017 |
| combined | camouflage | 10% | random_forest | +0.118 | +0.010 |
| combined | camouflage | 10% | rf_neighborhood | +0.098 | +0.007 |
| combined | camouflage | 20% | gcn | +0.090 | +0.023 |
| combined | camouflage | 20% | random_forest | +0.294 | +0.064 |
| combined | camouflage | 20% | rf_neighborhood | +0.296 | +0.083 |
| combined | camouflage | 30% | gcn | +0.104 | +0.088 |
| combined | camouflage | 30% | random_forest | +0.439 | +0.195 |
| combined | camouflage | 30% | rf_neighborhood | +0.319 | +0.198 |

## Paired comparison

| attack | strategy | ratio | comparison | mean diff | std | wins / n |
|---|---|---|---|---|---|---|
| none | - | 0% | gcn - random_forest (f1) | -0.319 | 0.124 | 0/5 |
| feature | camouflage | 5% | gcn - random_forest (f1) | -0.310 | 0.098 | 0/5 |
| feature | camouflage | 10% | gcn - random_forest (f1) | -0.239 | 0.108 | 0/5 |
| feature | camouflage | 20% | gcn - random_forest (f1) | -0.309 | 0.105 | 0/5 |
| feature | camouflage | 30% | gcn - random_forest (f1) | -0.247 | 0.136 | 0/5 |
| label | - | 5% | gcn - random_forest (f1) | -0.188 | 0.050 | 0/5 |
| label | - | 10% | gcn - random_forest (f1) | -0.145 | 0.199 | 1/5 |
| label | - | 20% | gcn - random_forest (f1) | -0.170 | 0.231 | 1/5 |
| label | - | 30% | gcn - random_forest (f1) | -0.047 | 0.261 | 3/5 |
| combined | camouflage | 10% | gcn - random_forest (f1) | -0.203 | 0.257 | 2/5 |
| combined | camouflage | 20% | gcn - random_forest (f1) | -0.116 | 0.162 | 1/5 |
| combined | camouflage | 30% | gcn - random_forest (f1) | +0.016 | 0.148 | 3/5 |
| none | - | 0% | gcn - rf_neighborhood (f1) | -0.258 | 0.131 | 0/5 |
| feature | camouflage | 5% | gcn - rf_neighborhood (f1) | -0.235 | 0.128 | 0/5 |
| feature | camouflage | 10% | gcn - rf_neighborhood (f1) | -0.278 | 0.084 | 0/5 |
| feature | camouflage | 20% | gcn - rf_neighborhood (f1) | -0.310 | 0.159 | 0/5 |
| feature | camouflage | 30% | gcn - rf_neighborhood (f1) | -0.278 | 0.098 | 0/5 |
| label | - | 5% | gcn - rf_neighborhood (f1) | -0.209 | 0.094 | 0/5 |
| label | - | 10% | gcn - rf_neighborhood (f1) | -0.209 | 0.124 | 0/5 |
| label | - | 20% | gcn - rf_neighborhood (f1) | -0.086 | 0.201 | 2/5 |
| label | - | 30% | gcn - rf_neighborhood (f1) | -0.108 | 0.273 | 2/5 |
| combined | camouflage | 10% | gcn - rf_neighborhood (f1) | -0.162 | 0.193 | 1/5 |
| combined | camouflage | 20% | gcn - rf_neighborhood (f1) | -0.052 | 0.095 | 2/5 |
| combined | camouflage | 30% | gcn - rf_neighborhood (f1) | -0.043 | 0.228 | 3/5 |

_Up to 5 seeds per cell. With small test sets, differences smaller than the std are not reliable._
