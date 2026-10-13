# Venom-GNN experiment summary

Test-node results, mean over seeds. Cell = F1 +/- std (ROC-AUC).
Synthetic data: simulated experiments, not real-world performance.

## attack

| setting | rf_behaviour | rf_graph | gcn_behaviour | gcn_graph |
|---|---|---|---|---|
| subtle/random | 0.36 +/- 0.07 (0.87) | 0.37 +/- 0.12 (0.87) | 0.25 +/- 0.08 (0.77) | 0.33 +/- 0.07 (0.73) |
| subtle/hub | 0.45 +/- 0.06 (0.89) | 0.97 +/- 0.03 (1.00) | 0.73 +/- 0.23 (0.94) | 0.68 +/- 0.21 (0.89) |
| subtle/clustered | 0.53 +/- 0.12 (0.89) | 0.54 +/- 0.12 (0.92) | 0.52 +/- 0.17 (0.91) | 0.56 +/- 0.23 (0.92) |
| obvious/random | 0.99 +/- 0.02 (1.00) | 0.98 +/- 0.04 (1.00) | 0.71 +/- 0.03 (0.98) | 0.78 +/- 0.14 (0.97) |
| obvious/hub | 1.00 +/- 0.00 (1.00) | 1.00 +/- 0.00 (1.00) | 0.85 +/- 0.08 (0.98) | 0.76 +/- 0.14 (0.96) |
| obvious/clustered | 0.99 +/- 0.02 (1.00) | 0.98 +/- 0.03 (1.00) | 0.83 +/- 0.11 (0.96) | 0.88 +/- 0.07 (0.96) |

_n = 5 seeds per cell._

## topology

| setting | rf_behaviour | rf_graph | gcn_behaviour | gcn_graph |
|---|---|---|---|---|
| barabasi_albert | 0.36 +/- 0.07 (0.87) | 0.37 +/- 0.12 (0.87) | 0.25 +/- 0.08 (0.77) | 0.33 +/- 0.07 (0.73) |
| erdos_renyi | 0.33 +/- 0.19 (0.87) | 0.39 +/- 0.10 (0.86) | 0.33 +/- 0.10 (0.79) | 0.34 +/- 0.13 (0.76) |
| watts_strogatz | 0.33 +/- 0.10 (0.87) | 0.37 +/- 0.15 (0.87) | 0.21 +/- 0.05 (0.66) | 0.13 +/- 0.12 (0.60) |

_n = 5 seeds per cell._

## poison_rate

| setting | rf_behaviour | rf_graph | gcn_behaviour | gcn_graph |
|---|---|---|---|---|
| 5% | 0.39 +/- 0.16 (0.93) | 0.36 +/- 0.14 (0.93) | 0.22 +/- 0.07 (0.78) | 0.21 +/- 0.21 (0.78) |
| 10% | 0.36 +/- 0.07 (0.87) | 0.37 +/- 0.12 (0.87) | 0.25 +/- 0.08 (0.77) | 0.33 +/- 0.07 (0.73) |
| 20% | 0.64 +/- 0.06 (0.91) | 0.62 +/- 0.06 (0.91) | 0.47 +/- 0.03 (0.75) | 0.46 +/- 0.07 (0.75) |

_n = 5 seeds per cell._

## label_noise

| setting | rf_behaviour | rf_graph | gcn_behaviour | gcn_graph |
|---|---|---|---|---|
| 0% | 0.36 +/- 0.07 (0.87) | 0.37 +/- 0.12 (0.87) | 0.25 +/- 0.08 (0.77) | 0.33 +/- 0.07 (0.73) |
| 10% | 0.41 +/- 0.16 (0.78) | 0.43 +/- 0.19 (0.79) | 0.20 +/- 0.12 (0.65) | 0.22 +/- 0.05 (0.63) |
| 20% | 0.35 +/- 0.11 (0.76) | 0.32 +/- 0.04 (0.73) | 0.24 +/- 0.05 (0.65) | 0.25 +/- 0.05 (0.62) |
| 30% | 0.35 +/- 0.08 (0.77) | 0.33 +/- 0.10 (0.80) | 0.25 +/- 0.10 (0.65) | 0.23 +/- 0.08 (0.64) |

_n = 5 seeds per cell._

## Paired comparison

| sweep | setting | comparison | mean diff | std | wins / n |
|---|---|---|---|---|---|
| attack | subtle/random | gcn_behaviour - rf_behaviour (f1) | -0.114 | 0.077 | 1/5 |
| attack | subtle/hub | gcn_behaviour - rf_behaviour (f1) | +0.282 | 0.185 | 5/5 |
| attack | subtle/clustered | gcn_behaviour - rf_behaviour (f1) | -0.004 | 0.118 | 2/5 |
| attack | obvious/random | gcn_behaviour - rf_behaviour (f1) | -0.282 | 0.049 | 0/5 |
| attack | obvious/hub | gcn_behaviour - rf_behaviour (f1) | -0.155 | 0.081 | 0/5 |
| attack | obvious/clustered | gcn_behaviour - rf_behaviour (f1) | -0.160 | 0.128 | 0/5 |
| topology | barabasi_albert | gcn_behaviour - rf_behaviour (f1) | -0.114 | 0.077 | 1/5 |
| topology | erdos_renyi | gcn_behaviour - rf_behaviour (f1) | -0.002 | 0.144 | 2/5 |
| topology | watts_strogatz | gcn_behaviour - rf_behaviour (f1) | -0.120 | 0.129 | 1/5 |
| poison_rate | 5% | gcn_behaviour - rf_behaviour (f1) | -0.170 | 0.170 | 1/5 |
| poison_rate | 10% | gcn_behaviour - rf_behaviour (f1) | -0.114 | 0.077 | 1/5 |
| poison_rate | 20% | gcn_behaviour - rf_behaviour (f1) | -0.169 | 0.079 | 0/5 |
| label_noise | 10% | gcn_behaviour - rf_behaviour (f1) | -0.216 | 0.165 | 0/5 |
| label_noise | 20% | gcn_behaviour - rf_behaviour (f1) | -0.105 | 0.082 | 0/5 |
| label_noise | 0% | gcn_behaviour - rf_behaviour (f1) | -0.114 | 0.077 | 1/5 |
| label_noise | 30% | gcn_behaviour - rf_behaviour (f1) | -0.093 | 0.102 | 1/5 |
| attack | subtle/random | gcn_graph - rf_graph (f1) | -0.044 | 0.128 | 2/5 |
| attack | subtle/hub | gcn_graph - rf_graph (f1) | -0.292 | 0.197 | 0/5 |
| attack | subtle/clustered | gcn_graph - rf_graph (f1) | +0.018 | 0.179 | 2/5 |
| attack | obvious/random | gcn_graph - rf_graph (f1) | -0.198 | 0.124 | 0/5 |
| attack | obvious/hub | gcn_graph - rf_graph (f1) | -0.235 | 0.138 | 0/5 |
| attack | obvious/clustered | gcn_graph - rf_graph (f1) | -0.097 | 0.096 | 1/5 |
| topology | barabasi_albert | gcn_graph - rf_graph (f1) | -0.044 | 0.128 | 2/5 |
| topology | erdos_renyi | gcn_graph - rf_graph (f1) | -0.054 | 0.115 | 2/5 |
| topology | watts_strogatz | gcn_graph - rf_graph (f1) | -0.234 | 0.157 | 0/5 |
| poison_rate | 5% | gcn_graph - rf_graph (f1) | -0.153 | 0.085 | 0/5 |
| poison_rate | 10% | gcn_graph - rf_graph (f1) | -0.044 | 0.128 | 2/5 |
| poison_rate | 20% | gcn_graph - rf_graph (f1) | -0.160 | 0.125 | 0/5 |
| label_noise | 10% | gcn_graph - rf_graph (f1) | -0.213 | 0.162 | 0/5 |
| label_noise | 20% | gcn_graph - rf_graph (f1) | -0.067 | 0.072 | 1/5 |
| label_noise | 0% | gcn_graph - rf_graph (f1) | -0.044 | 0.128 | 2/5 |
| label_noise | 30% | gcn_graph - rf_graph (f1) | -0.101 | 0.130 | 1/5 |
