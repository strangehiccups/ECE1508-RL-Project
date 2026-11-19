### Random scenarios (512 drones each, 10 s)

| method | calm success | calm rmse | v1_range success | v1_range rmse | train success | train rmse | strong success | strong rmse |
|---|---|---|---|---|---|---|---|---|
| nominal | 100% | 0.0 cm | 100% | 3.7 cm | 100% | 8.9 cm | 58% | 61.9 cm |
| res_robust | 100% | 0.5 cm | 100% | 0.7 cm | 100% | 1.8 cm | 73% | 19.1 cm |
| res_z0 | 100% | 0.7 cm | 100% | 1.8 cm | 100% | 4.5 cm | 76% | 30.5 cm |
| res_expert | 100% | 0.7 cm | 100% | 1.6 cm | 100% | 4.0 cm | 77% | 28.1 cm |
| res_rma | 100% | 0.7 cm | 100% | 1.6 cm | 100% | 4.1 cm | 75% | 29.9 cm |
| e2e_expert | 100% | 3.0 cm | 100% | 3.2 cm | 100% | 4.0 cm | 11% | 20.8 cm |
| e2e_rma | 100% | 3.5 cm | 100% | 3.7 cm | 99% | 4.8 cm | 2% | 23.1 cm |
| v1_rma | 100% | 3.7 cm | 100% | 12.2 cm | 100% | 37.2 cm | 11% | 110.1 cm |

### Overhead passes (64 drones each, 8 s)

| method | pass_0.75 success | pass_0.75 max_drop | pass_0.5 success | pass_0.5 max_drop | pass_0.25 success | pass_0.25 max_drop | pass_0.15 success | pass_0.15 max_drop | pass_0.25x2 success | pass_0.25x2 max_drop |
|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 100% | 3.2 cm | 100% | 4.3 cm | 100% | 6.4 cm | 100% | 8.2 cm | 100% | 12.2 cm |
| res_robust | 100% | 0.3 cm | 100% | 0.3 cm | 100% | 0.4 cm | 100% | 0.9 cm | 100% | 1.9 cm |
| res_z0 | 100% | 0.3 cm | 100% | 0.6 cm | 100% | 1.4 cm | 100% | 2.3 cm | 100% | 4.0 cm |
| res_expert | 100% | 0.3 cm | 100% | 0.6 cm | 100% | 1.4 cm | 100% | 2.3 cm | 100% | 4.0 cm |
| res_rma | 100% | 0.3 cm | 100% | 0.6 cm | 100% | 1.4 cm | 100% | 2.3 cm | 100% | 4.0 cm |
| e2e_expert | 100% | 1.7 cm | 100% | 1.7 cm | 100% | 1.8 cm | 100% | 1.8 cm | 100% | 1.8 cm |
| e2e_rma | 100% | 2.5 cm | 100% | 2.6 cm | 100% | 2.8 cm | 100% | 3.0 cm | 100% | 3.1 cm |
| v1_rma | 100% | 1.3 cm | 100% | 1.8 cm | 100% | 3.0 cm | 100% | 4.4 cm | 100% | 7.7 cm |

