# Project 1 Experiment Summary

| Run | Model | Experiment | Best Dev Acc | Test Acc | Test Loss |
| --- | --- | --- | ---: | ---: | ---: |
| mlp_baseline | MLP | baseline | 0.9336 | 0.9380 | 1.2288 |
| cnn_baseline | CNN | baseline | 0.9735 | 0.9749 | 0.0820 |
| mlp_dropout | MLP | dropout | 0.9364 | 0.9400 | 0.6240 |
| cnn_dropout | CNN | dropout | 0.9721 | 0.9738 | 0.0937 |

## Quick Comparisons

- `cnn_baseline` vs `mlp_baseline`: test accuracy delta +0.0369
- `mlp_dropout` vs `mlp_baseline`: test accuracy delta +0.0020
- `cnn_dropout` vs `mlp_baseline`: test accuracy delta +0.0358
