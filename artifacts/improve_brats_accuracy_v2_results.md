# Честные рычаги поверх n_subclasses=12 (BraTS has_tumor)

| Рычаг | val_accuracy | fit(), с |
|---|---|---|
| 1_filter_dependent | 0.6651 | 63.4 |
| 2_freeze_basis_auto | 0.6651 | 57.6 |
| 3_center_windowed | 0.6651 | 57.3 |
| 4_stratified_sampling | 0.6555 | 65.5 |
| 5_more_data_150pc | 0.6679 | 247.0 |
| baseline (для сравнения) | 0.6651 | — |
| combined ({}) | 0.6679 | 262.4 |

**Финал на test**: до новых рычагов 0.6447 (0.6563 со сглаживанием) -> после 0.5992 (0.6002 со сглаживанием).