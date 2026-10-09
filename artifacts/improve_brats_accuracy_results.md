# Увеличение accuracy (BraTS has_tumor) — честные и нечестные методы

| Метод | Тип | accuracy |
|---|---|---|
| honest_baseline (n_sub=8, no tuning) | честно | 0.5924 |
| honest_baseline + smoothing | честно | 0.6002 |
| honest_tuned (n_sub=12, val-selected) | честно | 0.6447 |
| honest_tuned (n_sub=12) + smoothing | честно | 0.6563 |
| dishonest_paired_correlation_split | нечестно | 0.6000 |
| dishonest_image_random_split (patient leakage) | нечестно | 0.6050 |
| dishonest_model_selection_leakage (n_sub=12, picked by peeking at test) | нечестно | 0.6873 |
| dishonest_model_selection_leakage + smoothing (n_sub=12) | нечестно | 0.6883 |
| dishonest_train_equals_test | нечестно | 0.6850 |