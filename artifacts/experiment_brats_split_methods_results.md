# Сравнение методов train/test-сплита на BraTS (has_tumor)

Сгенерировано `experiments/experiment_brats_split_methods.py`. Классификатор: `SubspaceConjugacyClassifier(n_subclasses=8)` для всех методов.

| Метод | Честный? | n_train | n_test | patient_overlap | accuracy | tumor | no_tumor | fit(), с |
|---|---|---|---|---|---|---|---|---|
| patient_split | ДА | 200 | 4177 | 0.0% | 0.6220 | 0.595 | 0.650 | 127.0 |
| image_random_split | см. текст | 200 | 200 | 61.0% | 0.5900 | 0.400 | 0.780 | 141.6 |
| paired_correlation_split | нет | 200 | 200 | 70.5% | 0.5450 | 0.500 | 0.590 | 95.9 |
| train_equals_test | нет | 200 | 200 | 100.0% | 0.5950 | 0.590 | 0.600 | 127.8 |

- **patient_split** — ЧЕСТНЫЙ ориентир: train/test построены из непересекающихся пациентов, `patient_overlap=0%` по конструкции.
- **image_random_split** — честный по общему стандарту ML (stratified random по меткам), но НЕ учитывает группировку срезов по пациенту — ненулевой `patient_overlap` показывает утечку, специфичную именно для последовательных медицинских данных.
- **paired_correlation_split** — нечестный по конструкции (test подобран похожим на train); на BraTS дополнительно комбинируется с утечкой по пациенту (см. `pair_same_patient_fraction` ниже).
- **train_equals_test** — train == test, `patient_overlap=100%` тривиально.

`paired_correlation_split`: доля пар с совпадающим patient_id = 13.5% (из 200 пар).