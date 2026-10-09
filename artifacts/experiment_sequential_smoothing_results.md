# Статья 4 — результаты эксперимента (сглаживание смежных срезов)

Сгенерировано `experiments/experiment_sequential_smoothing.py`. Датасет: `datasets/brats_flair_slices/` (BraTS2020, FLAIR, 160 пациентов). Классификатор: `SubspaceConjugacyClassifier(n_subclasses=8)`, обучен на 200 срезах из 128 train-пациентов (fit() заняло 115.4s), оценён на ВСЕХ срезах 32 held-out тестовых пациентов (сплит по пациентам, не по отдельным срезам — см. docstring модуля).

| Конфигурация | n_test | accuracy | tumor | no_tumor |
|---|---|---|---|---|
| meaningful_smoothed | 4177 | 0.6198 | 0.594 | 0.647 |
| meaningful_unsmoothed | 4177 | 0.6220 | 0.595 | 0.650 |
| shuffled_smoothed | 4177 | 0.4846 | 0.609 | 0.353 |

**H1** (сглаживание помогает при содержательном порядке: meaningful_smoothed > meaningful_unsmoothed): 0.6198 > 0.6220 -> **False**.
**H2** (сглаживание случайного порядка не даёт выигрыша: shuffled_smoothed <= meaningful_unsmoothed): 0.4846 <= 0.6220 -> **True**.

`meaningful_smoothed` сглаживается ОТДЕЛЬНО по каждому пациенту (физические границы между пациентами не смешиваются), `shuffled_smoothed` — одним глобальным вызовом на перемешанный массив (контроль, воспроизводит некорректное применение метода).