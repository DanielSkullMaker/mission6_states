# Статья 4 (нейросетевой подход) — результаты эксперимента

`experiments/experiment_article4_cnn_sequential.py`. Датасет: `datasets/brats_flair_slices/` (BraTS2020, FLAIR, 160 пациентов, тот же сплит по пациентам, что в классическом эксперименте). Архитектура: компактная свёрточная сеть (32->64->128->256 каналов), стек соседних срезов как входные каналы, 96x96, 6 эпох, Adam(lr=0.001).

| Конфигурация | n_test | accuracy | tumor | no_tumor |
|---|---|---|---|---|
| no_context_k1 | 4177 | 0.8518 | 0.842 | 0.862 |
| meaningful_neighbors | 4177 | 0.8449 | 0.913 | 0.773 |
| shuffled_neighbors | 4177 | 0.5638 | 0.565 | 0.563 |

**H1** (контекст соседних срезов помогает: meaningful_neighbors > no_context_k1): 0.8449 > 0.8518 -> **False**.
**H2** (случайные 'соседи' не лучше физически смежных: shuffled_neighbors <= meaningful_neighbors): 0.5638 <= 0.8449 -> **True**.