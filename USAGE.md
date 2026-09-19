# Полная инструкция по использованию subspace_conjugacy

Это подробное руководство «как сделать X», в отличие от [`README.md`](README.md)
(обзор проекта и статус фаз рефакторинга). Все примеры ниже реально
выполнены и проверены на этой библиотеке — при копировании кода
последовательность вызовов гарантированно работает как описано.

## Содержание

1. [Установка](#1-установка)
2. [Ключевые понятия](#2-ключевые-понятия)
3. [Быстрый старт за 5 минут (синтетические данные)](#3-быстрый-старт-за-5-минут-синтетические-данные)
4. [Работа с данными и DatasetConfig](#4-работа-с-данными-и-datasetconfig)
5. [Препроцессинг сырых изображений (NB1-NB2)](#5-препроцессинг-сырых-изображений-nb1-nb2)
6. [Векторизация (NB3)](#6-векторизация-nb3)
7. [Кластеризация одного класса (Фазы A+B)](#7-кластеризация-одного-класса-фазы-ab)
8. [Классификация (Фаза C)](#8-классификация-фаза-c)
9. [Оценка качества](#9-оценка-качества)
10. [Сохранение и загрузка](#10-сохранение-и-загрузка)
11. [Оркестратор FursovPipeline (рекомендуемый путь)](#11-оркестратор-fursovpipeline-рекомендуемый-путь)
12. [Legacy/parity-режим — сверка с ноутбуками](#12-legacyparity-режим--сверка-с-ноутбуками)
13. [Справочник гиперпараметров](#13-справочник-гиперпараметров)
14. [Типичные проблемы](#14-типичные-проблемы)
15. [Куда дальше](#15-куда-дальше)

---

## 1. Установка

```bash
cd mission_6states
pip install -e .            # библиотека + рантайм-зависимости
pip install -e ".[dev]"     # + pytest, black, ruff, mypy
```

Проверка:

```python
import subspace_conjugacy as sc
print(sc.__version__)
```

Зависимости: Python ≥3.10, NumPy, scikit-learn, OpenCV (headless), Pillow —
все ставятся автоматически из `pyproject.toml`.

## 2. Ключевые понятия

| Термин | Что это |
|---|---|
| **Класс** | Патология: glioma / meningioma / pituitary. |
| **Подкласс** | Подпространство внутри класса (по умолчанию 8 на класс) — не то же самое, что класс. |
| **Базис подкласса `Y_s`** | Матрица `(N, k)`, k обычно 2 — "координатная система" подкласса. |
| **`R(x, Y)`** | Показатель сопряжённости, `[0, 1]` — насколько хорошо x объясняется подпространством Y. Чем больше, тем "больше x похож на Y". |
| **`cos(a, b)`** | Косинусное сходство, `[-1, 1]`. |
| **N** | Размерность вектора признаков (65536 для картинок 256×256). |
| **Канон** | Единственная правильная реализация алгоритма в `algorithms/` (не путать с legacy). |
| **Legacy** | Код в `algorithms/legacy/`, воспроизводящий алгоритм КАК ОН БУКВАЛЬНО написан в ноутбуках (включая расхождения с каноном) — только для сверки, не для продакшена. |

**Важно:** `R(x, Y)` и `cos(a, b)` **масштабно-инвариантны**:
`R(αx, Y) = R(x, Y)` для любого `α > 0`. Метод различает объекты по
**направлению** вектора признаков, а не по его величине. Если вы
конструируете свои тестовые данные — не разносите классы константным
сдвигом вдоль одной оси, иначе метод не сможет их различить (см. [§14](#14-типичные-проблемы)).

## 3. Быстрый старт за 5 минут (синтетические данные)

```python
import numpy as np
from subspace_conjugacy import SubspaceConjugacyClassifier

np.random.seed(0)

# Данные должны различаться НАПРАВЛЕНИЕМ (см. §2), не только сдвигом
n_features = 64
directions = np.random.randn(3, n_features)
directions /= np.linalg.norm(directions, axis=1, keepdims=True)

X_list, y_list = [], []
for idx, cls in enumerate(["glioma", "meningioma", "pituitary"]):
    magnitudes = np.random.uniform(8, 12, size=(40, 1))
    noise = np.random.randn(40, n_features) * 0.5
    X_list.append(magnitudes * directions[idx] + noise)
    y_list.append(np.full(40, cls))

X, y = np.vstack(X_list), np.concatenate(y_list)

clf = SubspaceConjugacyClassifier(n_subclasses=4, freeze_basis_at=2)
clf.fit(X, y)

print(clf.predict(X[:5]))          # предсказанные классы
print(clf.predict_proba(X[:5]))    # вероятности (softmax по R)
```

Дальше — то же самое, но на настоящих МРТ-изображениях, по шагам.

## 4. Работа с данными и DatasetConfig

`DatasetConfig` описывает, где лежат файлы на каждой стадии пайплайна:

```python
from subspace_conjugacy.config import DatasetConfig

config = DatasetConfig(
    root="data",                                       # корень датасета
    classes=["glioma", "meningioma", "pituitary"],
    n_subclasses=8,
    subclass_factor=2,       # k — векторов в базисе подкласса
    image_size=(256, 256),
    test_samples_per_class=25,
)

config.paths["glioma"]["raw"]        # data/glioma_raw
config.paths["glioma"]["resized"]    # data/2_glioma_resize
config.paths["glioma"]["centered"]   # data/3_glioma_centered
config.paths["glioma"]["vectors"]    # data/5_all_vectors/glioma

config.create_directories()          # создать всю структуру папок сразу
```

**⚠ Важный нюанс.** Если у вас уже есть готовый датасет с именами папок,
не совпадающими с этой схемой (например, просто `glioma_centered/` без
префикса `3_`, как в `datasets/` этого репозитория) — `DatasetConfig` его
не найдёт автоматически. Два варианта:

```python
# Вариант A: переименовать/пересоздать структуру под DatasetConfig
# (см. §5 — препроцессинг сам создаёт нужные папки).

# Вариант B: загрузить в обход DatasetConfig, напрямую по своим путям —
# именно так сделано в main.py этого репозитория:
from pathlib import Path
from subspace_conjugacy import load_and_vectorize_batch

paths = sorted(Path("datasets/glioma_centered").glob("*.png"))
X_glioma = load_and_vectorize_batch(paths, method="horizontal")
# Дальше X_glioma можно скармливать FursovClusterer/FursovPipeline.run_stage
# напрямую, минуя стадию "vectorize" — DatasetConfig всё ещё пригодится
# для путей экспорта базисов (get_subclass_bases_path и т.п.).
```

## 5. Препроцессинг сырых изображений (NB1-NB2)

Нужен, если на входе снимки произвольного размера/формата (как
Kaggle "Brain Tumor MRI Dataset" — JPG от 200×200 до 900×741, вперемешку
RGB/grayscale). Если у вас уже есть готовые 256×256 отцентрированные PNG —
пропустите этот раздел и переходите к [§6](#6-векторизация-nb3).

### 5.1. Поэлементно (in-memory, для одного изображения)

```python
import cv2
from subspace_conjugacy import resize_image, suppress_background, center_image

image = cv2.imread("raw/glioma1.jpg", cv2.IMREAD_COLOR)   # произвольный размер
resized = resize_image(image, size=(256, 256))
centered = center_image(resized, background_threshold=10, min_shift=2)
# centered.shape == (256, 256, 3)
```

### 5.2. Один объект на всё (ImagePreprocessor)

```python
from subspace_conjugacy import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(256, 256), background_threshold=10)

# В памяти:
processed = preprocessor.process(image)

# Файл -> файл:
preprocessor.process_file("raw/glioma1.jpg", "out/glioma1.png")

# Директория -> директория (без промежуточных файлов, перенумерует выход):
preprocessor.process_directory("raw_dir/", "out_dir/", output_prefix="glioma", pattern="*.jpg")
```

### 5.3. Через DatasetConfig (двухстадийно, как в оригинальных ноутбуках)

```python
config = DatasetConfig(root="data", classes=["glioma", "meningioma", "pituitary"])
config.create_directories(stages=["raw"])
# ... скопируйте сырые .jpg в config.paths["glioma"]["raw"] и т.д. ...

preprocessor = ImagePreprocessor()
# raw/ -> resized/ (NB1) -> centered/ (NB2), файлы пронумерованы glioma1.png...
preprocessor.process_class_via_config("glioma", config, pattern="*.jpg")
```

Не забудьте про класс `"test"` — та же логика:
`preprocessor.process_class_via_config("test", config, pattern="*.jpg")`.

## 6. Векторизация (NB3)

```python
from subspace_conjugacy import vectorize_image, load_and_vectorize_batch
from subspace_conjugacy.features.extraction import extract_class_vectors

# Один массив:
vector = vectorize_image(processed, method="horizontal")   # (65536,)

# Список файлов:
X = load_and_vectorize_batch(sorted(Path("out_dir").glob("*.png")), method="horizontal")

# Через DatasetConfig (нужен ТОЧНЫЙ count, иначе падает — см. §14):
X_glioma = extract_class_vectors(config, "glioma", stage="centered", count=100)
```

`method="vertical"` даёт столбцовую развёртку (`ravel(order="F")`) —
используется реже, но поддержана везде симметрично с "horizontal".

## 7. Кластеризация одного класса (Фазы A+B)

### 7.1. Рекомендуемый способ — FursovClusterer (канон, фасад)

```python
from subspace_conjugacy import FursovClusterer

clusterer = FursovClusterer(
    n_subclasses=8,
    freeze_basis_at=2,        # k — размер базиса для классификатора
    growth_strategy="default",  # или "master" (ratio к среднему, NB7)
    reg_param=1e-8,
)
clusterer.fit(X_glioma)

clusterer.subspaces_             # список из 8 матриц (N, 2)
clusterer.labels_                # (M,) — номер подкласса на объект
clusterer.get_subclass_sizes()   # np.array([...]) — сколько объектов в каждом подклассе
clusterer.get_initial_pair()     # (idx1, idx2) — результат фазы A.1
clusterer.get_center_indices()   # 8 индексов — результат фазы A.2-A.3
clusterer.get_initial_pairs()    # (8, 2) — результат фазы B.1
```

⚠ **`get_subclass_sizes()` почти всегда покажет сильный дисбаланс**
(например, `[2, 39, 2, 2, 3, 6, 2, 4]`) — это ожидаемое поведение
жадного алгоритма фазы B.2, не баг. Подробнее — [§14](#14-типичные-проблемы).

### 7.2. По шагам (для исследования/отладки одной фазы)

```python
from subspace_conjugacy.algorithms import (
    GlobalMinCosinePairFinder,
    ReferenceCenterBuilder,
    CosineSecondVectorAttacher,
    ConjugacyClusterGrowth,
)

# A.1
finder = GlobalMinCosinePairFinder()
finder.fit(X_glioma)
pair = finder.pair_indices_

# A.2-A.3
builder = ReferenceCenterBuilder(n_subclasses=8)
builder.fit(X_glioma, initial_pair=pair)
centers = builder.center_indices_

# B.1
attacher = CosineSecondVectorAttacher()
attacher.fit(X_glioma, centers)
pairs = attacher.pairs_

# B.2
growth = ConjugacyClusterGrowth(freeze_basis_at=2, strategy="default")
growth.fit(X_glioma, pairs)
subspaces, labels = growth.subspace_bases_, growth.labels_
```

Все `.fit()` в стиле sklearn — возвращают `self`, результат смотрите в
атрибутах с `_` на конце.

## 8. Классификация (Фаза C)

### 8.1. Обучение целиком на сырых векторах

```python
from subspace_conjugacy import SubspaceConjugacyClassifier

clf = SubspaceConjugacyClassifier(n_subclasses=8, freeze_basis_at=2)
clf.fit(X_train, y_train)   # X_train — все классы вместе, y_train — метки
```

Внутри для каждого класса независимо запускается `FursovClusterer` — это
единственный путь кластеризации (не переиспользуйте `models.clusterer` —
это лишь алиас на `FursovClusterer` для обратной совместимости).

### 8.2. Способы предсказания

```python
predictions = clf.predict(X_test)                 # метка класса, (M,)
probabilities = clf.predict_proba(X_test)          # softmax(R) по классам, (M, n_classes)
R_per_class = clf.predict_r_matrix(X_test)         # max R по подклассам класса, (M, n_classes)
R_flat = clf.predict_r_matrix_flat(X_test)         # R по КАЖДОМУ подклассу, (M, n_classes*n_subclasses)
flat_idx = clf.predict_subclass(X_test)            # глобальный индекс подкласса, (M,)
confidence = clf.predict_confidence_ratio(X_test)  # NB8: best_R/mean(others)-1, (M,)

# Каким классом владеет предсказанный подкласс (совпадает с predict()):
owner = clf.flat_subclass_labels_[flat_idx]
```

### 8.3. Сборка без переобучения — fit_from_subclass_bases

Полезно, если базисы уже посчитаны отдельно (например, вы кластеризовали
каждый класс сами через `FursovClusterer` и хотите просто собрать
классификатор, либо загрузили базисы из CSV):

```python
subspaces_by_class = {
    "glioma": glioma_clusterer.subspaces_,
    "meningioma": meningioma_clusterer.subspaces_,
    "pituitary": pituitary_clusterer.subspaces_,
}

clf = SubspaceConjugacyClassifier(n_subclasses=8, freeze_basis_at=2)
clf.fit_from_subclass_bases(subspaces_by_class)
```

## 9. Оценка качества

```python
from subspace_conjugacy.evaluation.metrics import evaluate_classifier, confidence_summary

y_pred = clf.predict(X_test)
report = evaluate_classifier(y_test, y_pred)

report["accuracy"]              # float
report["per_class_accuracy"]    # {label: accuracy}
report["confusion_matrix"]      # sklearn confusion_matrix, порядок report["labels"]

confidence_summary(clf.predict_confidence_ratio(X_test))
# {"mean": ..., "min": ..., "max": ..., "fraction_negative": ...}
```

**Реалистичные ожидания по accuracy**: на реальном датасете 3 класса ×
200 изображений это ~61-80% в зависимости от размера holdout (см. README,
раздел «Известные ограничения») — заметно выше случайных 33%, но метод не
идеален, особенно на классе meningioma.

## 10. Сохранение и загрузка

Два независимых способа — выбирайте по ситуации.

### 10.1. Pickle (просто, но версионно хрупко)

```python
from subspace_conjugacy import save_model, load_model

save_model(clf, "artifacts/model.pkl")
clf_loaded = load_model("artifacts/model.pkl")
```

### 10.2. CSV + JSON (переносимо между версиями библиотеки)

```python
from subspace_conjugacy import save_pipeline_artifact, load_pretrained_classifier

config = DatasetConfig(root="artifacts/pipeline", n_subclasses=8, subclass_factor=2)
save_pipeline_artifact(clf, config)   # пишет CSV базисов на класс + JSON метаданных

# На другой машине / через время:
clf_loaded = load_pretrained_classifier(config)  # fit_from_subclass_bases, без переобучения
```

### 10.3. Отдельные CSV-стадии (совместимость с ноутбуками NB3-NB7)

Если нужен доступ к промежуточным артефактам ноутбуков (например, для
сверки или ручного анализа):

```python
from subspace_conjugacy import (
    save_class_vectors, load_class_vectors,               # NB3
    save_initial_pair_indices, load_initial_pair_indices,  # NB4-5
    save_center_indices, load_center_indices,              # A.2-A.3 (NB5)
    save_subclass_pairs, load_subclass_pairs,              # B.1 (NB6)
    save_subclass_bases, load_subclass_bases_as_list,      # Export (NB6-7)
)

# load_subclass_pairs() отдаёт (S, 2) — можно скормить прямо в:
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth
pairs = load_subclass_pairs("glioma", config)
ConjugacyClusterGrowth(freeze_basis_at=2).fit(X_glioma, pairs)
```

## 11. Оркестратор FursovPipeline (рекомендуемый путь)

Для полного сценария "сырые снимки → обученная модель → accuracy" удобнее
не вызывать модули по отдельности, а использовать `FursovPipeline` —
он же используется в тестах на реальном датасете.

```python
from subspace_conjugacy import FursovPipeline
from subspace_conjugacy.config import DatasetConfig

config = DatasetConfig(root="data", classes=["glioma", "meningioma", "pituitary"], n_subclasses=8)
pipeline = FursovPipeline(config)

# 1. NB1+NB2 для каждого класса (и для "test")
for cls in list(config.classes) + ["test"]:
    pipeline.run_preprocessing(cls, raw_pattern="*.jpg")

# 2. NB3 (vectorize) -> канон (cluster) -> NB6-7 (export_subspaces), для всех классов
pipeline.run_all_classes()          # заполняет pipeline.clusterers_

# 3. Фаза C: собрать классификатор
classifier = pipeline.build_classifier()

# 4a. Оценить на тесте с явными метками (рекомендуется):
report = pipeline.classify_test(y_test=y_true)

# 4b. Либо без меток — ТОЛЬКО если тестовая папка физически упорядочена
#     блоками по config.classes, ровно test_samples_per_class на класс
#     (буквальное поведение NB8, хрупкое):
report = pipeline.classify_test()

print(report["report"]["accuracy"])
print(report["predictions"])
```

### 11.1. Точечный вызов одной стадии

```python
X = pipeline.run_stage("vectorize", class_name="glioma", save=False)
clusterer = pipeline.run_stage("cluster", X=X, n_subclasses=8, freeze_basis_at=2)
pipeline.run_stage("export_subspaces", clusterer=clusterer, class_name="glioma")
```

`run_stage` сам подставит `pipeline.config` в стадии, которым он нужен
(`resize`, `center`, `vectorize`, `export_subspaces`) — можно передать
`config=...` явно, если нужен другой конфиг для конкретного вызова.

Полный список стадий: `resize`, `center`, `vectorize`, `global_pair`,
`reference_centers`, `subclass_seed`, `subclass_growth`, `cluster`,
`export_subspaces`, `classify`, `legacy_notebook` (см. `pipeline/stages.py`
за точными сигнатурами каждой).

### 11.2. Комбинирование с ручной загрузкой (для датасетов вне конвенции DatasetConfig)

```python
# X получен в обход DatasetConfig (см. §4, вариант B)
clusterer = pipeline.run_stage("cluster", X=X_glioma, n_subclasses=8, freeze_basis_at=2)
pipeline.run_stage("export_subspaces", clusterer=clusterer, class_name="glioma")
pipeline.clusterers_["glioma"] = clusterer   # чтобы build_classifier() его увидел
```

## 12. Legacy/parity-режим — сверка с ноутбуками

Нужен только для исследовательских/QA-целей — воспроизвести, что
буквально делали ноутбуки (включая их расхождения с каноном). **Не
используйте в продакшене.**

```python
from subspace_conjugacy.algorithms.legacy import (
    compute_per_vector_pairs_nb4,     # NB4: per-vector trio_list
    NotebookStagedPipeline,           # staged NB5->NB6->NB7
)

trio_list = compute_per_vector_pairs_nb4(X_glioma)   # [(i, argmin_j, cos), ...]

# use_canonical_pair=True  -> берёт пару из канона (A.1), затем работает как ноутбуки
# use_canonical_pair=False -> буквально NB4 trio_list + ручной индекс, как в NB5
staged = NotebookStagedPipeline(n_subclasses=8, use_canonical_pair=True)
result = staged.run_full_pipeline(X_glioma)
result["flattened_bases"]   # (16, N) — тот же формат, что 8_glioma_subclasses_vectors.csv
```

Через оркестратор — то же самое стадией `"legacy_notebook"`:
`pipeline.run_stage("legacy_notebook", X=X_glioma, n_subclasses=8)`.

## 13. Справочник гиперпараметров

| Параметр | Где | По умолчанию | Смысл |
|---|---|---|---|
| `n_subclasses` | `FursovClusterer`, `SubspaceConjugacyClassifier`, `DatasetConfig` | 8 | Число подпространств на класс |
| `freeze_basis_at` | `FursovClusterer`, `SubspaceConjugacyClassifier`, `ConjugacyClusterGrowth` | 2 | Размер базиса `k` после кластеризации; `None` — без ограничения (базис растёт вместе с подклассом) |
| `growth_strategy` | `FursovClusterer`, `SubspaceConjugacyClassifier` | `"default"` | `"default"` — argmax R; `"master"` — ratio к среднему по подклассам (NB7) |
| `reg_param` | все алгоритмы + `core.metrics` | `1e-8` | Тихоновская регуляризация `(YᵀY + reg·I)⁻¹` — защита от вырожденности при коллинеарных базисах |
| `subclass_factor` | `DatasetConfig` | 2 | Должно совпадать с `freeze_basis_at` — используется при разборе плоских CSV базисов |
| `image_size` | `DatasetConfig`, `ImagePreprocessor` | `(256, 256)` | Размер после resize |
| `background_threshold` | `ImagePreprocessor`, `suppress_background` | 10 | Порог яркости фона (0-255) |
| `min_shift` | `ImagePreprocessor`, `shift_rows/columns` | 2 | Минимальный `\|delta\|` для применения сдвига при центрировании |
| `method` | векторизация | `"horizontal"` | `"horizontal"` (`flatten`) или `"vertical"` (`ravel('F')`) |

## 14. Типичные проблемы

**`FileNotFoundError: ...glioma31.png`, хотя файлы есть.**
`DatasetConfig.get_image_paths`/`extract_class_vectors` по умолчанию ждут
ровно 100 файлов на класс (или `test_samples_per_class * len(classes)` для
"test") — это зашито под оригинальный датасет ноутбуков. Если у вас другое
количество — передайте `count=` явно:
`extract_class_vectors(config, "glioma", count=30)`. Через
`FursovPipeline.run_stage("vectorize", ...)` эта проблема не возникает —
там `count` определяется автоматически подсчётом реальных файлов.

**Accuracy на моих синтетических данных ~33% (случайный уровень), хотя классы явно разные.**
Скорее всего, классы отличаются только сдвигом вдоль одного направления
(`X += class_idx * 10`), а не разными направлениями — метод
масштабно-инвариантен и не видит такое различие (см. [§2](#2-ключевые-понятия) и
пример в [§3](#3-быстрый-старт-за-5-минут-синтетические-данные)).

**`get_subclass_sizes()` показывает, что один подкласс забрал почти все объекты.**
Ожидаемое поведение (не баг): B.2 — жадный `argmax R` по всем подклассам
сразу, и подкласс с уже бо́льшим базисом обычно объясняет новый вектор не
хуже меньшего — получается положительная обратная связь. Наблюдается и на
синтетике, и на реальных МРТ. Если критично — попробуйте
`growth_strategy="master"` (нормализация на среднее по подклассам, NB7) или
меньшее `n_subclasses`.

**`ValueError: Позиционная разметка теста требует ровно N объектов...`**
Вы вызвали `pipeline.classify_test()` без `y_test`, и число тестовых файлов
не равно `test_samples_per_class * len(classes)`. Либо поправьте
`test_samples_per_class` в `DatasetConfig`, либо (надёжнее) передайте
`y_test` явно.

**`ValueError: ... reference (H, W) не совпадает с image (H, W)` в `suppress_background`.**
`reference` должен быть 2D grayscale той же высоты/ширины, что и `image`
(даже если `image` цветное, `(H, W, 3)`). Получите его через
`cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.

**Обучение одного класса занимает больше минуты.**
Фаза B.2 (`ConjugacyClusterGrowth`) — `O(M²·N)` на чистом numpy (M — число
объектов класса, N — размерность вектора). На 200 изображениях 256×256
это уже заметно; на 60 — единицы секунд. Для экспериментов используйте
подвыборку, для продакшена — считайте это одноразовой offline-стадией
обучения, не частью inference.

## 15. Куда дальше

- [`README.md`](README.md) — обзор проекта, статус фаз рефакторинга, известные ограничения.
- [`refactoring_plan.txt`](refactoring_plan.txt) — полный план с промптами по фазам.
- [`theory/`](theory) — исходные документы с математикой канона.
- [`main.py`](main.py) — рабочий end-to-end пример на реальном датасете.
- `tests/test_pipeline.py`, `tests/test_parity/` — больше живых примеров использования на реальных данных.
