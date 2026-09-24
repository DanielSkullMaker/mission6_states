# Subspace Conjugacy Method for Brain Tumor Classification

Библиотека `subspace_conjugacy` реализует метод Фурсова (подпространственная
сопряжённость) для классификации МРТ-изображений опухолей мозга. Проект —
рефакторинг набора Jupyter-ноутбуков (`scripts/`) в тестируемую Python-
библиотеку; ход рефакторинга подробно расписан в [`refactoring_plan.txt`](refactoring_plan.txt)
и в разделе [«Статус реализации по фазам»](#статус-реализации-по-фазам) ниже.

## Содержание

- [Описание метода](#описание-метода)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Статус реализации по фазам](#статус-реализации-по-фазам)
- [Сверка с опубликованной статьёй](#сверка-с-опубликованной-статьёй)
- [Известные ограничения и особенности метода](#известные-ограничения-и-особенности-метода)
- [Тестирование](#тестирование)
- [Данные](#данные)
- [Ссылки](#ссылки)

## Описание метода

Метод основан на разбиении пространства признаков каждого класса на
подпространства (**подклассы**) и вычислении показателя сопряжённости
`R(x, Y)` для классификации новых объектов.

### Классы опухолей

- **Glioma** — глиома
- **Meningioma** — менингиома
- **Pituitary** — опухоль гипофиза

### Параметры по умолчанию

- Размер изображений после препроцессинга: 256×256 px
- Размерность вектора признаков: 65536 (256×256, построчная развёртка)
- Количество подклассов на класс: 8 (`n_subclasses` — int на все классы или
  `Dict[class_label, int]` для настройки на класс отдельно)
- Базисных векторов в подклассе: 2 (`freeze_basis_at` — int, `None`
  для неограниченного роста, либо `"auto"` для равнения по минимальному
  фактически достигнутому размеру среди всех классов)
- Итого подпространств для классификации: 24 (3 класса × 8 подклассов)
- Опциональные (по умолчанию **выключены**, обратная совместимость)
  предобработочные фильтры эталонных векторов — `filter_dependent`
  (почти линейно зависимые) и `filter_low_informativeness` (малоинформативные,
  доля "белых" элементов < 50% от среднего) — см.
  [«Сверка с опубликованной статьёй»](#сверка-с-опубликованной-статьёй).

### Три фазы алгоритма (канон, `refactoring_plan.txt`, раздел 1)

**Фаза A — поиск центров подклассов** (на эталонных векторах одного класса):
- **A.1** — глобальная пара векторов с минимальным косинусным сходством
  (`GlobalMinCosinePairFinder`)
- **A.2–A.3** — последовательное добавление центров через минимум `R(x, Y)`
  против растущего базиса уже найденных центров (`ReferenceCenterBuilder`)

**Фаза B — формирование кластеров**:
- **B.1** — для каждого центра: второй вектор с минимальным косинусом
  (`CosineSecondVectorAttacher`) → S пар, S подпространств `(N×2)`
- **B.2** — последовательное наполнение кластеров: на каждой итерации к
  подклассу присоединяется ровно один вектор с глобальным максимумом
  `R(x, Y_s)` по всем подклассам (`ConjugacyClusterGrowth`)

**Фаза C — классификация**:
- `C × n_subclasses` подпространств `Y_{c,s}` размера `(N, 2)`
  (24 при 3 классах × 8 подклассов)
- Класс объекта = класс-владелец подпространства с максимальным `R(x, Y_{c,s})`
  среди всех `C × n_subclasses` подпространств (`SubspaceConjugacyClassifier`)

**Математические формулы:**

```
Показатель сопряжённости:  R(x, Y) = (xᵀ Y (Yᵀ Y)⁻¹ Yᵀ x) / (xᵀ x)   ∈ [0, 1]
Косинусное сходство:       cos(a, b) = (a·b) / (‖a‖ · ‖b‖)            ∈ [-1, 1]
```

## Установка

```bash
cd mission_6states

# Установить в editable режиме
pip install -e .

# + dev-зависимости (pytest, black, ruff, mypy)
pip install -e ".[dev]"
```

**Зависимости:** Python ≥3.10, NumPy ≥1.22, SciPy ≥1.8, scikit-learn ≥1.2,
opencv-python-headless ≥4.8, Pillow ≥10.

## Быстрый старт

### Препроцессинг сырых снимков произвольного размера (NB1-NB2)

```python
from subspace_conjugacy import ImagePreprocessor
from subspace_conjugacy.config import DatasetConfig

config = DatasetConfig(root="data", classes=["glioma", "meningioma", "pituitary"])
config.create_directories(stages=["raw"])
# ... скопировать сырые .jpg в config.paths["glioma"]["raw"] и т.д. ...

preprocessor = ImagePreprocessor(target_size=(256, 256))
# raw/ -> resized/ -> centered/ (двухстадийно, как в NB1+NB2)
centered_paths = preprocessor.process_class_via_config("glioma", config, pattern="*.jpg")
```

### Классификация изображений «с нуля» (изображения → предсказание)

```python
import numpy as np
from pathlib import Path
from subspace_conjugacy import (
    load_and_vectorize_batch,
    SubspaceConjugacyClassifier,
)

# 1. Векторизация PNG-изображений (уже отцентрированных — см. "Препроцессинг" выше)
def load_class(class_dir: Path) -> np.ndarray:
    paths = sorted(class_dir.glob("*.png"))
    return load_and_vectorize_batch(paths, method="horizontal")

X_glioma = load_class(Path("datasets/glioma_centered"))
X_meningioma = load_class(Path("datasets/meningioma_centered"))
X_pituitary = load_class(Path("datasets/pituitary_centered"))

X = np.vstack([X_glioma, X_meningioma, X_pituitary])
y = np.array(
    ["glioma"] * len(X_glioma)
    + ["meningioma"] * len(X_meningioma)
    + ["pituitary"] * len(X_pituitary)
)

# 2. Обучение (канонический FursovClusterer строится для каждого класса внутри)
clf = SubspaceConjugacyClassifier(n_subclasses=8, freeze_basis_at=2)
clf.fit(X, y)

# 3. Предсказание
predictions = clf.predict(X[:5])
probabilities = clf.predict_proba(X[:5])          # softmax по classes_
flat_subclass = clf.predict_subclass(X[:5])        # плоский индекс 0..23 (Фаза C)
confidence = clf.predict_confidence_ratio(X[:5])   # NB8: best_R/mean(others)-1
```

Готовый end-to-end скрипт с train/test split, отчётом и сохранением модели —
[`main.py`](main.py). Скрипт содержит два независимых эксперимента, выбираемых
через `--experiment`:

```bash
python main.py                              # = --experiment article (по умолчанию)
python main.py --experiment article         # эксперименты 2-3 опубликованной статьи (Kaggle archive/)
python main.py --experiment draft-method    # метод из черновика vs классический ML vs CNN (datasets/*_centered/)
python main.py --experiment mnist           # тот же эксперимент, что и draft-method, но на MNIST
python main.py --experiment tuned --dataset mri     # честный подбор гиперпараметров + кривая по объёму данных (МРТ)
python main.py --experiment tuned --dataset mnist   # то же самое на MNIST
```

`--experiment tuned` — честный (без подгонки под желаемый результат) подбор
гиперпараметров ВСЕХ трёх подходов через holdout-валидацию (тест не
участвует в подборе, одна и та же методология для всех трёх сторон), затем
**кривая эффективности по объёму обучающих данных** (10/20/30/40/60/80/120/160
на класс) — прямая проверка тезиса `theory/` о том, что метод сопряжённости
работоспособен на малых выборках, тогда как CNN/классическому ML нужно
больше данных. Результат публикуется как есть — метод может как выиграть,
так и проиграть на любой конкретной точке; см. `refactoring_plan.txt`,
раздел 11 для готовых цифр.

### Подробный отчёт в Word

`reporting/word_report.py` — модуль **вне** пакета `subspace_conjugacy`
(отдельные зависимости, не тянутся библиотекой), генерирует подробный
`.docx`-отчёт по JSON-результатам `--experiment tuned`: методология, ВСЕ
опробованные конфигурации подбора гиперпараметров (не только победитель),
таблицы и графики кривой эффективности, честные выводы, вычисленные из
фактических чисел отчёта (не захардкожены).

```bash
pip install -r reporting/requirements.txt   # python-docx, matplotlib — не входят в extras библиотеки
python reporting/word_report.py \
    --report artifacts/tuned_comparison_mri_report.json \
    --report artifacts/tuned_comparison_mnist_report.json \
    --output artifacts/tuned_comparison_report.docx
```

`--experiment draft-method` сравнивает на `datasets/{class}_centered/`:
расширенную сетку конфигураций `SubspaceConjugacyClassifier` (baseline,
отдельные фильтры статьи, метод из черновика `theory/Макет новой статьи.docx`
в разных комбинациях, перебор `n_subclasses`), классические ML-модели sklearn
(логрегрессия/линейный SVM/random forest/kNN поверх PCA) и несколько
намеренно упрощённых CNN (PyTorch). `--experiment mnist` — та же сетка
конфигураций и та же сравнительная таблица, но на MNIST (10 классов цифр,
28×28 grayscale, скачивается через `sklearn.datasets.fetch_openml` при первом
запуске и кэшируется локально) — проверка, обобщаются ли выводы на другой
домен. Оба эксперимента требуют `pip install torch` (extras `cnn-experiment`
в `pyproject.toml`, не входит в основные зависимости библиотеки). Подробности,
обоснование решений и результаты обоих прогонов — docstring `main.py` и
`refactoring_plan.txt`, раздел 11.

### Кластеризация одного класса (канонический алгоритм A+B напрямую)

```python
from subspace_conjugacy import FursovClusterer

clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
clusterer.fit(X_glioma)

clusterer.subspaces_          # 8 матриц (N, 2)
clusterer.labels_             # (M,) метка подкласса на объект
clusterer.get_subclass_sizes()
clusterer.get_initial_pair()  # результат Фазы A.1
clusterer.get_center_indices()  # результат Фазы A.2-A.3
```

### Показатель сопряжённости и косинусное сходство напрямую

```python
from subspace_conjugacy import conjugate_criterion, cosine_similarity_matrix

Y = X_glioma[:2].T                       # базис подпространства (N, 2)
R = conjugate_criterion(X_glioma, Y)     # (M,), диапазон [0, 1]

cos_matrix = cosine_similarity_matrix(X_glioma)  # (M, M), диапазон [-1, 1]
```

### Сохранение и загрузка обученной модели

```python
from subspace_conjugacy.config import DatasetConfig
from subspace_conjugacy import save_pipeline_artifact, load_pretrained_classifier

config = DatasetConfig(root="artifacts/pipeline", n_subclasses=8, subclass_factor=2)
save_pipeline_artifact(clf, config)   # CSV базисов на класс + JSON метаданных

clf_loaded = load_pretrained_classifier(config)  # fit_from_subclass_bases, без переобучения
```

Либо через pickle (`subspace_conjugacy.io.persistence.save_model`/`load_model`) —
проще, но версионно более хрупко, чем CSV+JSON выше.

### Полный пайплайн через оркестратор (Фаза 6)

```python
from subspace_conjugacy import FursovPipeline
from subspace_conjugacy.config import DatasetConfig

config = DatasetConfig(root="data", classes=["glioma", "meningioma", "pituitary"], n_subclasses=8)
pipeline = FursovPipeline(config)

# NB1+NB2: raw -> resized -> centered, для каждого класса
for cls in config.classes + ["test"]:
    pipeline.run_preprocessing(cls, raw_pattern="*.jpg")

# NB3 (vectorize) -> канон (cluster) -> NB6-7 (export_subspaces), для всех классов
pipeline.run_all_classes()

# Фаза C: собрать классификатор и оценить на тесте
classifier = pipeline.build_classifier()
report = pipeline.classify_test(y_test=y_true)  # или без y_test — позиционный NB8-фолбэк
print(report["report"]["accuracy"])
```

### Опциональные фильтры эталонных векторов (сверка со статьёй, находки №3, №5)

```python
from subspace_conjugacy import SubspaceConjugacyClassifier

clf = SubspaceConjugacyClassifier(
    n_subclasses=8,
    filter_dependent=True,              # исключить почти линейно зависимые векторы
    dependency_threshold=0.999,
    filter_low_informativeness=True,    # исключить малоинформативные (мало "белых" элементов)
    informativeness_min_fraction=0.5,   # статья: < 50% от среднего по классу
)
clf.fit(X, y)
clf.excluded_indices_by_class_  # {class_label: индексы X, исключённые хотя бы одним фильтром}
```

Оба фильтра применяются независимо для каждого класса, перед кластеризацией
(сначала `filter_low_informativeness`, затем `filter_dependent` — среди
выживших). По умолчанию оба выключены.

### Разбиение на похожие пары — метод из черновика (theory/Макет новой статьи.docx)

```python
from subspace_conjugacy import SubspaceConjugacyClassifier

clf = SubspaceConjugacyClassifier(
    n_subclasses=8,
    split_correlated_pairs=True,     # разбить эталонные векторы класса на пары похожих
    correlated_pairs_subset="a",     # какое из двух подмножеств использовать ("a" или "b")
)
clf.fit(X, y)
```

⚠ В отличие от `filter_dependent`/`filter_low_informativeness` (сверены с
**опубликованной** статьёй, раздел "Сверка с опубликованной статьёй"), этот
метод взят из **черновика другой, неопубликованной** статьи (про КТ грудной
клетки) — `theory/Макет новой статьи.docx`, "Первый этап": итеративно ищет
пары наиболее похожих (по косинусному сходству) векторов и делит каждую пару
между двумя подмножествами; для кластеризации берётся только одно из них.
Применяется как фаза 0c — после `filter_low_informativeness`/`filter_dependent`,
перед фазой A.1. Подробности и принятые решения по неоднозначностям
черновика — `algorithms/correlated_pair_splitter.py` (docstring) и
`refactoring_plan.txt`, раздел 11. По умолчанию выключено.

### Поиск гиперпараметров (grid search / random search)

```python
from subspace_conjugacy import grid_search_classifier, random_search_classifier, summarize_search_results

search = grid_search_classifier(X_train, y_train, cv=3)          # по умолчанию — DEFAULT_PARAM_GRID
search.best_params_, search.best_score_
search.best_estimator_.predict(X_test)

search = random_search_classifier(X_train, y_train, n_iter=15, cv=3, random_state=42)
for row in summarize_search_results(search, top_n=3):
    print(row["rank"], row["mean_test_score"], row["params"])
```

`SubspaceConjugacyClassifier` — обычный sklearn-эстиматор (`get_params`/
`set_params`/`clone` работают из коробки), поэтому `grid_search_classifier`/
`random_search_classifier` — тонкие обёртки над
`GridSearchCV`/`RandomizedSearchCV` без собственной логики перебора.

### Двухэтапная классификация: проекция → тип опухоли (статья, находка №4)

```python
from subspace_conjugacy import SubspaceConjugacyClassifier, SequentialClassifier, otsu_binarize

# Этап 1 — определение проекции (axial/sagittal/coronal) на Otsu-бинаризованных изображениях
stage1 = SubspaceConjugacyClassifier(n_subclasses=4)
stage1.fit(X_projection_train, y_projection_train)

# Этап 2 — отдельный классификатор типа опухоли на каждую проекцию
stage2 = {
    "axial": SubspaceConjugacyClassifier(n_subclasses=8),
    "sagittal": SubspaceConjugacyClassifier(n_subclasses=8),
    "coronal": SubspaceConjugacyClassifier(n_subclasses=8),
}

seq = SequentialClassifier(stage1, stage2)
seq.fit(X1_train, y1_train, X2_by_group_train, y2_by_group_train)

projection_pred, tumor_type_pred = seq.predict(X1_test, X2_test)
```

### Оценка качества классификации

```python
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

y_pred = clf.predict(X_test)
report = evaluate_classifier(y_test, y_pred)
report["accuracy"]             # общая точность
report["per_class_accuracy"]   # {class_label: accuracy}
report["confusion_matrix"]     # sklearn confusion_matrix
```

## Структура проекта

```
subspace_conjugacy/
├── __init__.py              # Публичный API (см. __all__)
├── core/
│   └── metrics.py           # conjugate_criterion, cosine_similarity_matrix, compute_gram_inverse
├── algorithms/               # Канонический алгоритм (Фазы A, B) + legacy для parity
│   ├── global_pair.py            # A.1  — GlobalMinCosinePairFinder
│   ├── reference_centers.py      # A.2-A.3 — ReferenceCenterBuilder
│   ├── reference_filter.py       # Фаза 0b — LinearDependencyFilter (сверка со статьёй, находка №3)
│   ├── informativeness_filter.py # Фаза 0a — LowInformativenessFilter (находка №5)
│   ├── correlated_pair_splitter.py # Фаза 0c — CorrelatedPairSplitter (метод из черновика, раздел 11)
│   ├── subclass_seed.py          # B.1  — CosineSecondVectorAttacher
│   ├── subclass_growth.py        # B.2  — ConjugacyClusterGrowth
│   ├── fursov_clusterer.py       # Фасад 0a→0b→A.1→A.3→B.1→B.2 — FursovClusterer
│   ├── subclass_export.py        # flatten/unflatten/export базисов в CSV (NB6-7 формат) + equalize_subspace_bases
│   └── legacy/                   # ТОЛЬКО для parity-тестов, не для production
│       ├── per_vector_pairs.py       # NB4 (per-vector trio_list, расходится с A.1)
│       └── notebook_pipeline.py      # Staged NB5→NB6→NB7 (canonical- или legacy-pair)
├── models/
│   ├── base.py                    # BaseSubspaceEstimator (sklearn-совместимый интерфейс)
│   ├── clusterer.py                # SubspaceClusterer = алиас FursovClusterer (backward-compat)
│   ├── classifier.py               # SubspaceConjugacyClassifier — Фаза C / NB8
│   └── sequential_classifier.py    # SequentialClassifier — этап 1 (проекция) → этап 2 (тип опухоли), находка №4
├── model_selection/            # Поиск гиперпараметров (grid search / random search)
│   ├── param_space.py             # DEFAULT_PARAM_GRID, DEFAULT_PARAM_DISTRIBUTIONS
│   └── search.py                  # grid_search_classifier, random_search_classifier, search_hyperparameters
├── preprocessing/             # NB1-NB2: сырое изображение -> 256x256, центрировано
│   ├── resize.py                  # NB1
│   ├── normalization.py           # NB2 — подавление фона
│   ├── centering.py               # NB2 — горизонтальное/вертикальное центрирование
│   ├── binarization.py            # otsu_threshold/otsu_binarize — этап определения проекции (находка №4)
│   └── preprocessor.py            # ImagePreprocessor — фасад + DatasetConfig-интеграция
├── pipeline/                  # Оркестратор end-to-end сценария
│   ├── stages.py                  # STAGE_REGISTRY — именованные стадии (включая binarize)
│   └── fursov_pipeline.py         # FursovPipeline — run_class/run_all_classes/classify_test
├── features/
│   ├── vectorization.py      # vectorize_image/_batch, load_and_vectorize[_batch] (NB3)
│   └── extraction.py         # extract_class_vectors/_all_classes/_training_data (по DatasetConfig)
├── evaluation/
│   └── metrics.py            # evaluate_classifier, per_class_accuracy, confidence_summary
├── io/
│   ├── vectors.py            # CSV IO для каждой стадии NB3-NB7 + save_pipeline_artifact
│   └── persistence.py        # save_model/load_model (pickle), export/import_subspaces_{npz,json}
├── config/
│   └── dataset.py            # DatasetConfig — пути вместо hardcoded macOS-путей ноутбуков
└── utils/
    └── validation.py         # check_array_X, check_X_y, check_basis_matrix, check_hyperparameters

tests/
├── conftest.py                # synthetic fixtures + notebook CSV fixtures (skip если нет данных)
├── test_installation.py       # smoke-тесты установки пакета
├── test_vectorization.py
├── test_csv_io.py
├── test_classifier.py
├── test_sequential_classifier.py  # SequentialClassifier (находка №4)
├── test_model_selection.py      # grid_search_classifier/random_search_classifier + is_classifier regression
├── test_preprocessing.py        # resize/normalization/centering/otsu + реальный Kaggle-датасет
├── test_pipeline.py             # FursovPipeline + STAGE_REGISTRY, включая end-to-end на archive
├── test_algorithms/            # unit-тесты каждого канонического модуля (0a-0b, A.1-B.2, facade, export)
├── test_theory/                 # @pytest.mark.theory — инварианты канона на synthetic-данных
└── test_parity/                 # @pytest.mark.notebook_parity — на реальных PNG из datasets/

main.py           # End-to-end скрипт: датасет -> train/test split -> classifier -> отчёт -> артефакты
scripts/          # Исходные Jupyter-ноутбуки (reference/legacy, см. "Известные ограничения")
theory/           # Исходные документы с описанием канонического алгоритма (.docx)
refactoring_plan.txt   # Полный план рефакторинга по фазам — источник этого README
```

## Статус реализации по фазам

Нумерация фаз соответствует `refactoring_plan.txt`.

### ✅ Фаза 0 — Инфраструктура пакета

`pyproject.toml`, `requirements.txt`, `pytest.ini` с маркерами `theory`/`notebook_parity`/`slow`,
`subspace_conjugacy/__init__.py` с публичным API, `config/dataset.py::DatasetConfig`
(конфигурируемые пути вместо `/Users/vladkorsikov/research/...`).

### ✅ Фаза 1 — Preprocessing (NB1-NB2)

`preprocessing/` принимает произвольные сырые снимки (любой размер, RGB или
grayscale) и приводит их к формату, который ожидает остальной пайплайн
(256×256, отцентрированное содержимое):

- `preprocessing/resize.py` — resize до целевого размера через
  `cv2.INTER_LANCZOS4` (аналог `PIL.Image.LANCZOS` из NB1) +
  `resize_directory()` для пакетной обработки.
- `preprocessing/normalization.py` — `suppress_background()`: векторизованный
  (без попиксельных циклов) аналог `normalization_colour`/`normalization_grey`
  из NB2 — одна функция вместо двух дублирующихся в ноутбуке, работает и для
  grayscale, и для цветных изображений через общую grayscale-маску.
- `preprocessing/centering.py` — `compute_horizontal_delta`/`compute_vertical_delta`
  + `shift_rows`/`shift_columns` (через `np.roll`) — аналог
  `find_horizontal_delta`/`image_correction_top_and_bottom`/`find_vertical_delta`/
  `image_correction_left_and_right` из NB2. **Два бага оригинального ноутбука
  сознательно не воспроизведены** (см. docstring модуля): скан "снизу"/"справа"
  через `image[-line]` с `line` от 0 читает `image[-0] == image[0]` вместо
  истинного последнего элемента; и асимметричное условие
  (`< -1` по горизонтали, но `< 1` по вертикали) для порога "игнорировать
  сдвиг на ±1 пиксель".
- `preprocessing/preprocessor.py::ImagePreprocessor` — фасад: `.process(image)`
  (в памяти), `.process_file()`, `.process_directory()` (произвольная
  директория → пронумерованные PNG), `.process_class_via_config()`
  (двухстадийный raw → resized → centered через `DatasetConfig`, как в
  оригинальных ноутбуках).

**Проверено на реальном датасете** (Kaggle "Brain Tumor MRI Dataset", не
входит в репозиторий — см. `tests/test_preprocessing.py::TestRealBrainTumorArchive`,
`BRAIN_MRI_ARCHIVE_ROOT`): на выборке из 90 изображений медианное отклонение
центроида содержимого от геометрического центра кадра падает с ~13 до ~5
пикселей после центрирования; полный путь raw `.jpg` произвольного размера →
`resize` → `center` → `vectorize` → `FursovClusterer.fit()` отработан целиком
без ошибок.

### ✅ Фаза 2 — Векторизация и CSV IO

`features/vectorization.py` (векторизация через `numpy.flatten`/`ravel('F')` —
на 2-3 порядка быстрее ручных циклов NB3), `features/extraction.py`,
`io/vectors.py` (базовые `save/load_vectors_csv`, `save/load_class_vectors`).

### ✅ Фаза 3 — Канонический алгоритм кластеризации (A + B) и legacy для parity

Все подфазы реализованы и покрыты unit- и theory-тестами:

| Подфаза | Модуль | Статус |
|---|---|---|
| 3.1 — A.1 (глобальная пара) | `algorithms/global_pair.py` | ✅ |
| 3.2 — A.2-A.3 (центры через min R) | `algorithms/reference_centers.py` | ✅ |
| 3.3 — B.1 (cosine-seed второй вектор) | `algorithms/subclass_seed.py` | ✅ |
| 3.4 — B.2 (наполнение через max R) | `algorithms/subclass_growth.py` | ✅ (`strategy="default"` и `"master"` из NB7) |
| 3.5 — Фасад `FursovClusterer` | `algorithms/fursov_clusterer.py` | ✅, `SubspaceClusterer` — алиас |
| 3.6 — Экспорт + legacy staged pipeline | `algorithms/subclass_export.py`, `algorithms/legacy/notebook_pipeline.py` | ✅ |
| 3.7 — Theory + parity тесты | `tests/test_theory/`, `tests/test_parity/` | ✅ |

**API:** `GlobalMinCosinePairFinder`, `ReferenceCenterBuilder`, `CosineSecondVectorAttacher`,
`ConjugacyClusterGrowth`, `FursovClusterer` — всё в стиле sklearn (`fit()` возвращает `self`,
результат — в атрибутах с завершающим `_`: `.pair_indices_`, `.center_indices_`, `.pairs_`,
`.labels_`/`.subspace_bases_`).

### ✅ Фаза 4 — Classifier & Evaluation (Фаза C теории, NB8)

- `models/classifier.py::SubspaceConjugacyClassifier` переписан на канонический
  `FursovClusterer` (единственная реализация кластеризации в библиотеке — старая
  самостоятельная логика и дублированные `conjugate_criterion`/`cosine_similarity_matrix`
  убраны из `models/clusterer.py` и `models/classifier.py`).
- Новый API: `fit_from_subclass_bases()`, `predict_r_matrix_flat()`, `predict_subclass()`,
  `predict_confidence_ratio()` (формула NB8: `best_R / mean(R_others) - 1`).
- `evaluation/metrics.py`: `evaluate_classifier`, `per_class_accuracy`, `confidence_summary`.
- `main.py` доведён до реальной работоспособности на `datasets/*_centered/*.png`
  (изначально поддерживал только `.npy`/`.csv`).

### ✅ Фаза 5 — IO & Persistence

`io/vectors.py` дополнен до полного покрытия стадий NB3-NB7 плюс сквозной артефакт:

| CSV ноутбука | Функция | Канон/Legacy |
|---|---|---|
| `{class}_horizontal_vector.csv` | `save/load_vectors_csv`, `save/load_class_vectors` | NB3 |
| `{class}_class_first_and_second_*.csv` | `save/load_initial_pair_indices` | Legacy NB4-5 |
| `{class}_class_core_vectors.csv` | `save/load_center_indices` | A.2-A.3 |
| `{class}_new_classes.csv` | `save/load_subclass_pairs` | B.1 (совместимо с `ConjugacyClusterGrowth.fit()` напрямую) |
| `8_{class}_subclasses_vectors.csv` | `save/load_subclass_bases[_as_list]` | Export |
| — | `save_pipeline_artifact` / `load_pretrained_classifier` | CSV+JSON аналог pickle |

### ✅ Фаза 6 — Pipeline Orchestrator

`pipeline/stages.py` — реестр `STAGE_REGISTRY` из 11 именованных стадий
(`resize`, `center`, `vectorize`, `global_pair`, `reference_centers`,
`subclass_seed`, `subclass_growth`, `cluster`, `export_subspaces`,
`classify`, `legacy_notebook`) — каждая тонкая обёртка над уже
существующим модулем (без новой логики). `pipeline/fursov_pipeline.py::FursovPipeline`
собирает их в высокоуровневый сценарий:

- `run_stage(name, **kwargs)` — точечный вызов одной стадии; если сигнатура
  стадии принимает `config` и он не передан явно, автоматически
  подставляется `self.config`.
- `run_preprocessing(class_name)` — NB1+NB2 (`resize` → `center`) для одного класса.
- `run_class(class_name)` / `run_all_classes()` — `vectorize` → `cluster`
  (**канон**, `FursovClusterer`) → `export_subspaces`, как в плане.
  `"cluster"` — единственный путь кластеризации здесь; `"legacy_notebook"`
  доступна только через явный `run_stage()`, не участвует в `run_class()`
  (refactoring_plan.txt, раздел 6, п.1).
- `build_classifier()` — собирает `SubspaceConjugacyClassifier` из
  `self.clusterers_` через `fit_from_subclass_bases()`, без повторной
  кластеризации.
- `classify_test(y_test=...)` — предсказание + `evaluate_classifier()`
  (Фаза C / NB8). Без `y_test` использует **буквальную** позиционную
  разметку NB8 (первые `test_samples_per_class` объектов — первый класс из
  `config.classes`, и т.д.) — задокументирована как хрупкая, явно
  рекомендуется передавать `y_test`.

Стадия `vectorize` сама определяет реальное количество изображений (через
`glob`), а не полагается на зашитые в `DatasetConfig` 100/25 на класс — иначе
пайплайн не работал бы ни на архивном Kaggle-датасете, ни на любом
датасете произвольного размера.

Проверено end-to-end на реальных данных archive (`raw .jpg` → `resize` →
`center` → `cluster` → `build_classifier` → `classify_test`) —
`tests/test_pipeline.py::TestFullPipelineOnRealArchive`.

### ❌ Фаза 7 — Thin Notebook Wrappers

**Не реализовано.** Ноутбуки в `scripts/` остаются в исходном (легаси, с
известными багами) виде — не переписаны в тонкие обёртки над
`FursovPipeline` (технически уже возможно после Фазы 6, но не сделано).

### 🟡 Фаза 8 — Тесты и покрытие

Theory-тесты (`tests/test_theory/`) и parity-тесты на реальных данных
(`tests/test_parity/`) реализованы и проходят, но:

- **Coverage ≥70% не измерялся и не enforced** — `pytest-cov` есть в
  dev-зависимостях, но CI/порог покрытия не настроены.
- **Буквальный parity с CSV оригинальных ноутбуков не проверяется** — файлов,
  сгенерированных исходными (macOS-путёвыми) ноутбуками, в репозитории нет и
  взять их неоткуда. Parity-тесты проверяют структурные инварианты на реальных
  изображениях (`datasets/`), а не побайтовое совпадение с чужим CSV.
- Целевая метрика плана «accuracy ≥0.70 на test75» (конкретный 25/25/25
  held-out набор оригинальных ноутбуков) не воспроизведена буквально — такого
  набора не существует в репозитории. См. следующий раздел.

## Сверка с опубликованной статьёй

Помимо рефакторинга ноутбуков, библиотека была сверена с опубликованной
статьёй Korshikov & Fursov ("Pathology Recognition Based on Conjugacy
Criteria with Subspaces of Reference Images", [`theory/`](theory)) — полный
разбор в `refactoring_plan.txt`, раздел 10. Из 6 найденных расхождений 5
реализованы как **опциональные** (по умолчанию выключенные) возможности —
обратная совместимость не нарушена:

| № | Расхождение со статьёй | Реализация |
|---|---|---|
| 1 | `n_subclasses` — единый int на все классы, а не per-class | `n_subclasses` принимает `Dict[class_label, int]` |
| 2 | `freeze_basis_at=2` отбрасывает результат роста подпространства (B.2) | `freeze_basis_at="auto"` — рост без ограничения + равнение до общего минимального k (`equalize_subspace_bases`) |
| 3 | Почти линейно зависимые эталонные векторы не исключаются | `filter_dependent=True` — `LinearDependencyFilter` (Фаза 0b) |
| 4 | Первый этап статьи (Otsu + классификация проекции) не реализован | `otsu_binarize`/`otsu_threshold` + `SequentialClassifier` |
| 5 | Малоинформативные изображения (< 50% "белых" элементов от среднего) не отфильтровываются | `filter_low_informativeness=True` — `LowInformativenessFilter` (Фаза 0a, перед 0b) |
| 6 | Датасет/сплит статьи (300/375/450, 80/20) отличается от репозиторного | Не баг — задокументировано как контекст, изменений не требует |

Фильтры Фазы 0 применяются в порядке 0a (`filter_low_informativeness`) →
0b (`filter_dependent`) — сначала отбрасывается низкое качество данных,
затем избыточность среди оставшихся; `FursovClusterer.excluded_indices_` —
объединение (`np.union1d`) исключений обоих фильтров, с отдельными
`excluded_by_informativeness_`/`excluded_by_dependency_` для интроспекции.

## Известные ограничения и особенности метода

- **Accuracy на реальном датасете:** на end-to-end прогоне (`main.py`, все 600
  изображений `datasets/*_centered`, 70/30 split, `n_subclasses=8`) —
  **61%** общая точность (glioma/pituitary recall ~0.82-0.87, meningioma —
  систематически хуже, ~0.15 recall). На меньшем holdout-сплите (50/10 на
  класс) — **80%**. Оба результата значительно выше случайного угадывания
  (33% для 3 классов), но ниже целевых 70% из плана — целевая метрика
  привязана к другому датасету/сплиту (test75 оригинальных ноутбуков),
  которого нет в репозитории; для приближения к 0.70 может потребоваться
  подбор `n_subclasses`/`growth_strategy`, что не входило в задачи Фаз 4-5.
- **B.2 («жадное» наполнение кластеров) не гарантирует сбалансированные
  подклассы.** Поскольку критерий отбора — глобальный `argmax R(x, Y_s)» по
  всем подклассам сразу, а больший базис почти всегда объясняет вектор не хуже
  меньшего, один подкласс может «съесть» большую часть оставшихся векторов
  (наблюдалось и на синтетике, и на реальных МРТ — напр. один подкласс из 8
  получил 39 из 60 векторов). Это прямое следствие канонического алгоритма из
  `refactoring_plan.txt` (раздел 1.3, B.2), а не ошибка реализации — при
  рефакторинге алгоритм умышленно не менялся.
- **`R(x, Y)` и `cos(a, b)` масштабно-инвариантны** (`R(αx, Y) = R(x, Y)` для
  `α > 0`) — метод различает объекты по **направлению** вектора признаков, а
  не по его величине. Синтетические датасеты для тестов/демо должны задавать
  классам разные направления, а не константный сдвиг вдоль одного направления
  (иначе классификатор не сможет их различить — это свойство метода, не баг).
- **Известные баги оригинальных ноутбуков сознательно не воспроизведены** в
  канонической реализации, но описаны в `refactoring_plan.txt` (раздел 2.8) и
  зафиксированы в legacy-модулях как задокументированное расхождение:
  `vector_norm(first_reference)` в NB4 (неверная норма второго вектора в
  косинусном критерии) и `count_num` захардкоженный в 0/1 в
  `7_2_hf`/`7_master` ноутбуках (фаза "fulfilling new subclasses" фактически
  не выполнялась).

## Тестирование

```bash
# Все тесты
pytest

# Только канонический алгоритм на synthetic-данных (быстро, без датасета)
pytest -m theory

# Только на реальных МРТ из datasets/ (требует датасет, медленнее — B.2 растёт как O(M²·N))
pytest -m notebook_parity

# Исключить медленные тесты полного пайплайна
pytest -m "not slow"

# С покрытием
pytest --cov=subspace_conjugacy --cov-report=html
```

На момент последнего прогона: **542 passed, 1 skipped**, покрытие **94%**
(пропущенный тест — в `tests/test_csv_io.py::TestNotebookParity`, ожидает
датасет по старой схеме путей `DatasetConfig(root="data")` с директориями
`{class}_raw`, которой нет — актуальный датасет лежит в
`datasets/{class}_centered/`, для него используются fixtures в
`tests/test_parity/conftest.py`).

`notebook_parity`/`slow` тесты на полном 200-изображенческом классе укладываются
в единицы-десятки секунд на подвыборках (обычно 50-60 изображений на класс) —
полный B.2 на 200 изображениях одного класса занимает больше минуты за счёт
`O(M²·N)` роста базиса на чистом numpy.

## Данные

Используются два независимых датасета, ни один не входит в git (см. `.gitignore`):

- **`datasets/{glioma,meningioma,pituitary}_centered/`** — по 200 PNG 256×256
  на класс, уже прошедших resize + центрирование (эквивалент выхода NB2 /
  входа NB3). Используется большинством `tests/test_parity/` и `main.py`.
- **Kaggle "Brain Tumor MRI Dataset"** (Masoud Nickparvar; 4 класса —
  glioma/meningioma/**notumor**/pituitary, `Training`/`Testing` split,
  изображения произвольного размера от 200×200 до 900×741 и смешанных
  режимов RGB/grayscale) — сырые данные для проверки Фазы 1
  (`preprocessing/`). Класс `notumor` вне скоупа проекта (не используется).
  Путь задаётся переменной окружения `BRAIN_MRI_ARCHIVE_ROOT`
  (по умолчанию — путь на машине автора); тесты, зависящие от него,
  автоматически пропускаются, если датасет не найден.

Отдельного held-out `test`-набора для `datasets/*_centered/` в репозитории
нет — при необходимости holdout делается вручную
(`sklearn.model_selection.train_test_split`, как в `main.py`) или через
ручной срез списка файлов (как в `tests/test_parity/`).

## Ссылки

- План рефакторинга (полный, по фазам, с промптами и критериями приёмки):
  [`refactoring_plan.txt`](refactoring_plan.txt)
- Источник теории канонического алгоритма: [`theory/`](theory) (`.docx`)
- Исходные ноутбуки (reference/legacy для parity): [`scripts/`](scripts)

## Лицензия

MIT

## Авторы

Mission 6 States Team
