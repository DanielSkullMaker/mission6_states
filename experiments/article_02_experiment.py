"""
Эксперимент для статьи 2: "Симбиоз классификаторов: ансамблевая интеграция
метода подпространственной сопряжённости с классическими моделями машинного
обучения и свёрточными нейронными сетями".

План — theory/article_plans/02_simbioz_arkhitektur.txt, раздел 5
("Методология и план эксперимента"). Использует функционал, добавленный в
библиотеку для этой статьи:
  - subspace_conjugacy.evaluation.compute_ensemble_diagnostics — диагностика
    согласованности ошибок (потенциал ансамбля / жёсткий потолок / оракул).
  - subspace_conjugacy.models.{PrefitVotingClassifier, PrefitStackingClassifier,
    SwitchingEnsembleClassifier} — три схемы объединения УЖЕ ОБУЧЕННЫХ
    разнородных моделей (включая CNN на PyTorch, которую sklearn не умеет
    клонировать и переобучать).

Датасет (theory/article_plans/02_simbioz_arkhitektur.txt, раздел 0):
  Полный HAM10000/ISIC2018 (10 015 дерматоскопических изображений, 7
  классов новообразований кожи), скачанный напрямую с Harvard Dataverse
  (doi:10.7910/DVN/DBW86T) — HAM10000_images_part_1.zip + part_2.zip
  (~2.7 ГБ вместе) и HAM10000_metadata.tab с разметкой класса для каждого
  изображения. В отличие от более раннего варианта этого эксперимента
  (сжатая версия MedMNIST 28x28), здесь используются подлинные
  изображения набора данных, изменённые в размере до IMG_SIZE x IMG_SIZE
  пикселей (см. ниже) — заметно детальнее, чем сжатая версия, но всё ещё
  практичного для метода сопряжённости размера. Официального разбиения на
  train/val/test у полного набора данных нет — оно строится здесь
  самостоятельно: для каждого класса случайно (фиксированный seed)
  отбираются непересекающиеся train/val/test-подвыборки из ВСЕХ доступных
  изображений этого класса, ограниченные объёмом самого малочисленного
  класса (дерматофиброма, 115 изображений всего). val — для диагностики
  согласованности ошибок и обучения метамодели/выбора чемпионов ансамблей
  (НЕ участвует в обучении базовых моделей — тот же принцип честного
  holdout, что и в main.py), test — исключительно для итоговой оценки.

Три базовые модели (все — "уже обученные" в терминах библиотеки):
  - subspace — SubspaceConjugacyClassifier (канон, ядро проекта).
  - classical_ml — LogisticRegression поверх PCA (тот же приём, что и в
    main.py::CLASSICAL_ML_MODELS — на сырых пиксельных векторах классический
    ML работал бы медленно/неустойчиво).
  - cnn — компактная свёрточная сеть на PyTorch (2 свёрточных блока,
    архитектура вдохновлена main.py::SmallCNN, но меньше — обучающая
    выборка здесь на порядок меньше MRI-экспериментов проекта).

Все три обучаются на ОДНОМ И ТОМ ЖЕ сбалансированном подмножестве обучающей
выборки (одинаковое число изображений на класс — тот же принцип честного
сравнения, что и в main.py::prepare_centered_split), чтобы различия в
точности отражали свойства МЕТОДОВ, а не разный объём/состав данных.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from subspace_conjugacy.algorithms import FursovClusterer
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.models.ensemble import (
    PrefitVotingClassifier,
    PrefitStackingClassifier,
    SwitchingEnsembleClassifier,
)
from subspace_conjugacy.evaluation import compute_ensemble_diagnostics, per_class_accuracy
from subspace_conjugacy.core.metrics import compute_gram_inverse, conjugate_criterion

# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
RANDOM_SEED = 42
# Самый малочисленный класс полного набора данных — дерматофиброма, 115
# изображений всего; 60+15+30=105 оставляет запас.
TRAIN_N_PER_CLASS = 60
VAL_N_PER_CLASS = 15
TEST_N_PER_CLASS = 30
IMG_SIZE = 128  # исходные снимки HAM10000 — 600x450; здесь приводятся к квадрату IMG_SIZE x IMG_SIZE
N_SUBCLASSES = 4
PCA_COMPONENTS = 30
CNN_EPOCHS = 40
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-3

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "ham10000_full" / "raw"
METADATA_PATH = RAW_DIR / "HAM10000_metadata.tab"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "ham10000_full" / "images"
OUTPUT_DIR = Path(__file__).resolve().parent / "article_02_output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"

# Официальная разметка классов MedMNIST DermaMNIST (= HAM10000/ISIC2018,
# https://medmnist.com/): индекс -> (короткий код, русское название).
CLASS_INFO = {
    0: ("akiec", "актинический кератоз"),
    1: ("bcc", "базальноклеточный рак"),
    2: ("bkl", "доброкачественный кератоз"),
    3: ("df", "дерматофиброма"),
    4: ("mel", "меланома"),
    5: ("nv", "меланоцитарный невус"),
    6: ("vasc", "сосудистое новообразование"),
}
CLASS_LABELS = [str(i) for i in range(7)]


def log(msg: str) -> None:
    print(f"[article_02] {msg}", flush=True)


def _json_safe_diagnostics(diag: Dict[str, Any]) -> Dict[str, Any]:
    """compute_ensemble_diagnostics возвращает pairwise_agreement/
    pairwise_error_correlation с ключами-кортежами (model_a, model_b) —
    json.dump требует строковые ключи, поэтому здесь они превращаются в
    "model_a__model_b" только для сохранения в JSON (сам объект diag,
    используемый внутри эксперимента, не изменяется)."""
    safe = dict(diag)
    for key in ("pairwise_agreement", "pairwise_error_correlation"):
        safe[key] = {f"{a}__{b}": v for (a, b), v in diag[key].items()}
    return safe


# ----------------------------------------------------------------------
# Загрузка и подготовка данных (полный HAM10000, скачанный с Harvard
# Dataverse — два ZIP-архива изображений + файл метаданных с разметкой
# класса; извлекается один раз в IMAGES_DIR, дальше изображения читаются
# с диска по имени файла)
# ----------------------------------------------------------------------
def ensure_images_extracted() -> None:
    """Распаковывает оба ZIP-архива в IMAGES_DIR, если это ещё не сделано."""
    import zipfile

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(IMAGES_DIR.glob("*.jpg"))
    if len(existing) >= 10000:
        log(f"Изображения уже распакованы ({len(existing)} файлов в {IMAGES_DIR}).")
        return

    for part in ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]:
        zip_path = RAW_DIR / part
        log(f"Распаковка {part}...")
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.endswith(".jpg"):
                    target = IMAGES_DIR / Path(member).name
                    if not target.exists():
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
    n_files = len(list(IMAGES_DIR.glob("*.jpg")))
    log(f"Распаковано {n_files} изображений в {IMAGES_DIR}.")


def load_metadata() -> Dict[str, List[str]]:
    """Читает HAM10000_metadata.tab -> {код_класса: [image_id, ...]}."""
    import csv

    by_class: Dict[str, List[str]] = {info[0]: [] for info in CLASS_INFO.values()}
    with open(METADATA_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dx = row["dx"].strip('"')
            image_id = row["image_id"].strip('"')
            if dx in by_class:
                by_class[dx].append(image_id)
    for code, ids in by_class.items():
        log(f"  класс {code}: {len(ids)} изображений всего")
    return by_class


def load_image(image_id: str) -> np.ndarray:
    """Читает {image_id}.jpg, приводит к IMG_SIZE x IMG_SIZE RGB."""
    path = IMAGES_DIR / f"{image_id}.jpg"
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def build_splits(
    by_class: Dict[str, List[str]], seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Строит сбалансированные train/val/test — по TRAIN_N_PER_CLASS/
    VAL_N_PER_CLASS/TEST_N_PER_CLASS изображений на класс, случайно и
    БЕЗ ПЕРЕСЕЧЕНИЙ отобранных из всех доступных изображений этого класса
    (фиксированный seed — воспроизводимо)."""
    rng = np.random.default_rng(seed)
    code_to_idx = {info[0]: str(idx) for idx, info in CLASS_INFO.items()}

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for code, ids in by_class.items():
        n_needed = TRAIN_N_PER_CLASS + VAL_N_PER_CLASS + TEST_N_PER_CLASS
        if len(ids) < n_needed:
            raise ValueError(f"Класс {code}: доступно {len(ids)} < {n_needed} требуемых.")
        chosen = rng.choice(len(ids), size=n_needed, replace=False)
        chosen_ids = [ids[i] for i in chosen]
        train_ids = chosen_ids[:TRAIN_N_PER_CLASS]
        val_ids = chosen_ids[TRAIN_N_PER_CLASS:TRAIN_N_PER_CLASS + VAL_N_PER_CLASS]
        test_ids = chosen_ids[TRAIN_N_PER_CLASS + VAL_N_PER_CLASS:n_needed]

        label = code_to_idx[code]
        log(f"  класс {code}: загрузка {n_needed} изображений с диска...")
        for image_id in train_ids:
            X_train_list.append(load_image(image_id))
            y_train_list.append(label)
        for image_id in val_ids:
            X_val_list.append(load_image(image_id))
            y_val_list.append(label)
        for image_id in test_ids:
            X_test_list.append(load_image(image_id))
            y_test_list.append(label)

    return (
        np.stack(X_train_list), np.array(y_train_list),
        np.stack(X_val_list), np.array(y_val_list),
        np.stack(X_test_list), np.array(y_test_list),
    )


def vectorize(images: np.ndarray) -> np.ndarray:
    """(M, IMG_SIZE, IMG_SIZE, 3) uint8 -> (M, IMG_SIZE*IMG_SIZE*3) float64
    в [0, 1], построчная развёртка — тот же принцип, что и
    features/vectorization.py (horizontal) для grayscale-изображений
    проекта."""
    M = images.shape[0]
    return images.reshape(M, -1).astype(np.float64) / 255.0


def save_dataset_montage(X_by_class_example: Dict[str, np.ndarray]) -> None:
    """Montage с одним примером на каждый из 7 классов."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tile = IMG_SIZE
    pad = 6
    cols = 4
    rows = 2
    montage = np.full((rows * (tile + pad) + pad, cols * (tile + pad) + pad, 3), 255, dtype=np.uint8)
    for cls_idx in range(7):
        img = X_by_class_example[str(cls_idx)]
        r, c = divmod(cls_idx, cols)
        y0 = pad + r * (tile + pad)
        x0 = pad + c * (tile + pad)
        montage[y0:y0 + tile, x0:x0 + tile] = img
    Image.fromarray(montage).resize(
        (montage.shape[1] * 2, montage.shape[0] * 2), Image.NEAREST
    ).save(FIGURES_DIR / "dataset_montage.png")
    log(f"Обзорный montage датасета сохранён в {FIGURES_DIR}")


def save_reconstruction_figure(
    X_train_vec: np.ndarray, y_train: np.ndarray, X_test_vec: np.ndarray, y_test: np.ndarray,
    demo_class: str = "4",
) -> Dict[str, Any]:
    """Исходное изображение против его проекции на выращенное подпространство
    метода сопряжённости (та же иллюстрация, что и в статье 1) — прямая
    визуализация показателя сопряжённости R(x, Y), а не тривиальная
    нормировка яркости (визуально неотличимая от исходника)."""
    log(f"Строим иллюстрацию проекции на подпространство (класс {demo_class})...")
    X_cls_train = X_train_vec[y_train == demo_class]

    from subspace_conjugacy.algorithms import (
        GlobalMinCosinePairFinder, ReferenceCenterBuilder,
        CosineSecondVectorAttacher, ConjugacyClusterGrowth,
    )
    pair_finder = GlobalMinCosinePairFinder().fit(X_cls_train)
    builder = ReferenceCenterBuilder(n_subclasses=N_SUBCLASSES).fit(X_cls_train, pair_finder.pair_indices_)
    attacher = CosineSecondVectorAttacher().fit(X_cls_train, builder.center_indices_)
    growth = ConjugacyClusterGrowth(freeze_basis_at=None).fit(X_cls_train, attacher.pairs_)

    X_cls_test = X_test_vec[y_test == demo_class]
    n_demo = min(5, len(X_cls_test))
    demo_images = X_cls_test[:n_demo]

    R_per_subclass = np.stack(
        [conjugate_criterion(demo_images, Y) for Y in growth.subspace_bases_], axis=1
    )
    best_subclass = R_per_subclass.argmax(axis=1)
    best_R = R_per_subclass.max(axis=1)

    tile = IMG_SIZE
    pad = 6
    montage = np.full((2 * (tile + pad) + pad, n_demo * (tile + pad) + pad, 3), 255, dtype=np.uint8)
    r_values = []
    for i in range(n_demo):
        x = demo_images[i]
        Y = growth.subspace_bases_[best_subclass[i]]
        gram_inv = compute_gram_inverse(Y)
        # Скобки обязательны: "Y @ gram_inv @ Y.T @ x" из-за левоассоциативности
        # @ вычислился бы как ((Y @ gram_inv) @ Y.T) @ x — второй шаг строит
        # промежуточную матрицу (N, N), при N=IMG_SIZE*IMG_SIZE*3=49152 это
        # ~19.3 ГБ и практическое исчерпание памяти. Явная группировка справа
        # налево — сначала Y.T @ x (k-мерный вектор), затем gram_inv @ (...)
        # (k-мерный), и только в конце Y @ (...) (N-мерный) — вычисляет ТОТ ЖЕ
        # результат математически, но ни разу не материализует матрицу N x N.
        projection = Y @ (gram_inv @ (Y.T @ x))
        orig_img = (x.reshape(tile, tile, 3) * 255.0).clip(0, 255).astype(np.uint8)
        recon_img = (projection.reshape(tile, tile, 3) * 255.0).clip(0, 255).astype(np.uint8)
        x0 = pad + i * (tile + pad)
        montage[pad:pad + tile, x0:x0 + tile] = orig_img
        montage[2 * pad + tile:2 * pad + 2 * tile, x0:x0 + tile] = recon_img
        r_values.append(float(best_R[i]))

    montage_img = Image.fromarray(montage)
    if IMG_SIZE < 64:
        montage_img = montage_img.resize((montage.shape[1] * 2, montage.shape[0] * 2), Image.NEAREST)
    montage_img.save(FIGURES_DIR / "subspace_projection.png")

    log(f"  R по показанным примерам: {[round(r, 3) for r in r_values]}")
    return {
        "demo_class": demo_class,
        "demo_class_label": CLASS_INFO[int(demo_class)][1],
        "n_demo": n_demo,
        "r_values": r_values,
    }


# ----------------------------------------------------------------------
# Базовая модель: компактная CNN (PyTorch) + sklearn-совместимая обёртка
# ----------------------------------------------------------------------
def train_cnn_and_wrap(
    X_train_img: np.ndarray, y_train: np.ndarray,
) -> Any:
    """Обучает компактную CNN на изображениях (M, IMG_SIZE, IMG_SIZE, 3) и
    возвращает sklearn-совместимую ОБЁРТКУ (classes_/predict/predict_proba,
    принимает ТЕ ЖЕ векторизованные (M, IMG_SIZE*IMG_SIZE*3) входы, что и
    остальные модели) — именно такую обёртку и должен написать пользователь
    библиотеки для произвольной внешней модели, как явно указано в
    docstring subspace_conjugacy/models/ensemble.py (специальная обёртка
    под конкретный фреймворк намеренно не включена в саму библиотеку, т.к.
    PyTorch не входит в её зависимости)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(RANDOM_SEED)
    classes_sorted = np.array(sorted(np.unique(y_train)))
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}

    class CompactCNN(nn.Module):
        """Три свёрточных блока (8/16/32 канала) — на один больше, чем в
        первом варианте эксперимента, т.к. вход здесь заметно крупнее
        (IMG_SIZE x IMG_SIZE вместо 28x28): три подвыборки уменьшают
        картинку до IMG_SIZE/8 x IMG_SIZE/8 перед линейным классификатором,
        не позволяя входу полносвязного слоя разрастись непрактично."""

        def __init__(self, num_classes: int, img_size: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            flat_dim = 32 * (img_size // 8) ** 2
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, num_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    model = CompactCNN(num_classes=len(classes_sorted), img_size=IMG_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X_train_img, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y_idx = torch.tensor([class_to_idx[c] for c in y_train], dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CNN_BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(CNN_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"    CNN эпоха {epoch + 1}/{CNN_EPOCHS}: loss={epoch_loss / len(dataset):.4f}")
    model.eval()

    class CNNWrapper:
        """sklearn-совместимая обёртка: принимает векторизованные
        (M, IMG_SIZE*IMG_SIZE*3) входы (как и остальные модели), внутри
        переводит их обратно в (M, IMG_SIZE, IMG_SIZE, 3) для CNN."""

        classes_ = classes_sorted

        def _to_tensor(self, X_vec: np.ndarray) -> "torch.Tensor":
            imgs = X_vec.reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            return torch.tensor(imgs).permute(0, 3, 1, 2)

        def predict_proba(self, X_vec: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                logits = model(self._to_tensor(X_vec))
                proba = torch.softmax(logits, dim=1).numpy()
            return proba

        def predict(self, X_vec: np.ndarray) -> np.ndarray:
            proba = self.predict_proba(X_vec)
            return self.classes_[np.argmax(proba, axis=1)]

    return CNNWrapper()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_images_extracted()
    log("Чтение метаданных (разметка класса для каждого изображения)...")
    by_class = load_metadata()

    log("Построение train/val/test и загрузка изображений с диска "
        f"(изменение размера до {IMG_SIZE}x{IMG_SIZE})...")
    X_train_img, y_train, X_val_img, y_val, X_test_img, y_test = build_splits(by_class, RANDOM_SEED)

    X_train_vec = vectorize(X_train_img)
    X_val_vec = vectorize(X_val_img)
    X_test_vec = vectorize(X_test_img)

    log(f"train={X_train_vec.shape} (сбалансировано, {TRAIN_N_PER_CLASS}/класс), "
        f"val={X_val_vec.shape}, test={X_test_vec.shape}")

    example_by_class = {}
    for cls_idx in range(7):
        idx = np.where(y_train == str(cls_idx))[0][0]
        example_by_class[str(cls_idx)] = X_train_img[idx]
    save_dataset_montage(example_by_class)

    reconstruction_demo = save_reconstruction_figure(X_train_vec, y_train, X_test_vec, y_test)

    # ------------------------------------------------------------------
    # Три базовые модели — обучены на ОДНОМ И ТОМ ЖЕ train-подмножестве
    # ------------------------------------------------------------------
    log("Обучение базовой модели: метод сопряжённости...")
    t0 = time.perf_counter()
    # ВАЖНО (правка методологии, обнаруженная и исправленная одновременно со
    # статьёй 1 — см. article_01_experiment.py): SubspaceConjugacyClassifier
    # (freeze_basis_at="auto") сначала растит каждый класс без ограничения,
    # а ЗАТЕМ равняет ВСЕ n_subclasses x len(классов) подпространств к
    # общему МИНИМАЛЬНОМУ размеру. Из-за неравномерности жадного роста фазы
    # B.2 хотя бы одно из 7*4=28 подпространств почти всегда останавливается
    # на 2 опорных векторах (исходная пара без единого шага роста) — и это
    # единственное "невезучее" подпространство схлопывает ВСЕ 28 к k=2.
    # Обходной путь — тот же, что и в статье 1: строим FursovClusterer
    # ОТДЕЛЬНО для каждого класса (freeze_basis_at=None, полный рост без
    # ограничения) и собираем итоговый классификатор БЕЗ равнения
    # (fit_from_subclass_bases(..., equalize=False)) — каждое подпространство
    # участвует в предсказании со своим реально построенным числом векторов.
    subspace_bases_by_class: Dict[str, List[np.ndarray]] = {}
    for cls in np.unique(y_train):
        X_cls = X_train_vec[y_train == cls]
        cls_clusterer = FursovClusterer(n_subclasses=N_SUBCLASSES, freeze_basis_at=None)
        cls_clusterer.fit(X_cls)
        subspace_bases_by_class[cls] = cls_clusterer.subspaces_
    subspace = SubspaceConjugacyClassifier()
    subspace.fit_from_subclass_bases(subspace_bases_by_class, equalize=False)
    time_subspace = time.perf_counter() - t0
    log(f"  готово за {time_subspace:.1f}с")

    log("Обучение базовой модели: классический ML (PCA + логистическая регрессия)...")
    t0 = time.perf_counter()
    classical_ml = Pipeline([
        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)),
        ("logreg", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
    ])
    classical_ml.fit(X_train_vec, y_train)
    time_classical = time.perf_counter() - t0
    log(f"  готово за {time_classical:.1f}с")

    log("Обучение базовой модели: свёрточная нейронная сеть...")
    t0 = time.perf_counter()
    cnn = train_cnn_and_wrap(X_train_img, y_train)
    time_cnn = time.perf_counter() - t0
    log(f"  готово за {time_cnn:.1f}с")

    base_models = [("subspace", subspace), ("classical_ml", classical_ml), ("cnn", cnn)]

    # ------------------------------------------------------------------
    # Задача 1: диагностика согласованности ошибок на валидационной выборке
    # ------------------------------------------------------------------
    log("Диагностика согласованности ошибок (на валидационной выборке)...")
    val_predictions = {name: est.predict(X_val_vec) for name, est in base_models}
    diagnostics_val = compute_ensemble_diagnostics(y_val, val_predictions)
    log(f"  accuracy_by_model={diagnostics_val['accuracy_by_model']}")
    log(f"  oracle_accuracy={diagnostics_val['oracle_accuracy']:.4f}, "
        f"fraction_exactly_one_wrong={diagnostics_val['fraction_exactly_one_wrong']:.4f}, "
        f"fraction_all_wrong={diagnostics_val['fraction_all_wrong']:.4f}")

    # ------------------------------------------------------------------
    # Задача 2: три схемы ансамблирования
    # ------------------------------------------------------------------
    log("Построение ансамблей (voting/stacking/switching)...")
    voting = PrefitVotingClassifier(estimators=base_models, voting="soft").fit()
    stacking = PrefitStackingClassifier(estimators=base_models).fit(X_val_vec, y_val)
    # Датасет сильно несбалансирован (класс nv ~ 67% выборки, раздел
    # "Датасет") — метамодель стекинга с настройками по умолчанию рискует
    # выродиться в "всегда предсказывать самый частый класс" (высокая ОБЩАЯ
    # точность ценой почти нулевой полноты на остальных классах). Отдельно
    # проверяем тот же стекинг с балансировкой классов метамодели
    # (class_weight="balanced" — стандартный приём sklearn: ошибка на
    # объекте редкого класса штрафуется в обучении сильнее, обратно
    # пропорционально частоте класса), чтобы увидеть, действительно ли
    # проблема в необученной на дисбаланс метамодели, а не в самих базовых
    # моделях/признаках.
    stacking_balanced = PrefitStackingClassifier(
        estimators=base_models,
        meta_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(X_val_vec, y_val)
    switching = SwitchingEnsembleClassifier(estimators=base_models).fit(X_val_vec, y_val)
    log(f"  чемпионы по классам (switching): {switching.class_champion_}")

    # ------------------------------------------------------------------
    # Задача 3: честная оценка на held-out тесте — 3 базовые модели + 4 ансамбля
    # ------------------------------------------------------------------
    log("Финальная оценка на тестовой выборке...")
    all_models = base_models + [
        ("voting", voting), ("stacking", stacking),
        ("stacking_balanced", stacking_balanced), ("switching", switching),
    ]
    test_results = {}
    for name, est in all_models:
        t0 = time.perf_counter()
        y_pred = est.predict(X_test_vec)
        predict_time = time.perf_counter() - t0
        accuracy = float(np.mean(y_pred == y_test))
        per_class = per_class_accuracy(y_test, y_pred)
        # Средняя (макро-усреднённая) точность по классам — честная метрика
        # при сильном дисбалансе: в отличие от общей точности, не позволяет
        # модели, которая всегда предсказывает самый частый класс, получить
        # искусственно высокий результат (тот же принцип, что и "mean
        # accuracy по классам" в остальных честных сравнениях проекта,
        # main.py — там датасеты были сбалансированы по построению, здесь
        # дисбаланс реальный, поэтому разница между двумя метриками
        # становится информативной сама по себе).
        macro_accuracy = float(np.mean(list(per_class.values())))
        test_results[name] = {
            "accuracy": accuracy,
            "macro_accuracy": macro_accuracy,
            "per_class_accuracy": per_class,
            "predict_time_seconds": predict_time,
        }
        log(f"  {name}: accuracy={accuracy:.4f}, macro_accuracy={macro_accuracy:.4f} ({predict_time:.2f}с)")

    diagnostics_test = compute_ensemble_diagnostics(
        y_test, {name: est.predict(X_test_vec) for name, est in base_models}
    )

    # ------------------------------------------------------------------
    # Сохранение результатов
    # ------------------------------------------------------------------
    results = {
        "config": {
            "dataset": (
                f"HAM10000/ISIC2018, полный набор данных, скачан напрямую с "
                f"Harvard Dataverse (doi:10.7910/DVN/DBW86T), изображения "
                f"приведены к {IMG_SIZE}x{IMG_SIZE}"
            ),
            "img_size": IMG_SIZE,
            "classes": {str(k): {"code": v[0], "label_ru": v[1]} for k, v in CLASS_INFO.items()},
            "class_totals_full_dataset": {code: len(ids) for code, ids in by_class.items()},
            "train_n_per_class": TRAIN_N_PER_CLASS,
            "val_n_per_class": VAL_N_PER_CLASS,
            "test_n_per_class": TEST_N_PER_CLASS,
            "n_train_total": int(X_train_vec.shape[0]),
            "n_val_total": int(X_val_vec.shape[0]),
            "n_test_total": int(X_test_vec.shape[0]),
            "n_subclasses": N_SUBCLASSES,
            "pca_components": PCA_COMPONENTS,
            "cnn_epochs": CNN_EPOCHS,
            "random_seed": RANDOM_SEED,
        },
        "reconstruction_demo": reconstruction_demo,
        "base_model_train_time_seconds": {
            "subspace": time_subspace, "classical_ml": time_classical, "cnn": time_cnn,
        },
        "diagnostics_val": _json_safe_diagnostics(diagnostics_val),
        "diagnostics_test": _json_safe_diagnostics(diagnostics_test),
        "switching_class_champion": switching.class_champion_,
        "switching_champion_metric": switching.champion_metric_,
        "test_results": test_results,
        "total_elapsed_seconds": time.perf_counter() - start,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    log(f"Готово за {results['total_elapsed_seconds']:.1f}с. Результаты: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
