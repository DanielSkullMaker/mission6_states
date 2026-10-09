"""Нейросетевая версия эксперимента Статьи 4 (dissertation_articles_plan.txt):

  «Использование последовательной структуры смежных срезов медицинских
  изображений для повышения точности классификации» — нейросетевой подход.

Контекст перехода: классическая версия эксперимента
(`experiment_sequential_smoothing.py`, `SubspaceConjugacyClassifier` +
ручное усреднение соседних векторов признаков) не дала статистически
значимого результата — на честном патиентском сплите эффект сглаживания
неотличим от шума и даже меняет знак при расширении тестовой выборки
(см. память project_article4_significance). Решение: заменить ручное
усреднение признаков на СВЁРТОЧНУЮ СЕТЬ (2.5D CNN), которая сама учится
использовать соседние по оси Z срезы, а не усредняет их по фиксированной
формуле. Архитектура энкодера (блоки Conv-Conv-MaxPool, 32->64->128->256
каналов) ориентирована на эталонный 2D U-Net для BraTS из
`state4_scripts/3d-mri-brain-tumor-segmentation-u-net*.ipynb` (тот же
семейный паттерн, что используется для сегментации на этом датасете),
упрощённая под бинарную классификацию (без decoder-части) и переписанная
на PyTorch (уже опциональная зависимость проекта, extras "experiments"),
а не Keras/TensorFlow, которых в проекте нет.

Методология (сопоставимость с классической версией):
  - Тот же сплит ПО ПАЦИЕНТАМ (RANDOM_SEED=42, 80/20) — идентичная функция
    `split_patients`, поэтому train/test пациенты совпадают 1:1 с
    `experiment_sequential_smoothing.py` — числа напрямую сравнимы.
  - Два обучаемых варианта модели, одна архитектура (SliceStackCNN),
    отличаются только числом входных каналов K:
      - K=1 (context_cnn=False) — классифицирует один срез, без контекста
        соседей (нейросетевой аналог meaningful_unsmoothed).
      - K=N_CONTEXT (по умолчанию 5) — вход: N_CONTEXT срезов, центрированных
        на классифицируемом, уложенные как каналы (2.5D) — модель сама
        решает, как их комбинировать (нейросетевой аналог meaningful_smoothed,
        но свёртка вместо усреднения).
  - H2-контроль (аналог shuffled_smoothed): ТА ЖЕ обученная K=N_CONTEXT
    модель оценивается на тех же тестовых срезах, но с "соседями",
    замененными на случайные срезы из тестового пула (вместо физически
    смежных) — если точность падает к уровню K=1, значит модель
    действительно использует физическую смежность, а не просто получает
    больше шумных данных на вход.
  - Обучающая выборка — ВСЕ срезы train-пациентов (с ограничением
    MAX_TRAIN_PER_CLASS по времени CPU-обучения), а не выборка 100/класс,
    как у классического метода — это намеренное отличие (CNN требует
    объёма данных, подпространственный метод — нет), оговаривается в тексте
    статьи, а не скрывается.

Запуск:
    python experiments/experiment_article4_cnn_sequential.py

Требует: pip install -e ".[experiments]" (torch), датасет уже
подготовлен `experiments/prepare_brats_dataset.py` -> datasets/brats_flair_slices/.
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from subspace_conjugacy.evaluation.metrics import evaluate_classifier

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
TEST_PATIENT_FRACTION = 0.2  # идентично experiment_sequential_smoothing.py

N_CONTEXT = 5  # ширина стека соседних срезов для "контекстной" модели
IMG_SIZE = 96  # уменьшено с 256 (датасет) для CPU-обучения, см. docstring
MAX_TRAIN_PER_CLASS = 1500  # ограничение по времени CPU-обучения, не по методологии
BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 1e-3

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_cnn_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_cnn_results.md"

Row = Dict[str, object]


def load_metadata() -> List[Row]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} не найден. Сначала запустите: "
            'BRATS_ROOT="<путь>" python experiments/prepare_brats_dataset.py'
        )
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "slice_idx": int(r["slice_idx"]),
                    "path": str(DATASET_ROOT / r["path"]),
                    "has_tumor": r["has_tumor"] == "True",
                }
            )
    return rows


def split_patients(rows: List[Row]) -> Tuple[List[str], List[str]]:
    """Идентична experiment_sequential_smoothing.py::split_patients — тот же
    RANDOM_SEED, та же перестановка -> те же train/test пациенты."""
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * TEST_PATIENT_FRACTION))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def group_by_patient(rows: List[Row], patient_ids: List[str]) -> Dict[str, List[Row]]:
    """Срезы каждого пациента, отсортированные по slice_idx (физический порядок)."""
    wanted = set(patient_ids)
    groups: Dict[str, List[Row]] = {pid: [] for pid in patient_ids}
    for r in rows:
        if r["patient_id"] in wanted:
            groups[r["patient_id"]].append(r)
    for pid in groups:
        groups[pid].sort(key=lambda r: r["slice_idx"])
    return groups


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class SliceStackDataset(Dataset):
    """Для каждого (patient, center_index) отдаёт стек из `n_context` срезов
    (центрированный на center_index, edge-padding на границах пациента) как
    каналы тензора (n_context, IMG_SIZE, IMG_SIZE), плюс бинарную метку
    has_tumor самого центрального среза."""

    def __init__(self, groups: Dict[str, List[Row]], centers: List[Tuple[str, int]], n_context: int):
        self.groups = groups
        self.centers = centers
        self.n_context = n_context
        self._cache: Dict[str, np.ndarray] = {}

    def _patient_images(self, pid: str) -> np.ndarray:
        if pid not in self._cache:
            self._cache[pid] = np.stack([load_image(r["path"]) for r in self.groups[pid]])
        return self._cache[pid]

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int):
        pid, center = self.centers[idx]
        images = self._patient_images(pid)
        n = images.shape[0]
        half = self.n_context // 2
        idxs = [min(max(center + off, 0), n - 1) for off in range(-half, -half + self.n_context)]
        stack = images[idxs]
        label = float(self.groups[pid][center]["has_tumor"])
        return torch.from_numpy(stack), torch.tensor(label, dtype=torch.float32)


class ShuffledNeighborDataset(Dataset):
    """H2-контроль: те же центральные срезы, но "соседи" — случайные срезы
    из всего тестового пула (не физически смежные). Центральный срез (и его
    метка) не трогается."""

    def __init__(self, groups: Dict[str, List[Row]], centers: List[Tuple[str, int]], n_context: int, seed: int):
        self.groups = groups
        self.centers = centers
        self.n_context = n_context
        self._cache: Dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)
        all_pairs = [(pid, i) for pid, imgs in groups.items() for i in range(len(imgs))]
        self._random_pool = all_pairs
        self._rng = rng

    def _patient_images(self, pid: str) -> np.ndarray:
        if pid not in self._cache:
            self._cache[pid] = np.stack([load_image(r["path"]) for r in self.groups[pid]])
        return self._cache[pid]

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int):
        pid, center = self.centers[idx]
        center_img = self._patient_images(pid)[center]
        n_neighbors = self.n_context - 1
        half_before = n_neighbors // 2
        half_after = n_neighbors - half_before
        picks = self._rng.integers(0, len(self._random_pool), size=self.n_context)
        stack = []
        inserted_center = False
        for slot in range(self.n_context):
            if slot == half_before and not inserted_center:
                stack.append(center_img)
                inserted_center = True
                continue
            other_pid, other_idx = self._random_pool[picks[slot]]
            stack.append(self._patient_images(other_pid)[other_idx])
        label = float(self.groups[pid][center]["has_tumor"])
        return torch.from_numpy(np.stack(stack)), torch.tensor(label, dtype=torch.float32)


class SliceStackCNN(nn.Module):
    """Компактный свёрточный энкодер (архитектура в стиле энкодера 2D U-Net
    из state4_scripts/*.ipynb, без decoder-части — задача классификации, не
    сегментация) + классификационная голова."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def build_center_list(groups: Dict[str, List[Row]], max_per_class: int, seed: int) -> List[Tuple[str, int]]:
    """Случайная стратифицированная (по has_tumor) подвыборка центров среди
    всех (patient, slice_index) — ограничивает время CPU-обучения."""
    by_label: Dict[bool, List[Tuple[str, int]]] = {True: [], False: []}
    for pid, imgs in groups.items():
        for i, r in enumerate(imgs):
            by_label[bool(r["has_tumor"])].append((pid, i))
    rng = np.random.default_rng(seed)
    selected: List[Tuple[str, int]] = []
    for label, items in by_label.items():
        idx = rng.permutation(len(items))[:max_per_class]
        selected.extend(items[i] for i in idx)
    rng.shuffle(selected)
    return selected


def all_centers(groups: Dict[str, List[Row]]) -> List[Tuple[str, int]]:
    return [(pid, i) for pid, imgs in groups.items() for i in range(len(imgs))]


def train_model(groups: Dict[str, List[Row]], centers: List[Tuple[str, int]], n_context: int) -> Tuple[nn.Module, float]:
    torch.manual_seed(RANDOM_SEED)
    dataset = SliceStackDataset(groups, centers, n_context)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = SliceStackCNN(in_channels=n_context)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for X, y in loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"   [K={n_context}] epoch {epoch + 1}/{EPOCHS}: loss={epoch_loss / max(n_batches, 1):.4f}")
    fit_seconds = time.perf_counter() - t0
    return model, fit_seconds


@torch.no_grad()
def evaluate_model(model: nn.Module, dataset: Dataset) -> Dict[str, object]:
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    all_preds: List[str] = []
    all_true: List[str] = []
    for X, y in loader:
        logits = model(X)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        all_preds.extend("tumor" if p == 1 else "no_tumor" for p in preds.tolist())
        all_true.extend("tumor" if t == 1 else "no_tumor" for t in y.long().tolist())

    report = evaluate_classifier(np.array(all_true), np.array(all_preds))
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    return {
        "n_test": len(all_true),
        "accuracy": report["accuracy"],
        "per_class_accuracy": per_class,
        "confusion_matrix": report["confusion_matrix"].tolist(),
        "labels": [str(label) for label in report["labels"]],
    }


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4 (нейросетевой подход): 2.5D CNN со стеком соседних срезов ===")

    rows = load_metadata()
    print(f"Загружено {len(rows)} срезов, {len({r['patient_id'] for r in rows})} пациентов.")

    train_patient_ids, test_patient_ids = split_patients(rows)
    print(f"Пациенты: {len(train_patient_ids)} train, {len(test_patient_ids)} test (тот же сплит, что в классическом эксперименте).")

    train_groups = group_by_patient(rows, train_patient_ids)
    test_groups = group_by_patient(rows, test_patient_ids)

    train_centers = build_center_list(train_groups, MAX_TRAIN_PER_CLASS, RANDOM_SEED)
    test_centers = all_centers(test_groups)
    print(f"n_train={len(train_centers)} (ограничено MAX_TRAIN_PER_CLASS={MAX_TRAIN_PER_CLASS}/класс), n_test={len(test_centers)} (все срезы test-пациентов).")

    print("\n--- Обучение K=1 (без контекста соседей) ---")
    model_k1, fit_seconds_k1 = train_model(train_groups, train_centers, n_context=1)

    print(f"\n--- Обучение K={N_CONTEXT} (со стеком соседних срезов) ---")
    model_kn, fit_seconds_kn = train_model(train_groups, train_centers, n_context=N_CONTEXT)

    print("\n--- Оценка на held-out тестовых пациентах ---")
    result_k1 = evaluate_model(model_k1, SliceStackDataset(test_groups, test_centers, n_context=1))
    print(f"K=1 (no_context): accuracy={result_k1['accuracy']:.4f}")

    result_kn_meaningful = evaluate_model(model_kn, SliceStackDataset(test_groups, test_centers, n_context=N_CONTEXT))
    print(f"K={N_CONTEXT} (meaningful_neighbors): accuracy={result_kn_meaningful['accuracy']:.4f}")

    result_kn_shuffled = evaluate_model(
        model_kn, ShuffledNeighborDataset(test_groups, test_centers, n_context=N_CONTEXT, seed=RANDOM_SEED)
    )
    print(f"K={N_CONTEXT} (shuffled_neighbors, H2-контроль): accuracy={result_kn_shuffled['accuracy']:.4f}")

    results = {
        "no_context_k1": result_k1,
        "meaningful_neighbors": result_kn_meaningful,
        "shuffled_neighbors": result_kn_shuffled,
    }

    acc = {name: r["accuracy"] for name, r in results.items()}
    h1 = acc["meaningful_neighbors"] > acc["no_context_k1"]
    h2 = acc["shuffled_neighbors"] <= acc["meaningful_neighbors"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "random_seed": RANDOM_SEED,
        "n_context": N_CONTEXT,
        "img_size": IMG_SIZE,
        "max_train_per_class": MAX_TRAIN_PER_CLASS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_train": len(train_centers),
        "fit_seconds_k1": fit_seconds_k1,
        "fit_seconds_kn": fit_seconds_kn,
        "train_patients": train_patient_ids,
        "test_patients": test_patient_ids,
        "results": results,
        "hypotheses": {
            "H1_context_helps": h1,
            "H2_shuffled_neighbors_not_better_than_meaningful": h2,
        },
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 (нейросетевой подход) — результаты эксперимента",
        "",
        f"`experiments/experiment_article4_cnn_sequential.py`. Датасет: "
        f"`datasets/brats_flair_slices/` (BraTS2020, FLAIR, "
        f"{len(train_patient_ids) + len(test_patient_ids)} пациентов, тот же "
        "сплит по пациентам, что в классическом эксперименте). Архитектура: "
        f"компактная свёрточная сеть (32->64->128->256 каналов), стек "
        f"соседних срезов как входные каналы, {IMG_SIZE}x{IMG_SIZE}, "
        f"{EPOCHS} эпох, Adam(lr={LEARNING_RATE}).",
        "",
        "| Конфигурация | n_test | accuracy | tumor | no_tumor |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        pc = r["per_class_accuracy"]
        lines.append(
            f"| {name} | {r['n_test']} | {r['accuracy']:.4f} | "
            f"{pc.get('tumor', float('nan')):.3f} | {pc.get('no_tumor', float('nan')):.3f} |"
        )
    lines += [
        "",
        f"**H1** (контекст соседних срезов помогает: meaningful_neighbors > "
        f"no_context_k1): {acc['meaningful_neighbors']:.4f} > {acc['no_context_k1']:.4f} -> **{h1}**.",
        f"**H2** (случайные 'соседи' не лучше физически смежных: "
        f"shuffled_neighbors <= meaningful_neighbors): "
        f"{acc['shuffled_neighbors']:.4f} <= {acc['meaningful_neighbors']:.4f} -> **{h2}**.",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
