"""Статья 4 (нейросетевой подход, задача изменена на СЕГМЕНТАЦИЮ):

  «Анализ последовательной структуры смежных срезов медицинских
  изображений методом свёрточных нейронных сетей»

Контекст второго перехода (dissertation_articles_plan.txt): классификация
has_tumor/no_tumor (первая нейросетевая версия,
`experiment_article4_cnn_sequential.py`) заменена на попиксельную бинарную
СЕГМЕНТАЦИЮ опухоли (опухоль/фон) — та же задача, что в эталонных
ноутбуках `state4_scripts/3d-mri-brain-tumor-segmentation-u-net*.ipynb`
(там 4 класса, здесь — бинарный вариант "whole tumor" ради CPU-пригодности,
см. обсуждение в dissertation_articles_plan.txt). Подпространственный метод
(`SubspaceConjugacyClassifier`) из статьи полностью убран — сравнение
больше не ведётся.

Ключевой методологический урок предыдущего шага (см. обсуждение с
пользователем): пиксельная accuracy — ОБМАНЧИВАЯ метрика для сегментации
при сильном классовом перекосе (опухоль — обычно 1-3% пикселей срeза):
модель, предсказывающая "везде фон", получит >97% accuracy, не научившись
ничему. Поэтому здесь основная метрика — коэффициент Дайса (Dice), как и
в ноутбуке-источнике, а accuracy приводится лишь для прозрачности.

Архитектура: компактный U-Net (энкодер 32->64->128->256 + decoder с
skip-connections) — тот же семейный паттерн, что в ноутбуке, но меньше
(без 512-канального уровня) для практичности на CPU, PyTorch вместо
Keras/TF.

Методология "последовательной структуры" (тема диссертации сохранена):
те же 3 конфигурации, что и в классификационной версии, но для
сегментации:
  - no_context_k1       — U-Net на одном срезе (1 входной канал).
  - meaningful_neighbors — та же архитектура, вход — стек из N_CONTEXT
    физически смежных срезов как каналов (2.5D).
  - shuffled_neighbors   — ТА ЖЕ обученная meaningful_neighbors-модель,
    но "соседи" заменены случайными срезами (H2-контроль).
  Центральный срез (и его маска) — всегда тот же.

Запуск:
    python experiments/experiment_article4_unet_segmentation.py

Требует: маски сегментации, подготовленные
`experiments/prepare_brats_segmentation_masks.py` (binary *_mask.png рядом
с каждым FLAIR-срезом в datasets/brats_flair_slices/).
"""

import csv
import functools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
TEST_PATIENT_FRACTION = 0.2

N_CONTEXT = 5
IMG_SIZE = 96  # уменьшено с 128 — свободной RAM в системе мало (~3.3GB), см. OOM при первом прогоне
MAX_TRAIN_PER_CLASS = 1000  # по наличию опухоли на срезе (has_tumor) — ограничение по времени/памяти CPU
BATCH_SIZE = 8  # уменьшено с 16 — активации U-Net с decoder-путём заметно тяжелее по памяти, чем у классификатора без decoder (см. OOM при первом прогоне)
EPOCHS = 6
LEARNING_RATE = 1e-3

# Гиперпараметры (IMG_SIZE/BATCH_SIZE/EPOCHS) оставлены БЕЗ изменений при
# переходе на GPU — переносится только устройство вычислений, чтобы
# сравнение архитектур/конфигураций оставалось "при прочих равных"
# (см. docstring experiment_article4_architectures.py).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cuDNN по умолчанию НЕдетерминирован (сам выбирает алгоритм свёртки,
# может отличаться между запусками даже при одинаковом torch.manual_seed) —
# в отличие от CPU. Без этого сравнение "при одинаковом RANDOM_SEED"
# на GPU было бы нечестной иллюзией воспроизводимости.
if DEVICE.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_unet_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_unet_results.md"

Row = Dict[str, object]


def load_metadata() -> List[Row]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} не найден.")
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img_path = DATASET_ROOT / r["path"]
            mask_path = img_path.with_name(img_path.stem + "_mask.png")
            if not mask_path.exists():
                continue
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "slice_idx": int(r["slice_idx"]),
                    "path": str(img_path),
                    "mask_path": str(mask_path),
                    "has_tumor": r["has_tumor"] == "True",
                    "tumor_pixel_fraction": float(r["tumor_pixel_fraction"]),
                }
            )
    return rows


def split_patients(rows: List[Row], seed: int = RANDOM_SEED) -> Tuple[List[str], List[str]]:
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * TEST_PATIENT_FRACTION))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def group_by_patient(rows: List[Row], patient_ids: List[str]) -> Dict[str, List[Row]]:
    wanted = set(patient_ids)
    groups: Dict[str, List[Row]] = {pid: [] for pid in patient_ids}
    for r in rows:
        if r["patient_id"] in wanted:
            groups[r["patient_id"]].append(r)
    for pid in groups:
        groups[pid].sort(key=lambda r: r["slice_idx"])
    return groups


@functools.lru_cache(maxsize=2048)
def load_image(path: str) -> np.ndarray:
    """Загружает один срез. Общий ограниченный (LRU) кэш на уровне модуля —
    переиспользуется ВСЕМИ
    датасетами (train/test, K=1/K=5, meaningful/shuffled), чтобы не
    держать в памяти по отдельной копии каждого пациента на каждый
    датасет (так ловился OOM при полном прогоне до этого исправления —
    предыдущий вариант кэшировал ЦЕЛОГО пациента на первое обращение к
    любому его срезу, что для ~130 срезов/пациента и 5 одновременных
    датасетов в памяти было избыточно)."""
    from PIL import Image

    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


@functools.lru_cache(maxsize=2048)
def load_mask(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return (np.asarray(img, dtype=np.float32) > 127).astype(np.float32)


class SegStackDataset(Dataset):
    """Стек из n_context соседних срезов (каналы) + маска центрального среза."""

    def __init__(self, groups: Dict[str, List[Row]], centers: List[Tuple[str, int]], n_context: int):
        self.groups = groups
        self.centers = centers
        self.n_context = n_context

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int):
        pid, center = self.centers[idx]
        patient_rows = self.groups[pid]
        n = len(patient_rows)
        half = self.n_context // 2
        idxs = [min(max(center + off, 0), n - 1) for off in range(-half, -half + self.n_context)]
        stack = np.stack([load_image(patient_rows[i]["path"]) for i in idxs])
        mask = load_mask(patient_rows[center]["mask_path"])
        return torch.from_numpy(stack), torch.from_numpy(mask).unsqueeze(0)


class ShuffledNeighborSegDataset(Dataset):
    """H2-контроль: центральный срез/маска те же, "соседи" случайны."""

    def __init__(self, groups: Dict[str, List[Row]], centers: List[Tuple[str, int]], n_context: int, seed: int):
        self.groups = groups
        self.centers = centers
        self.n_context = n_context
        self._rng = np.random.default_rng(seed)
        self._random_pool = [(pid, i) for pid, imgs in groups.items() for i in range(len(imgs))]

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int):
        pid, center = self.centers[idx]
        center_row = self.groups[pid][center]
        half_before = (self.n_context - 1) // 2
        picks = self._rng.integers(0, len(self._random_pool), size=self.n_context)
        stack = []
        inserted = False
        for slot in range(self.n_context):
            if slot == half_before and not inserted:
                stack.append(load_image(center_row["path"]))
                inserted = True
                continue
            other_pid, other_idx = self._random_pool[picks[slot]]
            stack.append(load_image(self.groups[other_pid][other_idx]["path"]))
        mask = load_mask(center_row["mask_path"])
        return torch.from_numpy(np.stack(stack)), torch.from_numpy(mask).unsqueeze(0)


def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    """BatchNorm после каждой 3x3-свёртки (добавлен по запросу пользователя
    — без нормализации эти блоки за малый бюджет эпох нестабильно
    застревали в тривиальном предсказании "всегда фон", см.
    project_article4_architectures_preprocessing в памяти проекта)."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Компактный U-Net (энкодер 32->64->128->256, decoder с skip-connections)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.bottleneck = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.flatten(1)
    target_flat = target.flatten(1)
    intersection = (probs_flat * target_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)


@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    preds_flat = preds.flatten(1)
    target_flat = target.flatten(1)
    intersection = (preds_flat * target_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def precision_recall_f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def wilson_ci(successes: float, n: float, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """95%-доверительный интервал Уилсона для биномиальной доли (по
    умолчанию z=1.96). Добавлен по требованию медицинского рецензента (см.
    medical_article_requirements.txt, п.3) — точечные оценки accuracy/
    precision/recall без интервала вводят в заблуждение при небольшом n.
    Возвращает None при n=0 (интервал неопределён)."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def build_center_list(groups: Dict[str, List[Row]], max_per_class: int, seed: int) -> List[Tuple[str, int]]:
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


def train_model(
    groups: Dict[str, List[Row]],
    centers: List[Tuple[str, int]],
    n_context: int,
    model_factory=None,
) -> Tuple[nn.Module, float]:
    """model_factory(in_channels) -> nn.Module; по умолчанию SmallUNet —
    параметризовано, чтобы experiment_article4_architectures.py мог
    переиспользовать тот же цикл обучения/гиперпараметры для других
    архитектур (Attention U-Net, Residual U-Net, без skip-connections)."""
    if model_factory is None:
        model_factory = SmallUNet

    torch.manual_seed(RANDOM_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
    dataset = SegStackDataset(groups, centers, n_context)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = model_factory(n_context).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    t0 = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = combined_loss(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_dice += dice_coefficient(logits, y)
            n_batches += 1
        print(
            f"   [K={n_context}] epoch {epoch + 1}/{EPOCHS}: "
            f"loss={epoch_loss / max(n_batches, 1):.4f}  train_dice={epoch_dice / max(n_batches, 1):.4f}"
        )
    fit_seconds = time.perf_counter() - t0
    return model, fit_seconds


@torch.no_grad()
def evaluate_model(model: nn.Module, dataset: Dataset) -> Dict[str, object]:
    """Помимо общего Dice по всем срезам, отдельно считает Dice для срезов
    с has_tumor=True и has_tumor=False — усреднённый по ВСЕМ срезам Dice
    смешивает две разные подзадачи: "узнать пустой срез" (сглаженный Dice
    тривиально близок к 1.0, если модель верно предсказала "весь фон", но
    резко падает от одного false-positive пикселя — смотри smooth=1.0 в
    dice_coefficient) и "точно сегментировать опухоль там, где она есть"
    (содержательная задача статьи). has_tumor берётся из dataset.groups/
    dataset.centers (есть у SegStackDataset и ShuffledNeighborSegDataset),
    без изменения __getitem__/порядка итерации (loader с shuffle=False)."""
    model = model.to(DEVICE)
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    dice_scores: List[float] = []
    pred_has_tumor_flags: List[bool] = []
    pixel_correct = 0
    pixel_total = 0
    tp_pixels = 0.0
    fp_pixels = 0.0
    fn_pixels = 0.0
    tn_pixels = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        for i in range(X.shape[0]):
            dice_scores.append(dice_coefficient(logits[i : i + 1], y[i : i + 1]))
        preds = (torch.sigmoid(logits) > 0.5).float()
        pixel_correct += (preds == y).sum().item()
        pixel_total += y.numel()
        tp_pixels += (preds * y).sum().item()
        fp_pixels += (preds * (1.0 - y)).sum().item()
        fn_pixels += ((1.0 - preds) * y).sum().item()
        tn_pixels += ((1.0 - preds) * (1.0 - y)).sum().item()
        pred_has_tumor_flags.extend((preds.flatten(1).sum(dim=1) > 0).cpu().tolist())

    has_tumor_flags = [bool(dataset.groups[pid][idx]["has_tumor"]) for pid, idx in dataset.centers]
    dice_with_tumor = [d for d, flag in zip(dice_scores, has_tumor_flags) if flag]
    dice_without_tumor = [d for d, flag in zip(dice_scores, has_tumor_flags) if not flag]

    # F1 №1 — пиксельный, агрегированный по ВСЕМУ тест-сету (один общий TP/FP/FN,
    # а не усреднение per-слайс, как у Dice выше). Математически на уровне ОДНОГО
    # изображения Dice == F1, но агрегированный по датасету F1 — другое число
    # (не среднее отношений, а отношение сумм) — часто вторая обязательная
    # метрика рядом со средним Dice в литературе по сегментации.
    pixel_precision, pixel_recall, pixel_f1 = precision_recall_f1(tp_pixels, fp_pixels, fn_pixels)

    # F1 №2 — детекция опухоли НА УРОВНЕ СРЕЗА (бинарная классификация: предсказал
    # ли срез хоть один пиксель опухоли, против истинного has_tumor) — отдельная
    # от точности контура метрика "находит ли сеть опухоль вообще".
    slice_tp = sum(1 for p, t in zip(pred_has_tumor_flags, has_tumor_flags) if p and t)
    slice_fp = sum(1 for p, t in zip(pred_has_tumor_flags, has_tumor_flags) if p and not t)
    slice_fn = sum(1 for p, t in zip(pred_has_tumor_flags, has_tumor_flags) if not p and t)
    slice_tn = sum(1 for p, t in zip(pred_has_tumor_flags, has_tumor_flags) if not p and not t)
    slice_precision, slice_recall, slice_f1 = precision_recall_f1(slice_tp, slice_fp, slice_fn)

    # Доверительные интервалы (Wilson, 95%) для биномиальных долей — по
    # требованию медицинского рецензента (medical_article_requirements.txt,
    # п.3): точечная оценка recall/precision без интервала вводит в
    # заблуждение, особенно при малом n. slice_detection_recall — доля
    # срезов С опухолью, где она была обнаружена (успехи=slice_tp,
    # n=n_with_tumor); slice_detection_precision — доля срезов, где
    # предсказано "есть опухоль", оказавшихся верными (успехи=slice_tp,
    # n=slice_tp+slice_fp).
    n_with_tumor = len(dice_with_tumor)
    n_without_tumor = len(dice_without_tumor)
    ci_detection_recall = wilson_ci(slice_tp, n_with_tumor)
    ci_detection_precision = wilson_ci(slice_tp, slice_tp + slice_fp)
    ci_pixel_recall = wilson_ci(tp_pixels, tp_pixels + fn_pixels)
    ci_pixel_precision = wilson_ci(tp_pixels, tp_pixels + fp_pixels)

    return {
        "n_test": len(dice_scores),
        "mean_dice": float(np.mean(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_f1": pixel_f1,
        "pixel_precision_ci95": ci_pixel_precision,
        "pixel_recall_ci95": ci_pixel_recall,
        "pixel_confusion_matrix": {
            "tp": tp_pixels, "fp": fp_pixels, "fn": fn_pixels, "tn": tn_pixels,
        },
        "slice_detection_precision": slice_precision,
        "slice_detection_recall": slice_recall,
        "slice_detection_f1": slice_f1,
        "slice_detection_precision_ci95": ci_detection_precision,
        "slice_detection_recall_ci95": ci_detection_recall,
        "slice_confusion_matrix": {
            "tp": slice_tp, "fp": slice_fp, "fn": slice_fn, "tn": slice_tn,
        },
        "pixel_accuracy": pixel_correct / pixel_total,
        "n_with_tumor": n_with_tumor,
        "mean_dice_with_tumor": float(np.mean(dice_with_tumor)) if dice_with_tumor else None,
        "median_dice_with_tumor": float(np.median(dice_with_tumor)) if dice_with_tumor else None,
        "n_without_tumor": n_without_tumor,
        "mean_dice_without_tumor": float(np.mean(dice_without_tumor)) if dice_without_tumor else None,
        "median_dice_without_tumor": float(np.median(dice_without_tumor)) if dice_without_tumor else None,
    }


def save_qualitative_examples(
    model_k1: nn.Module,
    model_kn: nn.Module,
    test_groups: Dict[str, List[Row]],
    n_examples: int = 3,
    seed: int = RANDOM_SEED,
) -> None:
    """Сохраняет сырые массивы (срез/разметка/3 предсказания) для нескольких
    показательных тестовых срезов — отдельный скрипт
    generate_article4_unet_figures.py строит из них иллюстрацию для статьи.
    Выбираются срезы с заметной долей опухоли (tumor_pixel_fraction) из
    РАЗНЫХ пациентов, чтобы не повторять одного и того же человека."""
    model_k1 = model_k1.to(DEVICE)
    model_kn = model_kn.to(DEVICE)
    model_k1.eval()
    model_kn.eval()

    candidates = [
        (pid, i, row["tumor_pixel_fraction"])
        for pid, rows in test_groups.items()
        for i, row in enumerate(rows)
    ]
    candidates.sort(key=lambda t: t[2], reverse=True)

    chosen: List[Tuple[str, int]] = []
    seen_patients = set()
    for pid, i, _frac in candidates:
        if pid in seen_patients:
            continue
        chosen.append((pid, i))
        seen_patients.add(pid)
        if len(chosen) >= n_examples:
            break

    images, gt_masks, pred_k1, pred_meaningful, pred_shuffled = [], [], [], [], []
    with torch.no_grad():
        for pid, center in chosen:
            ds_k1 = SegStackDataset(test_groups, [(pid, center)], n_context=1)
            ds_kn = SegStackDataset(test_groups, [(pid, center)], n_context=N_CONTEXT)
            ds_shuf = ShuffledNeighborSegDataset(test_groups, [(pid, center)], n_context=N_CONTEXT, seed=seed)

            x_k1, y = ds_k1[0]
            x_kn, _ = ds_kn[0]
            x_shuf, _ = ds_shuf[0]
            x_k1_dev = x_k1.unsqueeze(0).to(DEVICE)
            x_kn_dev = x_kn.unsqueeze(0).to(DEVICE)
            x_shuf_dev = x_shuf.unsqueeze(0).to(DEVICE)

            images.append(x_k1[0].numpy())
            gt_masks.append(y[0].numpy())
            pred_k1.append((torch.sigmoid(model_k1(x_k1_dev))[0, 0] > 0.5).cpu().numpy())
            pred_meaningful.append((torch.sigmoid(model_kn(x_kn_dev))[0, 0] > 0.5).cpu().numpy())
            pred_shuffled.append((torch.sigmoid(model_kn(x_shuf_dev))[0, 0] > 0.5).cpu().numpy())

    out_path = ARTIFACTS_DIR / "experiment_article4_unet_examples.npz"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        images=np.stack(images),
        gt_masks=np.stack(gt_masks),
        pred_k1=np.stack(pred_k1).astype(np.uint8),
        pred_meaningful=np.stack(pred_meaningful).astype(np.uint8),
        pred_shuffled=np.stack(pred_shuffled).astype(np.uint8),
        patient_ids=np.array([pid for pid, _ in chosen]),
    )
    print(f"Примеры для иллюстрации сохранены: {out_path}")


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4 (нейросетевой подход, сегментация): U-Net со стеком соседних срезов ===")
    print(f"Устройство вычислений: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    rows = load_metadata()
    print(f"Загружено {len(rows)} срезов (с масками), {len({r['patient_id'] for r in rows})} пациентов.")

    train_patient_ids, test_patient_ids = split_patients(rows)
    print(f"Пациенты: {len(train_patient_ids)} train, {len(test_patient_ids)} test.")

    train_groups = group_by_patient(rows, train_patient_ids)
    test_groups = group_by_patient(rows, test_patient_ids)

    train_centers = build_center_list(train_groups, MAX_TRAIN_PER_CLASS, RANDOM_SEED)
    test_centers = all_centers(test_groups)
    print(f"n_train={len(train_centers)}, n_test={len(test_centers)}.")

    print("\n--- Обучение K=1 (без контекста соседей) ---")
    model_k1, fit_seconds_k1 = train_model(train_groups, train_centers, n_context=1)

    print(f"\n--- Обучение K={N_CONTEXT} (со стеком соседних срезов) ---")
    model_kn, fit_seconds_kn = train_model(train_groups, train_centers, n_context=N_CONTEXT)

    print("\n--- Оценка на held-out тестовых пациентах (метрика: Dice) ---")
    result_k1 = evaluate_model(model_k1, SegStackDataset(test_groups, test_centers, n_context=1))
    print(f"K=1 (no_context): mean_dice={result_k1['mean_dice']:.4f}  pixel_acc={result_k1['pixel_accuracy']:.4f}")

    result_kn_meaningful = evaluate_model(model_kn, SegStackDataset(test_groups, test_centers, n_context=N_CONTEXT))
    print(f"K={N_CONTEXT} (meaningful_neighbors): mean_dice={result_kn_meaningful['mean_dice']:.4f}  pixel_acc={result_kn_meaningful['pixel_accuracy']:.4f}")

    result_kn_shuffled = evaluate_model(
        model_kn, ShuffledNeighborSegDataset(test_groups, test_centers, n_context=N_CONTEXT, seed=RANDOM_SEED)
    )
    print(f"K={N_CONTEXT} (shuffled_neighbors, H2-контроль): mean_dice={result_kn_shuffled['mean_dice']:.4f}  pixel_acc={result_kn_shuffled['pixel_accuracy']:.4f}")

    print("\n--- Сохранение показательных примеров для иллюстрации ---")
    save_qualitative_examples(model_k1, model_kn, test_groups)

    results = {
        "no_context_k1": result_k1,
        "meaningful_neighbors": result_kn_meaningful,
        "shuffled_neighbors": result_kn_shuffled,
    }

    dice = {name: r["mean_dice"] for name, r in results.items()}
    h1 = dice["meaningful_neighbors"] > dice["no_context_k1"]
    h2 = dice["shuffled_neighbors"] <= dice["meaningful_neighbors"]

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
            "H1_context_helps_dice": h1,
            "H2_shuffled_neighbors_not_better_than_meaningful": h2,
        },
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 (нейросетевой подход, сегментация) — результаты эксперимента",
        "",
        f"`experiments/experiment_article4_unet_segmentation.py`. Датасет: "
        f"`datasets/brats_flair_slices/` (BraTS2020, FLAIR, бинарные маски "
        f"опухоль/фон). Архитектура: компактный U-Net (32->64->128->256), "
        f"{IMG_SIZE}x{IMG_SIZE}, {EPOCHS} эпох, Adam(lr={LEARNING_RATE}), "
        "loss = BCE + Dice.",
        "",
        "| Конфигурация | n_test | mean_dice | median_dice | pixel_accuracy |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_test']} | {r['mean_dice']:.4f} | "
            f"{r['median_dice']:.4f} | {r['pixel_accuracy']:.4f} |"
        )
    lines += [
        "",
        "Dice отдельно для срезов с опухолью и без (стратификация по "
        "has_tumor — усреднение по ВСЕМ срезам смешивает лёгкую подзадачу "
        "'узнать пустой срез' со содержательной задачей сегментации опухоли):",
        "",
        "| Конфигурация | n_with_tumor | mean_dice (с опухолью) | n_without_tumor | mean_dice (без опухоли) |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_with_tumor']} | {r['mean_dice_with_tumor']:.4f} | "
            f"{r['n_without_tumor']} | {r['mean_dice_without_tumor']:.4f} |"
        )
    lines += [
        "",
        "F1-score — два разных уровня агрегации (пиксельный, по сумме TP/FP/FN "
        "всего тест-сета, НЕ по среднему Dice на срез — это разные числа, хотя "
        "формула 2TP/(2TP+FP+FN) совпадает с Dice на уровне ОДНОГО изображения; "
        "и 'детекция' — бинарная классификация среза has_tumor по наличию хотя "
        "бы одного предсказанного пикселя опухоли):",
        "",
        "| Конфигурация | pixel precision | pixel recall | pixel F1 | detection precision | detection recall | detection F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['pixel_precision']:.4f} | {r['pixel_recall']:.4f} | {r['pixel_f1']:.4f} | "
            f"{r['slice_detection_precision']:.4f} | {r['slice_detection_recall']:.4f} | {r['slice_detection_f1']:.4f} |"
        )

    def fmt_ci(ci) -> str:
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None else "—"

    lines += [
        "",
        "95%-доверительные интервалы Уилсона для precision/recall (требование "
        "медицинского рецензента — medical_article_requirements.txt, п.3: "
        "точечная оценка без интервала вводит в заблуждение, особенно при "
        "малом n; здесь n большой — срезы с опухолью n=" + str(results["meaningful_neighbors"]["n_with_tumor"]) +
        " — интервалы узкие, но для Статьи 1 (n=20-30) они были бы в разы шире):",
        "",
        "| Конфигурация | pixel precision CI95 | pixel recall CI95 | detection precision CI95 | detection recall CI95 |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {fmt_ci(r['pixel_precision_ci95'])} | {fmt_ci(r['pixel_recall_ci95'])} | "
            f"{fmt_ci(r['slice_detection_precision_ci95'])} | {fmt_ci(r['slice_detection_recall_ci95'])} |"
        )

    lines += [
        "",
        "Confusion matrix на уровне СРЕЗА (детекция 'есть ли опухоль на срезе', "
        "TP/FP/FN/TN count) — явная таблица ошибок по требованию медицинского "
        "рецензента (п.1), а не только агрегированные precision/recall:",
        "",
        "| Конфигурация | TP (нашли) | FP (ложная тревога) | FN (пропустили) | TN (верно: нет опухоли) |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        cm = r["slice_confusion_matrix"]
        lines.append(f"| {name} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} |")

    lines += [
        "",
        f"**H1** (контекст соседних срезов повышает Dice: meaningful_neighbors > "
        f"no_context_k1): {dice['meaningful_neighbors']:.4f} > {dice['no_context_k1']:.4f} -> **{h1}**.",
        f"**H2** (случайные 'соседи' не лучше физически смежных: "
        f"shuffled_neighbors <= meaningful_neighbors): "
        f"{dice['shuffled_neighbors']:.4f} <= {dice['meaningful_neighbors']:.4f} -> **{h2}**.",
        "",
        "Пиксельная accuracy приведена только для прозрачности — при сильном "
        "классовом перекосе (опухоль — единицы процентов пикселей) она "
        "систематически завышена и не является содержательной метрикой "
        "качества сегментации; основная метрика статьи — Dice.",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
