"""Статья 4 — целевой эксперимент: можно ли поднять accuracy на +20 п.п.
над честным ориентиром (patient_split, BraTS has_tumor) честным ИЛИ
нечестным способом?

Честный ориентир серии экспериментов (один и тот же во всех скриптах,
один и тот же RANDOM_SEED/сплит пациентов):
  experiment_brats_split_methods.py::patient_split      -> accuracy=0.5924
  experiment_sequential_smoothing.py::meaningful_unsmoothed -> accuracy=0.5924
  improve_brats_accuracy_v2.py финал (после ВСЕХ честных рычагов, n_sub=12)
                                                          -> accuracy=0.5992

+20 п.п. от 0.5924 -> цель 0.79.

ЧЕСТНАЯ ветка (honest_push)
---------------------------
improve_brats_accuracy.py (грид n_subclasses=4..12) и improve_brats_accuracy_v2.py
(5 рычагов: filter_dependent, freeze_basis_at="auto", center_strategy="windowed",
stratified-по-пациентам отбор, TRAIN_PER_CLASS=150) уже честно перебрали
основные рычаги ЭТОГО метода на ЭТИХ признаках — выигрыш нашёлся только у
"больше данных" (+0.3 п.п. validation). Здесь — продолжение этого же честного
протокола (train/val/test на НЕПЕРЕСЕКАЮЩИХСЯ пациентах, test не трогается до
финала) с ещё большим TRAIN_PER_CLASS и ещё большим n_subclasses, чем было
опробовано ранее (чтобы видеть, продолжается ли тренд или это плато).

НЕЧЕСТНАЯ ветка (dishonest_push)
---------------------------------
train_equals_test (theory/code/Run_200.m, dishonest_train_equals_test_report.py)
уже даёт +9.3 п.п. (0.685) при n_subclasses=8 — само по себе недостаточно для
+20 п.п. Механизм, который здесь демонстрируется ВПЕРВЫЕ в этой серии: n_subclasses
управляет ЁМКОСТЬЮ модели (при фиксированном n_train подклассы становятся всё
мельче -> каждое подпространство строится почти по одному обучающему объекту).
В комбинации с train==test это превращает классификацию в фактическое
запоминание: эталонный объект узнаёт сам себя почти идеально. Грид
n_subclasses в [8, 20, 35, 45] на train_equals_test (100/класс, 200 объектов,
min_required=2*n_subclasses <= 100 -> n_subclasses<=50) должен показать
плавно растущую инфляцию accuracy вплоть до близкой к 100% — то есть
"+20 п.п. и куда угодно дальше" достижимо ОДНИМ изменением гиперпараметра в
сочетании с утечкой train==test, без единой содержательной идеи по метрике/
признакам.

Артефакты: artifacts/experiment_article4_boost_accuracy_{report.json,results.md}.

Запуск (после experiments/prepare_brats_dataset.py):
    python experiments/experiment_article4_boost_accuracy.py
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    smooth_adjacent_vectors,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42

SPLIT_METHODS_REPORT_PATH = Path("artifacts/experiment_brats_split_methods_report.json")


def _load_honest_baseline_accuracy() -> float:
    """Честный ориентир = patient_split::accuracy из уже посчитанного отчёта —
    читается динамически, а не хранится константой, чтобы не рассинхронизи-
    роваться при изменении объёма датасета (см. memory 2026-10-07: 40 -> 160
    пациентов, захардкоженная константа 0.5924 осталась от старого прогона
    и завысила/занизила бы инфляцию относительно актуального ориентира)."""
    report = json.loads(SPLIT_METHODS_REPORT_PATH.read_text(encoding="utf-8"))
    return report["results"]["patient_split"]["accuracy"]


HONEST_BASELINE_ACCURACY = _load_honest_baseline_accuracy()
TARGET_ACCURACY = HONEST_BASELINE_ACCURACY + 0.20

# --- Честная ветка ---
HONEST_N_SUBCLASSES = 14          # выше максимума (12), опробованного в improve_brats_accuracy.py
HONEST_TRAIN_PER_CLASS = 200      # выше максимума (150), опробованного в improve_brats_accuracy_v2.py

# --- Нечестная ветка ---
DISHONEST_TRAIN_PER_CLASS = 100   # как в dishonest_train_equals_test_report.py
DISHONEST_N_SUBCLASSES_GRID = [8, 20, 35, 45]  # 2*45=90 <= 100 = min_required

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_boost_accuracy_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_boost_accuracy_results.md"

Row = Dict[str, Any]


def load_metadata() -> List[Row]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} не найден — сначала prepare_brats_dataset.py.")
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "slice_idx": int(r["slice_idx"]),
                    "path": DATASET_ROOT / r["path"],
                    "label": "tumor" if r["has_tumor"] == "True" else "no_tumor",
                }
            )
    return rows


def vectorize_rows(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray]:
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    return X, y


def split_patients(rows: List[Row]) -> Tuple[List[str], List[str]]:
    """ТА ЖЕ логика/seed, что во всей серии экспериментов — те же 32/8 пациентов."""
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * 0.2))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def select_rows(rows: List[Row], patient_ids: Sequence[str], per_class: int, seed: int) -> List[Row]:
    pool = [r for r in rows if r["patient_id"] in set(patient_ids)]
    rng = np.random.default_rng(seed)
    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in pool if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        selected.extend(cls_rows[i] for i in idx)
    return selected


def sample_pool(rows: List[Row], per_class: int, seed: int) -> List[Row]:
    """Как в experiment_brats_split_methods.py::sample_pool — patient_id игнорируется."""
    return select_rows(rows, {r["patient_id"] for r in rows}, per_class, seed)


def build_patient_grouped_rows(rows: List[Row], patient_ids: Sequence[str]) -> Tuple[List[Row], List[int]]:
    pool = [r for r in rows if r["patient_id"] in set(patient_ids)]
    ordered: List[Row] = []
    group_lengths: List[int] = []
    for pid in sorted(set(patient_ids)):
        group = sorted((r for r in pool if r["patient_id"] == pid), key=lambda r: r["slice_idx"])
        ordered.extend(group)
        group_lengths.append(len(group))
    return ordered, group_lengths


def smooth_per_patient_group(X: np.ndarray, group_lengths: List[int]) -> np.ndarray:
    blocks = []
    offset = 0
    for length in group_lengths:
        blocks.append(smooth_adjacent_vectors(X[offset : offset + length]))
        offset += length
    return np.vstack(blocks)


def run_honest_push(rows: List[Row]) -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print(f"ЧЕСТНАЯ ВЕТКА: n_subclasses={HONEST_N_SUBCLASSES}, "
          f"train_per_class={HONEST_TRAIN_PER_CLASS}, filter_dependent=True")
    print("=" * 88)

    train_ids, test_ids = split_patients(rows)
    train_rows = select_rows(rows, train_ids, HONEST_TRAIN_PER_CLASS, RANDOM_SEED)
    test_rows, test_group_lengths = build_patient_grouped_rows(rows, test_ids)

    X_train, y_train = vectorize_rows(train_rows)
    X_test, y_test = vectorize_rows(test_rows)
    X_test_smoothed = smooth_per_patient_group(X_test, test_group_lengths)

    print(f"n_train={X_train.shape[0]} (32 train-пациента), n_test={X_test.shape[0]} (8 test-пациентов, непересекающиеся)")

    clf = SubspaceConjugacyClassifier(
        n_subclasses=HONEST_N_SUBCLASSES, filter_dependent=True,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    acc = evaluate_classifier(y_test, clf.predict(X_test))["accuracy"]
    acc_smoothed = evaluate_classifier(y_test, clf.predict(X_test_smoothed))["accuracy"]
    print(f"fit()={fit_seconds:.1f}s  accuracy={acc:.4f}  accuracy_smoothed={acc_smoothed:.4f}")

    return {
        "n_subclasses": HONEST_N_SUBCLASSES,
        "train_per_class": HONEST_TRAIN_PER_CLASS,
        "clf_kwargs": {"filter_dependent": True},
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "fit_seconds": fit_seconds,
        "accuracy": acc,
        "accuracy_smoothed": acc_smoothed,
        "patient_overlap_fraction": 0.0,
    }


def run_dishonest_push(rows: List[Row]) -> List[Dict[str, Any]]:
    print("\n" + "=" * 88)
    print("НЕЧЕСТНАЯ ВЕТКА: train == test, грид n_subclasses "
          f"{DISHONEST_N_SUBCLASSES_GRID} (ёмкость -> память)")
    print("=" * 88)

    pool = sample_pool(rows, DISHONEST_TRAIN_PER_CLASS, RANDOM_SEED + 1)
    X, y = vectorize_rows(pool)
    print(f"n=n_train=n_test={X.shape[0]} (train==test, 100% пересечение по построению)")

    results = []
    for n_sub in DISHONEST_N_SUBCLASSES_GRID:
        clf = SubspaceConjugacyClassifier(n_subclasses=n_sub)
        t0 = time.perf_counter()
        clf.fit(X, y)
        fit_seconds = time.perf_counter() - t0
        acc = evaluate_classifier(y, clf.predict(X))["accuracy"]
        avg_per_subclass = X.shape[0] / 2 / n_sub
        print(f"   n_subclasses={n_sub:3d}  (~{avg_per_subclass:.1f} объектов/подкласс)  "
              f"fit={fit_seconds:.1f}s  accuracy={acc:.4f}  "
              f"inflation={acc - HONEST_BASELINE_ACCURACY:+.4f} п.п. над честным ориентиром")
        results.append({
            "n_subclasses": n_sub,
            "avg_objects_per_subclass": avg_per_subclass,
            "fit_seconds": fit_seconds,
            "accuracy": acc,
            "inflation_over_honest_baseline": acc - HONEST_BASELINE_ACCURACY,
        })
    return results


def save_markdown_report(honest: Dict[str, Any], dishonest: List[Dict[str, Any]]) -> None:
    lines = [
        "# Статья 4 — можно ли поднять accuracy на +20 п.п. (BraTS has_tumor)?",
        "",
        f"Честный ориентир серии экспериментов: **{HONEST_BASELINE_ACCURACY:.4f}** "
        f"(`experiment_brats_split_methods.py::patient_split`). Цель +20 п.п.: "
        f"**{TARGET_ACCURACY:.4f}**.",
        "",
        "## Честная ветка",
        "",
        "Продолжение честного протокола `improve_brats_accuracy*.py` (train/test — "
        "непересекающиеся пациенты, test использован один раз): больше данных и "
        "больше n_subclasses, чем было опробовано ранее.",
        "",
        "| n_subclasses | train/класс | accuracy | + сглаживание | fit(), с |",
        "|---|---|---|---|---|",
        f"| {honest['n_subclasses']} | {honest['train_per_class']} | {honest['accuracy']:.4f} | "
        f"{honest['accuracy_smoothed']:.4f} | {honest['fit_seconds']:.1f} |",
        "",
        f"Прирост над честным ориентиром: {honest['accuracy'] - HONEST_BASELINE_ACCURACY:+.4f} п.п. "
        f"— {'ЦЕЛЬ ДОСТИГНУТА' if honest['accuracy'] >= TARGET_ACCURACY else 'цель +20 п.п. ЧЕСТНО НЕ достигнута'}.",
        "",
        "## Нечестная ветка (train == test + ёмкость модели)",
        "",
        "`train_equals_test`: тот же пул данных используется и как обучающий, и как "
        "тестовый. n_subclasses управляет размером подпространств — при росте "
        "n_subclasses объекты в каждом подклассе становятся всё малочисленнее, "
        "подпространство всё точнее описывает именно ИХ, а не класс в целом "
        "(запоминание вместо обобщения).",
        "",
        "| n_subclasses | объектов/подкласс (~) | accuracy | инфляция над честным ориентиром | fit(), с |",
        "|---|---|---|---|---|",
    ]
    for r in dishonest:
        lines.append(
            f"| {r['n_subclasses']} | {r['avg_objects_per_subclass']:.1f} | {r['accuracy']:.4f} | "
            f"{r['inflation_over_honest_baseline']:+.4f} | {r['fit_seconds']:.1f} |"
        )
    max_inflation = max(r["inflation_over_honest_baseline"] for r in dishonest)
    lines += [
        "",
        f"Максимальная инфляция в гриде: {max_inflation:+.4f} п.п. — "
        f"{'ПРЕВЫШАЕТ' if max_inflation >= 0.20 else 'НЕ достигает'} +20 п.п., "
        "достигнута ОДНИМ изменением гиперпараметра (n_subclasses) в сочетании с "
        "уже известной утечкой train==test, без единой содержательной идеи по методу.",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown-отчёт: {REPORT_MD_PATH}")


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4: эксперимент +20 п.п. (честно/нечестно), BraTS has_tumor ===")

    rows = load_metadata()
    print(f"Загружено {len(rows)} срезов, {len({r['patient_id'] for r in rows})} пациентов.")

    honest_result = run_honest_push(rows)
    dishonest_results = run_dishonest_push(rows)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "random_seed": RANDOM_SEED,
        "honest_baseline_accuracy": HONEST_BASELINE_ACCURACY,
        "target_accuracy": TARGET_ACCURACY,
        "honest_push": honest_result,
        "dishonest_push_grid": dishonest_results,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")
    save_markdown_report(honest_result, dishonest_results)

    print("\n" + "=" * 88)
    print("ИТОГ")
    print("=" * 88)
    print(f"Честный ориентир:              {HONEST_BASELINE_ACCURACY:.4f}")
    print(f"Цель (+20 п.п.):                {TARGET_ACCURACY:.4f}")
    print(f"Честная ветка (push):          {honest_result['accuracy']:.4f} "
          f"({honest_result['accuracy'] - HONEST_BASELINE_ACCURACY:+.4f} п.п.)")
    best_dishonest = max(dishonest_results, key=lambda r: r["accuracy"])
    print(f"Нечестная ветка (лучшая):      {best_dishonest['accuracy']:.4f} "
          f"({best_dishonest['inflation_over_honest_baseline']:+.4f} п.п., "
          f"n_subclasses={best_dishonest['n_subclasses']})")
    print("=" * 88)

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
