"""Сравнение ВСЕХ методик train/test-сплита, уже используемых в проекте
(честных и нечестных — см. README "Сплит через попарное разнесение",
experiments/dishonest_*.py, dissertation_articles_plan.txt Статья 6),
применённых к ОДНОЙ И ТОЙ ЖЕ задаче: бинарная классификация "срез содержит
опухоль" на BraTS2020 (`datasets/brats_flair_slices/`,
experiments/experiment_sequential_smoothing.py).

BraTS даёт ВТОРУЮ, помимо корреляции, ось возможной утечки, которой нет в
Kaggle-датасете Статей 1/3 — срезы СГРУППИРОВАНЫ по пациенту. Поэтому здесь
сравниваются 4 метода:

  1. patient_split (ЧЕСТНЫЙ) — train и test строятся из НЕПЕРЕСЕКАЮЩИХСЯ
     множеств пациентов (32 train / 8 test) — см.
     experiment_sequential_smoothing.py::split_patients. Единственный метод
     здесь, где test гарантированно не содержит ни одного среза пациента,
     чьи другие срезы видела модель.
  2. image_random_split ("ЧЕСТНЫЙ" по общему стандарту ML — stratified
     random split по меткам, evaluation.paper_train_test_split — ИМЕННО
     этот метод считается честным для Статей 1/3/5/6 плана на Kaggle-
     датасете) — но здесь применяется к пулу срезов ИГНОРИРУЯ patient_id.
     Из-за группировки срезов по пациенту часть test-срезов разделяет
     пациента с train-срезами — утечка, специфичная для ПОСЛЕДОВАТЕЛЬНЫХ
     медицинских данных, которой не существует на независимых снимках.
  3. paired_correlation_split (НЕЧЕСТНЫЙ, theory/code/Вариант_2_1.docx) —
     применяется к ТОМУ ЖЕ пулу, что и метод 2 (прямое сравнение "тот же
     пул данных, другая стратегия сплита", как в
     experiments/dishonest_correlated_pairs_report.py). Гипотеза: на
     последовательных срезах пары "максимально коррелирующих" изображений
     — это преимущественно пары СОСЕДНИХ срезов ОДНОГО пациента, то есть
     этот метод, в отличие от Kaggle-датасета, КОМБИНИРУЕТ утечку по
     корреляции (метод сам по себе) с утечкой по пациенту (метод 2) —
     проверяется ниже прямым подсчётом доли пар с совпадающим patient_id.
  4. train_equals_test (НЕЧЕСТНЫЙ, самый грубый — theory/code/Run_200.m,
     experiments/dishonest_train_equals_test_report.py) — train и test это
     буквально один и тот же пул изображений.

Общий классификатор для всех 4 методов — SubspaceConjugacyClassifier
(n_subclasses=8), как в Статье 1/4 плана, для сопоставимости. Методы 2-4
используют ОДИН общий непатентный пул (POOL_PER_CLASS на класс, собран
случайно из ВСЕХ 40 пациентов без учёта patient_id) — метод 1 структурно
не может использовать тот же пул (ему нужен контроль на уровне пациента с
самого начала).

Для каждого метода считается:
  - accuracy, per-class recall (evaluate_classifier);
  - patient_overlap_fraction — доля тестовых срезов, чей patient_id
    встречается среди обучающих срезов (объективный, измеримый индикатор
    именно того механизма утечки, который специфичен для этого датасета).

Артефакты: artifacts/experiment_brats_split_methods_{report.json,results.md}.

Запуск (после experiments/prepare_brats_dataset.py):
    python experiments/experiment_brats_split_methods.py
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    paired_correlation_split,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
N_SUBCLASSES = 8

PATIENT_SPLIT_TRAIN_PER_CLASS = 100   # метод 1
POOL_PER_CLASS = 200                   # методы 2-3 (общий пул, 200/класс = 400)
TRAIN_EQUALS_TEST_PER_CLASS = 100     # метод 4 (отдельный, меньший пул)

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_brats_split_methods_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_brats_split_methods_results.md"

Row = Dict[str, Any]


def load_metadata() -> List[Row]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} не найден. Сначала: "
            'BRATS_ROOT="<путь>" python experiments/prepare_brats_dataset.py'
        )
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


def vectorize_rows(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    pid = np.array([r["patient_id"] for r in rows])
    return X, y, pid


def sample_pool(rows: List[Row], per_class: int, random_state: int) -> List[Row]:
    """Случайный пул per_class строк на класс, patient_id ИГНОРИРУЕТСЯ."""
    rng = np.random.default_rng(random_state)
    pool: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in rows if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        pool.extend(cls_rows[i] for i in idx)
    return pool


def patient_overlap_fraction(pid_train: np.ndarray, pid_test: np.ndarray) -> float:
    train_patients = set(pid_train.tolist())
    if len(pid_test) == 0:
        return 0.0
    overlap = sum(1 for p in pid_test if p in train_patients)
    return overlap / len(pid_test)


def method_patient_split(rows: List[Row]) -> Dict[str, Any]:
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * 0.2))
    test_patients = set(shuffled[:n_test].tolist())
    train_patients = set(shuffled[n_test:].tolist())

    train_pool = [r for r in rows if r["patient_id"] in train_patients]
    rng2 = np.random.default_rng(RANDOM_SEED)
    train_rows: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in train_pool if r["label"] == label]
        idx = rng2.permutation(len(cls_rows))[:PATIENT_SPLIT_TRAIN_PER_CLASS]
        train_rows.extend(cls_rows[i] for i in idx)

    test_rows = [r for r in rows if r["patient_id"] in test_patients]

    X_train, y_train, pid_train = vectorize_rows(train_rows)
    X_test, y_test, pid_test = vectorize_rows(test_rows)
    return {
        "X_train": X_train, "y_train": y_train, "pid_train": pid_train,
        "X_test": X_test, "y_test": y_test, "pid_test": pid_test,
        "honest": True,
        "description": "непересекающиеся пациенты (32 train / 8 test)",
    }


def method_image_random_split(pool: List[Row]) -> Dict[str, Any]:
    X, y, pid = vectorize_rows(pool)
    idx = np.arange(len(pool))
    idx_train, idx_test = train_test_split(
        idx, train_size=0.5, stratify=y, random_state=RANDOM_SEED
    )
    return {
        "X_train": X[idx_train], "y_train": y[idx_train], "pid_train": pid[idx_train],
        "X_test": X[idx_test], "y_test": y[idx_test], "pid_test": pid[idx_test],
        "honest": "общий стандарт ML — да; для ЭТОГО датасета — нет",
        "description": "stratified random по срезам, patient_id игнорируется",
    }


def method_paired_correlation_split(pool: List[Row]) -> Dict[str, Any]:
    X, y, pid = vectorize_rows(pool)
    result = paired_correlation_split(X, y, random_state=RANDOM_SEED)
    pid_train = pid[result["train_indices"]]
    pid_test = pid[result["test_indices"]]

    pairs = result["pairs"]
    same_patient_pairs = sum(1 for i, j in pairs if pid[i] == pid[j])
    pair_same_patient_fraction = same_patient_pairs / len(pairs) if pairs else 0.0

    return {
        "X_train": result["X_train"], "y_train": result["y_train"], "pid_train": pid_train,
        "X_test": result["X_test"], "y_test": result["y_test"], "pid_test": pid_test,
        "honest": False,
        "description": "попарное разнесение максимально коррелирующих срезов",
        "pair_same_patient_fraction": pair_same_patient_fraction,
        "n_pairs": len(pairs),
    }


def method_train_equals_test(rows: List[Row]) -> Dict[str, Any]:
    pool = sample_pool(rows, TRAIN_EQUALS_TEST_PER_CLASS, RANDOM_SEED + 1)
    X, y, pid = vectorize_rows(pool)
    return {
        "X_train": X, "y_train": y, "pid_train": pid,
        "X_test": X, "y_test": y, "pid_test": pid,
        "honest": False,
        "description": "train == test (100% пересечение)",
    }


def run_method(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n=== {name} ({data['description']}) ===")
    print(f"   n_train={data['X_train'].shape[0]}, n_test={data['X_test'].shape[0]}")

    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(data["X_train"], data["y_train"])
    fit_seconds = time.perf_counter() - t0

    y_pred = clf.predict(data["X_test"])
    report = evaluate_classifier(data["y_test"], y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    overlap = patient_overlap_fraction(data["pid_train"], data["pid_test"])

    print(f"   fit()={fit_seconds:.1f}s  accuracy={report['accuracy']:.4f}  patient_overlap={overlap:.1%}")

    result = {
        "honest": data["honest"],
        "description": data["description"],
        "n_train": int(data["X_train"].shape[0]),
        "n_test": int(data["X_test"].shape[0]),
        "fit_seconds": fit_seconds,
        "accuracy": report["accuracy"],
        "per_class_accuracy": per_class,
        "patient_overlap_fraction": overlap,
        "confusion_matrix": report["confusion_matrix"].tolist(),
        "labels": [str(label) for label in report["labels"]],
    }
    if "pair_same_patient_fraction" in data:
        result["pair_same_patient_fraction"] = data["pair_same_patient_fraction"]
        result["n_pairs"] = data["n_pairs"]
    return result


def print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    headers = ["Метод", "Честный?", "n_train", "n_test", "overlap", "accuracy", "tumor", "no_tumor"]
    widths = [26, 10, 8, 7, 8, 9, 7, 9]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА: методы train/test-сплита на BraTS has_tumor (одинаковый классификатор)")
    print("=" * 100)
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))

    for name, r in results.items():
        pc = r["per_class_accuracy"]
        honest_label = "ДА" if r["honest"] is True else ("нет" if r["honest"] is False else "см. текст")
        print(fmt_row([
            name,
            honest_label,
            str(r["n_train"]),
            str(r["n_test"]),
            f"{r['patient_overlap_fraction']:.1%}",
            f"{r['accuracy']:.4f}",
            f"{pc.get('tumor', float('nan')):.3f}",
            f"{pc.get('no_tumor', float('nan')):.3f}",
        ]))
    print("=" * 100)

    pcs_methods = [name for name, r in results.items() if "pair_same_patient_fraction" in r]
    for name in pcs_methods:
        r = results[name]
        print(
            f"\n{name}: доля пар 'максимально коррелирующих' срезов, принадлежащих "
            f"ОДНОМУ пациенту = {r['pair_same_patient_fraction']:.1%} (из {r['n_pairs']} пар) — "
            "комбинация утечки по корреляции с утечкой по пациенту."
        )

    honest_acc = {name: r["accuracy"] for name, r in results.items() if r["honest"] is True}
    if honest_acc:
        baseline_name, baseline_acc = next(iter(honest_acc.items()))
        print(f"\nЧестный ориентир ({baseline_name}): accuracy={baseline_acc:.4f}.")
        for name, r in results.items():
            if r["honest"] is not True:
                inflation = r["accuracy"] - baseline_acc
                print(
                    f"   {name}: accuracy={r['accuracy']:.4f} "
                    f"({'+' if inflation >= 0 else ''}{inflation:.4f} п.п. относительно честного ориентира)"
                )


def save_markdown_report(results: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# Сравнение методов train/test-сплита на BraTS (has_tumor)",
        "",
        "Сгенерировано `experiments/experiment_brats_split_methods.py`. "
        f"Классификатор: `SubspaceConjugacyClassifier(n_subclasses={N_SUBCLASSES})` для всех методов.",
        "",
        "| Метод | Честный? | n_train | n_test | patient_overlap | accuracy | tumor | no_tumor | fit(), с |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        pc = r["per_class_accuracy"]
        honest_label = "ДА" if r["honest"] is True else ("нет" if r["honest"] is False else "см. текст")
        lines.append(
            f"| {name} | {honest_label} | {r['n_train']} | {r['n_test']} | "
            f"{r['patient_overlap_fraction']:.1%} | {r['accuracy']:.4f} | "
            f"{pc.get('tumor', float('nan')):.3f} | {pc.get('no_tumor', float('nan')):.3f} | "
            f"{r['fit_seconds']:.1f} |"
        )

    lines += [
        "",
        "- **patient_split** — ЧЕСТНЫЙ ориентир: train/test построены из непересекающихся "
        "пациентов, `patient_overlap=0%` по конструкции.",
        "- **image_random_split** — честный по общему стандарту ML (stratified random по "
        "меткам), но НЕ учитывает группировку срезов по пациенту — ненулевой `patient_overlap` "
        "показывает утечку, специфичную именно для последовательных медицинских данных.",
        "- **paired_correlation_split** — нечестный по конструкции (test подобран похожим на "
        "train); на BraTS дополнительно комбинируется с утечкой по пациенту (см. "
        "`pair_same_patient_fraction` ниже).",
        "- **train_equals_test** — train == test, `patient_overlap=100%` тривиально.",
    ]
    for name, r in results.items():
        if "pair_same_patient_fraction" in r:
            lines.append(
                f"\n`{name}`: доля пар с совпадающим patient_id = "
                f"{r['pair_same_patient_fraction']:.1%} (из {r['n_pairs']} пар)."
            )

    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown-отчёт: {REPORT_MD_PATH}")


def main() -> None:
    start = time.perf_counter()
    print("=== Сравнение методов train/test-сплита (BraTS, has_tumor) ===")

    rows = load_metadata()
    print(f"Загружено {len(rows)} срезов, {len({r['patient_id'] for r in rows})} пациентов.")

    pool_2_3 = sample_pool(rows, POOL_PER_CLASS, RANDOM_SEED)
    print(f"Общий пул для methods 2-3: {len(pool_2_3)} срезов ({POOL_PER_CLASS}/класс).")

    results = {
        "patient_split": run_method("patient_split", method_patient_split(rows)),
        "image_random_split": run_method("image_random_split", method_image_random_split(pool_2_3)),
        "paired_correlation_split": run_method("paired_correlation_split", method_paired_correlation_split(pool_2_3)),
        "train_equals_test": run_method("train_equals_test", method_train_equals_test(rows)),
    }

    print_summary_table(results)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"random_seed": RANDOM_SEED, "n_subclasses": N_SUBCLASSES, "results": results},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")
    save_markdown_report(results)

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
