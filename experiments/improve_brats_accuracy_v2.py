"""Продолжение experiments/improve_brats_accuracy.py — проверяет ВСЕ честные
рычаги, предложенные по запросу "Есть идеи как можно повысить точность?":

  1. filter_dependent=True       — LinearDependencyFilter: убирает почти
     линейно зависимые эталоны (соседние срезы одного пациента — почти
     копии, именно предмет Статьи 4).
  2. freeze_basis_at="auto"      — не обрезает рост базиса подкласса на 2
     векторах (финдинг №2 README), выравнивание по минимальному k.
  3. center_strategy="windowed"  — MATLAB-находка №2 (позиционно
     ограниченный пул кандидатов для центров подклассов).
  4. stratified_sampling         — обучающий пул набирается НЕ чисто
     случайно, а round-robin по пациентам (не больше MAX_PER_PATIENT
     срезов с одного пациента) — избегает перекоса "много почти
     одинаковых срезов одного пациента, мало от другого".
  5. TRAIN_PER_CLASS=150         — больше эталонов (дороже по времени,
     O(M^2*N)), изолированно от остальных рычагов.

Методология — та же честная схема, что в improve_brats_accuracy.py:
n_subclasses=12 ЗАФИКСИРОВАН (уже выбран честно в прошлом запуске,
artifacts/improve_brats_accuracy_report.json, val_accuracy=0.6651 — лучший
в сетке). Каждый рычаг проверяется на validation-пациентах (24 search_train
-> fit, 8 validation -> выбор), test (8 пациентов, ТЕ ЖЕ, что everywhere в
этой серии экспериментов) не трогается до самого последнего шага. Рычаги,
улучшившие validation относительно baseline (0.6651), объединяются в ОДНУ
финальную конфигурацию, которая переобучается на всех 32 train-пациентах и
оценивается ОДИН раз на test (+ сглаживание по пациенту, Статья 4).

Артефакты: artifacts/improve_brats_accuracy_v2_{report.json,results.md}.

Запуск:
    python experiments/improve_brats_accuracy_v2.py
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
N_SUBCLASSES = 12  # честно выбран в improve_brats_accuracy.py
TRAIN_PER_CLASS = 100
TRAIN_PER_CLASS_BIG = 150  # рычаг 5
MAX_PER_PATIENT = 5  # рычаг 4

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_v2_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_v2_results.md"
PREVIOUS_REPORT_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_report.json"

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
    """ТА ЖЕ логика, что во всей этой серии экспериментов — те же 8 test-пациентов."""
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * 0.2))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def split_search_validation(train_ids: List[str]) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(train_ids)
    n_val = round(len(shuffled) * 0.25)
    val_ids = sorted(shuffled[:n_val].tolist())
    search_train_ids = sorted(shuffled[n_val:].tolist())
    return search_train_ids, val_ids


def select_training_rows(rows: List[Row], patient_ids: Sequence[str], per_class: int, seed: int = RANDOM_SEED) -> List[Row]:
    """Рычаг-baseline: чисто случайный отбор per_class на класс."""
    pool = [r for r in rows if r["patient_id"] in set(patient_ids)]
    rng = np.random.default_rng(seed)
    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in pool if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        selected.extend(cls_rows[i] for i in idx)
    return selected


def select_training_rows_stratified(
    rows: List[Row], patient_ids: Sequence[str], per_class: int,
    max_per_patient: int = MAX_PER_PATIENT, seed: int = RANDOM_SEED,
) -> List[Row]:
    """Рычаг 4: round-robin по пациентам, не больше max_per_patient срезов
    с одного пациента на класс — избегает перекоса в сторону немногих
    "многословных" пациентов."""
    rng = np.random.default_rng(seed)
    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        per_patient_lists: Dict[str, List[Row]] = {}
        for pid in patient_ids:
            cls_rows = [r for r in rows if r["patient_id"] == pid and r["label"] == label]
            idx = rng.permutation(len(cls_rows))
            per_patient_lists[pid] = [cls_rows[i] for i in idx][:max_per_patient]

        patient_order = list(rng.permutation(list(patient_ids)))
        picked: List[Row] = []
        round_idx = 0
        while len(picked) < per_class:
            added_this_round = False
            for pid in patient_order:
                if len(picked) >= per_class:
                    break
                lst = per_patient_lists[pid]
                if round_idx < len(lst):
                    picked.append(lst[round_idx])
                    added_this_round = True
            if not added_this_round:
                break
            round_idx += 1
        selected.extend(picked)
    return selected


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


def fit_and_eval(
    name: str,
    train_rows: List[Row],
    X_val: np.ndarray, y_val: np.ndarray,
    clf_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    X_train, y_train = vectorize_rows(train_rows)
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, **clf_kwargs)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    val_acc = evaluate_classifier(y_val, clf.predict(X_val))["accuracy"]
    print(f"   {name}: n_train={X_train.shape[0]}  fit={fit_seconds:.1f}s  val_acc={val_acc:.4f}")
    return {"name": name, "val_accuracy": val_acc, "fit_seconds": fit_seconds, "n_train": int(X_train.shape[0]), "clf_kwargs": clf_kwargs}


def main() -> None:
    start = time.perf_counter()
    print("=== Проверка ВСЕХ честных рычагов повышения accuracy (BraTS has_tumor) ===")

    rows = load_metadata()
    train_ids, test_ids = split_patients(rows)
    search_train_ids, val_ids = split_search_validation(train_ids)
    print(f"Пациенты: {len(search_train_ids)} search_train, {len(val_ids)} validation, {len(test_ids)} test.")

    val_rows, _ = build_patient_grouped_rows(rows, val_ids)
    X_val, y_val = vectorize_rows(val_rows)
    test_rows, test_group_lengths = build_patient_grouped_rows(rows, test_ids)
    X_test, y_test = vectorize_rows(test_rows)
    X_test_smoothed = smooth_per_patient_group(X_test, test_group_lengths)

    baseline_val_acc = 0.6651  # improve_brats_accuracy_report.json, grid n_subclasses=12
    if PREVIOUS_REPORT_PATH.exists():
        with open(PREVIOUS_REPORT_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        baseline_val_acc = next(
            r["val_accuracy"] for r in prev["grid_results"] if r["n_subclasses"] == N_SUBCLASSES
        )
    print(f"\nBaseline (n_subclasses={N_SUBCLASSES}, случайный отбор, {TRAIN_PER_CLASS}/класс): val_accuracy={baseline_val_acc:.4f} (из предыдущего запуска).")

    print("\n--- Проверка рычагов по отдельности (validation, test не используется) ---")
    lever_results: List[Dict[str, Any]] = []

    random_train = select_training_rows(rows, search_train_ids, TRAIN_PER_CLASS)
    lever_results.append(fit_and_eval("1_filter_dependent", random_train, X_val, y_val, {"filter_dependent": True}))
    lever_results.append(fit_and_eval("2_freeze_basis_auto", random_train, X_val, y_val, {"freeze_basis_at": "auto"}))
    lever_results.append(fit_and_eval("3_center_windowed", random_train, X_val, y_val, {"center_strategy": "windowed"}))

    stratified_train = select_training_rows_stratified(rows, search_train_ids, TRAIN_PER_CLASS)
    lever_results.append(fit_and_eval("4_stratified_sampling", stratified_train, X_val, y_val, {}))

    big_train = select_training_rows(rows, search_train_ids, TRAIN_PER_CLASS_BIG)
    lever_results.append(fit_and_eval(f"5_more_data_{TRAIN_PER_CLASS_BIG}pc", big_train, X_val, y_val, {}))

    winners = [r for r in lever_results if r["val_accuracy"] > baseline_val_acc]
    print(f"\nРычаги, улучшившие validation относительно baseline ({baseline_val_acc:.4f}):")
    for r in winners:
        print(f"   {r['name']}: val_accuracy={r['val_accuracy']:.4f} (+{r['val_accuracy'] - baseline_val_acc:.4f})")
    if not winners:
        print("   (ни один рычаг не улучшил validation — честный вывод: baseline остаётся лучшей конфигурацией)")

    combined_kwargs: Dict[str, Any] = {}
    combined_per_class = TRAIN_PER_CLASS
    combined_stratified = False
    for r in winners:
        combined_kwargs.update(r["clf_kwargs"])
        if r["name"].startswith("4_stratified"):
            combined_stratified = True
        if r["name"].startswith("5_more_data"):
            combined_per_class = TRAIN_PER_CLASS_BIG

    print(f"\n--- Комбинация победивших рычагов: kwargs={combined_kwargs}, stratified={combined_stratified}, per_class={combined_per_class} ---")
    combined_train_search = (
        select_training_rows_stratified(rows, search_train_ids, combined_per_class)
        if combined_stratified else
        select_training_rows(rows, search_train_ids, combined_per_class)
    )
    combo_result = fit_and_eval("combined", combined_train_search, X_val, y_val, combined_kwargs)

    use_combo = combo_result["val_accuracy"] > baseline_val_acc
    final_kwargs = combined_kwargs if use_combo else {}
    final_per_class = combined_per_class if use_combo else TRAIN_PER_CLASS
    final_stratified = combined_stratified if use_combo else False
    print(f"\nИтоговая честная конфигурация для финала: {'combined' if use_combo else 'baseline (ни один рычаг не помог)'} -> kwargs={final_kwargs}, stratified={final_stratified}, per_class={final_per_class}")

    print(f"\n--- Финал: переобучение на ВСЕХ 32 train-пациентах, ОДНА оценка на test ---")
    final_train_rows = (
        select_training_rows_stratified(rows, train_ids, final_per_class)
        if final_stratified else
        select_training_rows(rows, train_ids, final_per_class)
    )
    X_final_train, y_final_train = vectorize_rows(final_train_rows)
    final_clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, **final_kwargs)
    t0 = time.perf_counter()
    final_clf.fit(X_final_train, y_final_train)
    final_fit_seconds = time.perf_counter() - t0

    final_acc = evaluate_classifier(y_test, final_clf.predict(X_test))["accuracy"]
    final_acc_smoothed = evaluate_classifier(y_test, final_clf.predict(X_test_smoothed))["accuracy"]
    print(f"fit()={final_fit_seconds:.1f}s  final_accuracy={final_acc:.4f}  final_accuracy_smoothed={final_acc_smoothed:.4f}")

    previous_final = None
    previous_final_smoothed = None
    if PREVIOUS_REPORT_PATH.exists():
        previous_final = prev["summary"].get(f"honest_tuned (n_sub={N_SUBCLASSES}, val-selected)")
        previous_final_smoothed = prev["summary"].get(f"honest_tuned (n_sub={N_SUBCLASSES}) + smoothing")

    print("\n" + "=" * 88)
    print("ИТОГ: честные рычаги поверх n_subclasses=12 (BraTS has_tumor)")
    print("=" * 88)
    print(f"{'До новых рычагов (n_sub=12, случайный отбор)':55}{previous_final if previous_final is not None else float('nan'):.4f}")
    print(f"{'До новых рычагов + сглаживание':55}{previous_final_smoothed if previous_final_smoothed is not None else float('nan'):.4f}")
    print(f"{'После новых рычагов (' + ('combined' if use_combo else 'ни один не помог') + ')':55}{final_acc:.4f}")
    print(f"{'После новых рычагов + сглаживание':55}{final_acc_smoothed:.4f}")
    print("=" * 88)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "random_seed": RANDOM_SEED,
        "n_subclasses": N_SUBCLASSES,
        "baseline_val_accuracy": baseline_val_acc,
        "lever_results": lever_results,
        "winners": [r["name"] for r in winners],
        "combined_kwargs": combined_kwargs,
        "combined_val_accuracy": combo_result["val_accuracy"],
        "used_combo_for_final": use_combo,
        "final_kwargs": final_kwargs,
        "final_per_class": final_per_class,
        "final_stratified": final_stratified,
        "final_fit_seconds": final_fit_seconds,
        "final_accuracy": final_acc,
        "final_accuracy_smoothed": final_acc_smoothed,
        "previous_final_accuracy": previous_final,
        "previous_final_accuracy_smoothed": previous_final_smoothed,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    md_lines = [
        "# Честные рычаги поверх n_subclasses=12 (BraTS has_tumor)",
        "",
        "| Рычаг | val_accuracy | fit(), с |",
        "|---|---|---|",
    ]
    for r in lever_results:
        md_lines.append(f"| {r['name']} | {r['val_accuracy']:.4f} | {r['fit_seconds']:.1f} |")
    md_lines += [
        f"| baseline (для сравнения) | {baseline_val_acc:.4f} | — |",
        f"| combined ({combined_kwargs}) | {combo_result['val_accuracy']:.4f} | {combo_result['fit_seconds']:.1f} |",
        "",
        f"**Финал на test**: до новых рычагов {previous_final:.4f} ({previous_final_smoothed:.4f} со сглаживанием) "
        f"-> после {final_acc:.4f} ({final_acc_smoothed:.4f} со сглаживанием).",
    ]
    REPORT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
