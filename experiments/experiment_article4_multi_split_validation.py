"""Статья 4 — валидация H1/H2 по НЕСКОЛЬКИМ независимым случайным
разбиениям train/test (по требованию медицинского рецензента —
medical_article_requirements.txt, п.5: "рост/эффект нужно подтверждать
усреднением по нескольким независимым разбиениям, а не одной точкой").

Базовый эксперимент (`experiment_article4_unet_segmentation.py`)
использует ОДИН фиксированный `RANDOM_SEED=42` для разбиения по пациентам
на всё время работы над статьёй. Мы на практике убедились, что это риск:
один и тот же код на одном и том же seed дал РАЗНЫЕ результаты между
прогонами из-за недетерминизма cuDNN (см. memory проекта) — то есть
единственная точка ненадёжна вдвойне (и из-за разбиения, и из-за
обучения). Этот скрипт обучает базовый U-Net (с BatchNorm) на
N_SPLITS РАЗНЫХ случайных патиентских разбиениях (варьируется только
сид разбиения, не архитектура/гиперпараметры) и репортит mean/std/min/max
по H1 (meaningful > no_context) и H2 (shuffled <= meaningful) — если
гипотезы подтверждаются НА ВСЕХ разбиениях, а не только на seed=42,
результат можно считать устойчивым, а не удачным/неудачным стечением.

Масштаб: ради контроля суммарного времени (9 архитектур x N разбиений
было бы часами) валидируется ТОЛЬКО базовый U-Net (с BatchNorm) —
headline-модель статьи, на бюджете EPOCHS=6 (как в
experiment_article4_architectures.py), не на полной сетке 15 эпох.
Это ЧАСТИЧНОЕ, но честное выполнение требования — распространение на
все 9 архитектур является следующим шагом, если потребуется.

Запуск:
    python experiments/experiment_article4_multi_split_validation.py
"""

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch

import experiment_article4_unet_segmentation as base

N_SPLITS = 5
SPLIT_SEEDS = [42, 101, 202, 303, 404]
EPOCHS_MULTI_SPLIT = 6  # как в experiment_article4_architectures.py, не полная сетка 15 эпох

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_multi_split_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_multi_split_results.md"


def run_one_split(rows, split_seed: int, epochs: int) -> Dict[str, object]:
    train_ids, test_ids = base.split_patients(rows, seed=split_seed)
    train_groups = base.group_by_patient(rows, train_ids)
    test_groups = base.group_by_patient(rows, test_ids)
    train_centers = base.build_center_list(train_groups, base.MAX_TRAIN_PER_CLASS, base.RANDOM_SEED)
    test_centers = base.all_centers(test_groups)

    old_epochs = base.EPOCHS
    base.EPOCHS = epochs
    try:
        model_k1, _ = base.train_model(train_groups, train_centers, n_context=1)
        model_kn, _ = base.train_model(train_groups, train_centers, n_context=base.N_CONTEXT)
    finally:
        base.EPOCHS = old_epochs

    result_k1 = base.evaluate_model(model_k1, base.SegStackDataset(test_groups, test_centers, n_context=1))
    result_kn = base.evaluate_model(
        model_kn, base.SegStackDataset(test_groups, test_centers, n_context=base.N_CONTEXT)
    )
    result_shuf = base.evaluate_model(
        model_kn,
        base.ShuffledNeighborSegDataset(test_groups, test_centers, n_context=base.N_CONTEXT, seed=split_seed),
    )

    return {
        "split_seed": split_seed,
        "n_train_patients": len(train_ids),
        "n_test_patients": len(test_ids),
        "no_context_dice_tumor": result_k1["mean_dice_with_tumor"],
        "meaningful_dice_tumor": result_kn["mean_dice_with_tumor"],
        "shuffled_dice_tumor": result_shuf["mean_dice_with_tumor"],
        "meaningful_detection_recall": result_kn["slice_detection_recall"],
        "h1_holds": result_kn["mean_dice_with_tumor"] > result_k1["mean_dice_with_tumor"],
        "h2_holds": result_shuf["mean_dice_with_tumor"] <= result_kn["mean_dice_with_tumor"],
    }


def main() -> None:
    start = time.perf_counter()
    print(f"=== Статья 4: валидация H1/H2 по {N_SPLITS} независимым разбиениям (базовый U-Net, {EPOCHS_MULTI_SPLIT} эпох) ===")
    print(
        f"Устройство вычислений: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if base.DEVICE.type == "cuda" else "")
    )

    rows = base.load_metadata()
    print(f"Загружено {len(rows)} срезов, {len({r['patient_id'] for r in rows})} пациентов.")

    runs: List[Dict[str, object]] = []
    for i, seed in enumerate(SPLIT_SEEDS[:N_SPLITS], start=1):
        print(f"\n--- Разбиение {i}/{N_SPLITS} (split_seed={seed}) ---")
        t0 = time.perf_counter()
        result = run_one_split(rows, seed, EPOCHS_MULTI_SPLIT)
        result["fit_seconds"] = time.perf_counter() - t0
        runs.append(result)
        print(
            f"   no_context={result['no_context_dice_tumor']:.4f}  "
            f"meaningful={result['meaningful_dice_tumor']:.4f}  "
            f"shuffled={result['shuffled_dice_tumor']:.4f}  "
            f"H1={result['h1_holds']}  H2={result['h2_holds']}  "
            f"({result['fit_seconds']:.0f}s)"
        )

    def stats(key: str) -> Dict[str, float]:
        vals = [r[key] for r in runs]
        return {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    summary = {
        "no_context_dice_tumor": stats("no_context_dice_tumor"),
        "meaningful_dice_tumor": stats("meaningful_dice_tumor"),
        "shuffled_dice_tumor": stats("shuffled_dice_tumor"),
        "meaningful_detection_recall": stats("meaningful_detection_recall"),
    }
    h1_holds_count = sum(1 for r in runs if r["h1_holds"])
    h2_holds_count = sum(1 for r in runs if r["h2_holds"])

    print("\n=== Сводка по всем разбиениям ===")
    for key, s in summary.items():
        print(f"   {key}: mean={s['mean']:.4f}  std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}")
    print(f"   H1 подтверждена в {h1_holds_count}/{len(runs)} разбиениях.")
    print(f"   H2 подтверждена в {h2_holds_count}/{len(runs)} разбиениях.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "n_splits": len(runs),
        "epochs": EPOCHS_MULTI_SPLIT,
        "device": str(base.DEVICE),
        "runs": runs,
        "summary": summary,
        "h1_holds_count": h1_holds_count,
        "h2_holds_count": h2_holds_count,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 — валидация H1/H2 по нескольким независимым разбиениям train/test",
        "",
        f"{len(runs)} независимых случайных патиентских разбиений (split_seed = "
        f"{SPLIT_SEEDS[:N_SPLITS]}), базовый U-Net (с BatchNorm), {EPOCHS_MULTI_SPLIT} эпох, "
        f"K=5. Устройство: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})." if base.DEVICE.type == "cuda" else "."),
        "",
        "По требованию медицинского рецензента (medical_article_requirements.txt, п.5): "
        "один прогон на одном разбиении недостаточен для вывода — ниже тот же "
        "эксперимент на нескольких независимых разбиениях.",
        "",
        "| split_seed | no_context (с опухолью) | meaningful | shuffled | H1 | H2 |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['split_seed']} | {r['no_context_dice_tumor']:.4f} | {r['meaningful_dice_tumor']:.4f} | "
            f"{r['shuffled_dice_tumor']:.4f} | {r['h1_holds']} | {r['h2_holds']} |"
        )
    lines += [
        "",
        "| Метрика | mean | std | min | max |",
        "|---|---|---|---|---|",
    ]
    for key, s in summary.items():
        lines.append(f"| {key} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
    lines += [
        "",
        f"**H1 подтверждена в {h1_holds_count}/{len(runs)} разбиениях.** "
        f"**H2 подтверждена в {h2_holds_count}/{len(runs)} разбиениях.**",
        "",
        "Ограничение: валидация охватывает только базовый U-Net (headline-модель "
        "статьи) на бюджете 6 эпох, не все 8 сравниваемых архитектур и не полную "
        "сетку в 15 эпох — ради контроля суммарного времени эксперимента. "
        "Распространение на все архитектуры — следующий шаг при необходимости.",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
