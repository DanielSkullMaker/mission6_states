"""Статья 4 — обобщение на ДРУГОЕ учреждение/источник снимков (по
требованию медицинского рецензента — medical_article_requirements.txt,
п.7: "метод работает с сырой яркостью пикселей, снимки с разных
аппаратов/больниц отличаются яркостью и контрастом — проверяли ли вы
это?").

BraTS2020 агрегирует данные НЕСКОЛЬКИХ исходных источников — это видно
из `name_mapping.csv` (столбец BraTS_2017_subject_ID вида
"Brats17_<SITE>_xxx_1", где SITE — исходный контрибьютор: CBICA
(University of Pennsylvania), TCIA (The Cancer Imaging Archive —
агрегирует данные других госпиталей), "2013" (наследие BraTS2013)).
Среди 160 пациентов, использованных в этой статье: CBICA=88, "2013"=20,
TCIA=3, без метки (новые пациенты BraTS2019/2020 без легаси-ID)=49.

Эксперимент: обучение ТОЛЬКО на CBICA (88 пациентов, один источник),
проверка на TCIA+"2013" (23 пациента, определённо ДРУГИЕ источники,
пациенты без метки исключены как источник неопределённого происхождения).
Сравнение с эталонным случайным разбиением (experiment_article4_unet_report.json,
где train/test перемешаны по всем источникам) количественно отвечает на
вопрос рецензента: падает ли качество при переносе на данные, которых не
было в обучающей выборке НИ В КАКОМ ВИДЕ (не просто новый пациент, а
новый источник снимков).

Ограничение: это НЕ проверка на снимках другого аппарата в строгом
клиническом смысле (BraTS — уже курируемый, нормализованный набор),
а приближение "другой институт/контрибьютор" в пределах имеющихся
данных — честная, но ограниченная версия требования рецензента.

Запуск:
    python experiments/experiment_article4_cross_institution.py
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

import experiment_article4_unet_segmentation as base

NAME_MAPPING_PATH = Path("data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv")
TRAIN_SITE = "CBICA"
TEST_SITES = {"TCIA", "2013"}
EPOCHS_CROSS_INSTITUTION = 6  # как в experiment_article4_architectures.py

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_cross_institution_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_cross_institution_results.md"
BASE_REPORT_PATH = ARTIFACTS_DIR / "experiment_article4_unet_report.json"


def load_site_mapping() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(NAME_MAPPING_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bid = r["BraTS_2020_subject_ID"]
            legacy = r.get("BraTS_2017_subject_ID", "")
            if legacy and legacy != "NA":
                parts = legacy.split("_")
                mapping[bid] = parts[1] if len(parts) > 1 else "UNKNOWN"
            else:
                mapping[bid] = "UNKNOWN"
    return mapping


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4: генерализация на другой институт (train=CBICA, test=TCIA+2013) ===")
    print(
        f"Устройство вычислений: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if base.DEVICE.type == "cuda" else "")
    )

    rows = base.load_metadata()
    site_map = load_site_mapping()
    all_patient_ids = sorted({r["patient_id"] for r in rows})

    train_ids = sorted(pid for pid in all_patient_ids if site_map.get(pid) == TRAIN_SITE)
    test_ids = sorted(pid for pid in all_patient_ids if site_map.get(pid) in TEST_SITES)
    print(f"Train (источник {TRAIN_SITE}): {len(train_ids)} пациентов.")
    print(f"Test (источники {sorted(TEST_SITES)}): {len(test_ids)} пациентов.")
    assert not (set(train_ids) & set(test_ids)), "train/test пересекаются по пациентам!"

    train_groups = base.group_by_patient(rows, train_ids)
    test_groups = base.group_by_patient(rows, test_ids)
    train_centers = base.build_center_list(train_groups, base.MAX_TRAIN_PER_CLASS, base.RANDOM_SEED)
    test_centers = base.all_centers(test_groups)
    print(f"n_train={len(train_centers)}, n_test={len(test_centers)}.")

    old_epochs = base.EPOCHS
    base.EPOCHS = EPOCHS_CROSS_INSTITUTION
    try:
        model, fit_seconds = base.train_model(train_groups, train_centers, n_context=base.N_CONTEXT)
    finally:
        base.EPOCHS = old_epochs

    result = base.evaluate_model(model, base.SegStackDataset(test_groups, test_centers, n_context=base.N_CONTEXT))
    print(
        f"\nCross-institution: mean_dice={result['mean_dice']:.4f}  "
        f"mean_dice_with_tumor={result['mean_dice_with_tumor']:.4f}  "
        f"detection_recall={result['slice_detection_recall']:.4f}"
    )

    baseline_comparison = None
    if BASE_REPORT_PATH.exists():
        base_report = json.loads(BASE_REPORT_PATH.read_text(encoding="utf-8"))
        baseline_comparison = base_report["results"]["meaningful_neighbors"]
        print(
            f"Эталон (случайное разбиение, все источники вперемешку): "
            f"mean_dice_with_tumor={baseline_comparison['mean_dice_with_tumor']:.4f}  "
            f"detection_recall={baseline_comparison['slice_detection_recall']:.4f}"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "train_site": TRAIN_SITE,
        "test_sites": sorted(TEST_SITES),
        "n_train_patients": len(train_ids),
        "n_test_patients": len(test_ids),
        "epochs": EPOCHS_CROSS_INSTITUTION,
        "fit_seconds": fit_seconds,
        "result": result,
        "baseline_random_split_comparison": baseline_comparison,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 — генерализация на другой источник снимков (cross-institution)",
        "",
        f"Обучение ТОЛЬКО на пациентах источника {TRAIN_SITE} ({len(train_ids)} чел.), "
        f"проверка на пациентах источников {sorted(TEST_SITES)} ({len(test_ids)} чел.) — "
        "источники не пересекаются ни на уровне пациента, ни на уровне контрибьютора данных. "
        f"K=5, {EPOCHS_CROSS_INSTITUTION} эпох. Устройство: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})." if base.DEVICE.type == "cuda" else "."),
        "",
        "| | mean Dice (все) | mean Dice (с опухолью) | detection recall | detection recall CI95 |",
        "|---|---|---|---|---|",
        f"| Cross-institution (train={TRAIN_SITE}, test={'+'.join(sorted(TEST_SITES))}) | "
        f"{result['mean_dice']:.4f} | {result['mean_dice_with_tumor']:.4f} | "
        f"{result['slice_detection_recall']:.4f} | "
        f"{'[%.3f, %.3f]' % tuple(result['slice_detection_recall_ci95']) if result['slice_detection_recall_ci95'] else '—'} |",
    ]
    if baseline_comparison is not None:
        lines.append(
            f"| Эталон (случайное разбиение, все источники вперемешку) | "
            f"{baseline_comparison['mean_dice']:.4f} | {baseline_comparison['mean_dice_with_tumor']:.4f} | "
            f"{baseline_comparison['slice_detection_recall']:.4f} | "
            f"{'[%.3f, %.3f]' % tuple(baseline_comparison['slice_detection_recall_ci95']) if baseline_comparison.get('slice_detection_recall_ci95') else '—'} |"
        )
        gap = baseline_comparison["mean_dice_with_tumor"] - result["mean_dice_with_tumor"]
        lines += [
            "",
            f"**Разрыв генерализации**: {gap*100:+.1f} п.п. mean Dice (с опухолью) при переносе "
            "на не встречавшийся в обучении источник снимков по сравнению со случайным "
            "разбиением по всем источникам вперемешку.",
        ]
    lines += [
        "",
        "Ограничение: это приближение \"перенос на другой институт/контрибьютор данных\" "
        "в пределах курируемого датасета BraTS2020 (сам BraTS уже нормализует интенсивность "
        "по собственному протоколу), а не проверка на снимках другого МРТ-аппарата/протокола "
        "съёмки в строгом клиническом смысле.",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
