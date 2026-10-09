"""⚠️ ДЕМОНСТРАЦИЯ НЕЧЕСТНОГО МЕТОДА ОЦЕНКИ №3 — "интереса ради": полностью
нечестная ветка Статьи 4 (train == test + взвинченное число подпространств
n_subclasses=45), оформленная как САМОСТОЯТЕЛЬНЫЙ отчёт в стиле
dishonest_train_equals_test_report.py / dishonest_correlated_pairs_report.py.

Это НЕ новый эксперимент — тот же сценарий уже посчитан и честно разобран в
experiments/experiment_article4_boost_accuracy.py (решётка n_subclasses
[8, 20, 35, 45] на train==test, artifacts/article4_draft.docx, раздел
"Пределы методологически корректного и некорректного повышения точности").
Здесь ТОТ ЖЕ сценарий пересчитан отдельно и представлен так, КАК БУДТО это
самостоятельный, валидный результат — с полным набором "убедительных"
атрибутов хорошего отчёта (матрица ошибок, per-class accuracy, таблица,
уверенный тон) — чтобы наглядно показать, насколько правдоподобно может
выглядеть абсолютно некорректный результат, если его не сопроводить
предупреждением. Все предупреждения ниже оставлены намеренно — это учебный
артефакт, а не черновик для публикации.

Механизм (напоминание, подробно разобран в основной статье): при
TRAIN_EQUALS_TEST_PER_CLASS=100 и n_subclasses=45 на подпространство
приходится в среднем ~2.2 обучающих объекта — подпространство каждого
подкласса подгоняется практически под конкретные обучающие векторы, а не
под класс в целом. В сочетании с train==test это даёт почти полное
"узнавание себя", а не обобщающую способность.

Артефакты:
  - artifacts/dishonest_brats_capacity_n45_report.json
  - artifacts/dishonest_brats_capacity_n45_report.md

Запуск (после experiments/prepare_brats_dataset.py):
    python experiments/dishonest_brats_capacity_report.py
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from subspace_conjugacy import SubspaceConjugacyClassifier, load_and_vectorize_batch
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
TRAIN_EQUALS_TEST_PER_CLASS = 100  # как в experiment_article4_boost_accuracy.py
N_SUBCLASSES = 45  # "интереса ради" — почти предельное значение (2*45=90 <= 100)

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "dishonest_brats_capacity_n45_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "dishonest_brats_capacity_n45_report.md"

# Честный ориентир для сравнения — уже посчитан в experiment_brats_split_methods.py,
# не пересчитывается здесь заново (тот же датасет, та же задача, n_subclasses=8).
HONEST_REFERENCE_PATH = ARTIFACTS_DIR / "experiment_brats_split_methods_report.json"

Row = Dict[str, Any]


def load_metadata() -> List[Row]:
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "patient_id": r["patient_id"],
                "path": DATASET_ROOT / r["path"],
                "label": "tumor" if r["has_tumor"] == "True" else "no_tumor",
            })
    return rows


def sample_pool(rows: List[Row], per_class: int, seed: int) -> List[Row]:
    """patient_id ИГНОРИРУЕТСЯ — как в methods_split.py::sample_pool."""
    rng = np.random.default_rng(seed)
    pool: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in rows if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        pool.extend(cls_rows[i] for i in idx)
    return pool


def load_honest_reference() -> Dict[str, Any]:
    if not HONEST_REFERENCE_PATH.exists():
        return None
    with open(HONEST_REFERENCE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]["patient_split"]


def main() -> None:
    start = time.perf_counter()
    print(f"=== [НЕЧЕСТНЫЙ МЕТОД №3] train == test + n_subclasses={N_SUBCLASSES} (BraTS has_tumor) ===")
    print("Результат ниже НЕ является корректной оценкой качества модели.\n")

    rows = load_metadata()
    pool = sample_pool(rows, TRAIN_EQUALS_TEST_PER_CLASS, RANDOM_SEED + 1)
    print(f"Загрузка и векторизация {len(pool)} срезов (train == test, patient_id игнорируется)...")
    X = load_and_vectorize_batch([r["path"] for r in pool], method="horizontal")
    y = np.array([r["label"] for r in pool])
    print(f"X.shape={X.shape}")

    avg_per_subclass = X.shape[0] / 2 / N_SUBCLASSES
    print(f"\nОбучение SubspaceConjugacyClassifier(n_subclasses={N_SUBCLASSES}) на ВСЕХ "
          f"{X.shape[0]} срезах (~{avg_per_subclass:.1f} объектов на подпространство)...")
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(X, y)
    fit_seconds = time.perf_counter() - t0
    print(f"fit() заняло {fit_seconds:.1f}s")

    print("\nПредсказание на ТОМ ЖЕ САМОМ пуле (test = train, 100% пересечение)...")
    y_pred = clf.predict(X)
    report = evaluate_classifier(y, y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    print(f"accuracy (утечка + ёмкость)={report['accuracy']:.4f}, per_class={per_class}")

    leak_result = {
        "n_subclasses": N_SUBCLASSES,
        "avg_objects_per_subclass": avg_per_subclass,
        "n_train": int(X.shape[0]),
        "n_test": int(X.shape[0]),
        "overlap_fraction": 1.0,
        "fit_seconds": fit_seconds,
        "accuracy": report["accuracy"],
        "per_class_accuracy": per_class,
        "confusion_matrix": report["confusion_matrix"].tolist(),
        "labels": [str(l) for l in report["labels"]],
    }

    honest_ref = load_honest_reference()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "warning": "train==test (100% overlap) + n_subclasses=45 (capacity/memorization) "
                           "— accuracy here is NOT a valid quality metric, see report .md",
                "results": {"dishonest_capacity_n45": leak_result},
                "honest_reference_patient_split": honest_ref,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    save_markdown_report(leak_result, honest_ref)
    print(f"\n=== Завершено за {(time.perf_counter() - start):.1f} с ===")


def save_markdown_report(leak: Dict[str, Any], honest_ref: Dict[str, Any]) -> None:
    pc = leak["per_class_accuracy"]
    lines = [
        "# ⚠️ Отчёт по НЕЧЕСТНОМУ методу оценки — «train == test» + "
        f"n_subclasses={leak['n_subclasses']} (ёмкость/запоминание), BraTS has_tumor",
        "",
        "**Этот отчёт существует «интереса ради» — как иллюстрация того, "
        "насколько убедительно может выглядеть полностью некорректный "
        "результат, если его не сопроводить предупреждением. Число ниже "
        "НЕЛЬЗЯ цитировать как accuracy классификатора ни при каких "
        "обстоятельствах — полный разбор того же сценария с честным "
        "контролем и объяснением механизма уже есть в основном тексте "
        "Статьи 4 (раздел «Пределы методологически корректного и "
        "некорректного повышения точности»). Этот файл — тот же "
        "сценарий, представленный БЕЗ смягчающего контекста, каким он "
        "выглядел бы в недобросовестной работе.**",
        "",
        "## В чём нечестность метода (две накладывающиеся проблемы)",
        "",
        f"1. **Train == test**: все {leak['n_train']} срезов используются "
        "и как обучающая, и как тестовая выборка одновременно — "
        "пересечение 100%, `patient_id` при формировании пула "
        "игнорируется.",
        f"2. **Взвинченная ёмкость модели**: при n_subclasses="
        f"{leak['n_subclasses']} на подпространство приходится в среднем "
        f"~{leak['avg_objects_per_subclass']:.1f} обучающих объекта — "
        "подпространство каждого подкласса подгоняется практически под "
        "конкретные обучающие векторы, а не под класс в целом "
        "(запоминание вместо обобщения). По отдельности каждая из этих "
        "двух проблем уже недопустима; вместе они усиливают друг друга: "
        "чем выше n_subclasses при train==test, тем ближе предсказание "
        "к тривиальному «узнаванию себя».",
        "",
        "## Результат",
        "",
        "| Сценарий | n_subclasses | train | test | пересечение | accuracy | "
        "без опухоли | опухоль | fit(), с |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| **train == test + высокая ёмкость (нечестный)** | "
        f"{leak['n_subclasses']} | {leak['n_train']} | {leak['n_test']} | "
        f"100% | **{leak['accuracy']:.4f}** | "
        f"{pc.get('no_tumor', float('nan')):.3f} | "
        f"{pc.get('tumor', float('nan')):.3f} | {leak['fit_seconds']:.1f} |",
    ]

    if honest_ref is not None:
        hpc = honest_ref["per_class_accuracy"]
        gap = leak["accuracy"] - honest_ref["accuracy"]
        lines += [
            f"| честный ориентир (непересекающиеся пациенты, S=8, для "
            f"сравнения) | 8 | {honest_ref['n_train']} | "
            f"{honest_ref['n_test']} | 0% | {honest_ref['accuracy']:.4f} | "
            f"{hpc.get('no_tumor', float('nan')):.3f} | "
            f"{hpc.get('tumor', float('nan')):.3f} | "
            f"{honest_ref['fit_seconds']:.1f} |",
            "",
            f"**Разрыв (нечестный − честный) = {gap:+.4f} "
            f"({gap*100:+.1f} процентных пункта)** — почти "
            "«идеальная» точность получена БЕЗ единой содержательной "
            "идеи по методу или признаковому представлению, исключительно "
            "подбором одного гиперпараметра в сочетании с заведомо "
            "некорректным протоколом оценки. ⚠️ Сравнение не идеально "
            "согласовано по числу подпространств (S=8 в честной строке "
            "против S=45 в нечестной) — это намеренно: число подпространств "
            "здесь ЯВЛЯЕТСЯ частью демонстрируемой нечестности, а не "
            "посторонним фактором, который следовало бы уравнять.",
        ]
    else:
        lines.append(
            "\n(Честный ориентир из "
            "`artifacts/experiment_brats_split_methods_report.json` не "
            "найден — запустите `experiments/experiment_brats_split_methods.py`.)"
        )

    lines += [
        "",
        "## Вывод",
        "",
        f"Точность **{leak['accuracy']:.4f}** ({leak['accuracy']*100:.1f}%) "
        "в этом сценарии не характеризует качество метода подпространственной "
        "сопряжённости на задаче BraTS has_tumor — она почти целиком "
        "объясняется тем, что (а) модель оценена на данных, которые сама "
        "же видела при обучении, и (б) ёмкость модели подобрана так, "
        "чтобы облегчить дословное запоминание этих данных. Любая "
        "публикация, где фигурировало бы число такого порядка без "
        "сопутствующего описания протокола train/test-сплита и числа "
        "подпространств, должна вызывать прямое методологическое "
        "недоверие — полный честный разбор того же механизма (включая "
        "решётку n_subclasses и статистическую проверку другого "
        "результата статьи) приведён в основном тексте Статьи 4.",
    ]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
