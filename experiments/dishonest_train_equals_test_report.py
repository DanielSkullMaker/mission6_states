"""⚠️ ДЕМОНСТРАЦИЯ НЕЧЕСТНОГО МЕТОДА ОЦЕНКИ №2 — по прямому запросу: "пусть в
test окажется вся train выборка, то есть весь датасет станет test".

Это самая грубая и самая известная форма утечки данных (data leakage) —
оценка модели на ТЕХ ЖЕ объектах, на которых она обучалась. В отличие от
предыдущего эксперимента (`dishonest_correlated_pairs_report.py`, где test
состоял из ПОХОЖИХ, но всё же ФОРМАЛЬНО ДРУГИХ изображений), здесь train и
test — БУКВАЛЬНО ОДНО И ТО ЖЕ множество (100% пересечение, не просто высокая
корреляция).

Методология НЕЧЕСТНОГО сценария (буквально по формулировке задачи):
  1. Берём ВЕСЬ датасет — datasets/{glioma,meningioma,pituitary}_centered/,
     по 200 изображений на класс (600 итого). Это "train".
  2. Обучаем SubspaceConjugacyClassifier(n_subclasses=8) (параметры по
     умолчанию библиотеки, как в main.py/reproduce_article1.py) на этом train.
  3. "Test" — ТОТ ЖЕ САМЫЙ датасет целиком (600 изображений), без единого
     отличающегося объекта. train_indices == test_indices.

⚠️ ПОЧЕМУ ЭТО НЕЧЕСТНО: измеренная здесь "accuracy" не имеет отношения к
способности модели распознавать НОВЫЕ, ранее не виденные снимки — она
измеряет лишь то, насколько объекты, УЖЕ УЧАСТВОВАВШИЕ в построении
подпространств (фазы A/B алгоритма — глобальная пара, центры, наполнение
кластеров), затем "узнаются" этими же подпространствами. Часть обучающих
векторов (2*n_subclasses на класс, ставшие буквальными столбцами базиса при
freeze_basis_at=2) лежат в built-подпространстве СВОЕГО класса ТОЧНО
(R(x, Y) = 1.0 по построению) — для них правильный ответ гарантирован
конструкцией метода, а не обобщением. Остальные обучающие векторы класса
были отнесены к своему подклассу как раз потому, что ИХ подпространство дало
для них наибольшее R среди подпространств СВОЕГО класса на момент кластери-
зации — это тоже не независимая проверка. Подробный разбор этого же
механизма (без полного датасета, с разверткой по n_subclasses) — уже в
`experiments/experiment_data_leakage_demo.py`; этот скрипт — прямая,
отдельная демонстрация ровно по формулировке задачи, на полном датасете, с
явным отчётом и явным честным контролем для сравнения.

Для контекста используется УЖЕ ПОСЧИТАННЫЙ честный baseline
(`artifacts/article1_report.json`, сценарий "70_30" — независимый
стратифицированный 80/20... 70/30 сплит, accuracy 0.6111) — та же модель
(SubspaceConjugacyClassifier(n_subclasses=8)), тот же датасет, честная
методология. ⚠️ Это НЕ идеально согласованное сравнение по объёму (честный
baseline тестируется на 180 независимых изображениях, здесь — на всех 600
буквально тех же, что видела модель) — это неизбежное следствие самой сути
эксперимента (весь датасет — train и test одновременно, отдельных
"свободных" данных для честного контроля такого же объёма просто не
существует). Разница явно оговорена в отчёте.

Артефакты:
  - artifacts/dishonest_train_equals_test_report.json
  - artifacts/dishonest_train_equals_test_report.md

Запуск:
    python experiments/dishonest_train_equals_test_report.py
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from subspace_conjugacy import SubspaceConjugacyClassifier, load_and_vectorize_batch
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

DATASET_ROOT = Path("datasets")
CLASSES = ["glioma", "meningioma", "pituitary"]
N_PER_CLASS = 200
N_SUBCLASSES = 8  # параметры по умолчанию библиотеки (main.py, reproduce_article1.py)

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "dishonest_train_equals_test_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "dishonest_train_equals_test_report.md"

# Честный ориентир для сравнения — уже посчитан в reproduce_article1.py,
# не пересчитывается здесь заново (та же модель, тот же датасет).
HONEST_REFERENCE_PATH = ARTIFACTS_DIR / "article1_report.json"


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Загружает и векторизует datasets/{class}_centered/ целиком (200 на класс)."""
    X_list, y_list = [], []
    for cls in CLASSES:
        paths = sorted((DATASET_ROOT / f"{cls}_centered").glob("*.png"))
        if len(paths) != N_PER_CLASS:
            raise FileNotFoundError(
                f"{cls}: найдено {len(paths)} изображений в "
                f"{DATASET_ROOT / f'{cls}_centered'}, ожидалось {N_PER_CLASS}."
            )
        X_list.append(load_and_vectorize_batch(paths, method="horizontal"))
        y_list.append(np.full(len(paths), cls))
    return np.vstack(X_list), np.concatenate(y_list)


def load_honest_reference() -> Dict[str, Any]:
    """Честный baseline (70/30, тот же классификатор) — из уже посчитанного
    article1_report.json (experiments/reproduce_article1.py). Не пересчитывается,
    чтобы не дублировать ~5.5 минут fit() ради уже известного числа."""
    if not HONEST_REFERENCE_PATH.exists():
        return None
    with open(HONEST_REFERENCE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]["70_30"]


def main() -> None:
    start = time.perf_counter()
    print("=== [НЕЧЕСТНЫЙ МЕТОД №2] train == test (весь датасет — и train, и test) ===")
    print("Результаты ниже НЕ являются корректной оценкой качества модели.\n")

    print("Загрузка и векторизация datasets/*_centered...")
    X, y = load_data()
    print(f"X.shape={X.shape}")

    print(f"\nОбучение SubspaceConjugacyClassifier(n_subclasses={N_SUBCLASSES}) на ВСЕХ {X.shape[0]} изображениях...")
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(X, y)
    fit_seconds = time.perf_counter() - t0
    print(f"fit() заняло {fit_seconds:.1f}s")

    print("\nПредсказание на ТОМ ЖЕ САМОМ датасете (test = train, 100% пересечение)...")
    y_pred = clf.predict(X)
    report = evaluate_classifier(y, y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    print(f"accuracy (утечка)={report['accuracy']:.4f}, per_class={per_class}")

    leak_result = {
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
                "n_subclasses": N_SUBCLASSES,
                "dataset_size_per_class": N_PER_CLASS,
                "warning": "train==test (100% overlap) — accuracy here is NOT a valid quality metric, see report .md",
                "results": {"dishonest_train_equals_test": leak_result},
                "honest_reference_70_30": honest_ref,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    save_markdown_report(leak_result, honest_ref)

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


def save_markdown_report(leak: Dict[str, Any], honest_ref: Dict[str, Any]) -> None:
    pc = leak["per_class_accuracy"]
    lines = [
        "# ⚠️ Отчёт по НЕЧЕСТНОМУ методу оценки — «train == test» (весь датасет — и обучение, и тест)",
        "",
        "**Этот отчёт описывает самую грубую и известную форму утечки данных: "
        "модель оценивается на ТЕХ ЖЕ объектах, на которых она обучалась. "
        "Число ниже НЕЛЬЗЯ цитировать как accuracy классификатора ни при "
        "каких обстоятельствах — оно не измеряет обобщающую способность "
        "вообще, а измеряет лишь то, насколько метод \"узнаёт\" объекты, "
        "уже использованные при построении собственных же подпространств.**",
        "",
        "## В чём нечестность метода",
        "",
        f"Весь датасет `datasets/{{glioma,meningioma,pituitary}}_centered/` "
        f"({leak['n_train']} изображений, по {leak['n_train'] // len(CLASSES)} "
        "на класс) целиком используется и как train, и как test — "
        "`train_indices == test_indices`, пересечение 100%. Это не "
        "«тестовая выборка, похожая на обучающую» (как в предыдущем "
        "нечестном эксперименте с попарным разнесением по корреляции) — "
        "здесь test буквально совпадает с train поэлементно.",
        "",
        "Почему это в ПРИНЦИПЕ должно завышать результат (механизм, а не "
        "просто общие слова): при `freeze_basis_at=2` (по умолчанию) "
        f"РОВНО {2 * N_SUBCLASSES} обучающих векторов на класс (2 на каждый "
        "из 8 подклассов) буквально становятся СТОЛБЦАМИ базиса Y своего "
        "подпространства. Для такого вектора x показатель сопряжённости "
        "R(x, Y) = 1.0 ТОЧНО, по построению метода (x лежит в span(Y)) — для "
        f"этих {2 * N_SUBCLASSES} из 200 векторов класса (8%) правильный "
        "ответ гарантирован конструкцией, а не обобщением. Остальные ~92% "
        "обучающих векторов класса такой гарантии не имеют: они были "
        "отнесены к своему подклассу на этапе кластеризации по максимуму R "
        "относительно РАСТУЩЕГО на тот момент базиса, а не относительно "
        "финального замороженного 2-векторного базиса, который реально "
        "используется в `predict()` — так что даже при 100%-м пересечении "
        "train/test далеко не для всех обучающих векторов исход "
        "предопределён. Это же объясняет, почему измеренный эффект ниже "
        "ожидаемого (см. «Результат» ниже).",
        "",
        "## Результат",
        "",
        "| Сценарий | train | test | пересечение train/test | accuracy | glioma | meningioma | pituitary | fit(), с |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| **train == test (нечестный)** | {leak['n_train']} | {leak['n_test']} | 100% | "
        f"**{leak['accuracy']:.4f}** | {pc.get('glioma', float('nan')):.3f} | "
        f"{pc.get('meningioma', float('nan')):.3f} | {pc.get('pituitary', float('nan')):.3f} | "
        f"{leak['fit_seconds']:.1f} |",
    ]

    if honest_ref is not None:
        hpc = honest_ref["per_class_accuracy"]
        gap = leak["accuracy"] - honest_ref["accuracy"]
        lines += [
            f"| честный baseline (70/30, для сравнения) | {honest_ref['n_train']} | "
            f"{honest_ref['n_test']} | 0% | {honest_ref['accuracy']:.4f} | "
            f"{hpc.get('glioma', float('nan')):.3f} | {hpc.get('meningioma', float('nan')):.3f} | "
            f"{hpc.get('pituitary', float('nan')):.3f} | {honest_ref['fit_seconds']:.1f} |",
            "",
            f"**Разрыв (нечестный − честный) = {gap:+.4f}** — но ⚠️ это сравнение НЕ "
            "идеально согласовано по объёму: честная строка оценивается на 180 "
            "НЕЗАВИСИМЫХ изображениях (не участвовавших в обучении своей модели), "
            f"а нечестная — на всех {leak['n_test']} изображениях, буквально "
            "видевшихся моделью при обучении. Разница в объёме тестовой выборки "
            "— неизбежное следствие самой сути эксперимента (весь датасет — и "
            "train, и test одновременно, отдельных «свободных» данных для "
            "полностью согласованного honest-контроля такого же объёма не "
            "существует).",
            "",
            "⚠️ **Измеренный эффект оказался НАМНОГО скромнее, чем можно было "
            "бы ожидать от 100%-го пересечения train/test** — не 90-100%, "
            f"а всего {leak['accuracy']:.3f} (+{gap:.3f} к честному baseline). "
            "Причина — в самой конструкции метода (см. «В чём нечестность "
            "метода» выше): гарантированный R=1.0 есть только у 8% векторов "
            "класса (буквальных столбцов замороженного базиса), остальные "
            "92% при предсказании конкурируют по тому же правилу argmax "
            "R(x, Y), что и настоящие новые изображения — просто с чуть "
            "больше благоприятным для себя (но не гарантированным) исходом. "
            "**Это НЕ довод в пользу того, что метод сплита можно "
            "использовать** — направление эффекта верное (нечестный результат "
            "выше честного), а скромная величина показывает лишь, что ДЛЯ "
            "ЭТОГО КОНКРЕТНОГО классификатора (`n_subclasses=8`, "
            "`freeze_basis_at=2`) риск «переобучения через утечку» ограничен "
            "архитектурой метода, а не тем, что 100%-e пересечение train/test "
            "чем-то менее некорректно как дизайн оценки.",
        ]
    else:
        lines.append(
            "\n(Честный baseline из `artifacts/article1_report.json` не найден — "
            "запустите `experiments/reproduce_article1.py` для сравнения.)"
        )

    lines += [
        "",
        "## Вывод",
        "",
        f"Точность **{leak['accuracy']:.4f}** в этом сценарии — не характеристика "
        "качества метода подпространственной сопряжённости, а частичное "
        "следствие того, что test-объекты уже были использованы при "
        "построении подпространств, по которым их же классифицируют. По "
        "СТРОГОСТИ ДИЗАЙНА это худшая из всех нечестных методологий, "
        "рассмотренных в проекте — 100% пересечение train/test строго хуже, "
        "чем `paired_correlation_split` "
        "(`dishonest_correlated_pairs_report.md`), где test хотя бы формально "
        "состоял из ДРУГИХ файлов. Но по ИЗМЕРЕННОЙ ВЕЛИЧИНЕ эффекта разница "
        f"скромная (здесь +{leak['accuracy'] - (honest_ref['accuracy'] if honest_ref else 0):.3f} "
        "к честному baseline, там — от −0.003 до драматических +0.37…+0.50 в "
        "зависимости от схемы классификации) — **строгость дизайна оценки и "
        "величина её искажения — разные вещи**: полностью некорректный "
        "дизайн (100% пересечение) не гарантирует драматически более высокое "
        "число, если сам метод (архитектура с малым замороженным базисом) "
        "объективно ограничивает, сколько вектор может «выиграть» от того, "
        "что он уже виден моделью. Даже в этом самом нечестном из всех "
        "рассмотренных сценариев класс meningioma остаётся слабым (accuracy "
        "0.215 — ХУЖЕ случайного угадывания ~0.33 для 3 классов, даже видя "
        "собственные обучающие данные) — это сильный сигнал, что проблема с "
        "meningioma структурная (сама схема "
        "разбиения на подпространства плохо описывает этот класс), а не "
        "следствие честности/нечестности оценки.",
        "",
        "Единственная корректная методология для отчётных цифр качества — "
        "оценка на независимой, не участвовавшей в обучении выборке "
        "(`paper_train_test_split`/`train_test_split` со `stratify`), как "
        "зафиксировано в README (раздел «Известные ограничения») и в Статье 6 "
        "диссертационного плана — независимо от того, насколько сильно "
        "конкретный нечестный дизайн исказил число в конкретном прогоне.",
    ]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
