"""⚠️ ДЕМОНСТРАЦИЯ НЕЧЕСТНОГО МЕТОДА ОЦЕНКИ — по прямому запросу: воспроизвести
"метод коррелированного датасета" (theory/code/Вариант_2_1.docx / theory/state_v2/
Вариант_4.docx, "Первый этап") и явно задокументировать, ПОЧЕМУ его результат
нельзя выдавать за качество модели.

Методология НЕЧЕСТНОГО сплита (буквально по формулировке задачи):
  Датасет — datasets/{glioma,meningioma,pituitary}_centered/ (200 изображений
  на класс). Для каждого класса независимо: находим пары максимально
  коррелирующих между собой изображений (нормированный коэффициент корреляции
  Пирсона), из каждой пары одно изображение уходит в train, другое — в test.
  200 изображений класса делятся РОВНО пополам: 100 train + 100 test на класс
  (300/300 итого на 3 класса). Реализация — evaluation.paired_correlation_split
  (уже есть в библиотеке, ровно эта методология — см. её docstring).

⚠️ ПОЧЕМУ ЭТО НЕЧЕСТНО (data leakage через дизайн сплита, не через код модели):
  Тестовая выборка не независима от обучающей — каждому тестовому изображению
  СОЗНАТЕЛЬНО подобрана его "почти копия" (сильно коррелированное изображение)
  в обучающей выборке. Модель, которая просто запоминает "похоже на X из
  train -> тот же класс, что у X", получает искусственное преимущество,
  которого не будет на реальных, независимо снятых новых снимках. Это не
  ошибка классификатора и не находка метода — это ошибка ДИЗАЙНА ОЦЕНКИ.
  Сам черновик статьи признаёт цену такой методологии дословно (см. README,
  раздел "Сплит через попарное разнесение").

Для количественной демонстрации ЭФФЕКТА (а не просто утверждения на словах)
скрипт считает ДВА сценария на ОДИНАКОВОМ объёме данных (300 train / 300 test,
по 100/100 на класс), различающихся ТОЛЬКО стратегией сплита:
  - "dishonest_correlated_pairs" — сплит выше (paired_correlation_split).
  - "honest_random_baseline"     — тот же объём, но paper_train_test_split
    (train_fraction=0.5) — честный СТРАТИФИЦИРОВАННЫЙ СЛУЧАЙНЫЙ сплит,
    train и test не имеют искусственной корреляции по конструкции.
Одинаковый классификатор (SubspaceConjugacyClassifier, n_subclasses=8 —
параметры по умолчанию библиотеки, как в main.py/reproduce_article1.py),
одинаковый random_state, одинаковый размер train/test — единственная
переменная — это стратегия отбора train/test. Разница в accuracy между
сценариями измеряет ИМЕННО цену нечестного сплита, а не что-то ещё.

Артефакты:
  - artifacts/dishonest_correlated_pairs_report.json — точные числа
  - artifacts/dishonest_correlated_pairs_report.md    — читаемый отчёт с
    предупреждением о нечестности метода

Запуск:
    python experiments/dishonest_correlated_pairs_report.py
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    paired_correlation_split,
    paper_train_test_split,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

DATASET_ROOT = Path("datasets")
CLASSES = ["glioma", "meningioma", "pituitary"]
N_PER_CLASS = 200
RANDOM_SEED = 42
N_SUBCLASSES = 8  # параметры по умолчанию библиотеки (main.py, reproduce_article1.py)

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "dishonest_correlated_pairs_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "dishonest_correlated_pairs_report.md"


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


def run_scenario(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    print(f"\n=== Сценарий: {name} ===")
    for cls in CLASSES:
        print(f"   {cls}: train={int(np.sum(y_train == cls))}, test={int(np.sum(y_test == cls))}")

    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    print(f"   fit() заняло {fit_seconds:.1f}s")

    y_pred = clf.predict(X_test)
    report = evaluate_classifier(y_test, y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    print(f"   accuracy={report['accuracy']:.4f}, per_class={per_class}")

    return {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "fit_seconds": fit_seconds,
        "accuracy": report["accuracy"],
        "per_class_accuracy": per_class,
        "confusion_matrix": report["confusion_matrix"].tolist(),
        "labels": [str(l) for l in report["labels"]],
    }


def save_markdown_report(results: Dict[str, Dict[str, Any]]) -> None:
    dishonest = results["dishonest_correlated_pairs"]
    honest = results["honest_random_baseline"]
    gap = dishonest["accuracy"] - honest["accuracy"]

    lines = [
        "# ⚠️ Отчёт по НЕЧЕСТНОМУ методу оценки — «попарное разнесение коррелированных изображений»",
        "",
        "**Этот отчёт описывает методологически некорректный (нечестный) способ "
        "оценки точности. Числа ниже НЕЛЬЗЯ цитировать как характеристику "
        "качества метода подпространственной сопряжённости — они измеряют "
        "преимущественно эффект утечки данных между train и test, а не "
        "способность классификатора обобщать на новые, независимые снимки.**",
        "",
        "## В чём нечестность метода",
        "",
        "Датасет `datasets/{glioma,meningioma,pituitary}_centered/` (200 "
        "изображений на класс) делится РОВНО пополам по каждому классу — 100 "
        "train + 100 test (300/300 итого). Деление не случайное: для каждого "
        "класса независимо ищутся пары НАИБОЛЕЕ коррелирующих между собой "
        "изображений (нормированный коэффициент корреляции Пирсона), и из "
        "каждой такой пары одно изображение намеренно кладётся в train, "
        "другое — в test (`subspace_conjugacy.evaluation.paired_correlation_split`, "
        "буквальная реализация «Первого этапа» чернового варианта статьи "
        "Коршикова и Фурсова, `theory/code/Вариант_2_1.docx` / "
        "`theory/state_v2/Вариант_4.docx`).",
        "",
        "Из-за этого тестовая выборка НЕ независима от обучающей: почти "
        "каждому тестовому изображению соответствует «почти копия» — сильно "
        "похожее изображение — в обучающей выборке. Это завышает измеренную "
        "accuracy по сравнению с честной, независимой оценкой — сам "
        "первоисточник методологии признаёт эту цену дословно (см. README, "
        "раздел «Сплит через попарное разнесение»).",
        "",
        "## Методология эксперимента",
        "",
        "Чтобы измерить именно ЦЕНУ нечестного сплита (а не что-то ещё), оба "
        "сценария ниже используют ОДИНАКОВЫЙ объём данных (300 train / 300 "
        "test, по 100/100 на класс), ОДИНАКОВЫЙ классификатор "
        f"(`SubspaceConjugacyClassifier(n_subclasses={N_SUBCLASSES})`, параметры "
        "по умолчанию библиотеки) и одинаковый `random_state`. Единственное "
        "отличие — СТРАТЕГИЯ выбора train/test:",
        "",
        "- **dishonest_correlated_pairs** — нечестный сплит, описанный выше "
        "(`paired_correlation_split`).",
        "- **honest_random_baseline** — тот же объём (100/100 на класс, "
        "стратифицированно), но независимый случайный сплит "
        "(`paper_train_test_split(train_fraction=0.5)`) — train и test не "
        "связаны намеренной корреляцией.",
        "",
        "## Результаты",
        "",
        "| Сценарий | train | test | accuracy | glioma | meningioma | pituitary | fit(), с |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        pc = r["per_class_accuracy"]
        lines.append(
            f"| {name} | {r['n_train']} | {r['n_test']} | {r['accuracy']:.4f} | "
            f"{pc.get('glioma', float('nan')):.3f} | {pc.get('meningioma', float('nan')):.3f} | "
            f"{pc.get('pituitary', float('nan')):.3f} | {r['fit_seconds']:.1f} |"
        )

    pc_dis = dishonest["per_class_accuracy"]
    pc_hon = honest["per_class_accuracy"]

    lines += [
        "",
        f"**Разрыв (нечестный − честный) = {gap:+.4f}** на одинаковом объёме "
        "данных и одинаковой модели.",
        "",
        "## Вывод — результат НЕ такой, как можно было бы ожидать, и это важно",
        "",
    ]

    if gap > 0.01:
        lines.append(
            f"Нечестный сплит завысил accuracy на {gap:+.4f} относительно честного "
            "контроля того же объёма — ожидаемое направление эффекта утечки."
        )
    else:
        lines += [
            f"На каноническом `SubspaceConjugacyClassifier(n_subclasses={N_SUBCLASSES})` "
            f"нечестный сплит НЕ завысил общую accuracy — разрыв {gap:+.4f} "
            "практически нулевой (в этом прогоне даже слегка отрицательный). "
            "**Это не означает, что метод сплита честный** — дизайн сплита "
            "остаётся некорректным по построению (тестовая выборка намеренно "
            "подобрана похожей на обучающую), а нулевой суммарный эффект здесь "
            "— особенность ИМЕННО этой модели, а не отсутствие утечки как "
            "таковой. По классам эффект заметен и разнонаправлен: "
            f"meningioma {pc_hon.get('meningioma', float('nan')):.2f} -> "
            f"{pc_dis.get('meningioma', float('nan')):.2f} (утечка ПОМОГЛА слабому "
            f"классу), pituitary {pc_hon.get('pituitary', float('nan')):.2f} -> "
            f"{pc_dis.get('pituitary', float('nan')):.2f} (стало ХУЖЕ) — эффекты "
            "по классам компенсируют друг друга в среднем, не отменяя того, что "
            "на каждый отдельный класс сплит подействовал через утечку, а не "
            "через честное обобщение.",
            "",
            "Почему эффект здесь намного слабее, чем в «Таблице 9» "
            "`experiments/experiment_draft_variant2.py` (там разрыв драматичен: "
            "0.71–0.84 на нечестном сплите против 0.32–0.34 на честном, для "
            "того же класса методов и той же идеи разнесения пар): в том "
            "эксперименте классификатор строит ОДНО подпространство размера "
            "всего m=10..50 векторов на класс — при малом m «двойник» "
            "тестового изображения с высокой вероятностью оказывается ОДНИМ ИЗ "
            "НЕМНОГИХ отобранных опорных векторов, и предсказание становится "
            "почти тавтологией. Здесь же `SubspaceConjugacyClassifier` "
            f"использует ВСЕ {dishonest['n_train'] // len(CLASSES)} обучающих "
            f"изображений КАЖДОГО класса, разложенные на {N_SUBCLASSES} "
            "подклассов — присутствие "
            "конкретного «двойника» где-то в этом большом наборе гораздо "
            "меньше влияет на итоговый argmax R(x, Y) по всем подпространствам.",
        ]

    lines += [
        "",
        "**Методологический вывод не меняется от знака этого конкретного "
        "разрыва**: дизайн сплита `paired_correlation_split` остаётся "
        "некорректным для отчётных цифр качества независимо от того, помог он "
        "или навредил в конкретном прогоне — тестовая выборка здесь по "
        "построению не независима от обучающей. Единственная корректная "
        "методология для отчётных цифр — честный независимый сплит "
        "(`paper_train_test_split`/`train_test_split` со `stratify`), как "
        "зафиксировано в README (раздел «Известные ограничения») и в Статье 6 "
        "диссертационного плана. Величина и даже знак эффекта утечки, как "
        "показано выше, зависят от конкретной схемы классификации — это "
        "довод СИЛЬНЕЕ, а не слабее: нечестный сплит непредсказуемо искажает "
        "оценку, и единственный надёжный способ этого избежать — не "
        "использовать его вовсе, а не полагаться на то, что эффект будет "
        "«в ожидаемую сторону».",
    ]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown-отчёт: {REPORT_MD_PATH}")


def main() -> None:
    start = time.perf_counter()
    print("=== [НЕЧЕСТНЫЙ МЕТОД] Попарное разнесение коррелированных изображений ===")
    print("Результаты ниже НЕ являются корректной оценкой качества модели.\n")

    print("Загрузка и векторизация datasets/*_centered...")
    X, y = load_data()
    print(f"X.shape={X.shape}")

    print("\nНечестный сплит (paired_correlation_split)...")
    dishonest_split = paired_correlation_split(X, y, random_state=RANDOM_SEED)
    dishonest_result = run_scenario(
        "dishonest_correlated_pairs",
        dishonest_split["X_train"], dishonest_split["y_train"],
        dishonest_split["X_test"], dishonest_split["y_test"],
    )

    print("\nЧестный контроль того же объёма (paper_train_test_split, train_fraction=0.5)...")
    X_train_h, X_test_h, y_train_h, y_test_h = paper_train_test_split(
        X, y, train_fraction=0.5, random_state=RANDOM_SEED
    )
    honest_result = run_scenario(
        "honest_random_baseline", X_train_h, y_train_h, X_test_h, y_test_h
    )

    results = {
        "dishonest_correlated_pairs": dishonest_result,
        "honest_random_baseline": honest_result,
    }

    gap = dishonest_result["accuracy"] - honest_result["accuracy"]
    print(f"\n=== Разрыв (нечестный - честный): {gap:+.4f} ===")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_subclasses": N_SUBCLASSES,
                "random_seed": RANDOM_SEED,
                "dataset_size_per_class": N_PER_CLASS,
                "warning": "dishonest_correlated_pairs scenario is NOT a valid quality metric — see report .md",
                "results": results,
                "gap_dishonest_minus_honest": gap,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"JSON-отчёт: {REPORT_JSON_PATH}")

    save_markdown_report(results)

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
