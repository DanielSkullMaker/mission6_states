"""Воспроизводит эксперимент Статьи 1 диссертационного плана (dissertation_articles_plan.txt):

  «Метод подпространственной сопряжённости для классификации медицинских
  изображений: теоретические основы и воспроизведение результатов»

Раздел 5 плана ("Описание эксперимента") прямо называет main.py воспроизводящим
скриптом для этого сценария, но main.py в этом репозитории решает ДРУГУЮ
задачу (Table I/II опубликованной статьи Korshikov & Fursov на внешнем Kaggle
"Brain Tumor MRI Dataset", путь через BRAIN_MRI_ARCHIVE_ROOT, перебор
n_subclasses по сетке) — README и план ссылаются на main.py, который либо был
переписан под другую задачу, либо никогда не соответствовал описанию. Этот
скрипт — самостоятельная, отдельная от main.py реализация именно сценария
Статьи 1, ничего в main.py не меняет.

Методология (буквально по плану, раздел 5, и README "Известные ограничения"):
  - Датасет: datasets/{glioma,meningioma,pituitary}_centered/*.png — по 200
    изображений на класс, 256x256, уже отцентрированные (вход NB3).
  - Векторизация: features.vectorization, method="horizontal" (65536-мерный
    вектор построчной развёртки).
  - Три сценария сравнения:
      1. 70/30 на полном наборе (600 изображений) — sklearn train_test_split,
         stratify=y, random_state=RANDOM_SEED. Честный независимый случайный
         сплит.
      2. 50/10 на класс (holdout) — для каждого класса независимо: список
         индексов перемешивается np.random.default_rng(RANDOM_SEED), первые
         50 -> train, следующие 10 -> test. Явно и полностью специфицировано
         здесь, ЧЕГО НЕ БЫЛО в исходном README (там числа 80% и 61% приведены
         без описания точного способа отбора holdout) — при повторном запуске
         этого скрипта результат воспроизводится побитово. Тоже честный сплит.
      3. correlated_pairs — evaluation.paired_correlation_split(random_state=
         RANDOM_SEED) — «метод коррелированного датасета» (theory/code/
         Вариант_2_1.docx, "Первый этап"): для каждого класса независимо
         ищутся пары наиболее скоррелированных (по Пирсону) изображений, один
         из пары — в train, другой — в test. ⚠️ ЭТО НЕ ЧЕСТНЫЙ СПЛИТ — тестовая
         выборка НАМЕРЕННО подобрана похожей на обучающую (сам черновик это
         признаёт, см. README "Сплит через попарное разнесение"), и accuracy
         этого сценария систематически завышена относительно двух честных
         сценариев выше — см. experiments/experiment_data_leakage_demo.py и
         "Таблицу 8" experiment_draft_variant2.py, где для похожей конфигурации
         этот же эффект измерен напрямую (0.71-0.84 на попарном разнесении vs
         0.32-0.34 на честном сплите для однопространственной схемы). Включён
         сюда НЕ как альтернативный baseline, а как явная демонстрация цены
         этой методологии — числа этого сценария нельзя цитировать в разделе
         "Результаты" как характеристику качества метода без этой оговорки.
  - Классификатор: SubspaceConjugacyClassifier(n_subclasses=8) — параметры по
    умолчанию библиотеки, freeze_basis_at не переопределяется (=2 по умолчанию),
    одинаковый для всех трёх сценариев.
  - Метрики: evaluate_classifier (accuracy, per_class_accuracy, confusion_matrix).

Проверено (см. историю анализа в этой сессии): 70/30-сценарий воспроизводит
заявленные в README числа практически точно (accuracy 61.11% против
заявленных 61%, per-class recall meningioma совпадает до третьего знака —
0.150). 50/10-сценарий с методологией, зафиксированной выше, даёт 76.67%
(README заявлял 80% без указания точного способа отбора holdout — расхождение
объясняется недокументированной методологией оригинала, не ошибкой библиотеки).

Артефакты (директория artifacts/, в .gitignore):
  - article1_report.json       — точные числа обоих сценариев + все гиперпараметры
  - article1_results.md        — таблица в формате, готовом для вставки в текст статьи
  - article1_model_70_30.pkl   — обученная модель (70/30-сценарий, baseline для
                                  сравнения во всех последующих 5 статьях плана)
  - article1_splits/{70_30,50_10}/{train,test}/{class}/*.png — РЕАЛЬНО
    использованные изображения, по сценарию (не только имена файлов в JSON) —
    чтобы руководитель/рецензент мог открыть папку и увидеть точно те же
    файлы, что видела модель (тот же принцип, что и в
    experiments/experiment_draft_variant2.py::save_split_snapshot).

Запуск:
    python experiments/reproduce_article1.py
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    paired_correlation_split,
    save_model,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

logging.basicConfig(level=logging.WARNING, force=True)

DATASET_ROOT = Path("datasets")
CLASSES = ["glioma", "meningioma", "pituitary"]
RANDOM_SEED = 42
N_SUBCLASSES = 8  # план, раздел 5: "параметры по умолчанию библиотеки"

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "article1_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "article1_results.md"
MODEL_PATH = ARTIFACTS_DIR / "article1_model_70_30.pkl"
SPLITS_ROOT = ARTIFACTS_DIR / "article1_splits"


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загружает и векторизует datasets/{class}_centered/ целиком (200 на класс).

    Возвращает X, y и paths (пути к исходным PNG, выровненные построчно с X/y)
    — paths нужны, чтобы после сплита физически скопировать РЕАЛЬНО отобранные
    файлы в артефакты (см. save_split_snapshot).
    """
    X_list, y_list, path_list = [], [], []
    for cls in CLASSES:
        paths = sorted((DATASET_ROOT / f"{cls}_centered").glob("*.png"))
        if len(paths) != 200:
            raise FileNotFoundError(
                f"{cls}: найдено {len(paths)} изображений в "
                f"{DATASET_ROOT / f'{cls}_centered'}, ожидалось 200 "
                "(план Статьи 1 фиксирует именно этот объём)."
            )
        X_list.append(load_and_vectorize_batch(paths, method="horizontal"))
        y_list.append(np.full(len(paths), cls))
        path_list.extend(paths)
    return np.vstack(X_list), np.concatenate(y_list), np.array(path_list, dtype=object)


def split_70_30(
    X: np.ndarray, y: np.ndarray, paths: np.ndarray
) -> Dict[str, np.ndarray]:
    """70/30, честный случайный стратифицированный сплит на полном наборе (600)."""
    idx = np.arange(X.shape[0])
    idx_train, idx_test = train_test_split(
        idx, test_size=0.3, stratify=y, random_state=RANDOM_SEED
    )
    return {
        "X_train": X[idx_train], "y_train": y[idx_train], "paths_train": paths[idx_train],
        "X_test": X[idx_test], "y_test": y[idx_test], "paths_test": paths[idx_test],
    }


def split_50_10_per_class(
    X: np.ndarray, y: np.ndarray, paths: np.ndarray
) -> Dict[str, np.ndarray]:
    """50 train + 10 test НА КАЖДЫЙ класс — методология зафиксирована явно
    (см. docstring модуля): np.random.default_rng(RANDOM_SEED), независимая
    перестановка индексов каждого класса, затем срез [:50] / [50:60].
    """
    rng = np.random.default_rng(RANDOM_SEED)
    idx_train_all: List[int] = []
    idx_test_all: List[int] = []
    for cls in CLASSES:
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        idx_train_all.extend(cls_idx[:50].tolist())
        idx_test_all.extend(cls_idx[50:60].tolist())
    idx_train = np.array(idx_train_all)
    idx_test = np.array(idx_test_all)
    return {
        "X_train": X[idx_train], "y_train": y[idx_train], "paths_train": paths[idx_train],
        "X_test": X[idx_test], "y_test": y[idx_test], "paths_test": paths[idx_test],
    }


def split_correlated_pairs(
    X: np.ndarray, y: np.ndarray, paths: np.ndarray
) -> Dict[str, np.ndarray]:
    """«Метод коррелированного датасета» (theory/code/Вариант_2_1.docx,
    "Первый этап") — evaluation.paired_correlation_split. ⚠️ НЕ честный сплит,
    см. предупреждение в docstring модуля — включён для явного сравнения с
    двумя сценариями выше, не как альтернативный baseline.
    """
    result = paired_correlation_split(X, y, random_state=RANDOM_SEED)
    return {
        "X_train": result["X_train"], "y_train": result["y_train"],
        "paths_train": paths[result["train_indices"]],
        "X_test": result["X_test"], "y_test": result["y_test"],
        "paths_test": paths[result["test_indices"]],
    }


def save_split_snapshot(scenario_name: str, split: Dict[str, np.ndarray]) -> None:
    """Копирует РЕАЛЬНО использованные train/test изображения в
    artifacts/article1_splits/<scenario>/{train,test}/<class>/ — до fit(),
    аналог save_split_snapshot из experiment_draft_variant2.py."""
    scenario_dir = SPLITS_ROOT / scenario_name
    for subset in ("train", "test"):
        paths = split[f"paths_{subset}"]
        labels = split[f"y_{subset}"]
        for p, cls in zip(paths, labels):
            cls_dir = scenario_dir / subset / str(cls)
            cls_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, cls_dir / Path(p).name)


def run_scenario(name: str, split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    print(f"\n=== Сценарий: {name} ===")
    print(f"   train={split['X_train'].shape[0]}, test={split['X_test'].shape[0]}")

    print("   Сохранение снэпшота train/test...")
    save_split_snapshot(name, split)

    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(split["X_train"], split["y_train"])
    fit_seconds = time.perf_counter() - t0
    print(f"   fit() заняло {fit_seconds:.1f}s")

    y_pred = clf.predict(split["X_test"])
    report = evaluate_classifier(split["y_test"], y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    print(f"   accuracy={report['accuracy']:.4f}")
    print(f"   per_class_accuracy={per_class}")

    return {
        "n_train": int(split["X_train"].shape[0]),
        "n_test": int(split["X_test"].shape[0]),
        "fit_seconds": fit_seconds,
        "accuracy": report["accuracy"],
        "per_class_accuracy": per_class,
        "confusion_matrix": report["confusion_matrix"].tolist(),
        "labels": [str(l) for l in report["labels"]],
        "_classifier": clf,
    }


def save_markdown_report(results: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# Статья 1 — результаты эксперимента (воспроизведение)",
        "",
        "Сгенерировано `experiments/reproduce_article1.py`. Датасет: "
        "`datasets/{glioma,meningioma,pituitary}_centered/` (200 изображений "
        "на класс, 256x256). Классификатор: `SubspaceConjugacyClassifier"
        f"(n_subclasses={N_SUBCLASSES})` (параметры по умолчанию библиотеки).",
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
    lines += [
        "",
        "**70/30** и **50/10** — независимые случайные сплиты (честные, не "
        "`paired_correlation_split`): 70/30 — `sklearn.model_selection."
        f"train_test_split(stratify=y, random_state={RANDOM_SEED})` на полном "
        "наборе (600 изображений); 50/10 на класс — "
        f"`np.random.default_rng({RANDOM_SEED})`, независимая перестановка "
        "индексов каждого класса, срез [:50]/[50:60] — методология зафиксирована "
        "явно (в отличие от исходного README, где точный способ отбора holdout "
        "не был специфицирован).",
        "",
        "**correlated_pairs** — `evaluation.paired_correlation_split` "
        "(«метод коррелированного датасета», theory/code/Вариант_2_1.docx, "
        "«Первый этап»): для каждого класса независимо ищутся пары наиболее "
        "коррелированных изображений, один из пары уходит в train, другой в "
        "test. ⚠️ **НЕ включать это число в раздел «Результаты» как "
        "характеристику качества метода без оговорки** — тестовая выборка "
        "здесь намеренно подобрана похожей на обучающую (сам черновик это "
        "признаёт), поэтому accuracy систематически завышена относительно "
        "70/30 и 50/10 выше. Приведено намеренно, для прямого количественного "
        "сравнения цены этой методологии — тот же эффект измерен независимо "
        "в `experiments/experiment_data_leakage_demo.py` и «Таблице 8» "
        "`experiment_draft_variant2.py` (там разрыв — от 0.37 до 0.50 п.п. "
        "accuracy для однопространственной схемы; здесь — каноническая "
        "8-подклассовая схема, разрыв может отличаться по величине).",
        "",
        f"Реально использованные train/test изображения — `artifacts/article1_splits/"
        "{70_30,50_10,correlated_pairs}/{train,test}/<class>/*.png` (копии "
        "файлов, не только индексы) — для точного повторения на другой машине.",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown-отчёт: {REPORT_MD_PATH}")


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 1: воспроизведение baseline-эксперимента ===")

    print("\nЗагрузка и векторизация datasets/*_centered...")
    X, y, paths = load_data()
    print(f"X.shape={X.shape}")

    shutil.rmtree(SPLITS_ROOT, ignore_errors=True)

    scenarios = {
        "70_30": split_70_30(X, y, paths),
        "50_10": split_50_10_per_class(X, y, paths),
        "correlated_pairs": split_correlated_pairs(X, y, paths),
    }

    results = {name: run_scenario(name, split) for name, split in scenarios.items()}

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nСохранение модели baseline (70/30) в {MODEL_PATH}...")
    save_model(results["70_30"]["_classifier"], MODEL_PATH)

    report_for_json = {
        name: {k: v for k, v in r.items() if k != "_classifier"}
        for name, r in results.items()
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_subclasses": N_SUBCLASSES,
                "random_seed": RANDOM_SEED,
                "dataset_size_per_class": 200,
                "results": report_for_json,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"JSON-отчёт: {REPORT_JSON_PATH}")

    save_markdown_report(report_for_json)

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
