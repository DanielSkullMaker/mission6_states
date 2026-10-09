"""Методология оценки из статьи Korshikov & Fursov (Table I/II/III, Fig. 5/7).

Источник: theory/VI_Korshikov_VA_Fursov_Pathology_Recognition_Based_on_
Conjugacy.docx, раздел "Analysis of Experimental Results".

Статья описывает методологию оценки, ОТЛИЧНУЮ от k-fold кросс-валидации
(model_selection/search.py): для набора данных фиксированного размера
выборка один раз делится 80/20 (train/test — "Each set was divided into
samples: 80% training sample and 20% test sample"), затем для СЕТКИ
n_subclasses обучается классификатор и выбирается n_subclasses с
максимальной средней по классам точностью на test — "For each set, the
number of subspaces at which the highest accuracy was achieved was
identified" (Fig. 5/7). Table I/II статьи сравнивают точность при
filter_low_informativeness=False/True (эксперименты 2 и 3 статьи, находка
№5 из README).

⚠️ ЗАЩИТА ОТ УТЕЧКИ (мотивация — theory/code/Run_200.m):
  `Recogn_VSPU_S1(M_test_3, M_train_3)` в оригинальном MATLAB-коде тестирует
  классификацию на ТОМ ЖЕ пуле изображений (M_test_3), из которого
  `DATA_100.m` отобрал опорные векторы (M_train_3) — часть "тестовых"
  объектов буквально являются столбцами базиса, против которого их же
  классифицируют. paper_train_test_split() ниже гарантирует непересечение
  train/test индексов явной проверкой (а не только полагается на
  корректность sklearn.train_test_split) именно из-за этого прецедента.

⚠️ ОСТАЮЩАЯСЯ МЕТОДОЛОГИЧЕСКАЯ ОГОВОРКА (уже самой статьи, не библиотеки):
  find_best_n_subclasses() выбирает n_subclasses ПО ТОЙ ЖЕ выборке
  (X_eval/y_eval), на которой затем репортится итоговая точность — если
  вызывающий код использует один и тот же test-набор и для подбора, и для
  финального Table I/II отчёта. Это МЯГКАЯ форма переобучения под тест
  (model selection leakage) — НЕ то же самое, что утечка Run_200.m (объекты
  test здесь НИКОГДА не участвуют в построении подпространств — они
  участвуют только в SELECTING гиперпараметра n_subclasses), но тоже
  завышает репортируемую точность относительно строго независимой оценки.
  Для строгой (unbiased) оценки: передайте ОТДЕЛЬНЫЕ X_val/y_val в
  find_best_n_subclasses() для подбора n_subclasses, а итоговую точность
  измерьте ОДИН РАЗ на третьем, ни разу не использованном X_test/y_test —
  см. Examples ниже.

Источник №2 (paired_correlation_split): theory/code/Вариант_2_1.docx —
БОЛЕЕ РАННИЙ черновик статьи (не опубликованная статья выше), "Первый этап"
формирования обучающих подпространств, с пометками научного руководителя
прямо в тексте. Дословный совет руководителя (реконструкция по черновику):
"Подбираем для каждого из изображений класса наиболее коррелированное по
формуле (1). Из этой пары одно изображение закидываем в обучающую, другое
в тестовую выборку." ⚠️ Сам черновик ОТКРЫТО признаёт цену этой методологии
(дословная цитата): "...в тестовой выборке будут все изображения, похожие
на изображения, из которых конструируется образцовое (обучающее) множество,
ошибки различных методов распознавания будут отражать преимущественно
влияние метода формирования этого множества." Иными словами: test
специально подбирается похожим на train — это НЕ независимая held-out
оценка (в отличие от paper_train_test_split выше), а отдельная, другая,
задокументированная в источниках методология со своей известной ценой.
Используйте paired_correlation_split() только когда нужно ВОСПРОИЗВЕСТИ
методологию черновика (например, Table 1 черновика, accuracy 0.905-0.993)
— не как замену честного held-out теста.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from subspace_conjugacy.evaluation.metrics import evaluate_classifier
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

logger = logging.getLogger(__name__)

# Табличная сетка не задана в статье явно (Fig. 5/7 — график, не список
# значений) — разумный диапазон вокруг дефолта библиотеки (n_subclasses=8).
DEFAULT_N_SUBCLASSES_GRID: Tuple[int, ...] = (4, 6, 8, 10, 12, 16)


def paper_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.8,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Стратифицированный 80/20 сплит (статья: "80% training / 20% test").

    Parameters
    ----------
    X : np.ndarray
        Матрица признаков (M, N).
    y : np.ndarray
        Метки классов (M,).
    train_fraction : float, default=0.8
        Доля обучающей выборки (статья: 0.8).
    random_state : int, optional
        Seed для воспроизводимости разбиения.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Разбиение — стратифицированное по классам (sklearn.train_test_split,
        stratify=y), с явной гарантией непересечения индексов (см. docstring
        модуля — мотивация: theory/code/Run_200.m).

    Raises
    ------
    RuntimeError
        Если обнаружено пересечение индексов train/test (не должно
        происходить при корректной работе sklearn — защитная проверка).
    """
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        train_size=train_fraction,
        stratify=y,
        random_state=random_state,
    )

    overlap = set(idx_train.tolist()) & set(idx_test.tolist())
    if overlap:
        logger.error(
            "paper_train_test_split: обнаружено пересечение train/test "
            "индексов (%d объектов).", len(overlap),
        )
        raise RuntimeError(
            f"paper_train_test_split: {len(overlap)} индексов присутствуют "
            "и в train, и в test одновременно — это баг (аналогичный "
            "структурной утечке в theory/code/Run_200.m), а не ожидаемое "
            "поведение."
        )

    logger.info(
        "paper_train_test_split: train=%d, test=%d, train_fraction=%.2f, "
        "пересечений индексов нет (проверено).",
        len(y_train), len(y_test), train_fraction,
    )
    return X_train, X_test, y_train, y_test


def find_best_n_subclasses(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_subclasses_grid: Sequence[int] = DEFAULT_N_SUBCLASSES_GRID,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Перебирает n_subclasses_grid, выбирает лучший по mean per-class accuracy.

    Реализует Fig. 5/7 статьи: "For each set, the number of subspaces at
    which the highest accuracy was achieved was identified" — столбец
    "mean" в Table I/II/III статьи и есть mean_per_class_accuracy здесь.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Обучающая выборка (например, из paper_train_test_split()).
    X_eval, y_eval : np.ndarray
        Выборка для оценки и подбора n_subclasses. ⚠ См. предупреждение в
        docstring модуля про model selection leakage, если тот же набор
        затем используется как финальный "test" в отчёте.
    n_subclasses_grid : Sequence[int], default=(4, 6, 8, 10, 12, 16)
        Сетка значений n_subclasses для перебора.
    classifier_kwargs : dict, optional
        Дополнительные параметры конструктора SubspaceConjugacyClassifier
        (например, filter_low_informativeness=True для Table II статьи).

    Returns
    -------
    result : dict
        - "best_n_subclasses": int
        - "best_mean_per_class_accuracy": float
        - "best_estimator_": обученный SubspaceConjugacyClassifier
          (лучший по mean_per_class_accuracy)
        - "all_results": list[dict] — по одной записи на n_subclasses
          ({"n_subclasses", "accuracy", "per_class_accuracy",
          "mean_per_class_accuracy"})

    Examples
    --------
    Строгая (unbiased) оценка — раздельные val/test:

    >>> X_train, X_rest, y_train, y_rest = paper_train_test_split(X, y, 0.8)
    >>> X_val, X_test, y_val, y_test = paper_train_test_split(X_rest, y_rest, 0.5)
    >>> result = find_best_n_subclasses(X_train, y_train, X_val, y_val)
    >>> final_accuracy = evaluate_classifier(
    ...     y_test, result["best_estimator_"].predict(X_test)
    ... )["accuracy"]  # единственное измерение на неиспользованном test
    """
    kwargs = dict(classifier_kwargs or {})
    all_results = []
    best = None

    logger.info(
        "find_best_n_subclasses: старт, grid=%s, n_train=%d, n_eval=%d, "
        "classifier_kwargs=%s.", list(n_subclasses_grid), len(y_train), len(y_eval), kwargs,
    )

    for n_sub in n_subclasses_grid:
        clf = SubspaceConjugacyClassifier(n_subclasses=n_sub, **kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_eval)
        report = evaluate_classifier(y_eval, y_pred)
        mean_accuracy = float(np.mean(list(report["per_class_accuracy"].values())))

        point = {
            "n_subclasses": n_sub,
            "accuracy": report["accuracy"],
            "per_class_accuracy": report["per_class_accuracy"],
            "mean_per_class_accuracy": mean_accuracy,
        }
        all_results.append(point)
        logger.info(
            "find_best_n_subclasses: n_subclasses=%d -> mean_per_class_accuracy=%.4f "
            "(accuracy=%.4f).", n_sub, mean_accuracy, report["accuracy"],
        )

        if best is None or mean_accuracy > best["mean_per_class_accuracy"]:
            best = {**point, "estimator": clf}

    logger.info(
        "find_best_n_subclasses: готово, best_n_subclasses=%d "
        "(mean_per_class_accuracy=%.4f).",
        best["n_subclasses"], best["mean_per_class_accuracy"],
    )

    return {
        "best_n_subclasses": best["n_subclasses"],
        "best_mean_per_class_accuracy": best["mean_per_class_accuracy"],
        "best_estimator_": best["estimator"],
        "all_results": all_results,
    }


def compare_informativeness_filter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_subclasses_grid: Sequence[int] = DEFAULT_N_SUBCLASSES_GRID,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    informativeness_min_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Table I (без фильтра) vs Table II (с фильтром) — эксперименты 2 и 3 статьи.

    Статья, 3-й эксперимент: "images with the number of white pixels less
    than 50% of the average number of white pixels in the images of the set
    are cut off" (LowInformativenessFilter, находка №5 README).

    Parameters
    ----------
    X_train, y_train, X_eval, y_eval : np.ndarray
        См. find_best_n_subclasses.
    n_subclasses_grid : Sequence[int], default=(4, 6, 8, 10, 12, 16)
        См. find_best_n_subclasses.
    classifier_kwargs : dict, optional
        Общие для обеих таблиц параметры (кроме filter_low_informativeness/
        informativeness_min_fraction, которые фиксируются этой функцией).
    informativeness_min_fraction : float, default=0.5
        Порог фильтра (статья: 50%).

    Returns
    -------
    comparison : dict
        {"table_I_baseline": <find_best_n_subclasses result>,
         "table_II_filtered": <find_best_n_subclasses result>}
    """
    kwargs = dict(classifier_kwargs or {})
    if "filter_low_informativeness" in kwargs or "informativeness_min_fraction" in kwargs:
        raise ValueError(
            "classifier_kwargs не должен содержать filter_low_informativeness/"
            "informativeness_min_fraction — эта функция задаёт их сама для "
            "каждой из двух таблиц."
        )

    logger.info("compare_informativeness_filter: старт (Table I vs Table II статьи).")

    table_I = find_best_n_subclasses(
        X_train, y_train, X_eval, y_eval, n_subclasses_grid,
        classifier_kwargs={**kwargs, "filter_low_informativeness": False},
    )
    table_II = find_best_n_subclasses(
        X_train, y_train, X_eval, y_eval, n_subclasses_grid,
        classifier_kwargs={
            **kwargs,
            "filter_low_informativeness": True,
            "informativeness_min_fraction": informativeness_min_fraction,
        },
    )

    logger.info(
        "compare_informativeness_filter: Table I mean=%.4f (n_subclasses=%d) vs "
        "Table II mean=%.4f (n_subclasses=%d).",
        table_I["best_mean_per_class_accuracy"], table_I["best_n_subclasses"],
        table_II["best_mean_per_class_accuracy"], table_II["best_n_subclasses"],
    )

    return {"table_I_baseline": table_I, "table_II_filtered": table_II}


def paired_correlation_split(
    X: np.ndarray,
    y: np.ndarray,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Train/test сплит через попарное разнесение наиболее скоррелированных
    изображений (theory/code/Вариант_2_1.docx, "Первый этап").

    ⚠️ См. ОБЯЗАТЕЛЬНОЕ предупреждение в docstring модуля — test здесь
    СПЕЦИАЛЬНО подобран похожим на train (в отличие от paper_train_test_split,
    независимого случайного сплита). Авторы черновика сами признают: точность,
    измеренная этим способом, отражает метод формирования выборки, а не
    обобщающую способность на независимые изображения.

    Алгоритм (по классам независимо, дословно по черновику):
      1. Берём произвольный ещё не распределённый вектор класса.
      2. Находим среди ещё не распределённых вектор с максимальным
         нормированным коэффициентом корреляции (формула 1 — Пирсон,
         как в CorrelationCoreSelector/DATA_100.m).
      3. Один из пары — в train, другой — в test (какой именно — по
         тексту черновика "любом", т.е. не специфицировано; здесь решается
         монеткой через random_state, чтобы не вносить систематического
         смещения от порядка индексов).
      4. Повторяем на оставшихся нераспределённых векторах класса до
         исчерпания. Нечётный "хвост" (если размер класса нечётный) —
         единственный оставшийся без пары вектор — идёт в train.

    Parameters
    ----------
    X : np.ndarray
        Матрица признаков (M, N).
    y : np.ndarray
        Метки классов (M,).
    random_state : int, optional
        Seed для порядка обхода "произвольных начальных векторов" и монетки
        train/test внутри пары.

    Returns
    -------
    result : dict
        - "X_train", "y_train", "X_test", "y_test": np.ndarray
        - "train_indices", "test_indices": np.ndarray (индексы в исходном X)
        - "pairs": list[tuple[int, int]] — (train_index, test_index) для
          каждой найденной пары (в исходных индексах X); "осиротевший"
          нечётный вектор в пары не входит.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    rng = np.random.default_rng(random_state)

    train_indices: List[int] = []
    test_indices: List[int] = []
    pairs: List[Tuple[int, int]] = []

    for cls in np.unique(y_arr):
        cls_indices = np.where(y_arr == cls)[0]
        n_cls = len(cls_indices)
        X_cls = X_arr[cls_indices]

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X_cls)
        corr = np.nan_to_num(corr, nan=-np.inf)
        np.fill_diagonal(corr, -np.inf)

        visit_order = rng.permutation(n_cls)
        unassigned = set(range(n_cls))

        for local_i in visit_order:
            if local_i not in unassigned:
                continue
            unassigned.discard(int(local_i))

            if not unassigned:
                # Нечётный остаток без пары — некуда его "разнести",
                # оставляем в train (см. docstring).
                train_indices.append(int(cls_indices[local_i]))
                logger.debug(
                    "paired_correlation_split: класс '%s' — нечётный остаток "
                    "(локальный индекс %d) без пары, отправлен в train.",
                    cls, local_i,
                )
                break

            candidates = np.fromiter(unassigned, dtype=int)
            local_j = int(candidates[np.argmax(corr[local_i, candidates])])
            unassigned.discard(local_j)

            global_i = int(cls_indices[local_i])
            global_j = int(cls_indices[local_j])

            if rng.random() < 0.5:
                train_idx, test_idx = global_i, global_j
            else:
                train_idx, test_idx = global_j, global_i

            train_indices.append(train_idx)
            test_indices.append(test_idx)
            pairs.append((train_idx, test_idx))

        logger.info(
            "paired_correlation_split: класс '%s' — %d векторов -> %d пар "
            "(train=%d, test=%d).",
            cls, n_cls, n_cls // 2, n_cls - n_cls // 2, n_cls // 2,
        )

    train_indices_arr = np.array(sorted(train_indices), dtype=int)
    test_indices_arr = np.array(sorted(test_indices), dtype=int)

    overlap = set(train_indices_arr.tolist()) & set(test_indices_arr.tolist())
    if overlap:
        # Не должно происходить при корректной работе алгоритма (каждый
        # индекс распределяется ровно один раз) — защитная проверка,
        # симметричная paper_train_test_split.
        logger.error(
            "paired_correlation_split: обнаружено пересечение train/test "
            "индексов (%d объектов).", len(overlap),
        )
        raise RuntimeError(
            f"paired_correlation_split: {len(overlap)} индексов присутствуют "
            "и в train, и в test одновременно — это баг в реализации."
        )

    logger.info(
        "paired_correlation_split: готово, train=%d, test=%d, %d пар всего.",
        len(train_indices_arr), len(test_indices_arr), len(pairs),
    )

    return {
        "X_train": X_arr[train_indices_arr],
        "y_train": y_arr[train_indices_arr],
        "X_test": X_arr[test_indices_arr],
        "y_test": y_arr[test_indices_arr],
        "train_indices": train_indices_arr,
        "test_indices": test_indices_arr,
        "pairs": pairs,
    }
