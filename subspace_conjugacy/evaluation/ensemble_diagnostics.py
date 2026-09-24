"""Диагностика согласованности ошибок нескольких классификаторов (статья 2,
"Симбиоз классификаторов", theory/article_plans/02_simbioz_arkhitektur.txt,
Задача 1 методологии).

Предпосылка ансамблирования: если модели ошибаются НЕКОРРЕЛИРОВАННО (на
разных объектах), голосование/стекинг способно дать точность выше лучшей
отдельной модели. Если модели ошибаются на ОДНИХ И ТЕХ ЖЕ объектах (сильно
коррелированные ошибки — например, все три путают meningioma по одной и
той же причине: низкое качество исходных снимков), ансамбль выигрыша не
даст — это тоже честный, ожидаемый результат (раздел 8 плана статьи,
"риски и ограничения"), а не повод считать метрику сломанной.

compute_ensemble_diagnostics() — единственная точка входа этого модуля:
принимает предсказания уже обученных (в т.ч. вне библиотеки — например,
CNN на PyTorch) моделей на одном и том же тестовом наборе и считает:
  - точность каждой модели по отдельности;
  - долю объектов, где ОШИБАЕТСЯ РОВНО ОДНА модель — "потенциал ансамбля"
    (эти ошибки в принципе исправимы голосованием/переключением, раз
    большинство моделей на этих объектах право);
  - долю объектов, где ошибаются ВСЕ модели — "жёсткий потолок" (никакое
    комбинирование предсказаний уже обученных моделей не может исправить
    объект, который в принципе никто не распознал верно);
  - accuracy "оракула" — доля объектов, где ПРАВА хотя бы одна модель;
    это теоретический максимум accuracy идеального (всезнающего) способа
    выбора между моделями объект-за-объектом — практический ансамбль
    (раздел models/ensemble.py) может к нему приблизиться, но не
    превзойти;
  - попарное согласие моделей и попарную корреляцию индикаторов ошибок
    (высокая корреляция = модели ошибаются вместе = меньше пользы от
    их объединения).
"""

import logging
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def compute_ensemble_diagnostics(
    y_true: Sequence[Any], predictions_by_model: Mapping[str, Sequence[Any]]
) -> Dict[str, Any]:
    """Считает статистику согласованности ошибок нескольких моделей.

    Parameters
    ----------
    y_true : Sequence[Any]
        Истинные метки классов тестовой выборки (M,).
    predictions_by_model : Mapping[str, Sequence[Any]]
        {имя_модели: предсказанные метки (M,)} — минимум 2 модели,
        предсказания ВСЕХ моделей должны быть получены на ОДНОМ И ТОМ ЖЕ
        наборе объектов в ОДНОМ И ТОМ ЖЕ порядке, что и y_true (иначе
        результат бессмыслен — это ответственность вызывающего кода, как
        и в остальной библиотеке — например, prepare_centered_split
        строит единый test-сплит для всех подходов именно по этой причине).

    Returns
    -------
    diagnostics : Dict[str, Any]
        Словарь с ключами:
        - "n_samples": число объектов.
        - "model_names": список имён моделей в порядке predictions_by_model.
        - "accuracy_by_model": Dict[str, float] — accuracy каждой модели.
        - "n_models_wrong_histogram": Dict[int, int] — распределение
          объектов по числу моделей, ошибившихся на них (0..len(models)).
        - "fraction_exactly_one_wrong": float — доля объектов, где
          ошибается РОВНО ОДНА модель (потенциал ансамбля большинством).
        - "fraction_all_wrong": float — доля объектов, где ошибаются ВСЕ
          модели (жёсткий потолок, не исправимый комбинированием).
        - "oracle_accuracy": float — доля объектов, где права ХОТЯ БЫ ОДНА
          модель (1 - fraction_all_wrong); верхняя граница accuracy для
          любого способа выбора между уже обученными моделями.
        - "pairwise_agreement": Dict[Tuple[str, str], float] — доля
          объектов, где два предсказания СОВПАДАЮТ (независимо от
          правильности) для каждой пары моделей.
        - "pairwise_error_correlation": Dict[Tuple[str, str], float] —
          коэффициент корреляции Пирсона между бинарными индикаторами
          ошибки (1 = ошибка) каждой пары моделей; NaN, если у одной из
          моделей в паре ошибок нет вовсе (нулевая дисперсия индикатора).

    Raises
    ------
    ValueError
        Если передано меньше двух моделей или размеры не совпадают.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array(["a", "a", "b", "b"])
    >>> preds = {
    ...     "subspace": np.array(["a", "b", "b", "b"]),
    ...     "cnn": np.array(["a", "a", "a", "b"]),
    ... }
    >>> diag = compute_ensemble_diagnostics(y_true, preds)
    >>> diag["oracle_accuracy"]
    1.0
    """
    y_true_arr = np.asarray(y_true)
    model_names = list(predictions_by_model.keys())

    if len(model_names) < 2:
        logger.error(
            "compute_ensemble_diagnostics: передано %d моделей, требуется минимум 2.",
            len(model_names),
        )
        raise ValueError(
            f"Для диагностики ансамбля нужно минимум 2 модели, получено {len(model_names)}."
        )

    pred_arrays = {}
    for name in model_names:
        arr = np.asarray(predictions_by_model[name])
        if arr.shape[0] != y_true_arr.shape[0]:
            logger.error(
                "compute_ensemble_diagnostics: модель '%s' имеет %d предсказаний, "
                "ожидалось %d (по y_true).", name, arr.shape[0], y_true_arr.shape[0],
            )
            raise ValueError(
                f"Модель '{name}': {arr.shape[0]} предсказаний, ожидалось "
                f"{y_true_arr.shape[0]} (по числу объектов в y_true)."
            )
        pred_arrays[name] = arr

    n_samples = y_true_arr.shape[0]
    n_models = len(model_names)

    # error_matrix[i, j] = True, если модель i ошиблась на объекте j
    error_matrix = np.stack(
        [pred_arrays[name] != y_true_arr for name in model_names], axis=0
    )
    n_models_wrong = error_matrix.sum(axis=0)  # (n_samples,)

    accuracy_by_model = {
        name: float(np.mean(pred_arrays[name] == y_true_arr)) for name in model_names
    }

    histogram = {
        int(k): int(v)
        for k, v in zip(*np.unique(n_models_wrong, return_counts=True))
    }
    # Заполняем нулями отсутствующие значения 0..n_models — читателю отчёта
    # удобнее видеть полную гистограмму, а не только встретившиеся значения.
    full_histogram = {k: histogram.get(k, 0) for k in range(n_models + 1)}

    fraction_exactly_one_wrong = float(np.mean(n_models_wrong == 1))
    fraction_all_wrong = float(np.mean(n_models_wrong == n_models))
    oracle_accuracy = float(np.mean(n_models_wrong < n_models))

    pairwise_agreement: Dict[Any, float] = {}
    pairwise_error_correlation: Dict[Any, float] = {}
    for name_a, name_b in combinations(model_names, 2):
        agree = float(np.mean(pred_arrays[name_a] == pred_arrays[name_b]))
        pairwise_agreement[(name_a, name_b)] = agree

        err_a = (pred_arrays[name_a] != y_true_arr).astype(np.float64)
        err_b = (pred_arrays[name_b] != y_true_arr).astype(np.float64)
        if np.std(err_a) == 0.0 or np.std(err_b) == 0.0:
            # Одна из моделей ни разу не ошиблась (или ошиблась всегда) —
            # корреляция не определена (деление на нулевое std);
            # np.nan честнее, чем произвольная заглушка вроде 0.
            corr = float("nan")
            logger.debug(
                "compute_ensemble_diagnostics: корреляция ошибок (%s, %s) не "
                "определена — у одной из моделей нулевая дисперсия индикатора ошибки.",
                name_a, name_b,
            )
        else:
            corr = float(np.corrcoef(err_a, err_b)[0, 1])
        pairwise_error_correlation[(name_a, name_b)] = corr

    diagnostics = {
        "n_samples": n_samples,
        "model_names": model_names,
        "accuracy_by_model": accuracy_by_model,
        "n_models_wrong_histogram": full_histogram,
        "fraction_exactly_one_wrong": fraction_exactly_one_wrong,
        "fraction_all_wrong": fraction_all_wrong,
        "oracle_accuracy": oracle_accuracy,
        "pairwise_agreement": pairwise_agreement,
        "pairwise_error_correlation": pairwise_error_correlation,
    }
    logger.info(
        "compute_ensemble_diagnostics: %d объектов, %d моделей, "
        "accuracy_by_model=%s, oracle_accuracy=%.4f, "
        "fraction_exactly_one_wrong=%.4f, fraction_all_wrong=%.4f.",
        n_samples, n_models, accuracy_by_model, oracle_accuracy,
        fraction_exactly_one_wrong, fraction_all_wrong,
    )
    return diagnostics
