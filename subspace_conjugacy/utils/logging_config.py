"""Централизованная настройка логирования библиотеки.

Библиотека сама НЕ настраивает handlers на своих логгерах при импорте
(стандартная практика для библиотек — см. `logging.getLogger(__name__)` +
`NullHandler` в ``subspace_conjugacy/__init__.py``), чтобы не мешать
логированию приложения-потребителя. ``configure_logging`` — необязательный
помощник для быстрого включения читаемого вывода в консоль (в ноутбуке,
скрипте или интерактивной сессии), а не обязательный шаг для использования
библиотеки.

Иерархия логгеров: каждый модуль библиотеки использует
``logging.getLogger(__name__)``, поэтому все логгеры лежат под общим
префиксом ``subspace_conjugacy.*`` (``subspace_conjugacy.algorithms.global_pair``,
``subspace_conjugacy.io.vectors`` и т.д.) — уровень/handlers можно менять
как для всей библиотеки сразу (``configure_logging``), так и точечно для
одного подмодуля через обычный ``logging.getLogger(...).setLevel(...)``.
"""

import logging
from typing import Optional

_LIBRARY_LOGGER_NAME = "subspace_conjugacy"

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> logging.Logger:
    """Включает читаемый вывод логов ``subspace_conjugacy.*`` в консоль.

    Удобно вызвать один раз в начале ноутбука/скрипта, чтобы увидеть
    подробный трейс работы кластеризации, классификатора, IO и препроцессинга
    без ручной настройки ``logging.basicConfig``.

    Parameters
    ----------
    level : int, default=logging.INFO
        Минимальный уровень сообщений. Используйте ``logging.DEBUG`` для
        детального трейса внутренних шагов алгоритмов (по одному вектору за
        итерацию B.2, промежуточные центры A.2-A.3 и т.п.).
    fmt : str, optional
        Формат сообщения для ``logging.Formatter``. По умолчанию —
        ``"%(asctime)s %(levelname)-8s %(name)s: %(message)s"``.
    datefmt : str, optional
        Формат времени. По умолчанию — ``"%H:%M:%S"``.
    handler : logging.Handler, optional
        Готовый handler (например, для записи в файл). Если не задан,
        создаётся ``logging.StreamHandler()`` (вывод в stderr).

    Returns
    -------
    logger : logging.Logger
        Корневой логгер библиотеки (``logging.getLogger("subspace_conjugacy")``),
        уже настроенный и готовый к использованию.

    Examples
    --------
    >>> from subspace_conjugacy import configure_logging
    >>> import logging
    >>> configure_logging(level=logging.DEBUG)  # подробный трейс
    >>> # ... clusterer.fit(X) теперь печатает прогресс по каждой фазе ...

    Notes
    -----
    Повторные вызовы не дублируют handlers — существующие handlers,
    добавленные этой функцией, снимаются перед добавлением нового.
    """
    logger = logging.getLogger(_LIBRARY_LOGGER_NAME)

    for existing in list(logger.handlers):
        if getattr(existing, "_subspace_conjugacy_managed", False):
            logger.removeHandler(existing)

    stream_handler = handler if handler is not None else logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(fmt or _DEFAULT_FORMAT, datefmt=datefmt or _DEFAULT_DATEFMT)
    )
    stream_handler._subspace_conjugacy_managed = True  # noqa: SLF001 (внутренняя метка)

    logger.addHandler(stream_handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger
