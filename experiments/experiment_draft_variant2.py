"""Восполняет пробелы черновика theory/code/Вариант_2_1.docx — Таблицы 2-7.

Черновик ("Детерминированный подход к формированию обучающих подпространств
для распознавания медицинских изображений", Коршиков В.И., Фурсов В.А.) —
более ранняя, неопубликованная версия статьи, с пометками научного
руководителя прямо в тексте. Таблица 1 в черновике УЖЕ ЗАПОЛНЕНА (перенесена
сюда как TABLE1_LITERATURE — см. ниже). Таблицы 2, 3, 4 (методы
LogisticRegression/LinearSVM/RandomForest/KNN/CNN на 10/30/50 образцах,
случайный отбор) остались ПУСТЫМИ в черновике. Таблицы 5, 6, 7 (те же методы,
но отбор "по критерию (3)") руководитель прямо анонсировал как будущую
работу, но не создал даже пустыми: "Еще надо бы сделать еще три такие же
таблицы (5, 6 и 7)... когда я запущу программу и мы «сгенерируем» обучающие
образцы для 10, 30 и 50 патологий каждого класса".

Использует ОБА источника этого проекта:
  - "находка" — канон библиотеки, algorithms.GlobalMinCosinePairFinder (A.1)
    + algorithms.ReferenceCenterBuilder (A.2-A.3) — это и есть буквально
    "Второй этап"/критерий (3) черновика: детерминированный последовательный
    отбор m наиболее "непохожих" опорных векторов класса.
  - "совет научного руководителя" — evaluation.paired_correlation_split,
    буквальная реализация "Первого этапа" черновика (пары наиболее
    скоррелированных изображений разносятся между train/test) — родом из
    прошлого разговора в этом чате про theory/code/Вариант_2_1.docx.

⚠️ Методологическое ЕДИНСТВО с прошлым экспериментом: раз train/test
получены через paired_correlation_split, тестовая выборка СПЕЦИАЛЬНО
подобрана похожей на обучающую (черновик сам это признаёт — см. README,
раздел "Сплит через попарное разнесение"). Все accuracy ниже, включая
Table 1, измерены В ЭТОЙ ЖЕ методологии — сравнение методов между собой
внутри таблиц корректно, но абсолютные числа НЕ сравнимы напрямую с Table
I/II опубликованной статьи (там независимый случайный 80/20 сплит,
theory/VI_Korshikov_VA_Fursov_...docx).

⚠️ ФОРМУЛА (1) vs (2): черновик использует "нормированный коэффициент
корреляции" (Пирсон, как в CorrelationCoreSelector/paired_correlation_split)
и для "Первого", и для "Второго" этапа. Библиотечный канон
(GlobalMinCosinePairFinder, используется здесь для критерия (3)) использует
КОСИНУСНОЕ сходство сырых векторов — как в ОПУБЛИКОВАННОЙ статье, формула
(3) theory/VI_Korshikov_VA_Fursov_...docx, а не Пирсона черновика. Это
осознанный выбор в пользу уже протестированного канона, а не побитовая
реплика черновика — отличие небольшое (векторы неотрицательны, пиксели
0-255), но не нулевое.

ПРОВЕРКА ЭТОЙ ГИПОТЕЗЫ (добавлено): `select_by_criterion3(..., pair_metric=)`
теперь параметризован метрикой поиска НАЧАЛЬНОЙ пары (формула (2) черновика).
"cosine" (по умолчанию, канон, GlobalMinCosinePairFinder) — то, что уже
описано выше и даёт заниженное 0.71-0.84 вместо литературных 0.90-0.99.
"pearson" (find_min_pearson_pair, буквально формула (2) черновика —
нормированный коэффициент корреляции, как в paired_correlation_split/
CorrelationCoreSelector) — проверяет, объясняется ли разрыв с литературной
Table 1 именно заменой метрики начальной пары. Дальнейший рост подпространства
(критерий (3), формула R(x,Y)) не меняется в обоих случаях — это уже
каноническая формула, совпадающая с опубликованной статьёй, отличие только
в том, КАКАЯ пара векторов стартует рост. run_table1_pearson_pair_variant()
считает Table 1 с pair_metric="pearson" и печатает её рядом с cosine-версией
и литературными числами для прямого сравнения.

CNN (сверточная нейронная сеть, метод №5 в черновике): теперь В СКОУПЕ —
torch добавлен как опциональная зависимость (`pip install -e ".[experiments]"`).
Черновик называет архитектуру и гиперпараметры только в референсной
"Таблице 4" ("Сравнение с нейросетями"), но не в тексте: Архитектура =
SmallCNN, скорость обучения = 0.0003, L2-регуляризация = 0.0001 — это и
есть `SmallCNN`/`CNN_LR`/`CNN_WEIGHT_DECAY` ниже, максимально faithful
восстановление по этим трём цифрам (сама архитектура слоёв в черновике не
описана — собрана нами). Вторая сеть, `DeepCNN`, добавлена сверх черновика,
чтобы показать, помогает ли дополнительная ёмкость модели на тех же
10/30/50 образцах на класс, или на такой скудной выборке она только сильнее
переобучается — это ожидаемый и содержательный результат сам по себе,
не повод считать эксперимент неудавшимся.

Таблица 9 (НАШЕ расширение, отсутствует в черновике — см. нумерацию ниже):
проверяет, зависит ли точность метода сопряжённости (критерий 3) от ТОГО,
КАК ИМЕННО 200 изображений класса случайно разложены на обучающую и
тестовую выборки. В отличие от Table 1 (одно фиксированное разбиение через
paired_correlation_split — "Первый этап" черновика), здесь используется
evaluation.paper_train_test_split — честный СТРАТИФИЦИРОВАННЫЙ СЛУЧАЙНЫЙ
80/20-сплит (методология самой опубликованной статьи, см. main.py), заново
перевыбираемый TABLE8_N_SPLITS раз с разными random_state. На каждом сплите
отбор эталонов (критерий 3) и классификация проводятся заново с нуля.
Результат — mean/std/min/max accuracy по независимым случайным разбиениям
для каждого m: большой std означал бы, что опубликованная в Table 1 цифра —
везение конкретного разбиения, а не устойчивое свойство метода. (Внутренние
имена функций/переменных/ключ JSON остались "table8_..." — не переименованы,
чтобы не расходиться со ссылкой на них в dissertation_articles_plan.txt;
изменилась только ЧЕЛОВЕКОЧИТАЕМАЯ подпись "Таблица N" в выводе/markdown.)

⚠️ НУМЕРАЦИЯ ТАБЛИЦ ИСПРАВЛЕНА под theory/state_v2/Вариант_4.docx (более
полная версия черновика с уже заполненными Table 1-8, найдена позже, чем
писался этот скрипт). Реальный документ содержит 8 таблиц, а не 7:
  - Табл. 1 — сопряжённость, критерий (3), m=10..50 (= TABLE1_LITERATURE).
  - Табл. 2 — сопряжённость, СЛУЧАЙНЫЙ отбор, m=10..50 — теперь считается
    функцией run_table2_random() (добавлена позже первоначальной версии
    этого скрипта). ⚠️ Числа, ранее стоявшие в Вариант_4/4_1.docx под
    "Таблица 2", ПОДОЗРИТЕЛЬНО точно (до 4 знаков) совпадали с
    table1_reproduced этого скрипта — то есть с воспроизведением КРИТЕРИЯ
    (3), а не случайного отбора. Похоже на ошибку при сборке .docx (взяли
    не ту колонку/таблицу), а не на настоящий эксперимент со случайным
    отбором для классификатора сопряжённости. run_table2_random() — тот же
    train_pool_by_class/X_test (paired_correlation_split, методология
    черновика) и тот же classify-код (conjugacy_classify), что и Table 1,
    единственное отличие — random_select() вместо select_by_criterion3()
    на этапе выбора m эталонов из train pool каждого класса. Результат
    численно РАЗЛИЧАЕТСЯ с Table 1 (см. artifacts/draft_variant2_tables.md),
    что подтверждает: прежние числа документа под "Таблица 2" не могли быть
    получены случайным отбором.
  - Табл. 3-5 — классические методы, случайный отбор, m=10/30/50 (были
    "Таблица 2-4" в предыдущей версии этого скрипта).
  - Табл. 6-8 — классические методы, критерий (3), m=10/30/50 (были
    "Таблица 5-7"). ⚠️ В самом Вариант_4.docx столбец "Среднее" Таблицы 6
    (m=10, критерий 3) арифметически неверен для строк LR/LSVM/RF/KNN —
    там стоит копия столбца "Среднее" из Таблицы 3 (m=10, случайный
    отбор), а не пересчитанное среднее по собственным per-class числам
    этой таблицы (per-class числа в документе — верные и совпадают с этим
    скриптом). Табл. 7 и 8 документа такой ошибки не содержат.

PCA (Число главных компонент): как и в референсной таблице черновика
"Сравнение с нейросетями" (LogisticRegression/LinearSVM: 50 компонент,
KNN: 150) — но не более (n_train - 1), т.к. при m=10 обучающая выборка
(30 векторов) физически не вмещает 150 компонент. PCA обучается ТОЛЬКО на
отобранных m*3 обучающих векторах, не на тесте — иначе снова утечка
(тема всего этого чата).

УЛУЧШЕНИЕ БЕЙЗЛАЙНОВ (по запросу — честно поднять качество классических
методов и CNN, а не переписать числа в отчёте):
  - Классические методы (LR/LSVM/RF/KNN): добавлен StandardScaler ПЕРЕД PCA
    (обучается только на train_m, применяется к test как transform — без
    утечки) — на пиксельных признаках без масштабирования компоненты PCA
    доминируются немногими яркими пикселями, StandardScaler это устраняет.
    Фиксированные гиперпараметры (C=0.1/1.0, max_depth=20, n_neighbors=5)
    заменены на GridSearchCV(cv=3) по небольшой честной сетке — тюнинг
    ТОЛЬКО на train_m (never на test), результат — реально лучше подобранная
    модель, а не число, подогнанное под целевой разрыв.
  - CNN (SmallCNN/DeepCNn): (a) добавлена аугментация (случайное горизонтальное
    отражение на каждом батче — стандартный, недорогой приём для крайне малых
    датасетов; вертикальное отражение/повороты не добавлены — для МРТ-срезов
    это менее оправдано анатомически); (b) вместо ОДНОГО случайного seed —
    ансамбль из CNN_N_SEEDS независимо обученных моделей (soft-voting по
    усреднённым softmax-вероятностям) — снижает шанс катастрофического
    "коллапса" одной неудачной инициализации (см. SmallCNN m=10: 0.000/1.000/
    0.000 в исходном прогоне) и является стандартной техникой повышения
    точности, а не подгонкой под желаемый результат.
  Оба изменения применены ОДИНАКОВО ко всем m/selection-конфигурациям (не
  избирательно) и не касаются метода сопряжённости (Table 1/9) — сравнение
  остаётся честным, разрыв с методом сопряжённости может как сократиться,
  так и не сократиться — сообщается то, что получится.

Запуск:
    python experiments/experiment_draft_variant2.py
"""

import json
import logging
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, TensorDataset

from subspace_conjugacy import (
    CorrelationCoreSelector,
    conjugate_criterion,
    load_and_vectorize_batch,
    paired_correlation_split,
    paper_train_test_split,
)
from subspace_conjugacy.algorithms import GlobalMinCosinePairFinder, ReferenceCenterBuilder
from subspace_conjugacy.evaluation.metrics import evaluate_classifier
from subspace_conjugacy.preprocessing.resize import DEFAULT_SIZE

logging.basicConfig(level=logging.WARNING, force=True)

DATASET_ROOT = Path("datasets")
CLASSES = ["glioma", "meningioma", "pituitary"]
N_PER_CLASS = 200  # черновик: "по 200 медицинских изображений трех классов"
RANDOM_SEED = 42
REG_PARAM = 1e-8

# CNN: гиперпараметры из референсной "Таблицы 4" черновика (Архитектура =
# SmallCNN, скорость обучения = 0.0003, L2-регуляризация = 0.0001) — сама
# архитектура слоёв в черновике не приведена, собрана нами под эти цифры.
CNN_LR = 3e-4
CNN_WEIGHT_DECAY = 1e-4
CNN_EPOCHS = 60
CNN_BATCH_SIZE = 16
CNN_DEVICE = torch.device("cpu")


class SmallCNN(nn.Module):
    """Faithful-реконструкция под референс черновика (Таблица 4)."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # -> 16 x 4 x 4, не зависит от размера входа
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 4 * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """Сверх черновика: больше ёмкости — проверяем, помогает ли она на 10-50
    образцах/класс, или только усиливает переобучение (см. docstring модуля).
    """

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # 256 -> 128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # -> 64 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


CNN_ARCHITECTURES = {"SmallCNN": SmallCNN, "DeepCNN": DeepCNN}

TABLE1_SAMPLE_SIZES = [10, 20, 30, 40, 50]
TABLE_2_7_SAMPLE_SIZES = [10, 30, 50]
TABLE8_N_SPLITS = 20

# Table 1 черновика (theory/code/Вариант_2_1.docx) — переносим как есть,
# для сравнения с нашим воспроизведением на datasets/.
TABLE1_LITERATURE: Dict[int, Dict[str, float]] = {
    10: {"glioma": 1.0000, "meningioma": 0.7200, "pituitary": 0.9950, "mean": 0.9050},
    20: {"glioma": 0.9450, "meningioma": 0.9900, "pituitary": 1.0000, "mean": 0.9783},
    30: {"glioma": 0.9600, "meningioma": 0.9950, "pituitary": 1.0000, "mean": 0.9850},
    40: {"glioma": 0.9650, "meningioma": 1.0000, "pituitary": 1.0000, "mean": 0.9883},
    50: {"glioma": 0.9900, "meningioma": 0.9900, "pituitary": 1.0000, "mean": 0.9933},
}

CLASSICAL_METHOD_FACTORIES = {
    "LogisticRegression": lambda: LogisticRegression(C=0.1, max_iter=3000),
    "LinearSVM": lambda: LinearSVC(C=1.0, max_iter=5000),
    "RandomForest": lambda: RandomForestClassifier(
        max_depth=20, n_estimators=200, random_state=RANDOM_SEED
    ),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
}
# Небольшая честная сетка для GridSearchCV(cv=3) — тюнинг ТОЛЬКО на train_m
# (см. докстринг модуля, "УЛУЧШЕНИЕ БЕЙЗЛАЙНОВ"). n_neighbors ограничен по
# верхней границе отдельно в fit_classical() под фактический размер train_m
# при m=10 (30 векторов, cv=3 -> ~10 в фолде).
PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "LinearSVM": {"C": [0.01, 0.1, 1.0, 10.0]},
    "RandomForest": {"max_depth": [5, 10, 20, None]},
    "KNN": {"n_neighbors": [3, 5, 7]},
}
PCA_COMPONENTS_DESIRED = {
    "LogisticRegression": 50,
    "LinearSVM": 50,
    "RandomForest": 50,
    "KNN": 150,
}

CNN_N_SEEDS = 3  # ансамбль (soft-voting) вместо одного случайного seed — см. докстринг модуля

ALL_METHOD_NAMES: List[str] = [*CLASSICAL_METHOD_FACTORIES, *CNN_ARCHITECTURES]

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_draft_variant2_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "draft_variant2_tables.md"

# Снэпшоты реально использованных train/test изображений, ПО ОДНОЙ директории
# на каждую модель (научный руководитель просил именно это — чтобы можно было
# открыть папку модели и увидеть ровно те файлы, на которых она обучалась и
# тестировалась, без необходимости доверять коду/сиду). Пересоздаётся с нуля
# при каждом запуске (см. main()), чтобы не оставалось файлов от старых
# конфигураций.
SNAPSHOT_ROOT = ARTIFACTS_DIR / "train_test_by_model"


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загружает и векторизует datasets/{class}_centered/ (до N_PER_CLASS на класс).

    Возвращает также ``paths`` — np.ndarray[object] путей к исходным PNG,
    выровненный построчно с X/y (тот же порядок, каким строился X) — нужен,
    чтобы после любого индексирования/сплита X можно было восстановить,
    какие именно файлы отбирались, и физически скопировать их в снэпшот
    train/test для конкретной модели (см. save_split_snapshot).
    """
    X_list, y_list, path_list = [], [], []
    for cls in CLASSES:
        paths = sorted((DATASET_ROOT / f"{cls}_centered").glob("*.png"))[:N_PER_CLASS]
        if len(paths) < N_PER_CLASS:
            raise FileNotFoundError(
                f"{cls}: найдено {len(paths)} изображений в "
                f"{DATASET_ROOT / f'{cls}_centered'}, требуется {N_PER_CLASS}."
            )
        X_list.append(load_and_vectorize_batch(paths, method="horizontal"))
        y_list.append(np.full(len(paths), cls))
        path_list.extend(paths)
    # float64: load_and_vectorize_batch отдаёт uint8, а conjugacy_classify() ниже
    # вызывает conjugate_criterion напрямую (не через классы библиотеки, которые
    # кастуют сами) — на uint8 X**2 внутри неё переполняется молча (200**2 -> 64),
    # что давало полностью неверную (near-chance) точность в Table 9 (paper_train_test_split
    # не кастует, в отличие от paired_correlation_split) при внешне безошибочном запуске.
    return np.vstack(X_list).astype(np.float64), np.concatenate(y_list), np.array(path_list, dtype=object)


def find_min_pearson_pair(X_class: np.ndarray) -> Tuple[int, int]:
    """Глобальная пара с МИНИМАЛЬНЫМ нормированным коэффициентом корреляции
    Пирсона — буквально формула (2) черновика/Вариант_4.docx. Пирсон-аналог
    канонического GlobalMinCosinePairFinder (тот использует косинусное
    сходство сырых векторов, не Пирсона) — см. "ПРОВЕРКА ЭТОЙ ГИПОТЕЗЫ" в
    docstring модуля.
    """
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X_class)
    np.fill_diagonal(corr, np.nan)
    flat_argmin = np.nanargmin(corr)
    i, j = np.unravel_index(flat_argmin, corr.shape)
    return tuple(sorted((int(i), int(j))))


def select_by_criterion3(
    X_class: np.ndarray, m: int, pair_metric: str = "cosine"
) -> np.ndarray:
    """"Второй этап"/критерий (3) черновика = поиск начальной пары (формула (2))
    + A.2-A.3 (ReferenceCenterBuilder, формула (3)) — m наиболее "непохожих"
    векторов класса.

    Возвращает индексы (в X_class) — ОДНО растущее подпространство, БЕЗ
    подклассов (в черновике их нет вовсе — в отличие от FursovClusterer/
    SubspaceConjugacyClassifier, где n_subclasses означает число ОТДЕЛЬНЫХ
    подклассов, здесь m — просто итоговый размер единственного подпространства).

    Parameters
    ----------
    pair_metric : {"cosine", "pearson"}
        Метрика поиска НАЧАЛЬНОЙ пары (формула (2)). "cosine" — канон
        библиотеки (GlobalMinCosinePairFinder), используется во всех таблицах
        по умолчанию. "pearson" — буквально по черновику (find_min_pearson_pair)
        — см. run_table1_pearson_pair_variant() и "ПРОВЕРКА ЭТОЙ ГИПОТЕЗЫ" в
        docstring модуля. Рост подпространства (критерий (3)) не зависит от
        этого параметра — только выбор старта.
    """
    if pair_metric == "cosine":
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_class)
        initial_pair = pair_finder.pair_indices_
    elif pair_metric == "pearson":
        initial_pair = find_min_pearson_pair(X_class)
    else:
        raise ValueError(f"Неизвестный pair_metric: {pair_metric!r}")

    builder = ReferenceCenterBuilder(n_subclasses=m, reg_param=REG_PARAM)
    builder.fit(X_class, initial_pair)
    return builder.center_indices_


# Доля класса, отбираемая в "ядро" CorrelationCoreSelector — 0.5 буквально
# соответствует DATA_100.m (M_Vector_100=100 из M_Vector_200=200 — ровно
# половина), и сохраняет тот же коэффициент для Варианта B (честный сплит,
# 80 из 160 = тоже 0.5), см. theory/code/fixed/run_fixed_full.m.
DATA100_CORE_FRACTION = 0.5


def select_by_criterion3_from_core(
    X_class: np.ndarray, m: int, core_fraction: float = DATA100_CORE_FRACTION
) -> np.ndarray:
    """Критерий (3), но предварённый отбором "ядра" — БУКВАЛЬНОЕ поведение
    DATA_100.m (theory/code/DATA_100.m, строки 25-84), а не то, что случайно
    воспроизвёл theory/code/fixed/*.m.

    Контекст (см. artifacts/matlab_leakage_report.md, раздел 7): DATA_100.m
    на "Первом этапе" делит класс на ДВЕ равные половины через повторяющееся
    удаление более коррелированного из максимально коррелированной пары —
    "ядро" (вектор, каждый раз оказывающийся частью такой пары, накапливается
    в Y/A_train) и "выжившие" (A_tmp, ни разу не удалённые). "Второй этап"
    (рост базиса критерием (3)) в оригинале строится ИЗ ЯДРА (A_train = Y),
    а НЕ из выживших — что раньше по ошибке перепутал
    theory/code/fixed/truncate_by_correlation.m (возвращал keep_idx=выжившие).

    CorrelationCoreSelector (algorithms/correlation_core_selector.py) уже
    реализует именно "Первый этап" DATA_100.m — core_indices_ это точно то
    же самое множество, что и A_train/Y. Эта функция — недостающее звено:
    применяет критерий (3) (select_by_criterion3) К ЭТОМУ ЯДРУ, а не к
    сырому X_class, и переводит индексы обратно в пространство X_class.

    Parameters
    ----------
    core_fraction : float, default=0.5
        Доля X_class, отбираемая в "ядро" — см. DATA100_CORE_FRACTION.
    """
    n_total = X_class.shape[0]
    n_core = int(round(n_total * core_fraction))
    core_selector = CorrelationCoreSelector(n_target=n_core)
    core_selector.fit(X_class)
    X_core = X_class[core_selector.core_indices_]

    local_idx = select_by_criterion3(X_core, m)  # индексы внутри X_core
    return core_selector.core_indices_[local_idx]  # remap -> индексы в X_class


def random_select(n_total: int, m: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_total, size=m, replace=False)


def save_split_snapshot(
    model_name: str,
    selection_label: str,
    m: int,
    train_paths_by_class: Dict[str, Sequence[Path]],
    test_paths_by_class: Dict[str, Sequence[Path]],
    out_root: Path = SNAPSHOT_ROOT,
) -> None:
    """Копирует РЕАЛЬНО использованные train/test изображения модели
    ``model_name`` в ``out_root/<model_name>/...`` — вызывается ДО fit()
    соответствующего метода, чтобы содержимое папки гарантированно совпадало
    с тем, что видела модель (см. запрос руководителя — «повторю на такой же
    выборке»).

    Структура:
      out_root/<model_name>/test/<class>/<файл>.png           — одна и та же
        тестовая выборка для всех m/selection (копируется один раз при первом
        вызове для этой модели — экономит место, содержимое идентично).
      out_root/<model_name>/<selection_label>/m<m>/train/<class>/<файл>.png
      out_root/<model_name>/<selection_label>/m<m>/manifest.json — имена
        файлов + m + selection, для программной сверки без обхода директорий.
    """
    model_dir = out_root / model_name

    test_dir = model_dir / "test"
    if not test_dir.exists():
        for cls, paths in test_paths_by_class.items():
            cls_dir = test_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy2(p, cls_dir / Path(p).name)

    train_root = model_dir / selection_label / f"m{m}"
    train_dir = train_root / "train"
    manifest: Dict[str, Any] = {
        "model": model_name, "selection": selection_label, "m": m, "files": {},
    }
    for cls, paths in train_paths_by_class.items():
        cls_dir = train_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, cls_dir / Path(p).name)
        manifest["files"][cls] = [Path(p).name for p in paths]
    train_root.mkdir(parents=True, exist_ok=True)
    (train_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def conjugacy_classify(ref_by_class: Dict[str, np.ndarray], X_test: np.ndarray) -> np.ndarray:
    """Решающее правило (4) черновика: argmax R(x, Y_класса) по всем классам."""
    R = np.column_stack(
        [conjugate_criterion(X_test, ref_by_class[cls].T, reg_param=REG_PARAM) for cls in CLASSES]
    )
    return np.array(CLASSES)[np.argmax(R, axis=1)]


def _evaluate(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    report = evaluate_classifier(y_test, y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    return {"per_class_accuracy": per_class, "mean_accuracy": float(np.mean(list(per_class.values())))}


def run_table1(
    train_pool_by_class: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_pool_paths_by_class: Optional[Dict[str, np.ndarray]] = None,
    test_paths_by_class: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Воспроизводит Table 1 черновика (метод показателя сопряжённости, критерий 3)."""
    rows = {}
    for m in TABLE1_SAMPLE_SIZES:
        idx_by_class = {
            cls: select_by_criterion3(train_pool_by_class[cls], m) for cls in CLASSES
        }
        if train_pool_paths_by_class is not None and test_paths_by_class is not None:
            save_split_snapshot(
                model_name="SubspaceConjugacy",
                selection_label="criterion3",
                m=m,
                train_paths_by_class={
                    cls: train_pool_paths_by_class[cls][idx_by_class[cls]] for cls in CLASSES
                },
                test_paths_by_class=test_paths_by_class,
            )
        ref_by_class = {
            cls: train_pool_by_class[cls][idx_by_class[cls]] for cls in CLASSES
        }
        y_pred = conjugacy_classify(ref_by_class, X_test)
        rows[m] = _evaluate(y_test, y_pred)
        print(
            f"   Table 1, m={m}: {rows[m]['per_class_accuracy']}, "
            f"mean={rows[m]['mean_accuracy']:.4f} "
            f"(черновик: {TABLE1_LITERATURE[m]['mean']:.4f})"
        )
    return rows


def run_table2_random(
    train_pool_by_class: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_pool_paths_by_class: Optional[Dict[str, np.ndarray]] = None,
    test_paths_by_class: Optional[Dict[str, np.ndarray]] = None,
    seed_offset: int = 3,
) -> Dict[int, Dict[str, Any]]:
    """Воспроизводит Table 2 черновика (метод показателя сопряжённости,
    СЛУЧАЙНЫЙ отбор эталонов) — см. "ПРОВЕРКА ЭТОЙ ГИПОТЕЗЫ" в docstring
    модуля про то, почему прежние числа документа под этим номером
    подозрительны. Идентична run_table1() во всём, КРОМЕ способа отбора
    m эталонов из train_pool_by_class[cls]: random_select() вместо
    select_by_criterion3(). seed_offset=3 — не пересекается с
    seed_offset=1/2, используемыми run_classical_tables() для Table 3-8.
    """
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    rows = {}
    for m in TABLE1_SAMPLE_SIZES:
        idx_by_class = {
            cls: random_select(train_pool_by_class[cls].shape[0], m, rng) for cls in CLASSES
        }
        if train_pool_paths_by_class is not None and test_paths_by_class is not None:
            save_split_snapshot(
                model_name="SubspaceConjugacy",
                selection_label="random",
                m=m,
                train_paths_by_class={
                    cls: train_pool_paths_by_class[cls][idx_by_class[cls]] for cls in CLASSES
                },
                test_paths_by_class=test_paths_by_class,
            )
        ref_by_class = {
            cls: train_pool_by_class[cls][idx_by_class[cls]] for cls in CLASSES
        }
        y_pred = conjugacy_classify(ref_by_class, X_test)
        rows[m] = _evaluate(y_test, y_pred)
        print(
            f"   Table 2, m={m}: {rows[m]['per_class_accuracy']}, "
            f"mean={rows[m]['mean_accuracy']:.4f} (случайный отбор)"
        )
    return rows


def run_table1_data100_literal(
    X: np.ndarray,
    y: np.ndarray,
    core_fraction: float = DATA100_CORE_FRACTION,
    honest_train_fraction: float = 0.8,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Проверка гипотезы (пересмотренный вывод раздела 7
    artifacts/matlab_leakage_report.md): разрыв между литературной Table 1
    (0.905-0.993) и честным воспроизведением (0.713-0.840) объясняется
    ГЛАВНЫМ ОБРАЗОМ тем, что DATA_100.m растит эталоны из "дублирующей"
    половины (CorrelationCoreSelector.core_indices_), а не утечкой train/test
    САМОЙ ПО СЕБЕ. Если гипотеза верна — Вариант A (утечка, буквальный
    DATA_100.m) должен быть БЛИЗОК к литературе (как уже показал разовый
    Octave-прогон, matlab_leakage_report.md §3: 0.915 при m=10), а Вариант B
    (та же "дублирующая" логика, но БЕЗ утечки — честный 80/20) ответит на
    ключевой вопрос: добирается ли "дублирующий" отбор до литературной
    точности САМ ПО СЕБЕ, без утечки, или утечка всё-таки необходима в
    дополнение к нему.

    Вариант A — train_pool == test == весь X класса (200/класс), как в
    оригинальном Run_200.m/DATA_100.m: core_fraction=0.5 -> n_core=100,
    буквально M_Vector_100 из DATA_100.m.
    Вариант B — честный paper_train_test_split (80/20 -> 160 train/40 test
    на класс при N_PER_CLASS=200): core_fraction=0.5 -> n_core=80, тот же
    коэффициент, что и в theory/code/fixed/run_fixed_full.m (TRUNCATED_HONEST=80
    из TRAIN_POOL_SIZE=160) — для прямой сравнимости с MATLAB-числами оттуда.
    """
    results: Dict[str, Dict[int, Dict[str, Any]]] = {"A": {}, "B": {}}

    print(
        f"\nВариант A (DATA_100.m буквально, train_pool==test, "
        f"core_fraction={core_fraction})..."
    )
    X_by_class_A = {cls: X[y == cls] for cls in CLASSES}
    for m in TABLE1_SAMPLE_SIZES:
        ref_by_class = {
            cls: X_by_class_A[cls][
                select_by_criterion3_from_core(X_by_class_A[cls], m, core_fraction)
            ]
            for cls in CLASSES
        }
        y_pred = conjugacy_classify(ref_by_class, X)
        results["A"][m] = _evaluate(y, y_pred)
        print(
            f"   Table1-DATA100-A, m={m}: {results['A'][m]['per_class_accuracy']}, "
            f"mean={results['A'][m]['mean_accuracy']:.4f} "
            f"(литература: {TABLE1_LITERATURE[m]['mean']:.4f})"
        )

    print(
        f"\nВариант B (та же «дублирующая» логика, но честный "
        f"{honest_train_fraction:.0%}/{1 - honest_train_fraction:.0%} сплит)..."
    )
    X_train, X_test, y_train, y_test = paper_train_test_split(
        X, y, train_fraction=honest_train_fraction, random_state=random_state
    )
    X_train_by_class = {cls: X_train[y_train == cls] for cls in CLASSES}
    for m in TABLE1_SAMPLE_SIZES:
        ref_by_class = {
            cls: X_train_by_class[cls][
                select_by_criterion3_from_core(X_train_by_class[cls], m, core_fraction)
            ]
            for cls in CLASSES
        }
        y_pred = conjugacy_classify(ref_by_class, X_test)
        results["B"][m] = _evaluate(y_test, y_pred)
        print(
            f"   Table1-DATA100-B, m={m}: {results['B'][m]['per_class_accuracy']}, "
            f"mean={results['B'][m]['mean_accuracy']:.4f} "
            f"(литература: {TABLE1_LITERATURE[m]['mean']:.4f})"
        )

    return results


def run_table1_pearson_pair_variant(
    train_pool_by_class: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """Table 1, но начальная пара (формула (2)) ищется по Пирсону
    (find_min_pearson_pair), а не по канону (cosine) — проверка гипотезы
    "ФОРМУЛА (1) vs (2)" / "ПРОВЕРКА ЭТОЙ ГИПОТЕЗЫ" из docstring модуля:
    объясняется ли разрыв между литературными 0.90-0.99 и воспроизведёнными
    (cosine) 0.71-0.84 именно заменой метрики поиска старта.
    """
    rows = {}
    for m in TABLE1_SAMPLE_SIZES:
        ref_by_class = {
            cls: train_pool_by_class[cls][
                select_by_criterion3(train_pool_by_class[cls], m, pair_metric="pearson")
            ]
            for cls in CLASSES
        }
        y_pred = conjugacy_classify(ref_by_class, X_test)
        rows[m] = _evaluate(y_test, y_pred)
        print(
            f"   Table 1 (пара по Пирсону), m={m}: {rows[m]['per_class_accuracy']}, "
            f"mean={rows[m]['mean_accuracy']:.4f} "
            f"(черновик: {TABLE1_LITERATURE[m]['mean']:.4f})"
        )
    return rows


def run_table8_random_split_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    table1: Dict[int, Dict[str, Any]],
    n_splits: int = TABLE8_N_SPLITS,
) -> Dict[int, Dict[str, Any]]:
    """Table 8 (наше расширение — см. docstring модуля): для каждого m
    запускает n_splits НЕЗАВИСИМЫХ случайных стратифицированных 80/20-сплитов
    (paper_train_test_split — методология опубликованной статьи, НЕ
    paired_correlation_split из Table 1), на каждом сплите заново отбирает
    эталоны (критерий 3) и классифицирует тестовую часть. Возвращает
    mean/std/min/max accuracy по сплитам — насколько результат Table 1
    (одно конкретное разбиение) устойчив к тому, какие именно изображения
    попали в train, а какие в test.
    """
    rows: Dict[int, Dict[str, Any]] = {}

    for m in TABLE1_SAMPLE_SIZES:
        accuracies: List[float] = []

        for split_idx in range(n_splits):
            X_train, X_test, y_train, y_test = paper_train_test_split(
                X, y, train_fraction=0.8, random_state=RANDOM_SEED + 20_000 + m * 100 + split_idx
            )
            ref_by_class = {}
            for cls in CLASSES:
                X_cls = X_train[y_train == cls]
                ref_by_class[cls] = X_cls[select_by_criterion3(X_cls, m)]
            y_pred = conjugacy_classify(ref_by_class, X_test)
            accuracies.append(_evaluate(y_test, y_pred)["mean_accuracy"])

        acc_arr = np.array(accuracies)
        rows[m] = {
            "baseline_accuracy": table1[m]["mean_accuracy"],
            "accuracy_mean": float(acc_arr.mean()),
            "accuracy_std": float(acc_arr.std()),
            "accuracy_min": float(acc_arr.min()),
            "accuracy_max": float(acc_arr.max()),
            "n_splits": n_splits,
        }
        print(
            f"   Table 8, m={m}: accuracy={rows[m]['accuracy_mean']:.4f} "
            f"± {rows[m]['accuracy_std']:.4f} (min={rows[m]['accuracy_min']:.4f}, "
            f"max={rows[m]['accuracy_max']:.4f}) по {n_splits} случайным "
            f"80/20-сплитам; Table 1 (paired_correlation_split)={rows[m]['baseline_accuracy']:.4f}"
        )
    return rows


def fit_classical(
    method_name: str, X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[StandardScaler, PCA, Any, Dict[str, Any]]:
    """StandardScaler -> PCA -> GridSearchCV(cv=3) по PARAM_GRIDS[method_name].

    Оба шага (масштабирование и подбор гиперпараметров) обучаются ТОЛЬКО на
    X_train (train_m) — тест трансформируется как .transform() уже обученными
    scaler/pca в вызывающем коде, без утечки. См. докстринг модуля,
    "УЛУЧШЕНИЕ БЕЙЗЛАЙНОВ".
    """
    n_components = min(
        PCA_COMPONENTS_DESIRED[method_name], X_train.shape[0] - 1, X_train.shape[1]
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_reduced = pca.fit_transform(X_scaled)

    grid = dict(PARAM_GRIDS[method_name])
    if method_name == "KNN":
        # n_neighbors должен быть < числа объектов в наименьшем CV-фолде.
        max_k = max(1, X_train.shape[0] // 3 - 1)
        grid = {"n_neighbors": [k for k in grid["n_neighbors"] if k <= max_k] or [1]}

    base_clf = CLASSICAL_METHOD_FACTORIES[method_name]()
    search = GridSearchCV(base_clf, grid, cv=3, scoring="accuracy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X_reduced, y_train)
    return scaler, pca, search.best_estimator_, search.best_params_


def _to_image_tensor(X_flat: np.ndarray) -> torch.Tensor:
    """Разворачивает построчно-развёрнутые векторы (M, 65536) обратно в
    изображения (M, 1, 256, 256), нормированные в [0, 1] (features.vectorization
    использует C-order flatten, поэтому reshape без явного order восстанавливает
    исходную картинку)."""
    n = X_flat.shape[0]
    images = X_flat.reshape(n, DEFAULT_SIZE[0], DEFAULT_SIZE[1])
    return torch.from_numpy(images).float().unsqueeze(1) / 255.0


def _train_cnn_once(
    arch_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int
) -> torch.Tensor:
    """Обучает ОДНУ CNN с нуля на m*3 изображениях, возвращает softmax-вероятности
    на X_test (M_test, n_classes) — используется fit_predict_cnn_ensemble() для
    усреднения по нескольким seed (soft-voting, см. докстринг модуля).

    Аугментация: случайное горизонтальное отражение на каждом батче (p=0.5 на
    изображение) — единственная добавленная аугментация, недорогая и не
    искажающая тип опухоли (в отличие от произвольных поворотов/сдвигов).
    Гиперпараметры (lr, weight_decay) — из референсной Таблицы 4 черновика
    (см. докстринг модуля); архитектура — CNN_ARCHITECTURES[arch_name].
    """
    torch.manual_seed(seed)
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    y_idx = torch.tensor([label_to_idx[label] for label in y_train], dtype=torch.long)
    X_img = _to_image_tensor(X_train)

    model = CNN_ARCHITECTURES[arch_name](n_classes=len(CLASSES)).to(CNN_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(X_img, y_idx),
        batch_size=min(CNN_BATCH_SIZE, len(X_img)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model.train()
    for _epoch in range(CNN_EPOCHS):
        for X_batch, y_batch in loader:
            flip_mask = torch.rand(X_batch.shape[0]) < 0.5
            if flip_mask.any():
                X_batch = X_batch.clone()
                X_batch[flip_mask] = X_batch[flip_mask].flip(dims=[-1])
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(_to_image_tensor(X_test)), dim=1)
    return proba


def fit_predict_cnn(
    arch_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, base_seed: int
) -> np.ndarray:
    """Ансамбль из CNN_N_SEEDS независимо обученных CNN (soft-voting по
    усреднённым softmax-вероятностям) — снижает риск катастрофического
    коллапса одной неудачной инициализации на крайне малой выборке (m*3
    изображений). См. докстринг модуля, "УЛУЧШЕНИЕ БЕЙЗЛАЙНОВ".
    """
    proba_sum = None
    for rep in range(CNN_N_SEEDS):
        proba = _train_cnn_once(arch_name, X_train, y_train, X_test, seed=base_seed + rep * 9973)
        proba_sum = proba if proba_sum is None else proba_sum + proba
    pred_idx = (proba_sum / CNN_N_SEEDS).argmax(dim=1).numpy()
    return np.array(CLASSES)[pred_idx]


def run_classical_tables(
    train_pool_by_class: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    selection: str,
    seed_offset: int,
    train_pool_paths_by_class: Optional[Dict[str, np.ndarray]] = None,
    test_paths_by_class: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """selection: "random" (Table 3-5 документа) или "criterion3" (Table 6-8)."""
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    results_by_m: Dict[int, Dict[str, Dict[str, Any]]] = {}
    save_snapshots = train_pool_paths_by_class is not None and test_paths_by_class is not None

    for m in TABLE_2_7_SAMPLE_SIZES:
        X_parts, y_parts = [], []
        path_parts_by_class: Dict[str, np.ndarray] = {}
        for cls in CLASSES:
            X_cls = train_pool_by_class[cls]
            idx = (
                random_select(X_cls.shape[0], m, rng)
                if selection == "random"
                else select_by_criterion3(X_cls, m)
            )
            X_parts.append(X_cls[idx])
            y_parts.append(np.full(m, cls))
            if save_snapshots:
                path_parts_by_class[cls] = train_pool_paths_by_class[cls][idx]
        X_train_m = np.vstack(X_parts)
        y_train_m = np.concatenate(y_parts)

        results_by_m[m] = {}
        for method_idx, method_name in enumerate(ALL_METHOD_NAMES):
            if save_snapshots:
                # ДО обучения — снэпшот фиксирует ровно то, что увидит fit()
                # ниже, в отдельную директорию с именем этой модели.
                save_split_snapshot(
                    model_name=method_name,
                    selection_label=selection,
                    m=m,
                    train_paths_by_class=path_parts_by_class,
                    test_paths_by_class=test_paths_by_class,
                )

            best_params: Optional[Dict[str, Any]] = None
            if method_name in CLASSICAL_METHOD_FACTORIES:
                scaler, pca, clf, best_params = fit_classical(method_name, X_train_m, y_train_m)
                y_pred = clf.predict(pca.transform(scaler.transform(X_test)))
                n_pca_components: Optional[int] = int(pca.n_components_)
            else:
                cnn_seed = RANDOM_SEED + seed_offset * 100 + m + method_idx
                y_pred = fit_predict_cnn(method_name, X_train_m, y_train_m, X_test, base_seed=cnn_seed)
                n_pca_components = None

            point = _evaluate(y_test, y_pred)
            point["n_pca_components"] = n_pca_components
            point["best_params"] = best_params
            results_by_m[m][method_name] = point
            pca_label = n_pca_components if n_pca_components is not None else "—"
            params_label = f", params={best_params}" if best_params else ""
            print(
                f"   m={m}, {method_name} (pca={pca_label}{params_label}): "
                f"mean={point['mean_accuracy']:.4f}"
            )
    return results_by_m


def _print_method_table(table_number: int, m: int, selection_label: str, row: Dict[str, Dict[str, Any]]) -> None:
    print(f"\nТаблица {table_number}. Результаты различных методов для {m} образцов ({selection_label})")
    header = f"{'Метод':<20} | {'glioma':>8} | {'meningioma':>10} | {'pituitary':>9} | {'Среднее':>8} | PCA"
    print(header)
    for method_name, point in row.items():
        pc = point["per_class_accuracy"]
        pca_label = point["n_pca_components"] if point["n_pca_components"] is not None else "—"
        print(
            f"{method_name:<20} | {pc.get('glioma', float('nan')):>8.3f} | "
            f"{pc.get('meningioma', float('nan')):>10.3f} | "
            f"{pc.get('pituitary', float('nan')):>9.3f} | "
            f"{point['mean_accuracy']:>8.3f} | {pca_label}"
        )


def save_markdown_report(
    table1: Dict[int, Dict[str, Any]],
    table2: Dict[int, Dict[str, Any]],
    table1_pearson: Dict[int, Dict[str, Any]],
    table8: Dict[int, Dict[str, Any]],
    tables_2_4: Dict[int, Dict[str, Dict[str, Any]]],
    tables_5_7: Dict[int, Dict[str, Dict[str, Any]]],
    table1_data100: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
) -> None:
    lines = [
        "# Восполнение пробелов theory/state_v2/Вариант_4.docx",
        "",
        "Сгенерировано `experiment_draft_variant2.py` — см. docstring скрипта "
        "про методологию (находка №критерий-3 + совет руководителя "
        "paired_correlation_split), CNN (SmallCNN/DeepCNN, гиперпараметры из "
        "референсной Таблицы 4 черновика), Таблицу 9 (чувствительность к "
        "случайному train/test-сплиту) и ограничения (формула (1) vs (2), "
        "несравнимость с Table I/II опубликованной статьи).",
        "",
        "⚠️ Нумерация ниже соответствует theory/state_v2/Вариант_4.docx (8 таблиц "
        "в документе). Таблица 2 (метод сопряжённости, СЛУЧАЙНЫЙ отбор) теперь "
        "считается `run_table2_random()` — прежние числа документа под этим "
        "номером подозрительно совпадали с воспроизведением критерия (3) "
        "(Таблица 1), см. предупреждение в docstring модуля.",
        "",
        "## Таблица 1 — метод показателя сопряжённости, критерий (3) (воспроизведение, начальная пара по косинусу — канон)",
        "",
        "| Число образцов | glioma | meningioma | pituitary | Среднее (наше) | Среднее (черновик) |",
        "|---|---|---|---|---|---|",
    ]
    for m in TABLE1_SAMPLE_SIZES:
        row = table1[m]["per_class_accuracy"]
        lines.append(
            f"| {m} | {row['glioma']:.3f} | {row['meningioma']:.3f} | {row['pituitary']:.3f} "
            f"| {table1[m]['mean_accuracy']:.3f} | {TABLE1_LITERATURE[m]['mean']:.3f} |"
        )

    lines += [
        "",
        "## Таблица 2 — метод показателя сопряжённости, СЛУЧАЙНЫЙ отбор (воспроизведение)",
        "",
        "Тот же train_pool/test (paired_correlation_split) и тот же "
        "conjugacy_classify(), что и в Таблице 1 — единственное отличие: "
        "эталоны выбираются random_select(), а не select_by_criterion3(). "
        "«Среднее (черновик)» — число, ранее стоявшее в документе под этим "
        "номером (совпадало с критерием (3), см. предупреждение выше).",
        "",
        "| Число образцов | glioma | meningioma | pituitary | Среднее (наше) | Среднее (черновик) |",
        "|---|---|---|---|---|---|",
    ]
    for m in TABLE1_SAMPLE_SIZES:
        row = table2[m]["per_class_accuracy"]
        lines.append(
            f"| {m} | {row['glioma']:.3f} | {row['meningioma']:.3f} | {row['pituitary']:.3f} "
            f"| {table2[m]['mean_accuracy']:.3f} | {table1[m]['mean_accuracy']:.3f} |"
        )

    if table1_data100 is not None:
        lines += [
            "",
            "## Таблица 1 — буквальное воспроизведение DATA_100.m (рост из «дублирующей» половины)",
            "",
            "Проверка гипотезы из `artifacts/matlab_leakage_report.md` (раздел 7): "
            "DATA_100.m растит эталоны из «дублирующей» половины "
            "(`CorrelationCoreSelector.core_indices_`), а не из «выжившей», как "
            "по ошибке делает `theory/code/fixed/*.m`. Вариант A — утечка "
            "(train_pool==test==весь класс, буквально Run_200.m); Вариант B — "
            "та же «дублирующая» логика, но честный 80/20 сплит (без утечки).",
            "",
            "| m | A: glioma | A: meningioma | A: pituitary | A: mean | B: mean | Литература |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in TABLE1_SAMPLE_SIZES:
            row_a = table1_data100["A"][m]["per_class_accuracy"]
            lines.append(
                f"| {m} | {row_a['glioma']:.3f} | {row_a['meningioma']:.3f} | "
                f"{row_a['pituitary']:.3f} | {table1_data100['A'][m]['mean_accuracy']:.3f} "
                f"| {table1_data100['B'][m]['mean_accuracy']:.3f} | {TABLE1_LITERATURE[m]['mean']:.3f} |"
            )

    lines += [
        "",
        "## Таблица 1 (вариант) — та же критерий (3), но начальная пара (формула (2)) по Пирсону, не по косинусу",
        "",
        "Проверка гипотезы \"ФОРМУЛА (1) vs (2)\" из docstring модуля: объясняется "
        "ли разрыв между литературными числами и воспроизведением выше именно "
        "заменой метрики поиска начальной пары.",
        "",
        "| Число образцов | glioma | meningioma | pituitary | Среднее (Пирсон-пара) | Среднее (cosine-пара) | Среднее (черновик) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in TABLE1_SAMPLE_SIZES:
        row = table1_pearson[m]["per_class_accuracy"]
        lines.append(
            f"| {m} | {row['glioma']:.3f} | {row['meningioma']:.3f} | {row['pituitary']:.3f} "
            f"| {table1_pearson[m]['mean_accuracy']:.3f} | {table1[m]['mean_accuracy']:.3f} "
            f"| {TABLE1_LITERATURE[m]['mean']:.3f} |"
        )

    lines += [
        "",
        "## Таблица 9 (наше расширение) — чувствительность метода сопряжённости к случайному train/test-сплиту",
        "",
        f"Для каждого m — {TABLE8_N_SPLITS} независимых случайных стратифицированных "
        "80/20-сплитов (paper_train_test_split, методология статьи, НЕ "
        "paired_correlation_split из Table 1), отбор эталонов (критерий 3) и "
        "классификация заново на каждом сплите. `baseline` — значение Table 1 "
        "(один фиксированный сплит paired_correlation_split).",
        "",
        "| m | baseline (Table 1) | mean(accuracy) | std | min | max |",
        "|---|---|---|---|---|---|",
    ]
    for m in TABLE1_SAMPLE_SIZES:
        row = table8[m]
        lines.append(
            f"| {m} | {row['baseline_accuracy']:.3f} | {row['accuracy_mean']:.3f} "
            f"| {row['accuracy_std']:.4f} | {row['accuracy_min']:.3f} | {row['accuracy_max']:.3f} |"
        )

    for table_number, m, selection_label, tables in [
        (3, 10, "случайный отбор", tables_2_4), (4, 30, "случайный отбор", tables_2_4),
        (5, 50, "случайный отбор", tables_2_4),
        (6, 10, "критерий (3)", tables_5_7), (7, 30, "критерий (3)", tables_5_7),
        (8, 50, "критерий (3)", tables_5_7),
    ]:
        lines += [
            "",
            f"## Таблица {table_number} — {m} образцов, {selection_label}",
            "",
            "| Метод | glioma | meningioma | pituitary | Среднее | PCA компонент |",
            "|---|---|---|---|---|---|",
        ]
        for method_name, point in tables[m].items():
            pc = point["per_class_accuracy"]
            pca_label = point["n_pca_components"] if point["n_pca_components"] is not None else "—"
            lines.append(
                f"| {method_name} | {pc['glioma']:.3f} | {pc['meningioma']:.3f} | "
                f"{pc['pituitary']:.3f} | {point['mean_accuracy']:.3f} | {pca_label} |"
            )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown-отчёт (Таблицы 1, 1-Пирсон, 3-8, 9): {REPORT_MD_PATH}")


def main() -> None:
    start = time.perf_counter()
    print("=== Восполнение пробелов theory/state_v2/Вариант_4.docx (Table 3-8, 9) ===")
    print(f"Классы: {CLASSES}, N_PER_CLASS={N_PER_CLASS}")

    print("\nЗагрузка и векторизация датасета...")
    t0 = time.perf_counter()
    X, y, paths = load_data()
    print(f"готово за {time.perf_counter() - t0:.1f}s, X.shape={X.shape}")

    print(f"\nСнэпшоты train/test по моделям — пересоздаю {SNAPSHOT_ROOT}...")
    shutil.rmtree(SNAPSHOT_ROOT, ignore_errors=True)
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n«Первый этап» (совет руководителя, paired_correlation_split)...")
    split = paired_correlation_split(X, y, random_state=RANDOM_SEED)
    train_pool_by_class = {
        cls: split["X_train"][split["y_train"] == cls] for cls in CLASSES
    }
    X_test, y_test = split["X_test"], split["y_test"]

    # paths, выровненные так же, как X_train/X_test выше (train_indices/
    # test_indices из paired_correlation_split — индексы в исходном X/paths).
    train_paths_all = paths[split["train_indices"]]
    test_paths_all = paths[split["test_indices"]]
    train_pool_paths_by_class = {
        cls: train_paths_all[split["y_train"] == cls] for cls in CLASSES
    }
    test_paths_by_class = {
        cls: test_paths_all[split["y_test"] == cls] for cls in CLASSES
    }
    for cls in CLASSES:
        print(f"   {cls}: усечённое (train pool)={train_pool_by_class[cls].shape[0]}, "
              f"тестовое={int(np.sum(y_test == cls))}")

    print("\nTable 1 (воспроизведение, критерий 3 + метод сопряжённости, начальная пара по косинусу)...")
    table1 = run_table1(
        train_pool_by_class, X_test, y_test,
        train_pool_paths_by_class=train_pool_paths_by_class,
        test_paths_by_class=test_paths_by_class,
    )

    print(
        "\nTable 1, вариант с начальной парой по Пирсону (проверка гипотезы "
        "\"ФОРМУЛА (1) vs (2)\" — см. docstring модуля)..."
    )
    table1_pearson = run_table1_pearson_pair_variant(train_pool_by_class, X_test, y_test)

    print("\nTable 2 (воспроизведение, СЛУЧАЙНЫЙ отбор + метод сопряжённости)...")
    table2 = run_table2_random(
        train_pool_by_class, X_test, y_test,
        train_pool_paths_by_class=train_pool_paths_by_class,
        test_paths_by_class=test_paths_by_class,
    )

    print(
        "\nTable 1, буквальное воспроизведение DATA_100.m (рост эталонов из "
        "«дублирующей» половины CorrelationCoreSelector, Вариант A — утечка, "
        "Вариант B — честный сплит; проверка гипотезы из "
        "artifacts/matlab_leakage_report.md, раздел 7)..."
    )
    table1_data100 = run_table1_data100_literal(X, y)

    print(
        f"\nTable 9 (наше расширение — зависит ли accuracy метода сопряжённости "
        f"от случайного разбиения на train/test, {TABLE8_N_SPLITS} независимых "
        f"80/20-сплитов на m)... [снэпшоты НЕ сохраняются — 20 сплитов x 5 m "
        f"дали бы избыточные ~100 конфигураций, здесь важна сама изменчивость "
        f"по сплитам, а не конкретный набор файлов]"
    )
    table8 = run_table8_random_split_sensitivity(X, y, table1)

    print("\nTable 3-5 (заполняем — случайный отбор из усечённого множества)...")
    tables_2_4 = run_classical_tables(
        train_pool_by_class, X_test, y_test, selection="random", seed_offset=1,
        train_pool_paths_by_class=train_pool_paths_by_class,
        test_paths_by_class=test_paths_by_class,
    )

    print("\nTable 6-8 (заполняем — отбор по критерию (3), как в Table 1)...")
    tables_5_7 = run_classical_tables(
        train_pool_by_class, X_test, y_test, selection="criterion3", seed_offset=2,
        train_pool_paths_by_class=train_pool_paths_by_class,
        test_paths_by_class=test_paths_by_class,
    )

    for table_number, m, label, tables in [
        (3, 10, "случайный отбор", tables_2_4), (4, 30, "случайный отбор", tables_2_4),
        (5, 50, "случайный отбор", tables_2_4),
        (6, 10, "критерий (3)", tables_5_7), (7, 30, "критерий (3)", tables_5_7),
        (8, 50, "критерий (3)", tables_5_7),
    ]:
        _print_method_table(table_number, m, label, tables[m])

    print("\n=== Итог: критерий (3) vs случайный отбор (m=50) ===")
    conj_50 = table1[50]["mean_accuracy"]
    random_50 = np.mean([tables_2_4[50][m]["mean_accuracy"] for m in CLASSICAL_METHOD_FACTORIES])
    crit3_50 = np.mean([tables_5_7[50][m]["mean_accuracy"] for m in CLASSICAL_METHOD_FACTORIES])
    cnn_random_50 = np.mean([tables_2_4[50][m]["mean_accuracy"] for m in CNN_ARCHITECTURES])
    cnn_crit3_50 = np.mean([tables_5_7[50][m]["mean_accuracy"] for m in CNN_ARCHITECTURES])
    print(f"Метод сопряжённости (критерий 3):       {conj_50:.3f}")
    print(f"Классические ML (4 метода), случайно:   {random_50:.3f}")
    print(f"Классические ML (4 метода), критерий 3: {crit3_50:.3f}")
    print(f"CNN (SmallCNN+DeepCNN), случайно:       {cnn_random_50:.3f}")
    print(f"CNN (SmallCNN+DeepCNN), критерий 3:     {cnn_crit3_50:.3f}")
    for arch_name in CNN_ARCHITECTURES:
        print(
            f"   {arch_name}: случайно={tables_2_4[50][arch_name]['mean_accuracy']:.3f}, "
            f"критерий (3)={tables_5_7[50][arch_name]['mean_accuracy']:.3f}"
        )

    print("\n=== Итог: объясняет ли метрика начальной пары (формула (2)) разрыв с литературной Table 1? ===")
    for m in TABLE1_SAMPLE_SIZES:
        lit = TABLE1_LITERATURE[m]["mean"]
        cos_acc = table1[m]["mean_accuracy"]
        pearson_acc = table1_pearson[m]["mean_accuracy"]
        print(
            f"   m={m}: черновик={lit:.4f}, cosine-пара={cos_acc:.4f} "
            f"(разрыв {lit - cos_acc:+.4f}), Пирсон-пара={pearson_acc:.4f} "
            f"(разрыв {lit - pearson_acc:+.4f})"
        )
    gap_cosine = np.mean([TABLE1_LITERATURE[m]["mean"] - table1[m]["mean_accuracy"] for m in TABLE1_SAMPLE_SIZES])
    gap_pearson = np.mean(
        [TABLE1_LITERATURE[m]["mean"] - table1_pearson[m]["mean_accuracy"] for m in TABLE1_SAMPLE_SIZES]
    )
    print(f"Средний разрыв с черновиком: cosine-пара={gap_cosine:.4f}, Пирсон-пара={gap_pearson:.4f}")
    print(
        "Вывод: замена метрики начальной пары "
        + ("ОБЪЯСНЯЕТ основную часть разрыва" if gap_pearson < gap_cosine / 2 else "НЕ объясняет разрыв")
        + " с литературной Table 1 (см. docstring модуля, \"ФОРМУЛА (1) vs (2)\")."
    )

    print("\n=== Итог: зависит ли accuracy метода сопряжённости от случайного train/test-сплита? ===")
    max_std = max(row["accuracy_std"] for row in table8.values())
    max_range = max(row["accuracy_max"] - row["accuracy_min"] for row in table8.values())
    print(f"Максимальный std(accuracy) по {TABLE8_N_SPLITS} случайным сплитам (по всем m): {max_std:.4f}")
    print(f"Максимальный разброс max-min по всем m: {max_range:.4f}")
    for m in TABLE1_SAMPLE_SIZES:
        row = table8[m]
        print(
            f"   m={m}: Table 1 (один сплит)={row['baseline_accuracy']:.4f}, "
            f"среднее по {TABLE8_N_SPLITS} сплитам={row['accuracy_mean']:.4f} "
            f"± {row['accuracy_std']:.4f}"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_per_class": N_PER_CLASS,
                "random_seed": RANDOM_SEED,
                "table1_literature": TABLE1_LITERATURE,
                "table1_reproduced": table1,
                "table1_pearson_pair_variant": table1_pearson,
                "table2_reproduced_random": table2,
                "table1_data100_literal": table1_data100,
                "table8_random_split_sensitivity": table8,
                "tables_2_4_random": tables_2_4,
                "tables_5_7_criterion3": tables_5_7,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    save_markdown_report(
        table1, table2, table1_pearson, table8, tables_2_4, tables_5_7,
        table1_data100=table1_data100,
    )

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
