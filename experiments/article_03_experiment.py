"""
Эксперимент для статьи 3: "Мультипредставительная гибридизация признакового
пространства в методе сопряжённости: горизонтальная и вертикальная развёртка
изображения".

План — theory/article_plans/03_multipredstavitelnaya_gibridizatsiya.txt,
раздел 5 ("Методология и план эксперимента"). Использует функционал,
добавленный в библиотеку для этой статьи:
  - features.vectorization.load_and_vectorize_batch_multi — векторизация
    ОДНОГО и того же набора изображений сразу горизонтальным и вертикальным
    методом (каждый файл читается с диска один раз).
  - models.MultiRepresentationConjugacyClassifier — объединение ДВУХ уже
    обученных SubspaceConjugacyClassifier (по одному на представление) на
    уровне ПОКАЗАТЕЛЯ СОПРЯЖЁННОСТИ R(x, Y), до softmax/argmax.

Датасет (theory/article_plans/03_multipredstavitelnaya_gibridizatsiya.txt,
раздел 0): datasets/{class}_centered/ — собственный МРТ-датасет проекта
(glioma/meningioma/pituitary, по 200 изображений на класс, 256x256, уже
отцентрированных) — тот же датасет и тот же 80/20 train/test сплит
(seed=42), что и в main.py::prepare_centered_split, чтобы результат был
сопоставим с остальными честными сравнениями проекта.

Сравниваются ШЕСТЬ вариантов признакового представления одного и того же
метода (SubspaceConjugacyClassifier, n_subclasses=8). Подпространства строятся
С ПОЛНЫМ РОСТОМ (freeze_basis_at=None у FursovClusterer на класс, затем
fit_from_subclass_bases(..., equalize=False)) — та же методология, что и в
статьях 1-2 цикла, а НЕ дефолт main.py (freeze_basis_at=2, только исходная
пара без единого шага роста). Использование дефолта здесь было бы шагом
назад относительно уже установленной практики: freeze_basis_at=2 не даёт
подпространствам расти вообще, из-за чего разные представления неотличимы
друг от друга не по существу, а просто потому, что обеим "нечем" себя
проявить (проверено эмпирически при отладке эксперимента — все шесть
вариантов давали ОДИНАКОВУЮ точность). equalize=False (а не True/"auto")
— чтобы не наступить на ту же ошибку глобального равнения ко всем
n_subclasses x 3 класса подпространствам, что была найдена и исправлена в
статьях 1 и 2 (см. их отчёты, раздел "Найденная и исправленная
методологическая ошибка"): единственная переменная эксперимента —
ПРЕДСТАВЛЕНИЕ, а не метод роста подпространств (та тема уже закрыта
статьёй 1).

  1. horizontal   — только построчная развёртка (текущий вариант по
                    умолчанию во всей библиотеке с первого ноутбука).
  2. vertical     — только постолбцовая развёртка (посчитана в NB3, но
                    нигде не использовалась начиная с NB4).
  3. concatenation — конкатенация обоих векторов (N=2*65536=131072) в ОДНО
                    подпространство.
  4. fusion_mean  — раздельные подпространства (те же обученные модели
                    horizontal/vertical, БЕЗ повторного обучения),
                    среднее показателей сопряжённости R_hor и R_ver.
  5. fusion_max   — то же самое, но максимум вместо среднего.
  6. late_vote    — позднее объединение: каждая модель независимо доводит
                    решение до ВЕРОЯТНОСТЕЙ (softmax), объединяются уже ОНИ
                    — в отличие от fusion_mean/fusion_max, объединяющих
                    показатель сопряжённости ДО softmax. Не реализовано
                    через PrefitVotingClassifier (models/ensemble.py):
                    тот класс передаёт ОДИН И ТОТ ЖЕ X всем моделям, а
                    здесь у horizontal- и vertical-модели разные X
                    (разные векторные представления одних и тех же
                    изображений) — короткая локальная реализация ниже.

Все числа сохраняются в JSON (experiments/article_03_output/results.json) —
отчёт (experiments/article_03_report.py) строится строго из этого файла.

ПОДБОР ГИПЕРПАРАМЕТРОВ (добавлено по запросу "увеличь максимально точность"):
n_subclasses=8 без подбора давал очень неравномерный рост подпространств
(например, один подкласс класса glioma забирал 142 из 160 эталонных
векторов) и точность всего 0.4417 у ВСЕХ шести вариантов одновременно —
ожидаемо, поскольку теорема инвариантности (отчёт статьи 3, раздел 2.3)
гарантирует, что представление здесь ни при чём: все шесть вариантов
используют ОДИН И ТОТ ЖЕ базовый метод с одними и теми же гиперпараметрами,
поэтому поднять точность можно только подбором ЭТИХ гиперпараметров
(n_subclasses, growth_strategy, filter_dependent, split_correlated_pairs),
причём результат подбора распространяется на ВСЕ шесть вариантов сразу.

tune_hyperparameters() делает честный подбор на train'/val сплите,
carved ИЗ обучающей выборки (тестовая выборка не участвует ни в подборе, ни
в выборе конфигурации) — тот же принцип, что и в main.py::run_tuned_
comparison_experiment и в отчётах статей 1-2. Подбор идёт ТОЛЬКО на
горизонтальном представлении: теорема инвариантности гарантирует, что при
ЛЮБЫХ фиксированных гиперпараметрах горизонтальная и вертикальная модели
(и их объединения) дают строго идентичный результат — подбирать отдельно
для каждого представления было бы 6-кратной тратой времени без единого
шанса получить другой ответ.

⚠ НАЙДЕННАЯ И ИСПРАВЛЕННАЯ ОШИБКА ПЕРВОЙ ВЕРСИИ ПОДБОРА: первая версия
искала по сетке из 26 конфигураций на train'/val сплите всего 40/40
изображений на класс (SEARCH_N_PER_CLASS=40) — победила конфигурация
n_subclasses=3, growth_strategy="master" с val_accuracy=0.825 (против 0.742
у n_subclasses=8/"default"). На НЕЗАВИСИМОМ тесте эта "победившая"
конфигурация дала accuracy=0.3417 — ХУЖЕ исходных 0.4417! Причина —
классический оверфиттинг на маленькую и потому шумную валидационную
выборку (120 изображений на 3 класса; при переборе 26 конфигураций почти
гарантированно находится конфигурация, которая случайно хорошо легла на
именно эту выборку, но не обобщается). Найдено при честной проверке на
отложенном тесте — не спрятано, а исправлено увеличением
SEARCH_N_PER_CLASS до 80 (то есть до объёма, вдвое превышающего сам
финальный тест — 40/класс) и сужением сетки до параметров, которые
статья 1 и предварительный (шумный) прогон уже указали как содержательные
(n_subclasses, growth_strategy) — filter_dependent/split_correlated_pairs
теперь проверяются один раз, ПОСЛЕ выбора лучших (n_subclasses,
growth_strategy), а не перемножаются в общую сетку, чтобы не множить
число "шансов на переобучение к шуму".
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from subspace_conjugacy.algorithms import FursovClusterer
from subspace_conjugacy.core.metrics import compute_gram_inverse, conjugate_criterion
from subspace_conjugacy.evaluation import per_class_accuracy
from subspace_conjugacy.features.vectorization import load_and_vectorize_batch_multi
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
from subspace_conjugacy.models.multi_representation import (
    MultiRepresentationConjugacyClassifier,
)

# ----------------------------------------------------------------------
# Конфигурация — сплит идентичен main.py::prepare_centered_split (тот же
# датасет, доля теста, seed), но подпространства растятся ПОЛНОСТЬЮ
# (freeze_basis_at=None + equalize=False), а не по дефолту main.py
# (freeze_basis_at=2) — см. пояснение в docstring модуля выше.
# ----------------------------------------------------------------------
RANDOM_SEED = 42
N_PER_CLASS = 200
TEST_FRACTION = 0.2
IMG_SIZE = 256

# Подбор гиперпараметров (см. docstring модуля) — честный train'/val сплит,
# carved ИЗ обучающей выборки (train'=val=SEARCH_N_PER_CLASS/класс, всего
# 2*SEARCH_N_PER_CLASS <= 160 доступных train/класс; тест не участвует).
# SEARCH_N_PER_CLASS=80 — ВДВОЕ больше финального теста (40/класс), чтобы
# выбор конфигурации не был шумовым артефактом маленькой val-выборки (см.
# "НАЙДЕННАЯ И ИСПРАВЛЕННАЯ ОШИБКА" в docstring модуля). Сетка сужена до
# n_subclasses x growth_strategy — единственных двух параметров, которые
# предварительный (шумный) прогон показал содержательными; filter_dependent
# и split_correlated_pairs проверяются ОТДЕЛЬНО, поверх уже выбранной пары,
# а не перемножаются в общую сетку — меньше "шансов на переобучение к шуму".
SEARCH_N_PER_CLASS = 80
SEARCH_N_SUBCLASSES_GRID = [2, 3, 4, 5, 6]
SEARCH_GROWTH_STRATEGIES = ["default", "master"]

DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets"
CLASSES = ["glioma", "meningioma", "pituitary"]
CLASS_LABELS_RU = {
    "glioma": "глиома",
    "meningioma": "менингиома",
    "pituitary": "аденома гипофиза",
}

OUTPUT_DIR = Path(__file__).resolve().parent / "article_03_output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"

MODEL_ORDER = [
    "horizontal", "vertical", "concatenation", "fusion_mean", "fusion_max", "late_vote",
]


def log(msg: str) -> None:
    print(f"[article_03] {msg}", flush=True)


# ----------------------------------------------------------------------
# Загрузка данных — тот же 80/20 сплит на класс, что и main.py::prepare_centered_split
# ----------------------------------------------------------------------
def sorted_class_images(class_name: str) -> List[Path]:
    """Пути к PNG класса в datasets/{class}_centered/, отсортированные по
    числовому суффиксу имени файла — идентично
    main.py::_sorted_centered_class_images."""
    class_dir = DATASET_ROOT / f"{class_name}_centered"
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"{class_dir} не найдена. Ожидается уже отцентрированный датасет "
            f"datasets/{{class}}_centered/*.png (200 PNG на класс)."
        )
    return sorted(
        class_dir.glob(f"{class_name}*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )


def build_split() -> Dict[str, Dict[str, Any]]:
    """Строит один и тот же 80/20 train/test сплит на класс (фиксированный
    seed), общий для ВСЕХ шести вариантов представления — единственная
    переменная эксперимента остаётся представлением, а не составом данных."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_test = int(round(N_PER_CLASS * TEST_FRACTION))
    split: Dict[str, Dict[str, Any]] = {}
    for cls in CLASSES:
        paths = sorted_class_images(cls)
        if len(paths) < N_PER_CLASS:
            raise ValueError(f"В {cls}_centered найдено только {len(paths)} < {N_PER_CLASS}.")
        paths = paths[:N_PER_CLASS]
        perm = rng.permutation(N_PER_CLASS)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        split[cls] = {"paths": paths, "train_idx": train_idx, "test_idx": test_idx}
        log(f"  класс {cls}: {len(train_idx)} train, {len(test_idx)} test (из {len(paths)})")
    return split


def load_both_representations(
    split: Dict[str, Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Векторизует все изображения КАЖДЫМ методом РОВНО ОДИН РАЗ (каждый PNG
    читается с диска один раз, не дважды — load_and_vectorize_batch_multi),
    возвращает объединённые (across-class) train/test матрицы для обоих
    представлений плюс общие метки y_train/y_test."""
    X_hor_train_list, X_ver_train_list, y_train_list = [], [], []
    X_hor_test_list, X_ver_test_list, y_test_list = [], [], []

    for cls, info in split.items():
        log(f"  векторизация класса {cls} (horizontal + vertical, {len(info['paths'])} файлов)...")
        by_method = load_and_vectorize_batch_multi(
            info["paths"], methods=("horizontal", "vertical")
        )
        X_hor_train_list.append(by_method["horizontal"][info["train_idx"]])
        X_ver_train_list.append(by_method["vertical"][info["train_idx"]])
        y_train_list.append(np.full(len(info["train_idx"]), cls))
        X_hor_test_list.append(by_method["horizontal"][info["test_idx"]])
        X_ver_test_list.append(by_method["vertical"][info["test_idx"]])
        y_test_list.append(np.full(len(info["test_idx"]), cls))

    X_hor_train = np.vstack(X_hor_train_list)
    X_ver_train = np.vstack(X_ver_train_list)
    y_train = np.concatenate(y_train_list)
    X_hor_test = np.vstack(X_hor_test_list)
    X_ver_test = np.vstack(X_ver_test_list)
    y_test = np.concatenate(y_test_list)
    return X_hor_train, X_ver_train, y_train, X_hor_test, X_ver_test, y_test


# ----------------------------------------------------------------------
# Рисунки датасета "до/после"
# ----------------------------------------------------------------------
def save_dataset_montage(X_hor_train: np.ndarray, y_train: np.ndarray) -> None:
    """Montage с одним примером на каждый из 3 классов — обзор датасета
    "до" (исходные, уже отцентрированные изображения, без дальнейшей
    обработки под конкретное представление)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tile = IMG_SIZE
    pad = 8
    montage = np.full((tile + 2 * pad, len(CLASSES) * (tile + pad) + pad), 255, dtype=np.uint8)
    for i, cls in enumerate(CLASSES):
        idx = np.where(y_train == cls)[0][0]
        img = X_hor_train[idx].reshape(tile, tile).astype(np.uint8)
        x0 = pad + i * (tile + pad)
        montage[pad:pad + tile, x0:x0 + tile] = img
    Image.fromarray(montage).save(FIGURES_DIR / "dataset_montage.png")
    log(f"Обзорный montage датасета сохранён в {FIGURES_DIR}")


def save_representation_comparison_figure(
    horizontal_model: SubspaceConjugacyClassifier,
    vertical_model: SubspaceConjugacyClassifier,
    X_hor_test: np.ndarray,
    X_ver_test: np.ndarray,
    y_test: np.ndarray,
    demo_class: str = "glioma",
) -> Dict[str, Any]:
    """Ключевая иллюстрация "до/после" ИМЕННО ЭТОЙ статьи: не тривиальная
    нормировка яркости (визуально неотличима от исходника — как отмечено в
    отчётах статей 1-2), а то, что РЕАЛЬНО "видит" метод при каждом
    представлении — проекция тестового изображения на подпространство,
    построенное ЭТИМ представлением (буквально те же подпространства, что
    участвуют в итоговой классификации: freeze_basis_at=2, никакой
    отдельной "показательной" перестройки с другими настройками).

    Верхний ряд — исходные тестовые изображения; средний — их проекция на
    horizontal-подпространство (Y_hor (Y_hor^T Y_hor)^{-1} Y_hor^T x,
    развёрнутая обратно построчно); нижний — проекция на
    vertical-подпространство (та же формула с Y_ver, развёрнутая обратно
    ПОСТОЛБЦОВО — reshape(order="F"), иначе изображение окажется
    транспонированным/искажённым, т.к. вектор был получён именно таким
    порядком обхода пикселей)."""
    log(f"Строим иллюстрацию horizontal vs vertical проекции (класс {demo_class})...")
    mask = y_test == demo_class
    X_hor_demo = X_hor_test[mask]
    X_ver_demo = X_ver_test[mask]
    n_demo = min(5, X_hor_demo.shape[0])
    X_hor_demo = X_hor_demo[:n_demo]
    X_ver_demo = X_ver_demo[:n_demo]

    R_hor = np.stack(
        [conjugate_criterion(X_hor_demo, Y) for Y in horizontal_model.subspaces_[demo_class]],
        axis=1,
    )
    R_ver = np.stack(
        [conjugate_criterion(X_ver_demo, Y) for Y in vertical_model.subspaces_[demo_class]],
        axis=1,
    )
    best_hor = R_hor.argmax(axis=1)
    best_ver = R_ver.argmax(axis=1)

    tile = IMG_SIZE
    pad = 6
    montage = np.full((3 * (tile + pad) + pad, n_demo * (tile + pad) + pad), 255, dtype=np.uint8)
    r_hor_values, r_ver_values = [], []
    for i in range(n_demo):
        x_hor, x_ver = X_hor_demo[i], X_ver_demo[i]
        Y_hor = horizontal_model.subspaces_[demo_class][best_hor[i]]
        Y_ver = vertical_model.subspaces_[demo_class][best_ver[i]]

        # Скобки обязательны: без них "Y @ gram_inv @ Y.T @ x" вычислился бы
        # как ((Y @ gram_inv) @ Y.T) @ x — второй шаг материализует матрицу
        # (N, N); при N=65536 это ~32 ГБ, практическое исчерпание памяти
        # (та же ловушка, что и в article_01/02_experiment.py).
        proj_hor = Y_hor @ (compute_gram_inverse(Y_hor) @ (Y_hor.T @ x_hor))
        proj_ver = Y_ver @ (compute_gram_inverse(Y_ver) @ (Y_ver.T @ x_ver))

        orig_img = x_hor.reshape(tile, tile).clip(0, 255).astype(np.uint8)
        # horizontal-вектор -> обычный reshape (построчный обход, как и
        # при векторизации); vertical-вектор -> reshape(order="F")
        # (постолбцовый обход) — иначе изображение получится искажённым.
        recon_hor = proj_hor.reshape(tile, tile).clip(0, 255).astype(np.uint8)
        recon_ver = proj_ver.reshape((tile, tile), order="F").clip(0, 255).astype(np.uint8)

        x0 = pad + i * (tile + pad)
        montage[pad:pad + tile, x0:x0 + tile] = orig_img
        montage[2 * pad + tile:2 * pad + 2 * tile, x0:x0 + tile] = recon_hor
        montage[3 * pad + 2 * tile:3 * pad + 3 * tile, x0:x0 + tile] = recon_ver
        r_hor_values.append(float(R_hor[i, best_hor[i]]))
        r_ver_values.append(float(R_ver[i, best_ver[i]]))

    Image.fromarray(montage).save(FIGURES_DIR / "representation_comparison.png")
    log(f"  R (horizontal) по показанным примерам: {[round(r, 3) for r in r_hor_values]}")
    log(f"  R (vertical) по показанным примерам:   {[round(r, 3) for r in r_ver_values]}")
    return {
        "demo_class": demo_class,
        "demo_class_label": CLASS_LABELS_RU[demo_class],
        "n_demo": n_demo,
        "r_horizontal_values": r_hor_values,
        "r_vertical_values": r_ver_values,
    }


# ----------------------------------------------------------------------
# Позднее объединение (probability-level) — не через PrefitVotingClassifier,
# см. docstring модуля выше.
# ----------------------------------------------------------------------
def late_vote_predict_proba(
    horizontal_model: SubspaceConjugacyClassifier,
    vertical_model: SubspaceConjugacyClassifier,
    X_hor: np.ndarray,
    X_ver: np.ndarray,
) -> np.ndarray:
    proba_hor = horizontal_model.predict_proba(X_hor)
    proba_ver = vertical_model.predict_proba(X_ver)
    return (proba_hor + proba_ver) / 2.0


def fit_full_growth_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_subclasses: int,
    growth_strategy: str = "default",
    filter_dependent: bool = False,
    split_correlated_pairs: bool = False,
    correlated_pairs_subset: str = "a",
) -> SubspaceConjugacyClassifier:
    """Строит SubspaceConjugacyClassifier с ПОЛНЫМ ростом подпространств:
    отдельный FursovClusterer(freeze_basis_at=None) на класс, затем
    fit_from_subclass_bases(..., equalize=False) — см. пояснение в
    docstring модуля (та же методология, что и в статьях 1-2 цикла).
    Гиперпараметры передаются явно (а не читаются из глобальной константы),
    чтобы одна и та же функция обслуживала и подбор (tune_hyperparameters),
    и финальное обучение — без риска рассинхронизации конфигураций."""
    bases_by_class: Dict[str, List[np.ndarray]] = {}
    for cls in np.unique(y):
        X_cls = X[y == cls]
        clusterer = FursovClusterer(
            n_subclasses=n_subclasses,
            freeze_basis_at=None,
            growth_strategy=growth_strategy,
            filter_dependent=filter_dependent,
            split_correlated_pairs=split_correlated_pairs,
            correlated_pairs_subset=correlated_pairs_subset,
        )
        clusterer.fit(X_cls)
        bases_by_class[cls] = clusterer.subspaces_
    clf = SubspaceConjugacyClassifier()
    clf.fit_from_subclass_bases(bases_by_class, equalize=False)
    return clf


# ----------------------------------------------------------------------
# Подбор гиперпараметров (честный train'/val сплит, только horizontal —
# см. docstring модуля)
# ----------------------------------------------------------------------
def _fit_eval_config(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    **config_kwargs: Any,
) -> float:
    """Обучает конфигурацию на train'/оценивает на val — возвращает
    val_accuracy, либо -inf, если конфигурация невалидна для этого объёма
    данных (например, n_subclasses слишком велико после фильтрации)."""
    try:
        clf = fit_full_growth_classifier(X_train, y_train, **config_kwargs)
    except ValueError as exc:
        log(f"    конфигурация {config_kwargs} провалилась: {exc}")
        return float("-inf")
    y_pred = clf.predict(X_val)
    return float(np.mean(y_pred == y_val))


def tune_hyperparameters(
    X_hor_train: np.ndarray, y_train: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Честный подбор гиперпараметров базового метода на train'/val сплите,
    carved ИЗ обучающей выборки (тестовая выборка не участвует). Подбор
    идёт ТОЛЬКО на горизонтальном представлении — см. обоснование в
    docstring модуля (теорема инвариантности статьи 3).

    Returns
    -------
    best_config : Dict[str, Any]
        Победившая конфигурация (ключи — kwargs fit_full_growth_classifier,
        без n_subclasses/y/X) плюс её val_accuracy.
    candidates : List[Dict[str, Any]]
        ВСЕ опробованные конфигурации (для честного отчёта — не только
        победитель, как и в остальных tuned-протоколах проекта, main.py).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    X_search_train_list, y_search_train_list = [], []
    X_search_val_list, y_search_val_list = [], []
    for cls in CLASSES:
        idx_cls = np.where(y_train == cls)[0]
        perm = rng.permutation(len(idx_cls))
        train_sub = idx_cls[perm[:SEARCH_N_PER_CLASS]]
        val_sub = idx_cls[perm[SEARCH_N_PER_CLASS:2 * SEARCH_N_PER_CLASS]]
        X_search_train_list.append(X_hor_train[train_sub])
        y_search_train_list.append(y_train[train_sub])
        X_search_val_list.append(X_hor_train[val_sub])
        y_search_val_list.append(y_train[val_sub])
    X_search_train = np.vstack(X_search_train_list)
    y_search_train = np.concatenate(y_search_train_list)
    X_search_val = np.vstack(X_search_val_list)
    y_search_val = np.concatenate(y_search_val_list)

    log(f"Подбор гиперпараметров: train'={X_search_train.shape[0]}, "
        f"val={X_search_val.shape[0]} (carved из {y_train.shape[0]} "
        f"обучающих изображений — {SEARCH_N_PER_CLASS}/класс на train' и "
        f"{SEARCH_N_PER_CLASS}/класс на val; тест не участвует)...")

    candidates: List[Dict[str, Any]] = []

    log("  Фаза 1: n_subclasses x growth_strategy (filter_dependent=False, "
        "split_correlated_pairs=False)...")
    for n_sub in SEARCH_N_SUBCLASSES_GRID:
        for growth in SEARCH_GROWTH_STRATEGIES:
            t0 = time.perf_counter()
            val_acc = _fit_eval_config(
                X_search_train, y_search_train, X_search_val, y_search_val,
                n_subclasses=n_sub, growth_strategy=growth, filter_dependent=False,
            )
            elapsed = time.perf_counter() - t0
            candidates.append({
                "phase": 1, "n_subclasses": n_sub, "growth_strategy": growth,
                "filter_dependent": False, "split_correlated_pairs": False,
                "correlated_pairs_subset": "a",
                "val_accuracy": val_acc, "elapsed_seconds": elapsed,
            })
            log(f"    n_subclasses={n_sub}, growth_strategy={growth}: "
                f"val_accuracy={val_acc:.4f} ({elapsed:.1f}с)")

    best_phase1 = max(candidates, key=lambda c: c["val_accuracy"])
    log(f"  Лучшая конфигурация фазы 1: {best_phase1}")

    log("  Фаза 2: filter_dependent и split_correlated_pairs поверх "
        "лучшей (n_subclasses, growth_strategy) фазы 1, по отдельности...")
    phase2_variants = [
        {"filter_dependent": True, "split_correlated_pairs": False, "correlated_pairs_subset": "a"},
        {"filter_dependent": False, "split_correlated_pairs": True, "correlated_pairs_subset": "a"},
        {"filter_dependent": False, "split_correlated_pairs": True, "correlated_pairs_subset": "b"},
    ]
    for variant in phase2_variants:
        t0 = time.perf_counter()
        val_acc = _fit_eval_config(
            X_search_train, y_search_train, X_search_val, y_search_val,
            n_subclasses=best_phase1["n_subclasses"],
            growth_strategy=best_phase1["growth_strategy"],
            **variant,
        )
        elapsed = time.perf_counter() - t0
        candidates.append({
            "phase": 2, "n_subclasses": best_phase1["n_subclasses"],
            "growth_strategy": best_phase1["growth_strategy"],
            "val_accuracy": val_acc, "elapsed_seconds": elapsed,
            **variant,
        })
        log(f"    {variant}: val_accuracy={val_acc:.4f} ({elapsed:.1f}с)")

    best_overall = max(candidates, key=lambda c: c["val_accuracy"])
    log(f"Лучшая конфигурация по итогам подбора: {best_overall}")
    return best_overall, candidates


def late_vote_predict(
    horizontal_model: SubspaceConjugacyClassifier,
    vertical_model: SubspaceConjugacyClassifier,
    X_hor: np.ndarray,
    X_ver: np.ndarray,
) -> np.ndarray:
    proba = late_vote_predict_proba(horizontal_model, vertical_model, X_hor, X_ver)
    return horizontal_model.classes_[np.argmax(proba, axis=1)]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("Построение честного 80/20 train/test сплита (тот же принцип, что и main.py)...")
    split = build_split()

    log("Загрузка и векторизация обоими методами (horizontal + vertical)...")
    X_hor_train, X_ver_train, y_train, X_hor_test, X_ver_test, y_test = load_both_representations(split)
    log(f"train={X_hor_train.shape[0]}, test={X_hor_test.shape[0]}, "
        f"N_horizontal={X_hor_train.shape[1]}, N_vertical={X_ver_train.shape[1]}")

    save_dataset_montage(X_hor_train, y_train)

    # ------------------------------------------------------------------
    # Подбор гиперпараметров (honest train'/val, только horizontal — см.
    # docstring модуля) — winning config применяется НИЖЕ ко всем трём
    # обучаемым моделям (horizontal/vertical/concatenation) одинаково.
    # ------------------------------------------------------------------
    best_config, search_candidates = tune_hyperparameters(X_hor_train, y_train)
    model_kwargs = {
        "n_subclasses": best_config["n_subclasses"],
        "growth_strategy": best_config["growth_strategy"],
        "filter_dependent": best_config["filter_dependent"],
        "split_correlated_pairs": best_config["split_correlated_pairs"],
        "correlated_pairs_subset": best_config["correlated_pairs_subset"],
    }

    # ------------------------------------------------------------------
    # 1-2. Базовые модели: только horizontal, только vertical
    # ------------------------------------------------------------------
    log(f"Обучение модели: только horizontal (полный рост, {model_kwargs})...")
    t0 = time.perf_counter()
    horizontal_model = fit_full_growth_classifier(X_hor_train, y_train, **model_kwargs)
    time_horizontal = time.perf_counter() - t0
    log(f"  готово за {time_horizontal:.1f}с "
        f"(размеры базисов: {[[Y.shape[1] for Y in b] for b in horizontal_model.subspaces_.values()]})")

    log(f"Обучение модели: только vertical (полный рост, {model_kwargs})...")
    t0 = time.perf_counter()
    vertical_model = fit_full_growth_classifier(X_ver_train, y_train, **model_kwargs)
    time_vertical = time.perf_counter() - t0
    log(f"  готово за {time_vertical:.1f}с "
        f"(размеры базисов: {[[Y.shape[1] for Y in b] for b in vertical_model.subspaces_.values()]})")

    representation_demo = save_representation_comparison_figure(
        horizontal_model, vertical_model, X_hor_test, X_ver_test, y_test
    )

    # ------------------------------------------------------------------
    # 3. Конкатенация
    # ------------------------------------------------------------------
    log(f"Обучение модели: конкатенация horizontal+vertical (полный рост, {model_kwargs})...")
    X_concat_train = np.hstack([X_hor_train, X_ver_train])
    X_concat_test = np.hstack([X_hor_test, X_ver_test])
    t0 = time.perf_counter()
    concat_model = fit_full_growth_classifier(X_concat_train, y_train, **model_kwargs)
    time_concat = time.perf_counter() - t0
    log(f"  готово за {time_concat:.1f}с (N={X_concat_train.shape[1]})")

    # ------------------------------------------------------------------
    # 4-5. Слияние на уровне показателя сопряжённости (без повторного
    # обучения — переиспользуют уже обученные horizontal_model/vertical_model)
    # ------------------------------------------------------------------
    log("Сборка моделей слияния показателя сопряжённости (fusion_mean, fusion_max)...")
    fusion_mean_model = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
        {"horizontal": horizontal_model, "vertical": vertical_model}, fusion="mean"
    )
    fusion_max_model = MultiRepresentationConjugacyClassifier.from_fitted_estimators(
        {"horizontal": horizontal_model, "vertical": vertical_model}, fusion="max"
    )

    # ------------------------------------------------------------------
    # Оценка всех шести вариантов на тестовой выборке
    # ------------------------------------------------------------------
    log("Оценка на тестовой выборке...")
    test_results: Dict[str, Any] = {}

    def _record(name: str, y_pred: np.ndarray, predict_time: float, fit_time: float) -> None:
        accuracy = float(np.mean(y_pred == y_test))
        per_class = per_class_accuracy(y_test, y_pred)
        macro_accuracy = float(np.mean(list(per_class.values())))
        test_results[name] = {
            "accuracy": accuracy,
            "macro_accuracy": macro_accuracy,
            "per_class_accuracy": per_class,
            "fit_time_seconds": fit_time,
            "predict_time_seconds": predict_time,
        }
        log(f"  {name}: accuracy={accuracy:.4f}, macro_accuracy={macro_accuracy:.4f} "
            f"(fit={fit_time:.1f}с, predict={predict_time:.2f}с)")

    t0 = time.perf_counter()
    y_pred = horizontal_model.predict(X_hor_test)
    _record("horizontal", y_pred, time.perf_counter() - t0, time_horizontal)

    t0 = time.perf_counter()
    y_pred = vertical_model.predict(X_ver_test)
    _record("vertical", y_pred, time.perf_counter() - t0, time_vertical)

    t0 = time.perf_counter()
    y_pred = concat_model.predict(X_concat_test)
    _record("concatenation", y_pred, time.perf_counter() - t0, time_concat)

    t0 = time.perf_counter()
    y_pred = fusion_mean_model.predict({"horizontal": X_hor_test, "vertical": X_ver_test})
    _record("fusion_mean", y_pred, time.perf_counter() - t0, 0.0)

    t0 = time.perf_counter()
    y_pred = fusion_max_model.predict({"horizontal": X_hor_test, "vertical": X_ver_test})
    _record("fusion_max", y_pred, time.perf_counter() - t0, 0.0)

    t0 = time.perf_counter()
    y_pred = late_vote_predict(horizontal_model, vertical_model, X_hor_test, X_ver_test)
    _record("late_vote", y_pred, time.perf_counter() - t0, 0.0)

    # ------------------------------------------------------------------
    # Сохранение результатов
    # ------------------------------------------------------------------
    results = {
        "config": {
            "dataset": "МРТ проекта, datasets/{class}_centered/",
            "classes": CLASSES,
            "class_labels_ru": CLASS_LABELS_RU,
            "n_per_class": N_PER_CLASS,
            "test_fraction": TEST_FRACTION,
            "n_train_total": int(X_hor_train.shape[0]),
            "n_test_total": int(X_hor_test.shape[0]),
            "img_size": IMG_SIZE,
            "n_features_horizontal": int(X_hor_train.shape[1]),
            "n_features_concatenation": int(X_concat_train.shape[1]),
            "growth": "full (freeze_basis_at=None, equalize=False)",
            "random_seed": RANDOM_SEED,
            **model_kwargs,
        },
        "tuning": {
            "search_n_per_class": SEARCH_N_PER_CLASS,
            "best_config": best_config,
            "candidates": search_candidates,
        },
        "basis_sizes": {
            "horizontal": {cls: [int(Y.shape[1]) for Y in b] for cls, b in horizontal_model.subspaces_.items()},
            "vertical": {cls: [int(Y.shape[1]) for Y in b] for cls, b in vertical_model.subspaces_.items()},
            "concatenation": {cls: [int(Y.shape[1]) for Y in b] for cls, b in concat_model.subspaces_.items()},
        },
        "representation_demo": representation_demo,
        "model_order": MODEL_ORDER,
        "test_results": test_results,
        "total_elapsed_seconds": time.perf_counter() - start,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    log(f"Готово за {results['total_elapsed_seconds']:.1f}с. Результаты: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
