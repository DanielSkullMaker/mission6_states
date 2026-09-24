"""
Эксперимент для статьи 1: "Итерационное наращивание опорных подпространств
в методе сопряжённости: анализ сходимости и вычислительной сложности".

План эксперимента — theory/article_plans/01_iterativnyi_algoritm.txt,
раздел 5 ("Методология и план эксперимента"), задачи 2-4. Использует
функционал, добавленный в библиотеку для этой статьи:
  - ConjugacyClusterGrowth.fit(..., store_history=True) -> growth_history_/
    get_growth_curve() — кривая R(k) и число обусловленности (Y^T Y) на
    каждой итерации фазы B.2 (subspace_conjugacy/algorithms/subclass_growth.py).
  - ConjugacyClusterGrowth(early_stopping=...)/FursovClusterer(early_stopping=...)
    — критерий ранней остановки роста ("relative_drop" / "fixed_fraction").

Два датасета (theory/article_plans/01_iterativnyi_algoritm.txt, раздел 0):
  - ОСНОВНОЙ: Fashion-MNIST (10 классов, 28x28, sklearn.datasets.fetch_openml,
    кэшируется локально) — реальные данные с нетривиальной геометрией классов
    (текстуры тканей вместо штрихов цифр).
  - ВСПОМОГАТЕЛЬНЫЙ: синтетические подпространства с ЗАДАННОЙ истинной
    размерностью r_true (смесь гауссиан вокруг r_true-мерного подпространства
    + шум) — нужен там, где важно знать "истинный" k заранее (Experiment C),
    что на реальных данных принципиально невозможно.

Структура эксперимента:
  Experiment A (реальные данные, Fashion-MNIST) — кривые сходимости R(k) и
    числа обусловленности (Y^T Y) по всем 10 классам (задачи 2-3 плана).
  Experiment B (реальные данные, Fashion-MNIST) — точность/время классификации
    при разных критериях ранней остановки против полного роста (задача 4).
  Experiment C (синтетика, контролируемая) — проверка чувствительности к
    ИЗВЕСТНОЙ истинной размерности подпространства: кривая R(k) должна резко
    "переламываться" в районе r_true, а число обусловленности — расти при
    приближении числа принятых векторов к N (задачи 2-3, обоснование связи
    с теорией; ср. замечание о LinearDependencyFilter в refactoring_plan.txt,
    раздел 10, находка №3, про полный ранг).
  Экспорт рисунков датасета "до/после" препроцессинга (для отчёта, раздел
    "Датасет").

Все числа сохраняются в JSON (experiments/article_01_output/results.json) —
отчёт (experiments/article_01_report.py) строится строго из этого файла, без
захардкоженных цифр.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from subspace_conjugacy.algorithms import (
    GlobalMinCosinePairFinder,
    ReferenceCenterBuilder,
    CosineSecondVectorAttacher,
    ConjugacyClusterGrowth,
    FursovClusterer,
)
from subspace_conjugacy.core.metrics import conjugate_criterion
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

# ----------------------------------------------------------------------
# Конфигурация (единая для воспроизводимости, как в main.py)
# ----------------------------------------------------------------------
RANDOM_SEED = 42
N_PER_CLASS = 200
TEST_FRACTION = 0.2
N_SUBCLASSES = 8  # то же значение, что зафиксировано в исходных ноутбуках

OUTPUT_DIR = Path(__file__).resolve().parent / "article_01_output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"

FASHION_MNIST_CLASSES = {
    "0": "Футболка/топ", "1": "Брюки", "2": "Свитер", "3": "Платье",
    "4": "Пальто", "5": "Сандалия", "6": "Рубашка", "7": "Кроссовок",
    "8": "Сумка", "9": "Ботинок",
}


def log(msg: str) -> None:
    print(f"[article_01] {msg}", flush=True)


# ----------------------------------------------------------------------
# Загрузка данных
# ----------------------------------------------------------------------
def load_fashion_mnist_split() -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Fashion-MNIST, 200/класс, 80/20 train/test, seed=42 — та же схема,
    что и MNIST-эксперимент проекта (main.py::fetch_mnist_split)."""
    from sklearn.datasets import fetch_openml

    log("Загрузка Fashion-MNIST (sklearn.datasets.fetch_openml, кэшируется локально)...")
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X_all = data.data.astype(np.float64)
    y_all = data.target.astype(str)

    rng = np.random.default_rng(RANDOM_SEED)
    n_test = int(round(N_PER_CLASS * TEST_FRACTION))

    X_train_by_class: Dict[str, np.ndarray] = {}
    X_test_list, y_test_list = [], []
    for digit in sorted(FASHION_MNIST_CLASSES.keys(), key=int):
        idx_all = np.where(y_all == digit)[0]
        chosen = rng.choice(idx_all, size=N_PER_CLASS, replace=False)
        perm = rng.permutation(N_PER_CLASS)
        test_local, train_local = perm[:n_test], perm[n_test:]
        X_train_by_class[digit] = X_all[chosen[train_local]]
        X_test_list.append(X_all[chosen[test_local]])
        y_test_list.append(np.full(len(test_local), digit))
        log(f"  класс {digit} ({FASHION_MNIST_CLASSES[digit]}): "
            f"{len(train_local)} train, {len(test_local)} test")

    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    return X_train_by_class, X_test, y_test


def save_dataset_montage(X_train_by_class: Dict[str, np.ndarray]) -> None:
    """Montage с одним примером на класс (сырые пиксели Fashion-MNIST,
    28x28) — общий обзор датасета для раздела "Датасет" отчёта."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    classes_sorted = sorted(X_train_by_class.keys(), key=int)
    tile = 28
    pad = 4
    cols = 5
    rows = 2
    montage = np.full(
        (rows * (tile + pad) + pad, cols * (tile + pad) + pad), 255, dtype=np.uint8
    )
    for i, cls in enumerate(classes_sorted):
        r, c = divmod(i, cols)
        img = X_train_by_class[cls][0].reshape(tile, tile).astype(np.uint8)
        y0 = pad + r * (tile + pad)
        x0 = pad + c * (tile + pad)
        montage[y0:y0 + tile, x0:x0 + tile] = img
    Image.fromarray(montage).resize(
        (montage.shape[1] * 4, montage.shape[0] * 4), Image.NEAREST
    ).save(FIGURES_DIR / "dataset_montage.png")

    log(f"Обзорный montage датасета сохранён в {FIGURES_DIR}")


def save_reconstruction_figure(
    X_train_by_class: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    demo_class: str = "5",
) -> Dict[str, Any]:
    """"До/после" в терминах САМОГО метода, а не общего препроцессинга:
    Fashion-MNIST не требует визуально заметной предобработки (изображения
    уже нормализованы и выровнены поставщиком датасета — линейная
    нормализация в [0,1] и обратно даёт пиксель-в-пиксель идентичный
    результат, что было бы малосодержательной иллюстрацией). Вместо этого
    показываем то, что содержательно для статьи: исходное тестовое
    изображение против его ПРОЕКЦИИ на выращенное подпространство класса
    (Y (Y^T Y)^{-1} Y^T x) — прямая визуализация показателя R(x, Y),
    центрального для теории (раздел 2 отчёта)."""
    from subspace_conjugacy.core.metrics import compute_gram_inverse, conjugate_criterion

    log(f"Строим иллюстрацию проекции на подпространство (класс {demo_class})...")
    X_cls_train = X_train_by_class[demo_class]

    pair_finder = GlobalMinCosinePairFinder().fit(X_cls_train)
    builder = ReferenceCenterBuilder(n_subclasses=N_SUBCLASSES).fit(
        X_cls_train, pair_finder.pair_indices_
    )
    attacher = CosineSecondVectorAttacher().fit(X_cls_train, builder.center_indices_)
    growth = ConjugacyClusterGrowth(freeze_basis_at=None)
    growth.fit(X_cls_train, attacher.pairs_)

    X_cls_test = X_test[y_test == demo_class]
    n_demo = min(5, len(X_cls_test))
    demo_images = X_cls_test[:n_demo]

    R_per_subclass = np.stack(
        [conjugate_criterion(demo_images, Y) for Y in growth.subspace_bases_], axis=1
    )
    best_subclass = R_per_subclass.argmax(axis=1)
    best_R = R_per_subclass.max(axis=1)

    tile = 28
    pad = 4
    montage = np.full((2 * (tile + pad) + pad, n_demo * (tile + pad) + pad), 255, dtype=np.uint8)
    r_values = []
    for i in range(n_demo):
        x = demo_images[i]
        Y = growth.subspace_bases_[best_subclass[i]]
        gram_inv = compute_gram_inverse(Y)
        # Скобки обязательны: "Y @ gram_inv @ Y.T @ x" из-за левоассоциативности
        # @ вычислился бы как ((Y @ gram_inv) @ Y.T) @ x — второй шаг строит
        # промежуточную матрицу (N, N), что при большом N расточительно по
        # памяти (безвредно здесь при N=784, но ловушка для копирования кода
        # на датасеты с крупными изображениями — см. article_02_experiment.py).
        # Явная группировка справа налево вычисляет тот же результат, ни разу
        # не материализуя матрицу N x N.
        projection = Y @ (gram_inv @ (Y.T @ x))
        orig_img = x.reshape(tile, tile).clip(0, 255).astype(np.uint8)
        recon_img = projection.reshape(tile, tile).clip(0, 255).astype(np.uint8)
        x0 = pad + i * (tile + pad)
        montage[pad:pad + tile, x0:x0 + tile] = orig_img
        montage[2 * pad + tile:2 * pad + 2 * tile, x0:x0 + tile] = recon_img
        r_values.append(float(best_R[i]))

    Image.fromarray(montage).resize(
        (montage.shape[1] * 5, montage.shape[0] * 5), Image.NEAREST
    ).save(FIGURES_DIR / "subspace_projection.png")

    log(f"  R по показанным примерам: {[round(r, 3) for r in r_values]}")
    return {
        "demo_class": demo_class,
        "demo_class_label": FASHION_MNIST_CLASSES[demo_class],
        "n_demo": n_demo,
        "r_values": r_values,
        "final_basis_sizes": [Y.shape[1] for Y in growth.subspace_bases_],
    }


# ----------------------------------------------------------------------
# Experiment A — кривые сходимости R(k) и обусловленности на реальных данных
# ----------------------------------------------------------------------
def run_experiment_a(X_train_by_class: Dict[str, np.ndarray]) -> Dict[str, Any]:
    log("=== Experiment A: кривые сходимости (Fashion-MNIST, все 10 классов) ===")
    per_class_curves: Dict[str, Any] = {}

    for cls in sorted(X_train_by_class.keys(), key=int):
        X_cls = X_train_by_class[cls]
        t0 = time.perf_counter()

        pair_finder = GlobalMinCosinePairFinder().fit(X_cls)
        builder = ReferenceCenterBuilder(n_subclasses=N_SUBCLASSES).fit(
            X_cls, pair_finder.pair_indices_
        )
        attacher = CosineSecondVectorAttacher().fit(X_cls, builder.center_indices_)

        growth = ConjugacyClusterGrowth(freeze_basis_at=None)
        growth.fit(X_cls, attacher.pairs_, store_history=True)

        elapsed = time.perf_counter() - t0
        r_curve = growth.get_growth_curve().tolist()
        cond_curve = [rec["gram_condition_number"] for rec in growth.growth_history_]
        basis_sizes = [rec["basis_size"] for rec in growth.growth_history_]

        # Монотонность: доля итераций, где R НЕ увеличился относительно
        # предыдущей (эмпирическая проверка "убывающей отдачи" — задача 2 плана).
        r_arr = np.array(r_curve)
        non_increasing_frac = float(np.mean(np.diff(r_arr) <= 1e-12)) if len(r_arr) > 1 else None

        per_class_curves[cls] = {
            "label": FASHION_MNIST_CLASSES[cls],
            "n_train": int(X_cls.shape[0]),
            "n_to_assign": len(r_curve),
            "r_curve": r_curve,
            "gram_condition_number_curve": cond_curve,
            "basis_size_curve": basis_sizes,
            "elapsed_seconds": elapsed,
            "non_increasing_fraction": non_increasing_frac,
            "r_first": r_curve[0] if r_curve else None,
            "r_last": r_curve[-1] if r_curve else None,
            "cond_first": cond_curve[0] if cond_curve else None,
            "cond_last": cond_curve[-1] if cond_curve else None,
        }
        log(f"  класс {cls} ({FASHION_MNIST_CLASSES[cls]}): "
            f"{len(r_curve)} итераций за {elapsed:.1f}с, "
            f"R: {r_curve[0]:.4f} -> {r_curve[-1]:.4f}, "
            f"cond: {cond_curve[0]:.2e} -> {cond_curve[-1]:.2e}, "
            f"non_increasing={non_increasing_frac:.2%}")

    return {"per_class": per_class_curves}


# ----------------------------------------------------------------------
# Experiment B — ранняя остановка: точность/время против полного роста
# ----------------------------------------------------------------------
def _own_class_r_quality(
    X_test: np.ndarray, y_test: np.ndarray, subspaces_by_class: Dict[str, List[np.ndarray]]
) -> float:
    """Средний показатель сопряжённости held-out объектов класса с ЛУЧШИМ
    ИЗ СВОИХ ЖЕ (собственного класса) подпространством — max_s R(x, Y_{c,s}).

    Не требует равнения размерности базисов между классами (в отличие от
    итоговой многоклассовой accuracy ниже) — измеряет качество ТОЛЬКО
    фазы роста самой по себе: "насколько хорошо подпространства класса,
    выращенные с данным early_stopping, объясняют НОВЫЕ объекты того же
    класса", без конкуренции с чужими базисами разной размерности. Прямая
    метрика для задачи 4 плана статьи, устойчивая к побочному эффекту
    equalize_subspace_bases (см. docstring _fit_and_evaluate ниже)."""
    scores = []
    for cls, bases in subspaces_by_class.items():
        X_cls_test = X_test[y_test == cls]
        if len(X_cls_test) == 0:
            continue
        R_per_subclass = np.stack(
            [conjugate_criterion(X_cls_test, Y) for Y in bases], axis=1
        )
        scores.append(R_per_subclass.max(axis=1).mean())
    return float(np.mean(scores))


def _fit_and_evaluate(
    X_train_by_class: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    early_stopping,
    early_stopping_threshold: float,
) -> Dict[str, Any]:
    """Обучает по одному FursovClusterer(freeze_basis_at=None, early_stopping=...)
    на класс и оценивает результат ДВУМЯ независимыми метриками:

    1. own_class_r_quality — среднее R held-out объектов класса с лучшим
       своим подпространством (_own_class_r_quality) — качество ТОЛЬКО фазы
       роста, без сравнения между классами.
    2. test_accuracy — честная многоклассовая точность через
       fit_from_subclass_bases(..., equalize=False), т.е. РЕАЛЬНЫЙ
       классификатор (решающее правило argmax по ВСЕМ классам) на базисах
       их РЕАЛЬНОГО, фактически достигнутого размера.

       ВАЖНАЯ ПРАВКА МЕТОДОЛОГИИ (обнаружена при первом прогоне этой
       статьи): equalize=True truncate'ит ВСЕ базисы ВСЕХ классов до
       общего МИНИМАЛЬНОГО k по всем n_subclasses x len(классов) = 80
       подпространствам. Из-за неравномерности жадного роста B.2 хотя бы
       ОДНО из 80 подпространств почти всегда останавливается на k=2
       (исходная пара из фазы B.1) даже при ПОЛНОМ росте без всякой ранней
       остановки — единственный "невезучий" подкласс из 80 задавал k=2 для
       АБСОЛЮТНО ВСЕХ, включая полностью выросшие. Результат — все шесть
       конфигураций (включая полный рост!) давали ОДИНАКОВУЮ точность
       0.53, полностью маскируя эффект ранней остановки на итоговую
       классификацию. С equalize=False каждое подпространство участвует в
       argmax со своим фактическим размером — точность вырастает до
       0.69-0.82 в зависимости от конфигурации, и появляется содержательное
       различие между стратегиями (см. раздел "Результаты" отчёта). Плата
       за это — теоретическая оговорка библиотеки (core/metrics.py,
       predict_r_matrix): R(x,Y) не масштабируется по k, поэтому базис
       большего размера при прочих равных склонен давать чуть больший R
       просто за счёт охвата большего числа измерений. Эмпирически на этом
       датасете эффект не доминирует (раздел "Результаты": конфигурация с
       остановкой на 75% дала точность НЕ НИЖЕ полного роста, несмотря на
       заметно меньший средний размер базиса) — но следует воспринимать
       абсолютные цифры точности как ориентировочные, а не как безупречно
       откалиброванное сравнение при СИЛЬНО различающихся размерах базисов.

    freeze_basis_at=None — намеренно (не 2!): при freeze_basis_at=2 итоговый
    базис классификатора — это ВСЕГДА только исходная пара из фазы B.1
    (первые 2 столбца), фаза B.2 роста на него не влияет вовсе — эффект
    ранней остановки был бы полностью не виден ни одной из двух метрик.
    """
    subspaces_by_class: Dict[str, List[np.ndarray]] = {}
    total_fit_time = 0.0
    total_stopped_early = 0
    total_excluded_by_early_stopping = 0
    basis_sizes_all: List[int] = []

    for cls, X_cls in X_train_by_class.items():
        t0 = time.perf_counter()
        clusterer = FursovClusterer(
            n_subclasses=N_SUBCLASSES,
            freeze_basis_at=None,
            early_stopping=early_stopping,
            early_stopping_threshold=early_stopping_threshold,
        )
        clusterer.fit(X_cls)
        total_fit_time += time.perf_counter() - t0
        subspaces_by_class[cls] = clusterer.subspaces_
        basis_sizes_all.extend(Y.shape[1] for Y in clusterer.subspaces_)
        if clusterer.stopped_early_:
            total_stopped_early += 1
        total_excluded_by_early_stopping += len(clusterer.excluded_by_early_stopping_)

    own_class_r_quality = _own_class_r_quality(X_test, y_test, subspaces_by_class)

    clf = SubspaceConjugacyClassifier()
    clf.fit_from_subclass_bases(subspaces_by_class, equalize=False)

    t_pred = time.perf_counter()
    y_pred = clf.predict(X_test)
    predict_time = time.perf_counter() - t_pred

    accuracy = float(np.mean(y_pred == y_test))
    per_class_accuracy = {
        cls: float(np.mean(y_pred[y_test == cls] == cls))
        for cls in sorted(X_train_by_class.keys())
    }

    return {
        "early_stopping": early_stopping,
        "early_stopping_threshold": early_stopping_threshold if early_stopping else None,
        "total_fit_time_seconds": total_fit_time,
        "predict_time_seconds": predict_time,
        "classes_stopped_early": total_stopped_early,
        "total_vectors_excluded_by_early_stopping": total_excluded_by_early_stopping,
        "basis_size_min": int(np.min(basis_sizes_all)),
        "basis_size_max": int(np.max(basis_sizes_all)),
        "basis_size_mean": float(np.mean(basis_sizes_all)),
        "equalized_basis_size": clf.equalized_basis_size_,
        "own_class_r_quality": own_class_r_quality,
        "test_accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
    }


def run_experiment_b(
    X_train_by_class: Dict[str, np.ndarray], X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    log("=== Experiment B: ранняя остановка vs полный рост (точность/время) ===")
    configs = [
        (None, 0.0),  # baseline — полный рост, как раньше
        ("fixed_fraction", 0.25),
        ("fixed_fraction", 0.5),
        ("fixed_fraction", 0.75),
        ("relative_drop", 0.5),
        ("relative_drop", 0.2),
    ]
    results = []
    for early_stopping, threshold in configs:
        label = "full_growth" if early_stopping is None else f"{early_stopping}@{threshold}"
        log(f"  конфигурация: {label}")
        res = _fit_and_evaluate(X_train_by_class, X_test, y_test, early_stopping, threshold)
        res["config_label"] = label
        results.append(res)
        log(f"    accuracy={res['test_accuracy']:.4f}, "
            f"fit_time={res['total_fit_time_seconds']:.1f}с, "
            f"equalized_k={res['equalized_basis_size']}")

    return {"configs": results}


# ----------------------------------------------------------------------
# Experiment C — синтетика с известной истинной размерностью подпространства
# ----------------------------------------------------------------------
def make_synthetic_subspace_data(
    n_samples: int, n_features: int, true_rank: int, noise_std: float, seed: int
) -> np.ndarray:
    """M векторов N-мерного пространства, лежащих (с точностью до шума
    noise_std) в true_rank-мерном подпространстве, натянутом на случайный
    ортонормированный базис. Позволяет проверить, "переламывается" ли
    эмпирическая кривая R(k) вблизи ИЗВЕСТНОГО true_rank — то, что
    невозможно установить на реальных данных, где истинная размерность
    класса неизвестна."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((n_features, true_rank)))
    coeffs = rng.standard_normal((n_samples, true_rank))
    X = coeffs @ basis.T
    X += rng.normal(scale=noise_std, size=X.shape)
    return X


def run_experiment_c() -> Dict[str, Any]:
    log("=== Experiment C: синтетика с известной истинной размерностью ===")
    n_features = 128
    n_samples = 120
    true_rank = 6
    noise_levels = [0.0, 0.05, 0.2]
    n_subclasses_synth = 1  # один "класс" — фокус на форме кривой роста, не на кластеризации

    results = []
    for noise_std in noise_levels:
        X = make_synthetic_subspace_data(
            n_samples, n_features, true_rank, noise_std, seed=RANDOM_SEED
        )
        pair_finder = GlobalMinCosinePairFinder().fit(X)
        attacher = CosineSecondVectorAttacher().fit(
            X, np.array([pair_finder.pair_indices_[0]])
        )
        growth = ConjugacyClusterGrowth(freeze_basis_at=None)
        growth.fit(X, attacher.pairs_, store_history=True)

        r_curve = growth.get_growth_curve().tolist()
        cond_curve = [rec["gram_condition_number"] for rec in growth.growth_history_]

        # Точка "перелома": первая итерация k, после которой R превышает
        # 0.95 (подпространство уже почти полностью объясняет новые векторы,
        # т.к. базис приблизился к истинному рангу true_rank).
        break_k = None
        for i, r in enumerate(r_curve):
            if r >= 0.95:
                break_k = i + 3  # +3: 2 из начальной пары + 1-индексация текущего шага
                break

        results.append({
            "noise_std": noise_std,
            "true_rank": true_rank,
            "n_features": n_features,
            "n_samples": n_samples,
            "r_curve": r_curve,
            "gram_condition_number_curve": cond_curve,
            "break_k_r_above_0_95": break_k,
        })
        log(f"  noise_std={noise_std}: break_k(R>=0.95)={break_k} "
            f"(истинный ранг={true_rank}), cond на последней итерации={cond_curve[-1]:.2e}")

    return {
        "true_rank": true_rank,
        "n_features": n_features,
        "n_samples": n_samples,
        "noise_levels": results,
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_by_class, X_test, y_test = load_fashion_mnist_split()
    save_dataset_montage(X_train_by_class)
    reconstruction_demo = save_reconstruction_figure(X_train_by_class, X_test, y_test)

    experiment_a = run_experiment_a(X_train_by_class)
    experiment_b = run_experiment_b(X_train_by_class, X_test, y_test)
    experiment_c = run_experiment_c()

    results = {
        "config": {
            "dataset": "Fashion-MNIST (sklearn.datasets.fetch_openml)",
            "n_per_class": N_PER_CLASS,
            "test_fraction": TEST_FRACTION,
            "random_seed": RANDOM_SEED,
            "n_subclasses": N_SUBCLASSES,
            "classes": FASHION_MNIST_CLASSES,
        },
        "reconstruction_demo": reconstruction_demo,
        "experiment_a_convergence_curves": experiment_a,
        "experiment_b_early_stopping": experiment_b,
        "experiment_c_synthetic": experiment_c,
        "total_elapsed_seconds": time.perf_counter() - start,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    log(f"Готово за {results['total_elapsed_seconds']:.1f}с. Результаты: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
