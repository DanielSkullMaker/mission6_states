"""ДЕМОНСТРАЦИЯ УТЕЧКИ ДАННЫХ (DATA LEAKAGE) — НЕ РЕАЛЬНЫЙ РЕЗУЛЬТАТ.

Показывает, КАК легко получить фиктивные accuracy 90%+ методологической
ошибкой (оценка на объектах, которые участвовали в обучении), и почему
такое число НИКОГДА нельзя выдавать за качество модели — особенно в задаче
классификации МРТ-опухолей, где доверие к фиктивно высокой точности может
привести к реальному вреду.

Механизм утечки здесь — не находки из theory/code/*.m (они ни при чём: ни
CorrelationCoreSelector, ни center_strategy="windowed", ни
smooth_adjacent_vectors не могут законно давать +20-30 п.п. без нарушения
train/test изоляции), а классическая ошибка оценки: классификатор
тестируется на ПОДМНОЖЕСТВЕ СВОЕЙ ЖЕ обучающей выборки, выданном за "тест".

Почему эффект зависит от n_subclasses (эмпирически найдено ниже):
  Фаза B.1 (CosineSecondVectorAttacher) делает n_subclasses начальных пар —
  это 2*n_subclasses обучающих векторов, которые становятся ЛИТЕРАЛЬНЫМИ
  столбцами базиса Y при freeze_basis_at=2 (по умолчанию). Для такого
  вектора x: R(x, Y) = 1.0 ТОЧНО (x лежит в span(Y) по построению) —
  классификация гарантированно верна не потому, что модель обобщает, а
  потому что она видит свой же обучающий пример.
    - При малом n_subclasses (мало пар относительно размера класса) доля
      "буквально запомненных" векторов мала — утечка почти незаметна.
    - При n_subclasses, приближающемся к половине размера класса, 2*n_subclasses
      покрывает БОЛЬШИНСТВО обучающих векторов — утечка становится огромной
      И ПРИ ЭТОМ ЧЕСТНАЯ holdout-точность может не расти или даже падать
      (переобучение: много мелких подклассов с малым числом векторов на
      подкласс хуже обобщают) — РАСХОЖДЕНИЕ МЕЖДУ ЧЕСТНОЙ И "ПРОТЕЧЕННОЙ"
      ACCURACY, РАСТУЩЕЕ ВМЕСТЕ С n_subclasses, — это и есть классический
      тревожный признак утечки/переобучения, который стоит искать в любом
      чужом отчёте.

Как распознать такую утечку в чужом отчёте (чек-лист):
  1. Совпадают ли индексы/файлы test и train? (здесь: намеренно да)
  2. Участвовал ли объект в построении подпространства (пара B.1/центр
     A.1-A.3), для которого его же потом "предсказывают"?
  3. Растёт ли разрыв между train- и holdout-accuracy вместе со сложностью
     модели (здесь — n_subclasses)? Растущий разрыв — красный флаг, а не
     повод раздувать n_subclasses ради красивого числа.

Запуск (только для демонстрации, результат НЕ для отчётов):
    python experiments/experiment_data_leakage_demo.py
"""

import logging

from experiment_matlab_findings import apply_core_selection, load_split, smooth_per_class, stack
from subspace_conjugacy import SubspaceConjugacyClassifier
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

logging.basicConfig(level=logging.ERROR, force=True)  # подавляем служебные WARNING о регуляризации (experiment_matlab_findings уже вызвал basicConfig при импорте — без force=True это было бы no-op)

BANNER = "=" * 78

# Специально растущий n_subclasses — чтобы наглядно показать, как утечка
# усиливается вместе с долей обучающих векторов, буквально попадающих в
# базис (2*n_subclasses / n_train_per_class).
N_SUBCLASSES_SWEEP = [8, 15, 20]


def main() -> None:
    print(BANNER)
    print("ДЕМОНСТРАЦИЯ УТЕЧКИ ДАННЫХ — ЦИФРЫ НИЖЕ НЕ ЯВЛЯЮТСЯ РЕАЛЬНОЙ ОЦЕНКОЙ")
    print("КАЧЕСТВА МОДЕЛИ. НЕ ЦИТИРОВАТЬ КАК ACCURACY КЛАССИФИКАТОРА.")
    print(BANNER)

    print("\nЗагрузка датасета (тот же сплит, что в experiment_matlab_findings.py)...")
    X_train_by_class, X_test_by_class = load_split()
    X_test_smoothed_by_class = smooth_per_class(X_test_by_class)

    # Честная находка №1 (ядро эталонных векторов) — тот же train, что и в
    # легитимном эксперименте, чтобы разница была видна именно из-за утечки,
    # а не из-за другого train-набора.
    X_train_core_by_class = apply_core_selection(X_train_by_class)
    X_train, y_train = stack(X_train_core_by_class)
    X_test, y_test = stack(X_test_smoothed_by_class)
    n_train_per_class = X_train.shape[0] // 3
    print(f"n_train_per_class={n_train_per_class} (после находки №1, core_fraction=0.5)")

    print(f"\n{'n_subclasses':>12} | {'доля в базисе':>14} | {'честная (holdout)':>18} | {'[УТЕЧКА] на train':>18} | {'разрыв':>8}")
    print("-" * 82)

    for n_sub in N_SUBCLASSES_SWEEP:
        clf = SubspaceConjugacyClassifier(n_subclasses=n_sub, freeze_basis_at=2)
        clf.fit(X_train, y_train)

        honest = evaluate_classifier(y_test, clf.predict(X_test))
        leak = evaluate_classifier(y_train, clf.predict(X_train))  # оценка на ВСЕЙ обучающей выборке — утечка

        basis_fraction = min(1.0, 2 * n_sub / n_train_per_class)
        gap = leak["accuracy"] - honest["accuracy"]

        print(
            f"{n_sub:>12} | {basis_fraction:>13.0%} | {honest['accuracy']:>18.3f} | "
            f"{leak['accuracy']:>18.3f} | {gap:>+8.3f}"
        )

    print(f"\n{BANNER}")
    print("ВЫВОД")
    print(BANNER)
    print(
        "При n_subclasses=20 «утечка» показывает accuracy около 0.96 — выглядит\n"
        "как выдающийся результат, но измеряет только то, насколько обучающие\n"
        "векторы лежат в построенных ПО НИМ ЖЕ подпространствах (тривиально\n"
        "близко к 1.0 по конструкции метода), а не способность классифицировать\n"
        "НОВЫЕ снимки. Честная holdout-точность при этом же n_subclasses=20\n"
        "не растёт вместе с «утечкой» (а часто падает из-за переобучения) —\n"
        "именно расходящийся разрыв в таблице выше и есть признак утечки,\n"
        "который стоит искать, а не повод сообщать 90%+ как результат модели."
    )


if __name__ == "__main__":
    main()
