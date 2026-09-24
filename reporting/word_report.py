"""Генератор подробного Word-отчёта по результатам эксперимента "tuned".

НАМЕРЕННО расположен ВНЕ пакета subspace_conjugacy: это вспомогательный
инструмент для оформления результатов экспериментов (main.py), а не часть
библиотеки метода — библиотека не должна тянуть за собой зависимости
документооборота (python-docx, matplotlib). Читает JSON-отчёты, которые
main.py --experiment tuned сохраняет в artifacts/tuned_comparison_{mri,
mnist}_report.json (см. main.py::run_tuned_comparison_experiment), и строит
единый .docx с теоретическим введением (математическая основа каждого
сравниваемого метода, формулы отрендерены через matplotlib mathtext в PNG —
без зависимости от LaTeX/Word Equation Editor, ключевые различия подходов и
объяснение, что именно улучшает метод из чернового варианта статьи внутри
подпространственного подхода), методологией эксперимента, результатами
подбора гиперпараметров (включая ВСЕ опробованные конфигурации с описанием
каждого гиперпараметра — а не только победителя: отчёт должен быть понятен
и проверяем читателем, незнакомым ни с этой библиотекой, ни с ML вообще),
кривой эффективности по объёму данных (таблица + графики) и честными
выводами, вычисленными из фактических чисел отчёта (а не захардкоженными —
при повторном прогоне с другой сеткой гиперпараметров или другими данными
выводы должны пересчитаться автоматически).

Зависимости (НЕ входят в основные/extras зависимости subspace_conjugacy —
см. reporting/requirements.txt):
    pip install -r reporting/requirements.txt

Запуск:
    python reporting/word_report.py \
        --report artifacts/tuned_comparison_mri_report.json \
        --report artifacts/tuned_comparison_mnist_report.json \
        --output artifacts/tuned_comparison_report.docx
"""

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # без дисплея — только рендер в файл
import matplotlib.pyplot as plt

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

# Единая, брендово-нейтральная цветовая схема для всех графиков отчёта —
# три подхода всегда обозначаются одним и тем же цветом на всех графиках и
# во всех разделах, чтобы читатель не путался между секциями.
APPROACH_COLORS = {
    "subspace": "#2563EB",      # синий — подпространственная сопряжённость
    "classical_ml": "#16A34A",  # зелёный — классический ML
    "cnn": "#DC2626",           # красный — CNN
}
APPROACH_LABELS = {
    "subspace": "Сопряжённость",
    "classical_ml": "Классический ML",
    "cnn": "CNN",
}
TIE_EPSILON = 0.01  # разница accuracy <= этого значения считается "ничьей"

# Плейн-текстовые описания гиперпараметров, встречающихся в таблицах подбора
# (разделы 2.1-2.3 каждого датасета) — читатель, не знакомый ни с библиотекой,
# ни с ML вообще, должен понимать, что означает каждая колонка, не заглядывая
# в исходный код. Ключи — точные имена полей из JSON-отчёта main.py.
HYPERPARAM_DESCRIPTIONS: Dict[str, str] = {
    "phase": (
        "Этап двухфазного поиска гиперпараметров подпространственного метода: "
        "фаза 1 — перебор n_subclasses и growth_strategy; фаза 2 — донастройка "
        "фильтров статьи поверх уже найденной лучшей пары из фазы 1."
    ),
    "n_subclasses": (
        "Число подпространств (подклассов), на которые делится каждый класс "
        "при кластеризации. Больше подклассов — точнее описывается форма "
        "класса, но каждому достаётся меньше эталонных векторов и выше риск "
        "переобучения."
    ),
    "growth_strategy": (
        "Правило присоединения очередного вектора к подклассу на фазе роста: "
        "'default' — максимум показателя сопряжённости R(x, Y); 'master' — "
        "максимум отношения R к среднему R по остальным подклассам (эвристика "
        "оригинального ноутбука NB7)."
    ),
    "split_correlated_pairs": (
        "Включён ли экспериментальный метод из чернового варианта статьи (см. "
        "раздел «Теоретические основы»): True — эталонные векторы класса "
        "делятся на пары похожих, для кластеризации используется только "
        "половина."
    ),
    "correlated_pairs_subset": (
        "Какая из двух половин разбиения ('a' или 'b') используется — по "
        "построению алгоритма они равноценны; отличия результата между ними — "
        "следствие конкретного состава пар, а не отдельный содержательный "
        "параметр."
    ),
    "filter_dependent": (
        "Исключать ли из обучающей выборки почти линейно зависимые эталонные "
        "векторы перед кластеризацией (сверка с опубликованной статьёй, "
        "находка №3 — refactoring_plan.txt)."
    ),
    "filter_low_informativeness": (
        "Исключать ли малоинформативные (почти однородные по яркости) "
        "изображения перед кластеризацией (находка №5)."
    ),
    "pca_components": (
        "Число главных компонент (см. раздел «Метод главных компонент»), до "
        "которого сжимается вектор изображения перед классическим ML. Больше "
        "компонент — меньше потерь информации, но выше риск переобучения и "
        "медленнее работа."
    ),
    "C": (
        "Обратный коэффициент регуляризации: меньшие значения сильнее "
        "штрафуют сложные модели (жёстче ограничивают веса), большие — почти "
        "не ограничивают, приближая обучение к простой минимизации ошибки на "
        "train."
    ),
    "n_estimators": (
        "Число независимых деревьев решений в случайном лесу — их прогнозы "
        "усредняются голосованием. Больше деревьев обычно стабильнее, но "
        "дольше обучение."
    ),
    "max_depth": (
        "Максимальная глубина одного дерева решений — ограничивает число "
        "последовательных вопросов дерева; None — без ограничения (до "
        "полностью однородных листьев)."
    ),
    "n_neighbors": (
        "Число k ближайших обучающих примеров, среди которых kNN ищет "
        "большинство классов при классификации нового объекта."
    ),
    "architecture": (
        "Архитектура CNN (см. раздел «Свёрточные нейронные сети») — "
        "определяет число и размер свёрточных блоков, то есть ёмкость сети."
    ),
    "learning_rate": (
        "Скорость обучения — размер шага, на который оптимизатор Adam "
        "изменяет веса сети на каждой итерации градиентного спуска."
    ),
    "weight_decay": (
        "Коэффициент L2-регуляризации весов сети — штрафует большие веса, "
        "снижая риск переобучения (аналог параметра C у логрегрессии/SVM, но "
        "для нейросети)."
    ),
}


def load_report(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def add_hyperparam_legend(doc: Document, keys: List[str]) -> None:
    """Печатает список 'параметр — описание' для тех из ``keys``, что есть в
    HYPERPARAM_DESCRIPTIONS (ключи без описания — например, val_accuracy —
    пропускаются как самоочевидные)."""
    relevant = [k for k in keys if k in HYPERPARAM_DESCRIPTIONS]
    if not relevant:
        return
    p = doc.add_paragraph()
    p.add_run("Описание параметров таблицы:").bold = True
    for k in relevant:
        doc.add_paragraph(f"{k} — {HYPERPARAM_DESCRIPTIONS[k]}", style="List Bullet")


def _fmt_hyperparams(d: Dict[str, Any], skip: Tuple[str, ...] = ()) -> str:
    items = [f"{k}={v}" for k, v in d.items() if k not in skip]
    return ", ".join(items)


def _set_cell_bold(cell, bold: bool = True) -> None:
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.bold = bold


def add_table(
    doc: Document, headers: List[str], rows: List[List[str]],
    bold_row_indices: Optional[List[int]] = None,
) -> None:
    """Добавляет таблицу со стилем 'Table Grid' (встроен в любой шаблон
    python-docx, в отличие от именованных акцентных стилей, которые могут
    отсутствовать) — жирным выделяются строки из ``bold_row_indices``
    (например, лучшая по val_accuracy конфигурация)."""
    bold_row_indices = set(bold_row_indices or [])
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = header
        _set_cell_bold(cell, True)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
        if i in bold_row_indices:
            for cell in cells:
                _set_cell_bold(cell, True)
    doc.add_paragraph()


def render_line_chart(
    x: List[float], series: Dict[str, List[float]], title: str, xlabel: str, ylabel: str,
    log_y: bool = False,
) -> Path:
    """Рисует линейный график (по одной линии на подход) и сохраняет во
    временный PNG — возвращает путь для Document.add_picture()."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, values in series.items():
        ax.plot(
            x, values, marker="o", label=APPROACH_LABELS.get(key, key),
            color=APPROACH_COLORS.get(key),
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fd, tmp_str = tempfile.mkstemp(suffix=".png")
    os.close(fd)  # mkstemp оставляет дескриптор открытым — на Windows это
    # блокирует последующее удаление файла (unlink) чужим процессом/библиотекой
    tmp = Path(tmp_str)
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    return tmp


def render_formula(formula: str, fontsize: int = 20) -> Path:
    """Рендерит формулу через matplotlib mathtext (встроенный упрощённый
    LaTeX-движок matplotlib — НЕ требует установленного LaTeX/Word Equation
    Editor) в плотно обрезанный PNG. ``formula`` — без обрамляющих $...$
    (добавляются здесь).

    Почему не OMML (родные формулы Word): python-docx не имеет built-in
    поддержки уравнений, а сборка OMML XML вручную для произвольных формул
    существенно сложнее и хрупче, чем рендер в изображение — плюс картинка
    гарантированно выглядит одинаково независимо от версии Word читателя.
    """
    fig = plt.figure(figsize=(6, 1))
    fig.text(0.01, 0.5, f"${formula}$", fontsize=fontsize, va="center")

    fd, tmp_str = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    tmp = Path(tmp_str)
    fig.savefig(tmp, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return tmp


def add_formula(doc: Document, formula: str, tmp_files: List[Path], width: float = 3.2) -> None:
    """Вставляет формулу как центрированное изображение и регистрирует
    временный файл для последующей очистки в generate_report()."""
    path = render_formula(formula)
    tmp_files.append(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def add_title_page(doc: Document, title: str, reports: List[Tuple[str, Dict[str, Any]]]) -> None:
    heading = doc.add_heading(title, level=0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        f"Автоматически сгенерированный отчёт — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    run.italic = True
    run.font.size = Pt(11)

    doc.add_paragraph()
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.add_run("Датасеты в отчёте: " + ", ".join(name for name, _ in reports)).bold = True

    doc.add_paragraph(
        "Методология: подбор гиперпараметров через holdout-валидацию (тестовая "
        "выборка не участвует в подборе), единая для всех трёх сравниваемых "
        "подходов; кривая эффективности по объёму обучающих данных построена с "
        "гиперпараметрами, зафиксированными по итогам подбора (не переподбирается "
        "заново на каждый размер выборки). Источник метода — "
        "theory/Макет новой статьи.docx и опубликованная статья Korshikov & "
        "Fursov; полные детали реализации — refactoring_plan.txt."
    )
    doc.add_page_break()


def add_theory_section(doc: Document, tmp_files: List[Path]) -> None:
    """Теоретическое введение — общее для ВСЕХ датасетов отчёта, поэтому
    печатается один раз, сразу после титульной страницы. Рассчитано на
    читателя, НЕ знакомого ни с этой библиотекой, ни с ML вообще: каждая
    формула сопровождается словесным объяснением смысла величин.
    """
    add_heading(doc, "Теоретические основы сравниваемых методов", level=0)
    doc.add_paragraph(
        "В этом разделе объясняется математическая основа каждого из трёх "
        "сравниваемых подходов, в чём они принципиально различаются и что "
        "именно меняет в подпространственном методе экспериментальная "
        "доработка из чернового варианта статьи. Раздел не предполагает "
        "предварительного знакомства ни с библиотекой subspace_conjugacy, ни "
        "с методами машинного обучения — каждая формула сопровождается "
        "объяснением смысла входящих в неё величин."
    )

    # --------------------------------------------------------------------
    add_heading(doc, "1. Метод подпространственной сопряжённости (наш метод)", level=1)
    doc.add_paragraph(
        "Идея метода: каждое изображение представляется вектором признаков "
        "(яркость пикселей, развёрнутая построчно в один длинный список "
        "чисел). Для каждого класса патологии строится не одна, а несколько "
        "«эталонных плоскостей» (подпространств) — так, чтобы разные "
        "эталонные изображения одного класса, которые всё же заметно "
        "отличаются друг от друга, попадали в разные подпространства, а не "
        "смешивались в одном грубом усреднённом описании класса."
    )
    doc.add_paragraph(
        "Похожесть двух векторов x и y измеряется косинусным сходством — "
        "косинусом угла между ними:"
    )
    add_formula(doc, r"r(x_i,\,x_j) \;=\; \frac{x_i^{T} x_j}{\|x_i\|\cdot\|x_j\|}", tmp_files)
    doc.add_paragraph(
        "Значение r близко к 1, если векторы почти сонаправлены (изображения "
        "похожи), и близко к 0, если они почти перпендикулярны (изображения "
        "сильно различаются)."
    )
    doc.add_paragraph(
        "Ключевая величина метода — показатель сопряжённости R(x, Y): "
        "насколько хорошо вектор x объясняется (приближается) линейной "
        "комбинацией столбцов матрицы Y (базиса подпространства):"
    )
    add_formula(
        doc, r"R(x,\,Y) \;=\; \frac{x^{T} Y\,(Y^{T}Y)^{-1}\,Y^{T} x}{x^{T} x}", tmp_files, width=3.6,
    )
    doc.add_paragraph(
        "R принимает значения от 0 (x никак не объясняется подпространством "
        "Y) до 1 (x целиком лежит в Y). Геометрически x^T Y (Y^T Y)^{-1} Y^T x "
        "— это квадрат длины проекции x на подпространство Y, а деление на "
        "x^T x (квадрат длины самого x) нормирует результат в диапазон [0,1] "
        "независимо от масштаба вектора."
    )
    doc.add_paragraph(
        "Обучение (кластеризация внутри класса) происходит в три шага, без "
        "какой-либо итеративной оптимизации весов или случайной "
        "инициализации (в отличие от классического ML/CNN ниже):"
    )
    doc.add_paragraph(
        "Шаг A — поиск центров подклассов: сначала находится пара самых "
        "непохожих (по r) эталонных изображений класса — это первые два "
        "центра. Затем, пока центров меньше требуемого числа, к ним по "
        "очереди добавляется изображение с МИНИМАЛЬНЫМ показателем R "
        "относительно уже найденных центров (то есть максимально не похожее "
        "на всё, что уже накоплено) — так центры равномерно «расползаются» "
        "по разным направлениям формы класса.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Шаг B — наращивание подпространств: к каждому центру сначала "
        "присоединяется второй, самый непохожий (по r) вектор — получаются "
        "S двумерных подпространств. Затем все оставшиеся эталонные векторы "
        "по одному (строго по одному за шаг — присоединение сразу пачки "
        "векторов ухудшило бы качество последнего заполняемого "
        "подпространства) присоединяются к тому подпространству, с которым "
        "у них МАКСИМАЛЬНЫЙ показатель R.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Шаг C — классификация: новое изображение относится к тому классу, "
        "которому принадлежит подпространство (среди ВСЕХ подпространств "
        "ВСЕХ классов) с максимальным R(x, Y).",
        style="List Bullet",
    )

    add_heading(doc, "1.1. Метод из чернового варианта статьи — CorrelatedPairSplitter", level=2)
    doc.add_paragraph(
        "Черновик другой (неопубликованной) статьи авторов метода предлагает "
        "дополнительный шаг ПЕРЕД шагом A: вместо поиска непохожих пар "
        "(как в шаге A), здесь ищутся пары, наоборот, максимально ПОХОЖИХ "
        "друг на друга эталонных векторов — тем же косинусным сходством r, "
        "но с противоположным критерием отбора (argmax вместо argmin). "
        "Каждая такая пара делится между двумя половинами (A и B); для "
        "обучения используется только одна половина, вторая отбрасывается."
    )
    doc.add_paragraph(
        "Почему это может помочь, а не просто теряет данные: два почти "
        "одинаковых эталонных изображения несут для построения подпространств "
        "почти одну и ту же геометрическую информацию — одно из них избыточно. "
        "Отбрасывая «более похожего» из каждой пары, метод (а) вдвое сокращает "
        "M — число эталонных векторов класса, что снижает стоимость шага B "
        "(она растёт как O(M²·N), где N — размерность признаков, поэтому вдвое "
        "меньшее M ускоряет обучение примерно в 4 раза), и (б) на практике "
        "часто НЕ ухудшает, а иногда улучшает итоговую точность — оставшийся "
        "набор эталонов менее «захламлён» почти-дубликатами, из-за чего центры "
        "подклассов (шаг A) и рост подпространств (шаг B) меньше «увязают» в "
        "направлениях, которые и так уже хорошо покрыты."
    )

    # --------------------------------------------------------------------
    add_heading(doc, "2. Классические методы машинного обучения", level=1)
    doc.add_paragraph(
        "В отличие от подпространственного метода, классические модели ML "
        "используются здесь не на «сырых» 65536- или 784-мерных векторах "
        "пикселей, а после понижения размерности методом главных компонент "
        "(PCA) — иначе на малом числе обучающих примеров эти модели либо "
        "работали бы непрактично медленно, либо статистически неустойчиво."
    )
    add_heading(doc, "2.0. Метод главных компонент (PCA)", level=2)
    doc.add_paragraph(
        "PCA ищет направления (главные компоненты) в пространстве признаков, "
        "вдоль которых данные разбросаны сильнее всего, и проецирует данные "
        "на несколько первых таких направлений — это стандартный способ "
        "сжать изображение в компактный вектор, сохранив как можно больше "
        "содержательной изменчивости. Математически главные компоненты — это "
        "собственные векторы ковариационной матрицы данных Σ:"
    )
    add_formula(doc, r"\Sigma v \;=\; \lambda v", tmp_files, width=1.6)
    doc.add_paragraph(
        "где v — направление (главная компонента), а λ — доля дисперсии "
        "данных, приходящаяся на это направление; компоненты берутся по "
        "убыванию λ."
    )

    add_heading(doc, "2.1. Логистическая регрессия", level=2)
    doc.add_paragraph(
        "Линейная модель: вычисляет взвешенную сумму признаков объекта и "
        "преобразует её сигмоидой в число от 0 до 1 — оценку вероятности "
        "класса:"
    )
    add_formula(doc, r"P(y=1\,|\,x) \;=\; \frac{1}{1+e^{-(w^{T}x+b)}}", tmp_files, width=3.0)
    doc.add_paragraph(
        "Веса w и смещение b подбираются минимизацией логарифмической "
        "функции потерь (штраф тем больше, чем увереннее модель ошиблась) "
        "методом градиентного спуска. Гиперпараметр C (см. таблицы подбора "
        "ниже) управляет тем, насколько сильно веса штрафуются за величину "
        "— меньший C заставляет модель быть «проще»."
    )

    add_heading(doc, "2.2. Линейный SVM (метод опорных векторов)", level=2)
    doc.add_paragraph(
        "Ищет прямую (гиперплоскость) w^T x + b = 0, разделяющую классы с "
        "МАКСИМАЛЬНЫМ зазором (margin) между ближайшими к границе точками "
        "каждого класса — интуиция «чем шире полоса между классами, тем "
        "надёжнее граница»:"
    )
    add_formula(
        doc,
        r"\min_{w,b}\ \frac{1}{2}\|w\|^2 + C\sum_i \xi_i,\quad y_i(w^{T}x_i+b)\geq 1-\xi_i",
        tmp_files, width=4.2,
    )
    doc.add_paragraph(
        "ξᵢ — «штрафные» переменные, допускающие отдельным точкам нарушать "
        "границу (данные редко разделимы идеально); C — тот же по смыслу "
        "параметр, что и у логистической регрессии: компромисс между "
        "шириной зазора и числом допущенных нарушений."
    )

    add_heading(doc, "2.3. Случайный лес (Random Forest)", level=2)
    doc.add_paragraph(
        "Ансамбль из n_estimators независимых деревьев решений: каждое "
        "дерево обучается на случайной подвыборке данных и признаков и "
        "разбивает пространство признаков последовательностью вопросов "
        "«признак > порог?», выбирая на каждом шаге вопрос, максимально "
        "уменьшающий неоднородность классов в получившихся группах — "
        "стандартная мера неоднородности (критерий Джини):"
    )
    add_formula(doc, r"\mathrm{Gini} \;=\; 1 - \sum_k p_k^2", tmp_files, width=2.4)
    doc.add_paragraph(
        "где pₖ — доля объектов класса k в узле дерева. Итоговый прогноз "
        "леса — голосование всех деревьев. max_depth ограничивает "
        "максимальную длину цепочки вопросов одного дерева."
    )

    add_heading(doc, "2.4. Метод k ближайших соседей (kNN)", level=2)
    doc.add_paragraph(
        "Самый простой из четырёх классических методов: вообще не строит "
        "явную модель на этапе обучения (просто запоминает все обучающие "
        "примеры), а при классификации нового объекта находит k ближайших "
        "(по расстоянию в пространстве после PCA) обучающих примеров и "
        "присваивает объекту класс большинства среди них:"
    )
    add_formula(doc, r"\hat{y} \;=\; \mathrm{mode}\{\,y_i : x_i \in N_k(x)\,\}", tmp_files, width=3.0)
    doc.add_paragraph(
        "где N_k(x) — множество k ближайших к x обучающих объектов. "
        "Единственный гиперпараметр здесь (помимо PCA) — само k "
        "(n_neighbors): меньшее k чувствительнее к шуму, большее — сглаживает "
        "границы между классами."
    )

    # --------------------------------------------------------------------
    add_heading(doc, "3. Свёрточные нейронные сети (CNN)", level=1)
    doc.add_paragraph(
        "В отличие от классического ML и подпространственного метода, CNN "
        "работает НЕ с вектором пикселей, а с изображением как двумерной "
        "сеткой, и сама обучается извлекать полезные признаки (а не "
        "получает их готовыми через PCA/векторизацию). Базовая операция — "
        "свёртка: скользящее окно (ядро) K умножается поэлементно на "
        "фрагмент изображения I и суммируется, давая одно число на каждое "
        "положение окна:"
    )
    add_formula(
        doc, r"(I*K)(i,j) \;=\; \sum_{m,n} I(i+m,\,j+n)\,K(m,n)", tmp_files, width=4.0,
    )
    doc.add_paragraph(
        "Значения ядра K подбираются обучением — сеть сама учится, какие "
        "локальные узоры (границы, пятна, текстуры) полезно искать на "
        "изображении. После свёртки применяется нелинейность ReLU "
        "(max(0, x) — отсекает отрицательные значения) и уменьшение "
        "разрешения (MaxPool — берёт максимум по маленькому окну, огрубляя "
        "картинку и снижая объём вычислений). После нескольких таких "
        "блоков «свёртка → ReLU → MaxPool» результат разворачивается в "
        "вектор и подаётся на полносвязный слой, а затем — на softmax, "
        "превращающий выходы сети в вероятности классов:"
    )
    add_formula(doc, r"p_c \;=\; \frac{e^{z_c}}{\sum_k e^{z_k}}", tmp_files, width=2.4)
    doc.add_paragraph(
        "Обучение — минимизация кросс-энтропийной функции потерь (штраф за "
        "низкую предсказанную вероятность истинного класса) методом "
        "обратного распространения ошибки и оптимизатором Adam, "
        "обновляющим каждый вес сети на каждом шаге по правилу:"
    )
    add_formula(
        doc,
        r"\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}",
        tmp_files, width=3.6,
    )
    doc.add_paragraph(
        "где η — скорость обучения (learning_rate, см. таблицы ниже), а "
        "m̂ₜ/v̂ₜ — сглаженные по времени оценки среднего и дисперсии градиента "
        "(это отличает Adam от простого градиентного спуска — шаг "
        "автоматически подстраивается под «шумность» градиента у каждого "
        "конкретного веса). weight_decay добавляет к этому правилу "
        "дополнительный штраф, пропорциональный текущему значению веса, "
        "снижая риск переобучения — сеть в этом эксперименте обучается "
        "CNN_EPOCHS проходов (эпох) по всей обучающей выборке."
    )

    # --------------------------------------------------------------------
    add_heading(doc, "4. Ключевые различия подходов", level=1)
    headers = ["", "Сопряжённость", "Классический ML", "CNN"]
    rows = [
        [
            "Как устроена модель",
            "Геометрический объект (набор подпространств), строится напрямую",
            "Параметрическая формула (веса) или явное правило (дерево/соседи)",
            "Глубокая параметрическая сеть с автоматическим извлечением признаков",
        ],
        [
            "Как «обучается»",
            "Прямое пошаговое построение (Фазы A/B), БЕЗ итеративной оптимизации",
            "Минимизация функции потерь (LogReg/SVM) или прямое построение (RF/kNN)",
            "Итеративная минимизация функции потерь градиентным спуском (Adam)",
        ],
        [
            "Случайность/детерминизм",
            "Полностью детерминирован при фиксированных данных",
            "В основном детерминирован (кроме случайных подвыборок RF)",
            "Случайная инициализация весов + случайный порядок батчей",
        ],
        [
            "Работа с признаками",
            "Сырые векторы пикселей, без понижения размерности",
            "Требует понижения размерности (PCA) перед обучением",
            "Сама извлекает признаки из изображения (свёртки)",
        ],
        [
            "Что определяет стоимость обучения",
            "O(M²·N): число эталонных векторов M и размерность признаков N",
            "Число компонент PCA и размер обучающей выборки",
            "Число эпох, размер сети и объём обучающей выборки",
        ],
        [
            "Добавление нового класса/примера",
            "Новые эталоны просто добавляются в матрицу класса",
            "Требуется переобучение модели целиком",
            "Требуется переобучение (или дообучение) сети целиком",
        ],
    ]
    add_table(doc, headers, rows)
    doc.add_paragraph(
        "Практический вывод из этих различий: подпространственный метод — "
        "единственный из трёх, который в принципе не выполняет итеративную "
        "оптимизацию параметров и не использует случайность — его поведение "
        "полностью предопределено входными данными. Это объясняет, почему "
        "его стоимость обучения зависит от M и N (а не от числа эпох, как у "
        "CNN) и почему метод из чернового варианта статьи (раздел 1.1), "
        "сокращающий M вдвое, даёт почти четырёхкратное ускорение — этот "
        "рычаг специфичен для подпространственного метода и не имеет прямого "
        "аналога у классического ML/CNN."
    )
    doc.add_page_break()


def add_methodology_section(doc: Document, report: Dict[str, Any]) -> None:
    add_heading(doc, "1. Методология и датасет", level=1)
    ds = report["dataset"]
    tuning = report["tuning"]

    doc.add_paragraph(f"Датасет: {ds.get('name', '—')}")
    doc.add_paragraph(f"Классы ({len(ds.get('classes', []))}): {', '.join(map(str, ds.get('classes', [])))}")
    doc.add_paragraph(f"Изображений на класс: {ds.get('n_per_class', '—')}")
    doc.add_paragraph(f"Доля теста: {ds.get('test_fraction', '—')}")
    doc.add_paragraph(
        f"Доля holdout-валидации при подборе гиперпараметров: {tuning.get('val_fraction')} "
        f"(random_seed={tuning.get('random_seed')})"
    )
    doc.add_paragraph(
        "Тестовая выборка используется РОВНО ОДИН РАЗ — для финальной оценки "
        "уже зафиксированных по итогам подбора гиперпараметров конфигураций "
        "(раздел 3) и для кривой эффективности (раздел 4). Она не участвует "
        "ни в одном из шагов подбора гиперпараметров (раздел 2)."
    )


def add_subspace_tuning_section(doc: Document, report: Dict[str, Any]) -> None:
    add_heading(doc, "2.1. Подпространственная сопряжённость", level=2)
    tuning = report["tuning"]
    candidates = tuning["subspace_candidates"]
    best_params = tuning["best_subspace_params"]

    doc.add_paragraph(
        f"Всего опробовано конфигураций: {len(candidates)} (двухфазный поиск — "
        "фаза 1: n_subclasses x growth_strategy; фаза 2: донастройка фильтров "
        "статьи на лучшей паре из фазы 1 — см. refactoring_plan.txt, раздел 11)."
    )
    doc.add_paragraph("Лучшая конфигурация: " + _fmt_hyperparams(best_params))
    add_hyperparam_legend(doc, [
        "phase", "n_subclasses", "growth_strategy", "split_correlated_pairs",
        "correlated_pairs_subset", "filter_dependent", "filter_low_informativeness",
    ])

    sorted_candidates = sorted(candidates, key=lambda c: c["val_accuracy"], reverse=True)
    headers = [
        "Фаза", "n_subclasses", "growth_strategy", "split_correlated_pairs",
        "correlated_pairs_subset", "filter_dependent", "filter_low_informativeness",
        "val_accuracy",
    ]
    rows = []
    for c in sorted_candidates:
        p = c["params"]
        rows.append([
            c.get("phase", "—"), p.get("n_subclasses"), p.get("growth_strategy"),
            p.get("split_correlated_pairs"), p.get("correlated_pairs_subset"),
            p.get("filter_dependent"), p.get("filter_low_informativeness"),
            f"{c['val_accuracy']:.3f}",
        ])
    add_table(doc, headers, rows, bold_row_indices=[0])


def add_classical_ml_tuning_section(doc: Document, report: Dict[str, Any]) -> None:
    add_heading(doc, "2.2. Классический ML", level=2)
    tuning = report["tuning"]
    by_model = tuning["classical_ml_candidates_by_model"]
    best_name = tuning["best_classical_name"]

    doc.add_paragraph(f"Лучшая модель по итогам подбора: {best_name}.")
    all_param_keys = sorted({
        k for info in by_model.values() for k in info.get("hyperparams", {}).keys()
    })
    add_hyperparam_legend(doc, all_param_keys)

    for name, info in by_model.items():
        all_candidates = info.get("all_candidates", [info])
        heading_text = f"{name} — перебрано {len(all_candidates)} конфигураций"
        if name == best_name:
            heading_text += " (ЛУЧШАЯ МОДЕЛЬ)"
        add_heading(doc, heading_text, level=3)

        sorted_candidates = sorted(all_candidates, key=lambda c: c["val_accuracy"], reverse=True)
        top = sorted_candidates[:10]
        param_keys = sorted({k for c in top for k in c["hyperparams"].keys()})
        headers = param_keys + ["val_accuracy"]
        rows = [
            [c["hyperparams"].get(k, "—") for k in param_keys] + [f"{c['val_accuracy']:.3f}"]
            for c in top
        ]
        if len(sorted_candidates) > 10:
            doc.add_paragraph(f"(показаны 10 лучших из {len(sorted_candidates)} — полный список в JSON-отчёте)")
        add_table(doc, headers, rows, bold_row_indices=[0])


def add_cnn_tuning_section(doc: Document, report: Dict[str, Any]) -> None:
    add_heading(doc, "2.3. Свёрточные нейросети (CNN)", level=2)
    tuning = report["tuning"]
    candidates = tuning["cnn_candidates"]
    best = tuning["best_cnn"]

    doc.add_paragraph(
        f"Всего опробовано конфигураций: {len(candidates)} (архитектура x "
        "learning_rate x weight_decay, сокращённое число эпох на этапе подбора)."
    )
    doc.add_paragraph(
        "Лучшая конфигурация: " + _fmt_hyperparams(
            {k: v for k, v in best.items() if k != "val_accuracy"}
        )
    )
    add_hyperparam_legend(doc, ["architecture", "learning_rate", "weight_decay"])

    sorted_candidates = sorted(candidates, key=lambda c: c["val_accuracy"], reverse=True)
    headers = ["architecture", "learning_rate", "weight_decay", "val_accuracy"]
    rows = [
        [c["architecture"], c["learning_rate"], c.get("weight_decay", "—"), f"{c['val_accuracy']:.3f}"]
        for c in sorted_candidates
    ]
    add_table(doc, headers, rows, bold_row_indices=[0])


def add_full_data_section(doc: Document, report: Dict[str, Any]) -> None:
    add_heading(doc, "3. Финальная оценка на полном объёме данных", level=1)
    doc.add_paragraph(
        "Тюнингованные (по итогам раздела 2) конфигурации переобучены на "
        "ПОЛНОМ train и впервые оценены на тестовой выборке."
    )
    fd = report["full_data_evaluation"]
    headers = ["Подход", "Accuracy", "Время (с)"]
    rows = []
    best_acc = max(fd["accuracy"].values())
    bold_idx = []
    for i, key in enumerate(("subspace", "classical_ml", "cnn")):
        rows.append([
            APPROACH_LABELS[key], f"{fd['accuracy'][key]:.3f}", f"{fd['time_seconds'][key]:.1f}",
        ])
        if abs(fd["accuracy"][key] - best_acc) < 1e-9:
            bold_idx.append(i)
    add_table(doc, headers, rows, bold_row_indices=bold_idx)


def add_data_efficiency_section(doc: Document, report: Dict[str, Any], tmp_files: List[Path]) -> None:
    add_heading(doc, "4. Кривая эффективности по объёму обучающих данных", level=1)
    doc.add_paragraph(
        "Прямая проверка тезиса theory/ Introduction о работоспособности метода "
        "сопряжённости на малых обучающих выборках: те же три (уже тюнингованные) "
        "конфигурации переобучаются на срезах train нарастающего размера и "
        "оцениваются на одном и том же полном тесте."
    )
    sweep = report["data_efficiency_sweep"]
    sizes = [r["n_train_per_class"] for r in sweep]

    headers = ["N/класс"] + [
        f"{APPROACH_LABELS[k]} — accuracy" for k in ("subspace", "classical_ml", "cnn")
    ] + [f"{APPROACH_LABELS[k]} — время (с)" for k in ("subspace", "classical_ml", "cnn")]
    rows = []
    for r in sweep:
        rows.append([
            r["n_train_per_class"],
            f"{r['subspace']['mean_accuracy']:.3f}",
            f"{r['classical_ml']['mean_accuracy']:.3f}",
            f"{r['cnn']['mean_accuracy']:.3f}",
            f"{r['subspace']['fit_time_seconds']:.2f}",
            f"{r['classical_ml']['fit_time_seconds']:.2f}",
            f"{r['cnn']['training_time_seconds']:.2f}",
        ])
    add_table(doc, headers, rows)

    acc_series = {
        "subspace": [r["subspace"]["mean_accuracy"] for r in sweep],
        "classical_ml": [r["classical_ml"]["mean_accuracy"] for r in sweep],
        "cnn": [r["cnn"]["mean_accuracy"] for r in sweep],
    }
    time_series = {
        "subspace": [r["subspace"]["fit_time_seconds"] for r in sweep],
        "classical_ml": [r["classical_ml"]["fit_time_seconds"] for r in sweep],
        "cnn": [r["cnn"]["training_time_seconds"] for r in sweep],
    }

    acc_chart = render_line_chart(
        sizes, acc_series, "Accuracy в зависимости от объёма обучающих данных",
        "Изображений на класс", "Mean accuracy",
    )
    time_chart = render_line_chart(
        sizes, time_series, "Время обучения в зависимости от объёма данных",
        "Изображений на класс", "Время (с, логарифмическая шкала)", log_y=True,
    )
    tmp_files.extend([acc_chart, time_chart])
    doc.add_picture(str(acc_chart), width=Inches(6))
    doc.add_picture(str(time_chart), width=Inches(6))


def add_verdict_section(doc: Document, report: Dict[str, Any]) -> None:
    """Выводы вычисляются ИЗ ФАКТИЧЕСКИХ чисел отчёта — не захардкожены, чтобы
    при повторном прогоне с другой сеткой/данными текст автоматически
    отражал реальный результат, а не переиспользовал прошлый нарратив."""
    add_heading(doc, "5. Выводы", level=1)
    sweep = report["data_efficiency_sweep"]
    fd = report["full_data_evaluation"]

    doc.add_paragraph(
        "Ниже — автоматически вычисленная (не переиспользованная из прошлых "
        "прогонов) сводка по фактическим числам этого отчёта."
    )

    win_sizes, tie_sizes, lose_sizes = [], [], []
    for r in sweep:
        accs = {k: r[k]["mean_accuracy"] for k in ("subspace", "classical_ml", "cnn")}
        best_key = max(accs, key=accs.get)
        sub_acc = accs["subspace"]
        best_other = max(v for k, v in accs.items() if k != "subspace")
        if best_key == "subspace" and sub_acc - best_other > TIE_EPSILON:
            win_sizes.append(r["n_train_per_class"])
        elif abs(sub_acc - best_other) <= TIE_EPSILON:
            tie_sizes.append(r["n_train_per_class"])
        else:
            lose_sizes.append(r["n_train_per_class"])

    def _fmt_sizes(sizes: List[int]) -> str:
        return ", ".join(str(s) for s in sizes) if sizes else "ни при одном из проверенных объёмов"

    doc.add_paragraph(
        f"Метод сопряжённости ЛУЧШЕ обоих конкурентов (разница accuracy > "
        f"{TIE_EPSILON}) при объёме на класс: {_fmt_sizes(win_sizes)}.",
        style="List Bullet",
    )
    doc.add_paragraph(
        f"Метод сопряжённости НАРАВНЕ (разница accuracy ≤ {TIE_EPSILON}) при "
        f"объёме на класс: {_fmt_sizes(tie_sizes)}.",
        style="List Bullet",
    )
    doc.add_paragraph(
        f"Метод сопряжённости УСТУПАЕТ при объёме на класс: {_fmt_sizes(lose_sizes)}.",
        style="List Bullet",
    )

    full_accs = fd["accuracy"]
    full_best = max(full_accs, key=full_accs.get)
    doc.add_paragraph(
        f"На полном объёме данных лучший результат показал: "
        f"{APPROACH_LABELS[full_best]} (accuracy={full_accs[full_best]:.3f}).",
        style="List Bullet",
    )

    fastest_at_full = min(fd["time_seconds"], key=fd["time_seconds"].get)
    doc.add_paragraph(
        f"На полном объёме данных быстрее всех обучился: "
        f"{APPROACH_LABELS[fastest_at_full]} ({fd['time_seconds'][fastest_at_full]:.1f}с).",
        style="List Bullet",
    )

    smallest, largest = sweep[0], sweep[-1]
    for key in ("classical_ml", "cnn"):
        gap_small = smallest["subspace"]["mean_accuracy"] - smallest[key]["mean_accuracy"]
        gap_large = largest["subspace"]["mean_accuracy"] - largest[key]["mean_accuracy"]
        trend = "сокращается" if gap_small > gap_large else "растёт" if gap_small < gap_large else "не меняется"
        doc.add_paragraph(
            f"Разрыв accuracy (сопряжённость − {APPROACH_LABELS[key]}) при уменьшении "
            f"объёма данных с {largest['n_train_per_class']} до {smallest['n_train_per_class']} "
            f"на класс: {trend} ({gap_large:+.3f} → {gap_small:+.3f}).",
            style="List Bullet",
        )


def add_cross_dataset_section(doc: Document, reports: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Сводная таблица, если отчётов больше одного (например, МРТ + MNIST) —
    сравнивает позиционирование метода между датаsetами."""
    if len(reports) < 2:
        return
    add_heading(doc, "6. Сравнение между датасетами", level=1)
    headers = ["Датасет", "Accuracy (полные данные)", "Лучший подход (полные данные)", "Наравне/лучше при N/класс"]
    rows = []
    for name, report in reports:
        fd = report["full_data_evaluation"]
        full_best = max(fd["accuracy"], key=fd["accuracy"].get)
        sweep = report["data_efficiency_sweep"]
        competitive_sizes = []
        for r in sweep:
            accs = {k: r[k]["mean_accuracy"] for k in ("subspace", "classical_ml", "cnn")}
            best_other = max(v for k, v in accs.items() if k != "subspace")
            if accs["subspace"] >= best_other - TIE_EPSILON:
                competitive_sizes.append(r["n_train_per_class"])
        rows.append([
            name, f"{fd['accuracy']['subspace']:.3f}", APPROACH_LABELS[full_best],
            ", ".join(map(str, competitive_sizes)) if competitive_sizes else "—",
        ])
    add_table(doc, headers, rows)


def generate_report(report_paths: List[Path], output_path: Path, title: str) -> None:
    reports: List[Tuple[str, Dict[str, Any]]] = []
    for path in report_paths:
        data = load_report(path)
        name = data.get("dataset", {}).get("name") or path.stem
        reports.append((name, data))

    doc = Document()
    tmp_files: List[Path] = []

    add_title_page(doc, title, reports)
    add_theory_section(doc, tmp_files)

    for name, report in reports:
        add_heading(doc, name, level=0)
        add_methodology_section(doc, report)
        add_heading(doc, "2. Подбор гиперпараметров", level=1)
        add_subspace_tuning_section(doc, report)
        add_classical_ml_tuning_section(doc, report)
        add_cnn_tuning_section(doc, report)
        add_full_data_section(doc, report)
        add_data_efficiency_section(doc, report, tmp_files)
        add_verdict_section(doc, report)
        doc.add_page_break()

    add_cross_dataset_section(doc, reports)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)

    for tmp in tmp_files:
        tmp.unlink(missing_ok=True)

    print(f"Отчёт сохранён: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерирует подробный Word-отчёт по JSON-результатам main.py --experiment tuned."
    )
    parser.add_argument(
        "--report", action="append", required=True, dest="reports",
        help="Путь к JSON-отчёту (можно указать несколько раз для нескольких датасетов).",
    )
    parser.add_argument(
        "--output", default="artifacts/tuned_comparison_report.docx",
        help="Путь для сохранения .docx (по умолчанию artifacts/tuned_comparison_report.docx).",
    )
    parser.add_argument(
        "--title", default="Подпространственная сопряжённость vs классический ML vs CNN",
        help="Заголовок отчёта.",
    )
    args = parser.parse_args()

    generate_report(
        [Path(p) for p in args.reports], Path(args.output), args.title,
    )


if __name__ == "__main__":
    main()
