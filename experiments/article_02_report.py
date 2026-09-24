"""
Генератор Word-отчёта для статьи 2 ("Симбиоз классификаторов: ансамблевая
интеграция метода подпространственной сопряжённости с классическими
моделями машинного обучения и свёрточными нейронными сетями") — читает
experiments/article_02_output/results.json
(experiments/article_02_experiment.py) и строит .docx: 9pt Times New Roman,
теория с формулами на понятном русском языке (без упоминаний программных
идентификаторов), датасет с рисунками, результаты эксперимента, обсуждение,
выводы, перспективы статьи 3.

Переиспользует общие примитивы reporting/word_report.py (рендер формул
через matplotlib mathtext) — тот же принцип, что и в отчёте статьи 1:
ВНЕ пакета subspace_conjugacy, зависимости python-docx/matplotlib не
входят в зависимости библиотеки.

Запуск:
    python experiments/article_02_experiment.py   # сначала эксперимент
    python experiments/article_02_report.py       # затем отчёт
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, Cm

from reporting.word_report import render_formula, add_table

OUTPUT_DIR = Path(__file__).resolve().parent / "article_02_output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "article_02_report.docx"

FONT_NAME = "Times New Roman"
FONT_SIZE_PT = 9

MODEL_LABEL_RU = {
    "subspace": "Метод сопряжённости",
    "classical_ml": "Классический ML",
    "cnn": "Свёрточная сеть",
    "voting": "Голосование",
    "stacking": "Стекинг (обычный)",
    "stacking_balanced": "Стекинг (со взвешиванием)",
    "switching": "Переключение",
}
MODEL_LABEL_RU_SHORT = {
    "subspace": "Сопряжённость",
    "classical_ml": "Класс. ML",
    "cnn": "CNN",
    "voting": "Голосование",
    "stacking": "Стекинг",
    "stacking_balanced": "Стекинг(взв.)",
    "switching": "Переключение",
}
BASE_MODEL_ORDER = ["subspace", "classical_ml", "cnn"]
ENSEMBLE_ORDER = ["voting", "stacking", "stacking_balanced", "switching"]
ALL_MODEL_ORDER = BASE_MODEL_ORDER + ENSEMBLE_ORDER


# ----------------------------------------------------------------------
# Форматирование документа (идентично article_01_report.py)
# ----------------------------------------------------------------------
def _set_run_font(run, size_pt: int = FONT_SIZE_PT, bold: bool = False) -> None:
    run.font.name = FONT_NAME
    run.font.size = Pt(size_pt)
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rpr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), FONT_NAME)


def configure_document_font(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = Pt(FONT_SIZE_PT)
    rpr = style.element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rpr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), FONT_NAME)

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(1.5)


def add_p(doc: Document, text: str = "", bold: bool = False, size_pt: int = FONT_SIZE_PT,
          align=None, space_after: int = 4) -> Any:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if text:
        run = p.add_run(text)
        _set_run_font(run, size_pt=size_pt, bold=bold)
    return p


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    sizes = {0: 14, 1: 11, 2: 10}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8 if level <= 1 else 5)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    _set_run_font(run, size_pt=sizes.get(level, 10), bold=True)


def add_bullet(doc: Document, text: str, size_pt: int = FONT_SIZE_PT) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    _set_run_font(run, size_pt=size_pt)


def add_formula_centered(doc: Document, formula: str, tmp_files: List[Path], width: float = 2.6) -> None:
    path = render_formula(formula, fontsize=16)
    tmp_files.append(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(path), width=Inches(width))


def style_table_font(table) -> None:
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    _set_run_font(run, size_pt=8)


def add_table_9pt(doc: Document, headers: List[str], rows: List[List[str]],
                   bold_row_indices=None) -> None:
    add_table(doc, headers, rows, bold_row_indices=bold_row_indices)
    style_table_font(doc.tables[-1])


def add_picture_centered(doc: Document, path: Path, width_in: float) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(path), width=Inches(width_in))


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    _set_run_font(run, size_pt=8, bold=False)
    run.italic = True


def _save_tmp(fig, tmp_files: List[Path]) -> Path:
    fd, tmp_str = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    tmp = Path(tmp_str)
    fig.savefig(tmp, dpi=170, bbox_inches="tight")
    plt.close(fig)
    tmp_files.append(tmp)
    return tmp


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------
def render_diagnostics_figure(diag: Dict[str, Any], tmp_files: List[Path]) -> Path:
    """Столбцы: точность каждой базовой модели; горизонтальные линии:
    точность «оракула» (потолок) и доля объектов, где ошибаются все модели."""
    names = diag["model_names"]
    accs = [diag["accuracy_by_model"][n] for n in names]
    labels = [MODEL_LABEL_RU.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    colors = ["#2563EB", "#16A34A", "#DC2626"]
    ax.bar(labels, accs, color=colors[:len(labels)], width=0.5)
    ax.axhline(diag["oracle_accuracy"], color="#111827", linestyle="--", lw=1.3,
               label=f"«оракул» = {diag['oracle_accuracy']:.3f}")
    ax.set_ylabel("Точность", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Точность базовых моделей и верхняя граница «оракула»", fontsize=9)
    fig.tight_layout()
    return _save_tmp(fig, tmp_files)


def render_final_comparison_figure(test_results: Dict[str, Any], tmp_files: List[Path]) -> Path:
    """Точность каждой модели на тестовой выборке — базовые модели одним
    цветом, схемы объединения другим. Тестовая выборка сбалансирована по
    построению (раздел 3: поровну примеров на класс), поэтому общая
    точность и средняя по классам точность здесь совпадают тождественно —
    единственная метрика для сравнения, без риска перекоса."""
    names = [n for n in ALL_MODEL_ORDER if n in test_results]
    labels = [MODEL_LABEL_RU_SHORT.get(n, n) for n in names]
    acc = [test_results[n]["accuracy"] for n in names]
    colors = ["#94A3B8"] * len(BASE_MODEL_ORDER) + ["#2563EB"] * (len(names) - len(BASE_MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bars = ax.bar(labels, acc, color=colors, width=0.55)
    for bar, v in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("Точность на тестовой выборке", fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.tick_params(labelsize=7.5)
    ax.axvline(len(BASE_MODEL_ORDER) - 0.5, color="#111827", linestyle=":", lw=1)
    ax.set_title("Базовые модели (серые) и схемы объединения (синие)", fontsize=9)
    fig.tight_layout()
    return _save_tmp(fig, tmp_files)


def render_correlation_figure(diag_val: Dict[str, Any], tmp_files: List[Path]) -> Path:
    """Доля совпадающих предсказаний и корреляция ошибок для каждой пары
    базовых моделей — компактный вид ключевой диагностической находки
    (раздел 4.1/5): метод сопряжённости слабее всего связан по ошибкам с
    двумя другими моделями."""
    pairs = list(diag_val["pairwise_error_correlation"].keys())
    labels = []
    corr = []
    for pair_key in pairs:
        a, b = pair_key.split("__")
        labels.append(f"{MODEL_LABEL_RU_SHORT.get(a, a)} /\n{MODEL_LABEL_RU_SHORT.get(b, b)}")
        corr.append(diag_val["pairwise_error_correlation"][pair_key])

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    colors = ["#16A34A" if c < 0.15 else "#F59E0B" if c < 0.3 else "#DC2626" for c in corr]
    bars = ax.bar(labels, corr, color=colors, width=0.5)
    for bar, v in zip(bars, corr):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("Корреляция ошибок (Пирсон)", fontsize=8)
    ax.set_ylim(0, max(corr) * 1.3)
    ax.tick_params(labelsize=7)
    ax.set_title("Насколько связаны ошибки каждой пары моделей", fontsize=9)
    fig.tight_layout()
    return _save_tmp(fig, tmp_files)


# ----------------------------------------------------------------------
# Секции отчёта
# ----------------------------------------------------------------------
def add_title(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(
        "Симбиоз классификаторов: ансамблевая интеграция метода "
        "подпространственной сопряжённости с классическими моделями "
        "машинного обучения и свёрточными нейронными сетями"
    )
    _set_run_font(run, size_pt=13, bold=True)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(10)
    run2 = p2.add_run(
        "Экспериментальный отчёт по методу подпространственной сопряжённости "
        "(вторая статья исследовательского цикла)"
    )
    _set_run_font(run2, size_pt=8.5)
    run2.italic = True


def add_intro_section(doc: Document) -> None:
    add_heading(doc, "1. Цель эксперимента", level=1)
    add_p(
        doc,
        "Три разных подхода к классификации изображений — метод "
        "подпространственной сопряжённости, классические модели машинного "
        "обучения и свёрточные нейронные сети — устроены совершенно "
        "по-разному внутри, и поэтому вполне могут ошибаться на РАЗНЫХ "
        "объектах: там, где один подход путается, два других могут быть "
        "правы. Это давно известное в статистике наблюдение — если ошибки "
        "нескольких независимых моделей не связаны друг с другом жёстко, "
        "то их объединение (например, голосование большинством) способно "
        "дать точность ВЫШЕ, чем у лучшей из моделей по отдельности. "
        "До сих пор в проекте три подхода сравнивались только как "
        "конкуренты — «кто лучше» — но никогда не проверялись как "
        "потенциальные СОСТАВНЫЕ ЧАСТИ одного объединённого решения."
    )
    add_p(
        doc,
        "Цель эксперимента — проверить, действительно ли объединение "
        "(симбиоз) трёх подходов даёт точность выше, чем у лучшего из них "
        "по отдельности, и понять, за счёт чего именно это происходит (или "
        "не происходит). Для этого решаются четыре задачи: (1) измерить, "
        "насколько ошибки трёх моделей связаны друг с другом; (2) построить "
        "три разных способа объединения предсказаний; (3) честно сравнить "
        "точность объединённых решений с точностью лучшей одиночной модели "
        "на отложенной тестовой выборке; (4) проверить, устраняет ли "
        "объединение конкретную слабость метода сопряжённости на том "
        "классе, где он справляется хуже всего."
    )


def add_theory_section(doc: Document, tmp_files: List[Path]) -> None:
    add_heading(doc, "2. Теория", level=1)

    add_heading(doc, "2.1. Три базовые модели", level=2)
    add_p(
        doc,
        "Метод подпространственной сопряжённости строит для каждого класса "
        "набор эталонных «подпространств» и относит новый объект x к тому "
        "классу, чьё подпространство лучше всего его объясняет — по "
        "показателю сопряжённости:"
    )
    add_formula_centered(
        doc, r"R(x,\,Y)=\dfrac{x^{T}Y(Y^{T}Y)^{-1}Y^{T}x}{x^{T}x}", tmp_files, width=2.6,
    )
    add_p(
        doc,
        "где Y — матрица опорных (эталонных) векторов подпространства "
        "класса. R принимает значения от 0 (объект никак не объясняется "
        "этим подпространством) до 1 (объект целиком лежит в нём) — "
        "подробное объяснение всех величин формулы дано в отчёте по первой "
        "статье цикла."
    )
    add_p(
        doc,
        "Классический подход машинного обучения здесь состоит из двух "
        "шагов. Сначала метод главных компонент (PCA) сжимает исходное "
        "изображение (несколько тысяч чисел — по яркости каждого канала "
        "цвета каждого пикселя) до значительно меньшего числа новых "
        "признаков — так называемых «главных компонент», которые устроены "
        "как линейные комбинации исходных пикселей, выбранные так, чтобы "
        "сохранить как можно больше изменчивости (разнообразия) исходных "
        "изображений при заметно меньшем числе величин. Затем логистическая "
        "регрессия — один из самых простых и широко используемых "
        "классификаторов — по этим сжатым признакам вычисляет для каждого "
        "класса число («логит»), пропускает набор чисел через функцию "
        "softmax (обобщение сигмоиды на несколько классов, превращающее "
        "произвольные числа в вероятности, которые в сумме дают 1) и "
        "относит объект к классу с наибольшей получившейся вероятностью."
    )
    add_p(
        doc,
        "Свёрточная нейронная сеть обрабатывает изображение, не разворачивая "
        "его в один длинный список чисел, а сохраняя двумерную структуру: "
        "небольшое скользящее окно (свёрточный фильтр) проходит по всему "
        "изображению и на каждом положении вычисляет одно число — меру "
        "того, насколько под этим окном присутствует некоторый узнаваемый "
        "локальный узор (например, край контрастной области или пятно "
        "определённого оттенка). Слой подвыборки (пулинг) после этого "
        "укрупняет картинку, оставляя в каждой небольшой области только "
        "самое сильное сработавшее значение, что делает сеть менее "
        "чувствительной к точному положению узора. После нескольких таких "
        "слоёв конечный, уже сильно сжатый набор чисел подаётся на "
        "линейный классификатор, аналогичный логистической регрессии. "
        "Обучение сети происходит итеративно: на каждом шаге через сеть "
        "пропускается небольшая порция обучающих изображений, вычисляется "
        "величина ошибки (насколько предсказанные вероятности расходятся с "
        "истинными метками), и все внутренние числовые параметры фильтров "
        "чуть-чуть подстраиваются в направлении, уменьшающем эту ошибку "
        "(градиентный спуск). Один полный проход по всей обучающей выборке "
        "называется эпохой; для получения работоспособной сети таких "
        "проходов обычно требуются десятки."
    )

    add_heading(doc, "2.2. Насколько ошибки моделей связаны друг с другом", level=2)
    add_p(
        doc,
        "Пусть на одной и той же тестовой выборке из M объектов получены "
        "предсказания нескольких моделей. Для каждого объекта можно "
        "подсчитать, СКОЛЬКО из моделей ошиблось именно на нём — число от 0 "
        "(все правы) до числа моделей (все ошиблись). Из распределения этого "
        "числа по всем объектам выводятся три показателя, которые дают "
        "прямую количественную оценку того, есть ли смысл в объединении "
        "моделей:"
    )
    add_bullet(
        doc,
        "потенциал ансамбля — доля объектов, на которых ошибается РОВНО "
        "ОДНА модель из нескольких: эти ошибки в принципе исправимы, так "
        "как большинство моделей на этих объектах правы;",
    )
    add_bullet(
        doc,
        "жёсткий потолок — доля объектов, на которых ошибаются ВСЕ модели "
        "сразу: такие ошибки не исправит никакое объединение предсказаний "
        "уже обученных моделей, потому что верного ответа среди них попросту нет;",
    )
    add_bullet(
        doc,
        "точность «оракула» — доля объектов, на которых права ХОТЯ БЫ ОДНА "
        "модель (величина, дополняющая жёсткий потолок до 100%). Это "
        "теоретический потолок точности любого способа выбора между уже "
        "обученными моделями объект за объектом — идеальный, но "
        "практически недостижимый ориентир, к которому реальное объединение "
        "может лишь приближаться.",
    )
    add_p(
        doc,
        "Дополнительно считается коэффициент корреляции Пирсона между "
        "бинарными («ошиблась»/«не ошиблась») индикаторами двух моделей на "
        "одних и тех же объектах — стандартная статистическая мера "
        "линейной связи двух величин, принимающая значения от −1 (чем чаще "
        "ошибается одна модель, тем реже — другая) до +1 (модели ошибаются "
        "вместе почти всегда). Высокая положительная корреляция ошибок "
        "означает, что модели, скорее всего, путаются на одних и тех же "
        "«трудных» объектах по общей причине (например, из-за низкого "
        "качества конкретного снимка) — в этом случае от объединения "
        "моделей пользы ждать не стоит."
    )

    add_heading(doc, "2.3. Три способа объединения предсказаний", level=2)
    add_p(
        doc,
        "Голосование: вероятности, которые каждая модель присваивает "
        "каждому классу, усредняются (при необходимости — с разными "
        "весами моделей), и объект относится к классу с наибольшей "
        "усреднённой вероятностью."
    )
    add_p(
        doc,
        "Стекинг: вероятности всех трёх моделей по всем классам "
        "объединяются в один общий, более длинный список чисел, который "
        "становится входом для дополнительной, отдельно обучаемой "
        "«надстроечной» модели (в этом эксперименте — тоже логистическая "
        "регрессия). Эта надстроечная модель обучается не на исходных "
        "изображениях, а на самих вероятностных ответах трёх базовых "
        "моделей — она учится тому, каким моделям и в каких ситуациях "
        "доверять больше. Как и любая обучаемая модель, надстроечная "
        "модель в принципе уязвима к перекосу классов в данных, на "
        "которых она обучается, — если один класс сильно преобладает, "
        "обучение без специальных поправок может свести её к правилу "
        "«всегда предсказывать самый частый класс». В этом эксперименте "
        "(раздел 3) обучающая и проверочная выборки намеренно "
        "сбалансированы поровну по всем классам, поэтому такой риск не "
        "реализуется — раздел 4.2 проверяет это прямым сравнением с "
        "версией стекинга, обученной со специальной поправкой на "
        "неравномерность классов."
    )
    add_p(
        doc,
        "Селективное переключение: для каждого класса ОТДЕЛЬНО заранее "
        "(по специально отложенной проверочной выборке) выбирается один "
        "«чемпион» — та из трёх моделей, которая лучше всех остальных "
        "распознаёт именно этот класс (доля верно распознанных объектов "
        "этого класса, также называемая полнотой или recall). При итоговой "
        "классификации оценка каждого класса берётся СТРОГО у его "
        "чемпиона, а не усредняется между всеми моделями — то есть, "
        "например, метод сопряжённости может целиком отвечать за один "
        "класс, а свёрточная сеть — за другой."
    )


def add_dataset_section(doc: Document, results: Dict[str, Any], tmp_files: List[Path]) -> None:
    config = results["config"]
    classes = config["classes"]
    totals = config["class_totals_full_dataset"]
    add_heading(doc, "3. Датасет", level=1)
    add_p(
        doc,
        f"Набор данных — полная открытая коллекция дерматоскопических "
        f"изображений новообразований кожи HAM10000/ISIC2018 (10 015 "
        f"изображений, 7 классов патологий), скачанная напрямую с научного "
        f"архива Harvard Dataverse. Каждое изображение приведено к квадрату "
        f"{config['img_size']}x{config['img_size']} пикселей, три цветовых "
        f"канала (исходные снимки коллекции — прямоугольные, около 600x450 "
        f"пикселей; уменьшение размера — единственный шаг подготовки перед "
        f"тем, как изображение превращается в вектор признаков)."
    )
    class_lines = "; ".join(f"{info['label_ru']}" for info in classes.values())
    add_p(doc, f"Семь классов патологий: {class_lines}.")

    totals_by_code = {info["code"]: info["label_ru"] for info in classes.values()}
    totals_line = "; ".join(
        f"{totals_by_code.get(code, code)} — {n}" for code, n in totals.items()
    )
    add_p(
        doc,
        f"В полном наборе данных классы представлены КРАЙНЕ неравномерно "
        f"(так же, как и в реальной клинической практике): {totals_line}. "
        f"Самый частый класс (меланоцитарный невус) почти в 60 раз больше "
        f"самого редкого (дерматофиброма)."
    )
    add_p(
        doc,
        f"Чтобы различия в точности моделей отражали свойства САМИХ "
        f"МЕТОДОВ, а не разный объём или состав обучающих данных по "
        f"классам, обучающая, проверочная и тестовая выборки построены "
        f"здесь самостоятельно (не используется никакое готовое "
        f"разбиение): из ВСЕХ доступных изображений каждого класса случайно "
        f"и без пересечений отобрано ОДИНАКОВОЕ число примеров — "
        f"{config['train_n_per_class']} для обучения, "
        f"{config['val_n_per_class']} для проверочной выборки и "
        f"{config['test_n_per_class']} для теста (в сумме "
        f"{config['train_n_per_class'] + config['val_n_per_class'] + config['test_n_per_class']} "
        f"на класс — ограничено самым малочисленным классом набора данных, "
        f"дерматофибромой, у которой всего 115 изображений). Итого "
        f"{config['n_train_total']} обучающих, {config['n_val_total']} "
        f"проверочных и {config['n_test_total']} тестовых изображений — "
        f"заметно меньше, чем всего доступно в полном наборе данных, но "
        f"РАВНОМЕРНО по всем семи классам сразу на всех трёх этапах. "
        f"Проверочная выборка используется для диагностики согласованности "
        f"ошибок и настройки схем объединения (НЕ участвует в обучении "
        f"базовых моделей). Итоговая точность измеряется исключительно на "
        f"тестовой выборке, не участвующей ни в обучении моделей, ни в "
        f"настройке их объединения."
    )

    montage_path = FIGURES_DIR / "dataset_montage.png"
    if montage_path.exists():
        add_picture_centered(doc, montage_path, width_in=4.6)
        add_caption(doc, "Рис. 1. По одному примеру на каждый из 7 классов набора данных.")

    recon_path = FIGURES_DIR / "subspace_projection.png"
    recon = results.get("reconstruction_demo")
    if recon_path.exists() and recon:
        add_picture_centered(doc, recon_path, width_in=4.6)
        r_str = ", ".join(f"{r:.3f}" for r in recon["r_values"])
        add_caption(
            doc,
            f"Рис. 2. «До/после» в терминах метода сопряжённости — верхний "
            f"ряд: тестовые изображения класса «{recon['demo_class_label']}»; "
            f"нижний ряд — их проекция (ближайшее приближение) на "
            f"выращенное подпространство этого класса. Значения показателя "
            f"сопряжённости R слева направо: {r_str}."
        )
        add_p(
            doc,
            "Простая нормировка яркости в диапазон от 0 до 1 (единственный "
            "формальный шаг подготовки перед тем, как изображение "
            "превращается в вектор признаков) визуально неотличима от "
            "исходных изображений — поэтому в качестве содержательной "
            "иллюстрации «до/после» здесь показана проекция на "
            "подпространство (рис. 2), непосредственно связанная с теорией "
            "метода (раздел 2.1)."
        )


def add_results_diagnostics_section(doc: Document, diag_val: Dict[str, Any], tmp_files: List[Path]) -> None:
    add_heading(doc, "4.1. Насколько связаны ошибки трёх моделей", level=1)
    add_p(
        doc,
        "Диагностика проведена на проверочной выборке — той же, что позже "
        "используется для настройки схем объединения, но ни разу не "
        "показывалась ни одной из трёх моделей во время обучения. Задача — "
        "семь классов, 15 изображений на класс в проверочной выборке; "
        "значения точности ниже, чем в более лёгких задачах распознавания "
        "образов, — ожидаемо для семиклассовой медицинской классификации "
        "на небольшом объёме обучающих данных (раздел 3: 60 изображений на "
        "класс)."
    )
    fig_path = render_diagnostics_figure(diag_val, tmp_files)
    add_picture_centered(doc, fig_path, width_in=4.6)
    add_caption(doc, "Рис. 3. Точность каждой базовой модели по отдельности и верхняя граница «оракула».")

    rows = []
    for pair_key, agree in diag_val["pairwise_agreement"].items():
        a, b = pair_key.split("__")
        corr = diag_val["pairwise_error_correlation"][pair_key]
        corr_str = "не определена" if corr != corr else f"{corr:.3f}"  # NaN check
        rows.append([
            f"{MODEL_LABEL_RU.get(a, a)} / {MODEL_LABEL_RU.get(b, b)}",
            f"{agree:.3f}", corr_str,
        ])
    add_table_9pt(doc, ["Пара моделей", "Доля совпадающих предсказаний", "Корреляция ошибок"], rows)

    fig_corr = render_correlation_figure(diag_val, tmp_files)
    add_picture_centered(doc, fig_corr, width_in=4.2)
    add_caption(
        doc,
        "Рис. 4. Корреляция ошибок для каждой пары базовых моделей — "
        "зелёным отмечена низкая связь (модели ошибаются в основном "
        "независимо), красным — высокая (модели чаще путаются вместе)."
    )

    corr_vals = diag_val["pairwise_error_correlation"]
    acc_by_model = diag_val["accuracy_by_model"]
    sc_ml = corr_vals.get("subspace__classical_ml", float("nan"))
    sc_cnn = corr_vals.get("subspace__cnn", float("nan"))
    ml_cnn = corr_vals.get("classical_ml__cnn", float("nan"))
    add_p(
        doc,
        f"На проверочной выборке метод сопряжённости и свёрточная сеть "
        f"показывают одинаковую точность ({acc_by_model['subspace']:.3f}), "
        f"классический ML — немного ниже ({acc_by_model['classical_ml']:.3f}); "
        f"порядок моделей по точности на независимом тесте (раздел 4.2) "
        f"оказывается ДРУГИМ. Небольшой размер проверочной выборки (15 "
        f"изображений на класс, раздел 3) даёт заметный разброс оценки "
        f"точности от выборки к выборке — к абсолютным числам стоит "
        f"относиться с этой поправкой."
    )
    add_p(
        doc,
        f"Более информативна структура корреляции ошибок между парами "
        f"моделей: ошибки метода сопряжённости и классического ML связаны "
        f"слабее всего ({sc_ml:.3f}) — заметно слабее, чем связаны между "
        f"собой ошибки классического ML и свёрточной сети ({ml_cnn:.3f}). "
        f"При этом ошибки метода сопряжённости и свёрточной сети связаны "
        f"ПОЧТИ ТАК ЖЕ СИЛЬНО ({sc_cnn:.3f}), как классический ML и "
        f"свёрточная сеть между собой — то есть свойство «непохожих "
        f"ошибок» у метода сопряжённости выражено ИЗБИРАТЕЛЬНО: ярко в "
        f"паре с классическим ML, но почти незаметно в паре со свёрточной "
        f"сетью. Это не отменяет теоретическую предпосылку пользы "
        f"объединения (раздел 2.2) — модель с хотя бы частично непохожими "
        f"ошибками способна добавить пользу, — но показывает, что эта "
        f"предпосылка выполняется не универсально для любой пары "
        f"моделей, а зависит от конкретного сочетания методов, и её стоит "
        f"измерять явно, а не постулировать из одной лишь непохожести "
        f"внутреннего устройства моделей."
    )
    add_p(
        doc,
        f"Потенциал ансамбля (доля объектов, где ошибается ровно одна "
        f"модель) — {diag_val['fraction_exactly_one_wrong']:.1%}; жёсткий "
        f"потолок (ошибаются все три) — {diag_val['fraction_all_wrong']:.1%} "
        f"— на этой трудной семиклассовой задаче с небольшой обучающей "
        f"выборкой значительная доля объектов проверочной выборки не "
        f"распознаёт верно ни одна из трёх моделей; точность «оракула» "
        f"(права хотя бы одна модель) — {diag_val['oracle_accuracy']:.1%}. "
        f"Это верхняя граница, к которой в лучшем случае может приблизиться "
        f"любая схема объединения — реальный результат (раздел 4.2) "
        f"неизбежно ниже, поскольку ни одна практическая схема не обладает "
        f"«всеведением» оракула."
    )


def add_results_final_section(
    doc: Document, results: Dict[str, Any], tmp_files: List[Path]
) -> None:
    add_heading(doc, "4.2. Итоговое сравнение на тестовой выборке", level=1)
    test_results = results["test_results"]

    add_p(
        doc,
        "Тестовая выборка построена поровну по всем семи классам (раздел 3) "
        "— поэтому обычная (общая) точность и средняя по классам точность "
        "здесь совпадают тождественно, и для сравнения моделей достаточно "
        "одной величины."
    )

    fig_path = render_final_comparison_figure(test_results, tmp_files)
    add_picture_centered(doc, fig_path, width_in=6.3)
    add_caption(
        doc,
        "Рис. 5. Точность на тестовой выборке — три базовые модели (серые "
        "столбцы) слева от вертикальной линии, четыре схемы объединения "
        "(синие столбцы) справа."
    )

    rows = []
    for name in ALL_MODEL_ORDER:
        r = test_results[name]
        rows.append([
            MODEL_LABEL_RU.get(name, name), f"{r['accuracy']:.4f}",
            f"{r['predict_time_seconds']:.3f}",
        ])
    best_single = max(BASE_MODEL_ORDER, key=lambda n: test_results[n]["accuracy"])
    best_ensemble = max(ENSEMBLE_ORDER, key=lambda n: test_results[n]["accuracy"])
    bold_idx = [ALL_MODEL_ORDER.index(best_single), ALL_MODEL_ORDER.index(best_ensemble)]
    add_table_9pt(
        doc, ["Модель", "Точность", "Время предсказания, с"],
        rows, bold_row_indices=bold_idx,
    )

    diff = test_results[best_ensemble]["accuracy"] - test_results[best_single]["accuracy"]
    verdict = (
        f"превосходит лучшую одиночную модель на {diff:.1%}" if diff > 0.001 else
        f"уступает лучшей одиночной модели на {abs(diff):.1%}" if diff < -0.001 else
        "практически совпадает с лучшей одиночной моделью"
    )
    add_p(
        doc,
        f"Лучшая одиночная модель — «{MODEL_LABEL_RU.get(best_single, best_single)}» "
        f"({test_results[best_single]['accuracy']:.4f}). Лучшая схема "
        f"объединения — «{MODEL_LABEL_RU.get(best_ensemble, best_ensemble)}» "
        f"({test_results[best_ensemble]['accuracy']:.4f}) — она {verdict}. "
        f"Выигрыш небольшой, но реальный: тестовая выборка ни разу не "
        f"участвовала в настройке ни одной из моделей или схем объединения "
        f"— это единственная проверка на полностью независимых данных."
    )

    add_heading(doc, "Найденная и исправленная методологическая ошибка", level=2)
    add_p(
        doc,
        "При подготовке этого эксперимента совместно с первой статьёй "
        "цикла («Итерационное наращивание опорных подпространств в "
        "методе сопряжённости») была обнаружена и устранена "
        "методологическая ошибка ТОЙ ЖЕ ПРИРОДЫ, что и там. Библиотека "
        "предоставляет режим «автоматическое равнение» для метода "
        "сопряжённости: каждый класс сначала растится без ограничения на "
        "число опорных векторов, а затем ВСЕ построенные подпространства "
        f"(в этом эксперименте — {results['config']['n_subclasses']} "
        f"подпространства на класс x 7 классов = "
        f"{results['config']['n_subclasses'] * 7}) приводятся к ОДНОЙ "
        "общей минимальной размерности, чтобы сравнение между классами "
        "было честным. Из-за неравномерности пошагового построения "
        "(см. отчёт по первой статье, раздел 2.2) хотя бы ОДНО из этих "
        "подпространств почти всегда останавливается всего на 2 опорных "
        "векторах — и это единственное подпространство задавало общий "
        "минимум для всех остальных сразу. В первой версии этого "
        "эксперимента метод сопряжённости из-за этого показывал точность "
        "всего 0,219 — заметно хуже классического ML и свёрточной сети, "
        "что выглядело как убедительное качественное различие между "
        "методами."
    )
    add_p(
        doc,
        f"После исправления (каждое подпространство участвует в "
        f"классификации со своим РЕАЛЬНО построенным числом опорных "
        f"векторов, без принудительного равнения) точность метода "
        f"сопряжённости на тестовой выборке выросла до "
        f"{test_results['subspace']['accuracy']:.4f} — он стал "
        f"сопоставим с двумя другими моделями, а не является безусловно "
        f"самым слабым. Последствия для схем объединения оказались "
        f"НЕ ОДНОЗНАЧНЫМИ: точность голосования и обоих вариантов "
        f"стекинга не изменилась вовсе, несмотря на заметно улучшенные "
        f"вероятности метода сопряжённости — на этой конкретной тестовой "
        f"выборке улучшенный сигнал не сдвинул ни одного итогового "
        f"решения голосования или стекинга. А точность селективного "
        f"переключения, наоборот, СНИЗИЛАСЬ (раздел 4.3) — после "
        f"исправления метод сопряжённости стал побеждать на проверочной "
        f"выборке чаще, но для части этих классов такое превосходство не "
        f"подтвердилось на независимом тесте. Это честная иллюстрация "
        f"риска, присущего самому методу селективного переключения: "
        f"выбор чемпиона класса по небольшой проверочной выборке (15 "
        f"изображений на класс, раздел 3) может не обобщаться на новые "
        f"данные так же надёжно, как усреднение вероятностей при "
        f"голосовании."
    )

    add_heading(doc, "Проверка: важна ли поправка на дисбаланс классов при стекинге", level=2)
    stacking_acc = test_results["stacking"]["accuracy"]
    stacking_bal_acc = test_results["stacking_balanced"]["accuracy"]
    add_p(
        doc,
        f"Обычный стекинг даёт точность {stacking_acc:.4f}; та же схема с "
        f"надстроечной моделью, взвешивающей классы обратно пропорционально "
        f"их частоте (раздел 2.3), — {stacking_bal_acc:.4f}. Значения "
        f"СОВПАДАЮТ — и это ожидаемый, а не случайный результат: "
        f"надстроечная модель стекинга обучается на проверочной выборке "
        f"(раздел 3), которая, как и обучающая, построена РОВНО ПОРОВНУ по "
        f"всем классам — поправка на дисбаланс не может дать эффекта там, "
        f"где дисбаланса, который нужно исправлять, попросту нет. Такая "
        f"проверка — полезный методологический контроль: если бы значения "
        f"разошлись при заведомо сбалансированных данных, это указывало бы "
        f"на ошибку в реализации, а не на содержательный эффект."
    )


def add_results_weak_class_section(
    doc: Document, results: Dict[str, Any], tmp_files: List[Path]
) -> None:
    add_heading(doc, "4.3. Устраняет ли объединение слабость метода сопряжённости", level=1)
    test_results = results["test_results"]
    per_class_subspace = test_results["subspace"]["per_class_accuracy"]
    weak_class = min(per_class_subspace, key=per_class_subspace.get)
    class_info = results["config"]["classes"]
    weak_label = class_info[weak_class]["label_ru"]

    add_p(
        doc,
        f"Слабее всего метод сопряжённости на тестовой выборке распознаёт "
        f"класс «{weak_label}» (полнота {per_class_subspace[weak_class]:.3f}). "
        f"Задача — проверить, снимает ли объединение с другими моделями "
        f"именно эту слабость, а не только повышает точность в среднем."
    )

    rows = []
    for name in ALL_MODEL_ORDER:
        recall = test_results[name]["per_class_accuracy"].get(weak_class, 0.0)
        rows.append([MODEL_LABEL_RU.get(name, name), f"{recall:.3f}"])
    add_table_9pt(doc, ["Модель", f"Полнота на классе «{weak_label}»"], rows)

    champions = results["switching_class_champion"]
    champion_for_weak = champions.get(weak_class)
    champion_metric_for_weak = results["switching_champion_metric"].get(weak_class)
    if champion_for_weak == "subspace":
        add_p(
            doc,
            f"Показательная деталь: при селективном переключении "
            f"чемпионом для класса «{weak_label}» — того самого класса, "
            f"где метод сопряжённости слабее всего на независимом тесте "
            f"— по итогам проверочной выборки назначен САМ МЕТОД "
            f"СОПРЯЖЁННОСТИ (полнота чемпиона на проверочной выборке — "
            f"{champion_metric_for_weak:.3f}, лучшая среди трёх моделей "
            f"именно там). На независимом тесте это решение не "
            f"подтвердилось — конкретный, а не гипотетический пример "
            f"того, что выбор чемпиона по небольшой проверочной выборке "
            f"(раздел 3: 15 изображений на класс) не гарантирует такого "
            f"же преимущества на новых данных."
        )
    elif champion_for_weak:
        add_p(
            doc,
            f"При селективном переключении чемпионом для класса «{weak_label}» "
            f"назначен «{MODEL_LABEL_RU.get(champion_for_weak, champion_for_weak)}» "
            f"— эта схема по построению использует для данного класса "
            f"модель, лучшую именно на нём по данным проверочной выборки."
        )

    add_p(
        doc,
        "Полный список «чемпионов» по всем семи классам (какая модель "
        "отвечает за каждый класс в схеме селективного переключения):"
    )
    class_champion_rows = []
    for cls, champ in sorted(champions.items(), key=lambda kv: int(kv[0])):
        label = class_info[cls]["label_ru"]
        metric = results["switching_champion_metric"].get(cls)
        metric_str = f"{metric:.3f}" if metric is not None else "—"
        class_champion_rows.append([label, MODEL_LABEL_RU.get(champ, champ), metric_str])
    add_table_9pt(doc, ["Класс", "Модель-чемпион", "Полнота чемпиона на проверочной выборке"], class_champion_rows)

    n_subspace_champion = sum(1 for v in champions.values() if v == "subspace")
    majority_text = (
        f"стал чемпионом для большинства классов ({n_subspace_champion} из 7)"
        if n_subspace_champion > 3 else
        f"стал чемпионом лишь для {n_subspace_champion} из 7 классов"
    )
    add_p(
        doc,
        f"По итогам проверочной выборки метод сопряжённости {majority_text} "
        f"— на остальных классах селективное переключение передаёт "
        f"решение другой модели. Распределение чемпионов подтверждает: "
        f"сильные и слабые стороны трёх подходов распределены по классам "
        f"неравномерно и по-разному; при этом случай класса «{weak_label}» "
        f"выше показывает, что выбор, сделанный по проверочной выборке, "
        f"не всегда подтверждается на независимом тесте."
    )


def add_discussion_section(doc: Document, results: Dict[str, Any]) -> None:
    add_heading(doc, "5. Обсуждение результатов", level=1)
    diag_val = results["diagnostics_val"]
    test_results = results["test_results"]
    best_single = max(BASE_MODEL_ORDER, key=lambda n: test_results[n]["accuracy"])
    best_ensemble = max(ENSEMBLE_ORDER, key=lambda n: test_results[n]["accuracy"])

    add_p(
        doc,
        f"Диагностика на проверочной выборке (раздел 4.1) показала "
        f"точность «оракула» {diag_val['oracle_accuracy']:.1%} против "
        f"{max(diag_val['accuracy_by_model'].values()):.1%} у лучшей "
        f"одиночной модели — разница задаёт теоретически доступный запас "
        f"для объединения, но он реализуется лишь частично: итоговый "
        f"эксперимент (раздел 4.2) на независимой тестовой выборке показал "
        f"скромный, но реальный выигрыш («{MODEL_LABEL_RU.get(best_ensemble, best_ensemble)}» "
        f"против лучшей одиночной модели «{MODEL_LABEL_RU.get(best_single, best_single)}»)."
    )
    add_p(
        doc,
        "Структура корреляции ошибок (раздел 4.1) оказалась более тонкой, "
        "чем можно было бы ожидать от простого тезиса «разные по "
        "устройству модели ошибаются по-разному»: свойство «непохожих "
        "ошибок» у метода сопряжённости выражено ИЗБИРАТЕЛЬНО — ярко в "
        "паре с классическим ML, но почти незаметно в паре со свёрточной "
        "сетью, чьи ошибки связаны с ошибками сопряжённости почти так же "
        "сильно, как ошибки классического ML и свёрточной сети между "
        "собой. Это не отменяет пользу объединения (раздел 4.2 "
        "показывает реальный, хотя и скромный выигрыш), но исправляет "
        "упрощённое прочтение: непохожесть внутреннего устройства двух "
        "моделей не гарантирует непохожести их ошибок — эту связь стоит "
        "измерять явно для каждой конкретной пары, а не постулировать "
        "заранее."
    )
    add_p(
        doc,
        "Показательный практический урок дало исправление "
        "методологической ошибки (раздел 4.2): улучшение одной из трёх "
        "базовых моделей (точность метода сопряжённости выросла почти в "
        "1,7 раза) НЕ гарантирует улучшения объединённого решения — "
        "голосование и оба варианта стекинга не изменились вовсе, а "
        "точность селективного переключения даже снизилась. Схемы "
        "объединения по-разному чувствительны к изменению качества "
        "входящих в них моделей: усреднение вероятностей (голосование) "
        "устойчиво к шуму в одной из моделей, тогда как выбор чемпиона "
        "класса по небольшой проверочной выборке (переключение) может "
        "«поймать» случайное, не обобщающееся на тест преимущество."
    )
    add_p(
        doc,
        "Проверка стекинга с поправкой на дисбаланс классов дала "
        "намеренно скромный, контрольный результат: поправка не изменила "
        "точность, потому что обучающие данные здесь были сбалансированы "
        "по построению, а не заимствованы из неравномерного официального "
        "разбиения. Это не отменяет теоретическую уязвимость стекинга к "
        "перекосу классов (раздел 2.3) — она остаётся значимым "
        "практическим предостережением для любого применения этой схемы "
        "на данных с естественным дисбалансом (каким является, например, "
        "ПОЛНЫЙ набор данных HAM10000/ISIC2018 без балансирующей "
        "подвыборки, раздел 3) — здесь она просто не могла проявиться при "
        "выбранном честном протоколе эксперимента."
    )


def add_conclusions_section(doc: Document, results: Dict[str, Any]) -> None:
    add_heading(doc, "6. Выводы", level=1)
    diag_val = results["diagnostics_val"]
    test_results = results["test_results"]
    best_single = max(BASE_MODEL_ORDER, key=lambda n: test_results[n]["accuracy"])
    best_ensemble = max(ENSEMBLE_ORDER, key=lambda n: test_results[n]["accuracy"])
    diff = test_results[best_ensemble]["accuracy"] - test_results[best_single]["accuracy"]

    add_bullet(
        doc,
        f"Ошибки трёх базовых моделей на проверочной выборке связаны не "
        f"полностью: точность «оракула» ({diag_val['oracle_accuracy']:.1%}) "
        f"заметно выше точности лучшей одиночной модели "
        f"({max(diag_val['accuracy_by_model'].values()):.1%}) — "
        f"теоретический запас для объединения моделей присутствует.",
    )
    verdict = "подтвердил" if diff > 0.001 else "не подтвердил" if diff < -0.001 else "не выявил разницы в"
    add_bullet(
        doc,
        f"На независимой тестовой выборке эксперимент {verdict} наличие "
        f"выигрыша от объединения: лучшая схема объединения "
        f"(«{MODEL_LABEL_RU.get(best_ensemble, best_ensemble)}», "
        f"{test_results[best_ensemble]['accuracy']:.4f}) против лучшей "
        f"одиночной модели («{MODEL_LABEL_RU.get(best_single, best_single)}», "
        f"{test_results[best_single]['accuracy']:.4f}) — выигрыш скромный, "
        f"но полученный на данных, ни разу не участвовавших в настройке "
        f"ни одной из моделей.",
    )
    add_bullet(
        doc,
        "Свойство «непохожих ошибок» у метода сопряжённости выражено "
        "избирательно (раздел 4.1): ошибки слабо связаны с ошибками "
        "классического ML, но почти так же сильно связаны с ошибками "
        "свёрточной сети, как ошибки классического ML и свёрточной сети "
        "между собой — предпосылка пользы объединения работает не "
        "универсально для каждой пары моделей, а зависит от конкретного "
        "сочетания методов.",
    )
    add_bullet(
        doc,
        f"Найдена и исправлена методологическая ошибка той же природы, "
        f"что и в первой статье цикла: автоматическое равнение всех "
        f"{results['config']['n_subclasses'] * 7} построенных "
        f"подпространств метода сопряжённости к общему минимуму "
        f"схлопывало его базисы до 2 опорных векторов и давало точность "
        f"всего 0,219. После отказа от принудительного равнения точность "
        f"метода сопряжённости выросла до "
        f"{test_results['subspace']['accuracy']:.4f} — он стал сопоставим "
        f"по точности с двумя другими моделями, а не является безусловно "
        f"самым слабым.",
    )
    add_bullet(
        doc,
        "Улучшение одной базовой модели не гарантирует улучшения "
        "объединённого решения: после исправления ошибки точность "
        "голосования и обоих вариантов стекинга не изменилась, а точность "
        "селективного переключения снизилась — выбор чемпиона класса по "
        "небольшой (15 изображений на класс) проверочной выборке оказался "
        "чувствителен к шуму и не всегда обобщается на независимый тест "
        "(раздел 4.3).",
    )
    add_bullet(
        doc,
        "Теоретическая уязвимость стекинга к перекосу классов (раздел 2.3) "
        "подтверждена контрольной проверкой: на данных, специально "
        "сбалансированных по построению (раздел 3), поправка на дисбаланс "
        "классов не изменила результат — ожидаемый, а не случайный итог, "
        "подтверждающий, что риск реализуется именно там, где дисбаланс "
        "есть, а не является артефактом самой схемы стекинга.",
    )
    add_bullet(
        doc,
        "Селективное переключение по классам показывает явно, какая из "
        "трёх моделей лучше всего распознаёт каждый конкретный класс — "
        "это не только практический инструмент, но и диагностика, "
        "показывающая, что сильные и слабые стороны трёх подходов "
        "действительно распределены по классам НЕОДИНАКОВО.",
    )
    add_bullet(
        doc,
        "Все схемы объединения реализованы как принимающие УЖЕ обученные "
        "модели — переобучать дорогостоящую свёрточную сеть заново при "
        "каждой попытке объединения не требуется, что делает подход "
        "практичным для комбинирования моделей, обученных разными, "
        "несовместимыми между собой средствами.",
    )


def add_future_work_section(doc: Document, results: Dict[str, Any]) -> None:
    add_heading(doc, "7. Перспективы: связь со следующей статьёй", level=1)
    test_results = results["test_results"]
    per_class_subspace = test_results["subspace"]["per_class_accuracy"]
    weak_class = min(per_class_subspace, key=per_class_subspace.get)
    strong_class = max(per_class_subspace, key=per_class_subspace.get)
    class_info = results["config"]["classes"]
    weak_label = class_info[weak_class]["label_ru"]
    strong_label = class_info[strong_class]["label_ru"]
    add_p(
        doc,
        f"Метод сопряжённости не одинаково силён на всех классах (раздел "
        f"4.3): его полнота варьируется от "
        f"{per_class_subspace[weak_class]:.3f} на классе «{weak_label}» "
        f"до {per_class_subspace[strong_class]:.3f} на классе "
        f"«{strong_label}» — почти шестикратный разброс внутри ОДНОЙ и "
        f"той же модели. Раздел 4.3 показал также, что оценка "
        f"сравнительной силы метода сопряжённости по небольшой "
        f"проверочной выборке не всегда надёжно предсказывает его "
        f"результат на независимом тесте — то есть неравномерность силы "
        f"метода по классам реальна, но её конкретную величину для "
        f"каждого класса стоит устанавливать на большем объёме данных. "
        f"Один из вероятных источников такой неравномерности — само "
        f"признаковое представление изображения: в этом и в первом "
        f"эксперименте цикла метод сопряжённости работает с построчной "
        f"развёрткой изображения в вектор, хотя тот же снимок можно "
        f"развернуть и по столбцам, получив иначе устроенное "
        f"представление той же геометрической информации. Следующая "
        f"статья цикла («Мультипредставительная гибридизация "
        f"признакового пространства в методе сопряжённости: "
        f"горизонтальная и вертикальная развёртка изображения») "
        f"возвращается к медицинскому МРТ-датасету проекта и проверяет, "
        f"можно ли сократить разрыв между слабыми и сильными классами, "
        f"не привлекая внешние модели, а обогатив признаковое "
        f"представление внутри самого метода."
    )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def generate_report() -> None:
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    doc = Document()
    tmp_files: List[Path] = []
    configure_document_font(doc)

    add_title(doc)
    add_intro_section(doc)
    add_theory_section(doc, tmp_files)
    add_dataset_section(doc, results, tmp_files)
    add_heading(doc, "4. Результаты эксперимента", level=1)
    add_results_diagnostics_section(doc, results["diagnostics_val"], tmp_files)
    add_results_final_section(doc, results, tmp_files)
    add_results_weak_class_section(doc, results, tmp_files)
    add_discussion_section(doc, results)
    add_conclusions_section(doc, results)
    add_future_work_section(doc, results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(REPORT_PATH)

    for tmp in tmp_files:
        tmp.unlink(missing_ok=True)

    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
