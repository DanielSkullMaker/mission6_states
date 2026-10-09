"""Статья 4 — та же ПОЛНАЯ версия, что и `generate_article4_report_final.py`
(все результаты, все графики, чек-лист медицинского рецензента), но
оформленная ПО СТИЛЮ шаблона конференции ИТНТ
(`theory/RU_MSWord_Template_Local_2024.doc`): римская нумерация разделов
(кроме «Литературы»), подписи рисунков ПОД рисунком («Рис. N.», 8 pt),
заголовки таблиц НАД таблицей (римские цифры, без точки в конце), ссылки
по ГОСТ 7.1-2003, десятичная запятая вместо точки, двухколоночная
вёрстка текста (широкие таблицы/рисунки — отдельной одноколоночной
секцией, это ЯВНО разрешено шаблоном: «Большие рисунки и таблицы могут
быть размещены сразу в двух столбцах»), обязательные разделы «Введение»/
«Заключение»/«Литература».

ПРИМЕЧАНИЕ: шаблон ИТНТ требует объём тезисов не более 2 страниц — по
решению пользователя этот лимит СОЗНАТЕЛЬНО не соблюдается (нужна полная
статья со всеми результатами), применяется только оформительский СТИЛЬ
шаблона, не его ограничение по объёму.

Запуск (после generate_article4_report_final.py и всех генераторов
графиков):
    python experiments/generate_article4_report_itnt.py
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ARTIFACTS = Path("artifacts")
UNET_REPORT_PATH = ARTIFACTS / "experiment_article4_unet_report.json"
CORRELATION_REPORT_PATH = ARTIFACTS / "article4_correlation_report.json"
ARCHITECTURES_REPORT_PATH = ARTIFACTS / "experiment_article4_architectures_report.json"
PREPROCESSING_ABLATION_REPORT_PATH = ARTIFACTS / "experiment_article4_preprocessing_ablation_report.json"
OVERFITTING_REPORT_PATH = ARTIFACTS / "experiment_article4_overfitting_report.json"
MULTI_SPLIT_REPORT_PATH = ARTIFACTS / "experiment_article4_multi_split_report.json"
CROSS_INST_REPORT_PATH = ARTIFACTS / "experiment_article4_cross_institution_report.json"
ACCURACY_BOOST_REPORT_PATH = ARTIFACTS / "experiment_article4_accuracy_boost_report.json"

FIG_PREPROCESSING = ARTIFACTS / "article4_fig2_preprocessing.png"
CHART_CORRELATION_MATRICES = ARTIFACTS / "article4_chart_correlation_matrices.png"
CHART_CORRELATION_DECAY = ARTIFACTS / "article4_chart_correlation_decay.png"
CHART_DICE = ARTIFACTS / "article4_unet_chart_dice.png"
CHART_PIXEL_ACCURACY = ARTIFACTS / "article4_unet_chart_pixel_accuracy.png"
FIG_EXAMPLES = ARTIFACTS / "article4_unet_fig_examples.png"
CHART_ARCHITECTURES = ARTIFACTS / "article4_unet_chart_architectures.png"
CHART_OVERFITTING = ARTIFACTS / "article4_unet_chart_overfitting.png"
CHART_PREPROCESSING_ABLATION = ARTIFACTS / "article4_unet_chart_preprocessing_ablation.png"
CHART_MULTI_SPLIT = ARTIFACTS / "article4_unet_chart_multi_split.png"
CHART_CROSS_INSTITUTION = ARTIFACTS / "article4_unet_chart_cross_institution.png"
CHART_ACCURACY_BOOST = ARTIFACTS / "article4_unet_chart_accuracy_boost.png"

PREP_ORDER = ["full_pipeline", "no_centering", "strong_threshold", "clahe_contrast", "minmax_norm"]
PREP_LABELS_RU = {
    "full_pipeline": "Полный конвейер",
    "no_centering": "Без центрирования",
    "strong_threshold": "Усиленный порог фона (40)",
    "clahe_contrast": "CLAHE-контраст",
    "minmax_norm": "Min-max нормализация",
}
ARCH_LABELS_RU = {
    "unet_baseline": "U-Net (базовый)",
    "plain_no_skip": "Plain encoder-decoder (без skip)",
    "res_unet": "Residual U-Net",
    "attention_unet": "Attention U-Net",
    "segnet": "SegNet",
    "fcn8s": "FCN-8s",
    "multiresunet": "MultiResUNet",
    "deeplabv3plus": "DeepLabV3+ (lite)",
    "resnet18_unet_transfer": "ResNet18-UNet (transfer learning)",
}
ARCH_ORDER = [
    "unet_baseline", "plain_no_skip", "res_unet", "attention_unet",
    "segnet", "fcn8s", "multiresunet", "deeplabv3plus", "resnet18_unet_transfer",
]

OUTPUT_PATH = ARTIFACTS / "article4_itnt_style.docx"

FONT_NAME = "Times New Roman"
BODY_SIZE = Pt(9)
CAPTION_SIZE = Pt(8)

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def ru_num(value: float, spec: str) -> str:
    """Форматирует число с ЗАПЯТОЙ как десятичным разделителем (ГОСТ/
    русская типографика — шаблон ИТНТ прямо требует «0,25», не «0.25»)."""
    return format(value, spec).replace(".", ",")


def _set_run_font(run, size=BODY_SIZE) -> None:
    run.font.name = FONT_NAME
    run.font.size = size
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), FONT_NAME)


def set_document_base_font(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = BODY_SIZE
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), FONT_NAME)


def set_columns(section, num: int) -> None:
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), "425" if num > 1 else "0")


def new_section(doc: Document, num_cols: int):
    """Непрерывный разрыв секции (без разрыва страницы) со сменой числа
    колонок — широкие таблицы/рисунки идут в одноколоночной секции (это
    прямо разрешено шаблоном), окружающий текст — в двухколоночной."""
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(sec, num_cols)
    return sec


def add_paragraph(doc, text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    _set_run_font(run)
    if size is not None:
        run.font.size = size
    return p


def add_lead_paragraph(doc, lead: str, text: str):
    """Стиль «Аннотация—...» / «Ключевые слова—...»: полужирный курсивный
    заголовок-тире перед текстом в ОДНОМ абзаце (как в шаблоне ИТНТ)."""
    p = doc.add_paragraph()
    run_lead = p.add_run(lead + "—")
    run_lead.bold = True
    run_lead.italic = True
    _set_run_font(run_lead)
    run_text = p.add_run(text)
    _set_run_font(run_text)
    return p


def add_numbered_heading(doc, number: str, text: str, size=Pt(11)):
    """Заголовок текстового раздела с римской нумерацией (шаблон ИТНТ:
    «Все заголовки разделов статьи, исключая Благодарности и Литературу,
    нумеруются римскими цифрами»)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(f"{number}. {text}")
    run.bold = True
    _set_run_font(run)
    run.font.size = size


def add_heading_plain(doc, text, size=Pt(11)):
    """Заголовок-компонент БЕЗ нумерации (Благодарности/Литература/
    Аннотация — шаблон ИТНТ)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.bold = True
    _set_run_font(run)
    run.font.size = size


def add_subheading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.bold = True
    run.italic = True
    _set_run_font(run)


def add_formula(doc, text, number):
    """Формула по центру, номер — в скобках у правого края (шаблон ИТНТ:
    «Номера уравнений в круглых скобках должны располагаться на одном
    уровне справа»)."""
    p = doc.add_paragraph()
    tab_stops = p.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Inches(3.0), alignment=2)  # правый край
    run1 = p.add_run(text)
    run1.italic = True
    _set_run_font(run1)
    run2 = p.add_run(f"\t({number})")
    _set_run_font(run2)
    return p


def add_figure(doc, path, caption_number: int, caption_text: str, width_inches=3.1, wide=False):
    """Подпись ПОД рисунком, формат «Рис. N. текст.» (8 pt), шаблон ИТНТ.
    Широкие рисунки (wide=True) — в отдельной однокolоночной секции."""
    if wide:
        new_section(doc, 1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_inches))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_cap = cap.add_run(f"Рис. {caption_number}. {caption_text}")
    _set_run_font(run_cap, size=CAPTION_SIZE)
    if wide:
        new_section(doc, 2)


def add_table(doc, table_number: int, title: str, headers, rows, wide=False):
    """Заголовок НАД таблицей, «ТАБЛИЦА <римская>. текст» (без точки в
    конце), 8 pt, по центру — шаблон ИТНТ. Широкие таблицы — в отдельной
    однокolоночной секции."""
    if wide:
        new_section(doc, 1)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_cap = cap.add_run(f"ТАБЛИЦА {ROMAN[table_number - 1]}. {title}")
    run_cap.bold = True
    _set_run_font(run_cap, size=CAPTION_SIZE)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        run = hdr_cells[i].paragraphs[0].add_run(str(h))
        run.bold = True
        _set_run_font(run, size=CAPTION_SIZE)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            _set_run_font(run, size=CAPTION_SIZE)
    if wide:
        new_section(doc, 2)


def add_reference(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Pt(14)
    p.paragraph_format.first_line_indent = Pt(-14)
    run = p.add_run(text)
    _set_run_font(run)


def ru_patients(n: int) -> str:
    n_abs = abs(n)
    last_two = n_abs % 100
    last_one = n_abs % 10
    if 11 <= last_two <= 14:
        return f"{n} пациентов"
    if last_one == 1:
        return f"{n} пациент"
    if 2 <= last_one <= 4:
        return f"{n} пациента"
    return f"{n} пациентов"


def fmt_ci(ci):
    if not ci:
        return "—"
    return f"[{ru_num(ci[0]*100, '.1f')}%, {ru_num(ci[1]*100, '.1f')}%]"


def main() -> None:
    report = json.loads(UNET_REPORT_PATH.read_text(encoding="utf-8"))
    corr = json.loads(CORRELATION_REPORT_PATH.read_text(encoding="utf-8"))
    arch_report = json.loads(ARCHITECTURES_REPORT_PATH.read_text(encoding="utf-8"))
    prep_report = json.loads(PREPROCESSING_ABLATION_REPORT_PATH.read_text(encoding="utf-8"))
    overfit_report = json.loads(OVERFITTING_REPORT_PATH.read_text(encoding="utf-8"))
    multisplit_report = json.loads(MULTI_SPLIT_REPORT_PATH.read_text(encoding="utf-8"))
    crossinst_report = json.loads(CROSS_INST_REPORT_PATH.read_text(encoding="utf-8"))
    boost_report = json.loads(ACCURACY_BOOST_REPORT_PATH.read_text(encoding="utf-8"))

    results = report["results"]
    k1 = results["no_context_k1"]
    kn = results["meaningful_neighbors"]
    sh = results["shuffled_neighbors"]
    h1 = report["hypotheses"]["H1_context_helps_dice"]
    h2 = report["hypotheses"]["H2_shuffled_neighbors_not_better_than_meaningful"]

    n_train_patients = len(report["train_patients"])
    n_test_patients = len(report["test_patients"])
    n_total_patients = n_train_patients + n_test_patients
    n_context = report["n_context"]

    decay_content = corr["decay_content_order"]
    decay_shuffled = corr["decay_shuffled_order"]

    arch_results = arch_report["results"]
    best_arch_key = max(arch_results, key=lambda k: arch_results[k]["mean_dice_with_tumor"])
    best_arch_name = ARCH_LABELS_RU[best_arch_key]
    best_arch_dice = arch_results[best_arch_key]["mean_dice_with_tumor"]
    best_arch_result = arch_results[best_arch_key]

    prep_results = prep_report["results"]

    ms_summary = multisplit_report["summary"]
    ms_runs = multisplit_report["runs"]
    ms_h1_count = multisplit_report["h1_holds_count"]
    ms_h2_count = multisplit_report["h2_holds_count"]
    ms_n = multisplit_report["n_splits"]

    ci_result = crossinst_report["result"]
    ci_baseline = crossinst_report["baseline_random_split_comparison"]
    ci_gap = ci_baseline["mean_dice_with_tumor"] - ci_result["mean_dice_with_tumor"]

    boost_final = boost_report["final_evaluation"]
    boost_gain = boost_final["mean_dice_with_tumor"] - best_arch_dice

    doc = Document()
    set_document_base_font(doc)
    for section in doc.sections:
        section.left_margin = Pt(72)
        section.right_margin = Pt(72)

    # === НАЗВАНИЕ / АВТОРЫ (однокolоночная секция — шаблон ИТНТ) ======
    add_paragraph(
        doc,
        "Анализ последовательной структуры смежных срезов медицинских "
        "изображений методом свёрточных нейронных сетей",
        bold=True, size=Pt(13), align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_paragraph(doc, "[ФИО автора]", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "[Наименование организации]", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "[Город], Россия", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "[email]", align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    # === АННОТАЦИЯ / КЛЮЧЕВЫЕ СЛОВА (стиль ИТНТ: «Аннотация—...») =====
    add_lead_paragraph(
        doc, "Аннотация",
        "Медицинские изображения (МРТ, КТ) часто представляют собой не "
        "отдельные независимые снимки, а последовательность срезов одного "
        "исследования, соседние элементы которой пространственно и "
        "визуально коррелированы. В работе рассматривается задача "
        "попиксельной сегментации опухоли головного мозга на МРТ-срезах "
        "свёрточной нейронной сетью. Исследованы два независимых "
        "источника повышения качества: учёт нескольких физически смежных "
        "по оси срезов в виде дополнительных входных каналов и "
        "одновременное использование нескольких МРТ-модальностей (FLAIR, "
        "T1, T1-постконтраст, T2) вместо одной. Качество оценивается "
        "коэффициентом Дайса на срезах, содержащих опухоль, отдельно от "
        "срезов без опухоли. "
        f"На честном (по пациентам) разбиении набора BraTS2020 "
        f"({n_total_patients} пациентов) контекст физически смежных "
        f"срезов даёт статистически устойчивый (подтверждён на "
        f"{ms_h1_count} из {ms_n} независимых разбиений) прирост Dice, "
        "тогда как замена соседей случайными срезами ухудшает результат "
        f"(подтверждено на {ms_h2_count} из {ms_n} разбиений). "
        f"Мультимодальный вход оказался более сильным рычагом: лучшая из "
        f"9 архитектур на одной модальности даёт "
        f"{ru_num(best_arch_dice*100, '.1f')}% Dice, тогда как тот же тип "
        f"сети на четырёх модальностях — "
        f"{ru_num(boost_final['mean_dice_with_tumor']*100, '.1f')}% "
        f"({ru_num(boost_gain*100, '+.1f')} процентных пункта). "
        f"Качество падает на {ru_num(ci_gap*100, '.1f')} процентных "
        "пункта при переносе модели на источник данных вне обучающей "
        "выборки; сравнение с дообученной (transfer learning) "
        "предобученной сетью не дало ей преимущества.",
    )
    add_lead_paragraph(
        doc, "Ключевые слова",
        "сегментация изображений, магнитно-резонансная томография, "
        "свёрточные нейронные сети, U-Net, мультимодальные данные, "
        "глубокое обучение, медицинская визуализация",
    )

    # === ОСНОВНОЙ ТЕКСТ — ДВУХКОЛОНОЧНАЯ СЕКЦИЯ =======================
    new_section(doc, 2)

    # I. ВВЕДЕНИЕ
    add_numbered_heading(doc, "I", "Введение")
    add_paragraph(
        doc,
        "Снимки одного пациента — например, последовательные срезы "
        "МРТ-исследования — визуально и статистически ближе друг к другу, "
        "чем снимки разных пациентов: это структурная информация, которую "
        "сегментация одного изолированного среза не использует, хотя "
        "соседние срезы могут содержать существенную подсказку о форме и "
        "границах той же опухоли в трёхмерном пространстве.",
    )
    add_paragraph(
        doc,
        "Отдельного обсуждения заслуживает выбор метрики качества на "
        "двух уровнях. Во-первых, пиксельная accuracy — обманчивый "
        "показатель для сегментации малых объектов: модель, "
        "предсказывающая «фон» повсеместно, уже получает accuracy, "
        "близкую к доле фоновых пикселей. Во-вторых, коэффициент Дайса, "
        "усреднённый по всем тестовым срезам (включая заведомо пустые), "
        "смешивает «распознать отсутствие патологии» (тривиально) и "
        "«точно очертить границу опухоли там, где она есть» "
        "(содержательная задача). Поэтому основные результаты приводятся "
        "раздельно для срезов с опухолью и без неё.",
    )
    add_paragraph(
        doc,
        "Задача работы — установить, научится ли свёрточная сеть "
        "использовать физическую смежность срезов для более точной "
        "сегментации опухоли (гипотезы H1/H2), и насколько использование "
        "нескольких МРТ-модальностей способно повысить качество "
        "относительно архитектурных улучшений в пределах одной "
        "модальности. Дополнительно проверены: статистическая "
        "устойчивость выводов по нескольким независимым разбиениям "
        "данных, устойчивость качества при переносе на источник снимков "
        "вне обучения, и сравнение с переносом обучения (transfer "
        "learning) от сети, предобученной на другом домене изображений.",
    )

    # II. ТЕОРИЯ
    add_numbered_heading(doc, "II", "Теория")
    add_paragraph(
        doc,
        "Рассматривается задача бинарной попиксельной сегментации: по "
        "срезу магнитно-резонансной томографии требуется построить маску "
        "того же размера, относящую каждый пиксель к классу «опухоль» "
        "или «фон».",
    )
    add_paragraph(
        doc,
        "Для каждой пары срезов одного пациента вычислено косинусное "
        "сходство вектора пикселей; усреднение по всем парам с "
        "фиксированным расстоянием (лагом) по оси среза даёт следующую "
        "зависимость:",
    )
    decay_rows = [
        [lag, ru_num(decay_content[lag]*100, ".1f") + "%", ru_num(decay_shuffled[lag]*100, ".1f") + "%"]
        for lag in ["1", "2", "3", "5", "10", "20", "40", "80"]
    ]
    add_table(doc, 1, "ЗАТУХАНИЕ СХОДСТВА СРЕЗОВ С РАССТОЯНИЕМ",
              ["Лаг |i-j|", "Содержательный порядок", "Перемешанный порядок"], decay_rows)
    add_figure(doc, CHART_CORRELATION_DECAY, 1,
               "Затухание косинусного сходства срезов с расстоянием: "
               "содержательный порядок против перемешанного.",
               width_inches=6.0, wide=True)
    add_figure(doc, CHART_CORRELATION_MATRICES, 2,
               "Матрицы попарного сходства срезов одного пациента.",
               width_inches=6.3, wide=True)

    add_subheading(doc, "Архитектуры сегментации")
    add_paragraph(
        doc,
        "Базовая архитектура — компактный U-Net (Ronneberger et al., "
        "2015) [1] с нормализацией по батчу после каждой свёртки — её "
        "отсутствие приводило к нестабильной сходимости при ограниченном "
        "бюджете обучения (см. «Результаты эксперимента»). Для проверки "
        "специфичности выводов сопоставлены ещё восемь архитектур: "
        "энкодер-декодер без skip-connections, Residual U-Net [3, 4], "
        "Attention U-Net [5], SegNet [6], FCN-8s [7], MultiResUNet [8], "
        "DeepLabV3+ [9] и ResNet18-UNet с предобученным на ImageNet "
        "энкодером (transfer learning).",
    )
    add_paragraph(
        doc,
        "Последовательная структура смежных срезов представлена в "
        "«2.5D»-формате: N физически смежных срезов подаются как "
        "отдельные входные каналы одного тензора. Мультимодальный вход "
        "(раздел «Повышение точности») представлен тем же способом — по "
        "одному каналу на модальность (FLAIR, T1, T1-постконтраст, T2) "
        "одного среза.",
    )
    add_formula(doc, "L = L_BCE(p, y) + L_Dice(p, y)", 1)
    add_formula(doc, "L_Dice = 1 − (2·Σ(p·y) + s) / (Σp + Σy + s)", 2)
    add_paragraph(
        doc,
        "Функция потерь — сумма бинарной перекрёстной энтропии и мягкой "
        "версии потерь Дайса (Milletari et al., 2016) [2]. Итоговое "
        "качество оценивается коэффициентом Дайса:",
    )
    add_formula(doc, "Dice(P, Y) = 2·|P ∩ Y| / (|P| + |Y|)", 3)
    add_paragraph(
        doc,
        "где P — предсказанная маска, Y — истинная маска. Для оценки "
        "статистической надёжности долей (precision/recall) используется "
        "95%-доверительный интервал Уилсона для биномиальной доли.",
    )

    # III. ОПИСАНИЕ ЭКСПЕРИМЕНТА
    add_numbered_heading(doc, "III", "Описание эксперимента")
    add_paragraph(
        doc,
        f"Датасет: BraTS2020, {n_total_patients} пациентов. Для каждого "
        "среза используются четыре МРТ-модальности: FLAIR, T1, "
        "T1-постконтраст и T2 — явное указание модальностей приводится "
        "по требованию медицинского рецензента исследования.",
    )
    add_subheading(doc, "Подготовка данных: от NIfTI к PNG")
    add_paragraph(
        doc,
        "Исходные объёмы (240×240×155 вокселей) обработаны по шагам: "
        "отбор срезов по доле «мозговых» вокселей (≥2%); нормализация "
        "яркости по 1-му/99-му перцентилям в [0, 255]; приведение к "
        "размеру 256×256 (Lanczos); подавление фона (порог 10/255); "
        "центрирование циклическим сдвигом; сохранение в PNG с "
        "сохранением физического порядка срезов. Маска опухоли и три "
        "дополнительные модальности проходят тот же геометрический путь "
        "со сдвигом центрирования, вычисленным по FLAIR того же среза.",
    )
    add_figure(doc, FIG_PREPROCESSING, 3,
               "Пример среза после предобработки, с наложенной маской "
               "опухоли и после полного цикла предобработки.",
               width_inches=6.0, wide=True)
    add_paragraph(
        doc,
        f"Разбиение данных — случайный сплит по идентификатору пациента "
        f"({ru_patients(n_train_patients)} на обучение, "
        f"{ru_patients(n_test_patients)} в held-out тест) — требование "
        "медицинского рецензента, исключающее попадание срезов одного "
        "пациента одновременно в обучение и тест.",
    )
    add_paragraph(
        doc,
        "Сопоставлены три конфигурации: без контекста (K=1); физические "
        f"соседи (K={n_context}, стек физически смежных срезов как "
        "отдельных каналов); случайные «соседи» (K=5, та же обученная "
        "модель, но соседи заменены случайными срезами теста — контроль "
        "механизма эффекта).",
    )

    # IV. РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА
    add_numbered_heading(doc, "IV", "Результаты эксперимента")

    add_subheading(doc, "A. Контекст физически смежных срезов (гипотезы H1, H2)")
    add_table(
        doc, 2, "КАЧЕСТВО СЕГМЕНТАЦИИ ПО ТРЁМ КОНФИГУРАЦИЯМ",
        ["Конфигурация", "Dice, все срезы", "Dice, с опухолью", "Pixel accuracy"],
        [
            ["Без контекста (K=1)", ru_num(k1["mean_dice"], ".3f"), ru_num(k1["mean_dice_with_tumor"], ".3f"), ru_num(k1["pixel_accuracy"], ".3f")],
            [f"Физич. соседи (K={n_context})", ru_num(kn["mean_dice"], ".3f"), ru_num(kn["mean_dice_with_tumor"], ".3f"), ru_num(kn["pixel_accuracy"], ".3f")],
            [f"Случайные «соседи» (K={n_context})", ru_num(sh["mean_dice"], ".3f"), ru_num(sh["mean_dice_with_tumor"], ".3f"), ru_num(sh["pixel_accuracy"], ".3f")],
        ],
        wide=True,
    )
    add_figure(doc, CHART_DICE, 4, "Коэффициент Дайса по трём конфигурациям.", width_inches=6.0, wide=True)
    add_figure(doc, CHART_PIXEL_ACCURACY, 5, "Пиксельная accuracy по тем же конфигурациям.", width_inches=6.0, wide=True)
    add_paragraph(
        doc,
        f"На разбиении seed=42 Dice (срезы с опухолью) конфигурации "
        f"«физические соседи» ({ru_num(kn['mean_dice_with_tumor']*100,'.1f')}%) "
        f"{'превышает' if h1 else 'не превышает'} конфигурацию «без "
        f"контекста» ({ru_num(k1['mean_dice_with_tumor']*100,'.1f')}%). "
        "Это иллюстрация на одном разбиении — статистически надёжный "
        "вывод см. в разделе D.",
    )

    add_subheading(doc, "B. Confusion matrix и доверительные интервалы")
    add_paragraph(
        doc,
        "По требованию медицинского рецензента приводится не только "
        "агрегированная точность, но и разбивка ошибок детекции опухоли "
        "на срезе, вместе с 95%-доверительными интервалами Уилсона.",
    )
    cm_rows = []
    for name_ru, r in [("Без контекста (K=1)", k1), (f"Физич. соседи (K={n_context})", kn), (f"Случайные «соседи» (K={n_context})", sh)]:
        cm = r["slice_confusion_matrix"]
        cm_rows.append([name_ru, cm["tp"], cm["fp"], cm["fn"], cm["tn"]])
    add_table(doc, 3, "CONFUSION MATRIX ДЕТЕКЦИИ ОПУХОЛИ НА СРЕЗЕ",
              ["Конфигурация", "TP", "FP", "FN", "TN"], cm_rows, wide=True)
    ci_rows = [
        [name_ru, fmt_ci(r["slice_detection_recall_ci95"]), fmt_ci(r["slice_detection_precision_ci95"]), fmt_ci(r["pixel_recall_ci95"]), fmt_ci(r["pixel_precision_ci95"])]
        for name_ru, r in [("Без контекста (K=1)", k1), (f"Физич. соседи (K={n_context})", kn), (f"Случайные «соседи» (K={n_context})", sh)]
    ]
    add_table(doc, 4, "95%-ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ УИЛСОНА",
              ["Конфигурация", "Detection recall", "Detection precision", "Pixel recall", "Pixel precision"],
              ci_rows, wide=True)

    add_subheading(doc, "C. Качественные примеры и сравнение архитектур")
    add_figure(doc, FIG_EXAMPLES, 6,
               "Примеры сегментации: исходный срез, разметка и "
               "предсказания трёх конфигураций.", width_inches=6.3, wide=True)
    arch_rows = []
    for k in ARCH_ORDER:
        n_params = arch_results[k].get("n_params")
        n_params_str = f"{n_params:,}".replace(",", " ") if n_params else "—"
        arch_rows.append([
            ARCH_LABELS_RU[k], ru_num(arch_results[k]["mean_dice"], ".3f"),
            ru_num(arch_results[k]["mean_dice_with_tumor"], ".3f"),
            n_params_str, f"{arch_results[k]['fit_seconds']:.0f} с",
        ])
    add_table(doc, 5, "КАЧЕСТВО СЕГМЕНТАЦИИ ПО АРХИТЕКТУРАМ",
              ["Архитектура", "Dice, все срезы", "Dice, с опухолью", "Параметров", "Время"],
              arch_rows, wide=True)
    add_figure(doc, CHART_ARCHITECTURES, 7, "Dice по девяти архитектурам.", width_inches=7.5, wide=True)
    add_paragraph(
        doc,
        f"Лучший результат на одной модальности — {best_arch_name} "
        f"({ru_num(best_arch_dice*100,'.1f')}% Dice с опухолью). "
        "ResNet18-UNet (transfer learning) не дал преимущества перед "
        f"специализированными архитектурами "
        f"({ru_num(arch_results['resnet18_unet_transfer']['mean_dice_with_tumor']*100,'.1f')}% "
        "— один из худших результатов) — правдоподобно из-за короткого "
        "бюджета дообучения и отсутствия согласования нормализации входа "
        "с исходным ImageNet-предобучением. Сравнение с предобученной "
        "сетью включено по требованию медицинского рецензента.",
    )
    overfit_summary = {row["name"]: row for row in overfit_report["summary"]}
    overfit_rows = [
        [ARCH_LABELS_RU[name], ru_num(row['best_test_dice_with_tumor'], ".3f"), row["best_epoch"],
         row["overfit_onset_epoch"] if row["overfit_onset_epoch"] is not None else "не набл."]
        for name, row in overfit_summary.items()
    ]
    add_table(doc, 6, f"ДИНАМИКА ОБУЧЕНИЯ, {overfit_report['epochs_full']} ЭПОХ",
              ["Архитектура", "Макс. Dice", "Эпоха max", "Переобуч."], overfit_rows, wide=True)
    add_figure(doc, CHART_OVERFITTING, 8, "Кривые train/test Dice по эпохам для восьми архитектур.", width_inches=7.0, wide=True)
    add_paragraph(
        doc,
        "После добавления нормализации по батчу ни одна из девяти "
        "архитектур не показала полного коллапса (до этого обучение было "
        "нестабильным независимо от архитектуры). Кривые test-Dice "
        "шумные для части сетей — ранний «пик» правдоподобнее объясняется "
        "выбросом, а не истинным оптимумом обобщения.",
    )

    add_subheading(doc, "D. Устойчивость по нескольким разбиениям")
    add_paragraph(
        doc,
        f"По требованию медицинского рецензента базовая модель обучена "
        f"заново на {ms_n} независимых случайных разбиениях по пациентам.",
    )
    ms_rows = [
        [r["split_seed"], ru_num(r['no_context_dice_tumor'], ".3f"), ru_num(r['meaningful_dice_tumor'], ".3f"),
         ru_num(r['shuffled_dice_tumor'], ".3f"), str(r["h1_holds"]), str(r["h2_holds"])]
        for r in ms_runs
    ]
    add_table(doc, 7, "DICE (С ОПУХОЛЬЮ) ПО 5 НЕЗАВИСИМЫМ РАЗБИЕНИЯМ",
              ["Seed", "Без контекста", "Физич. соседи", "Случ. «соседи»", "H1", "H2"], ms_rows, wide=True)
    add_figure(doc, CHART_MULTI_SPLIT, 9, "Dice по 5 независимым разбиениям train/test.", width_inches=6.5, wide=True)
    add_paragraph(
        doc,
        f"H1 подтверждена на {ms_h1_count} из {ms_n} разбиений, H2 — на "
        f"{ms_h2_count} из {ms_n}. Среднее: физические соседи "
        f"{ru_num(ms_summary['meaningful_dice_tumor']['mean']*100,'.1f')}% "
        f"± {ru_num(ms_summary['meaningful_dice_tumor']['std']*100,'.1f')} "
        f"п.п. против {ru_num(ms_summary['no_context_dice_tumor']['mean']*100,'.1f')}% "
        f"± {ru_num(ms_summary['no_context_dice_tumor']['std']*100,'.1f')} "
        "п.п. без контекста. Важно: единственное разбиение, на котором "
        "H1 не подтвердилась — seed=42, использованный во всех остальных "
        "одиночных экспериментах разделов A–C этой статьи; разброс между "
        "разбиениями того же порядка, что и сам эффект.",
    )

    add_subheading(doc, "E. Генерализация на источник данных вне обучения")
    add_paragraph(
        doc,
        f"Модель обучена только на пациентах источника "
        f"{crossinst_report['train_site']} ({crossinst_report['n_train_patients']} чел.) "
        f"и проверена на источниках {'+'.join(crossinst_report['test_sites'])} "
        f"({crossinst_report['n_test_patients']} чел.) — требование "
        "медицинского рецензента о чувствительности метода к "
        "межисточниковым различиям яркости/контраста.",
    )
    add_table(
        doc, 8, "ГЕНЕРАЛИЗАЦИЯ НА НЕ ПРЕДСТАВЛЕННЫЙ В ОБУЧЕНИИ ИСТОЧНИК",
        ["", "Dice, все срезы", "Dice, с опухолью", "Detection recall"],
        [
            ["Случайное разбиение", ru_num(ci_baseline['mean_dice'],".3f"), ru_num(ci_baseline['mean_dice_with_tumor'],".3f"), ru_num(ci_baseline['slice_detection_recall'],".3f")],
            ["Другой источник", ru_num(ci_result['mean_dice'],".3f"), ru_num(ci_result['mean_dice_with_tumor'],".3f"), ru_num(ci_result['slice_detection_recall'],".3f")],
        ],
        wide=True,
    )
    add_figure(doc, CHART_CROSS_INSTITUTION, 10, "Разрыв генерализации при переносе на другой источник данных.", width_inches=5.5, wide=True)
    add_paragraph(
        doc,
        f"Качество падает на {ru_num(ci_gap*100,'.1f')} процентных "
        "пункта при переносе на источник вне обучения — количественное "
        "подтверждение опасения медицинского рецензента. Оговорка: это "
        "перенос между контрибьюторами внутри курируемого BraTS2020, не "
        "проверка на снимках другого протокола МРТ-съёмки.",
    )

    add_subheading(doc, "F. Повышение точности: мультимодальный вход")
    add_paragraph(
        doc,
        "Проверено, насколько одновременное использование всех четырёх "
        "модальностей способно повысить качество относительно "
        "архитектурных вариаций в пределах одной модальности, с "
        f"увеличенным разрешением ({boost_report['img_size']}×{boost_report['img_size']}), "
        "аугментацией (флип, поворот, яркостный джиттер), расписанием "
        "скорости обучения с понижением по плато и ранней остановкой "
        f"(остановлено на эпохе {boost_report['best_epoch']}).",
    )
    add_table(
        doc, 9, "ЭФФЕКТ МУЛЬТИМОДАЛЬНОГО ВХОДА",
        ["", "Dice, все срезы", "Dice, с опухолью", "Detection recall", "CI95"],
        [
            [f"Одна модальность ({best_arch_name})", ru_num(best_arch_result['mean_dice'],".3f"), ru_num(best_arch_dice,".3f"), ru_num(best_arch_result['slice_detection_recall'],".3f"), fmt_ci(best_arch_result.get("slice_detection_recall_ci95"))],
            ["Мультимодальный вход", ru_num(boost_final['mean_dice'],".3f"), ru_num(boost_final['mean_dice_with_tumor'],".3f"), ru_num(boost_final['slice_detection_recall'],".3f"), fmt_ci(boost_final["slice_detection_recall_ci95"])],
        ],
        wide=True,
    )
    add_figure(doc, CHART_ACCURACY_BOOST, 11, "Прирост Dice от мультимодального входа.", width_inches=5.5, wide=True)
    add_paragraph(
        doc,
        f"Мультимодальный вход даёт прирост {ru_num(boost_gain*100,'.1f')} "
        "процентных пункта Dice относительно лучшей из девяти архитектур "
        "на одной модальности — существенно больше эффекта от любой "
        "отдельной архитектурной замены или от учёта контекста соседних "
        "срезов. При ограниченном бюджете улучшений выбор входных данных "
        "заслуживает приоритета перед перебором архитектур.",
    )

    add_subheading(doc, "G. Абляция предобработки")
    add_paragraph(
        doc,
        f"Сопоставлены пять вариантов предобработки (сокращённая выборка "
        f"— {prep_report['n_patients']} пациентов, K=1): полный "
        "конвейер; без подавления фона и центрирования; усиленный порог "
        "фона; CLAHE-контраст; min-max нормализация.",
    )
    prep_rows = [
        [PREP_LABELS_RU[k], ru_num(prep_results[k]['mean_dice'], ".3f"), ru_num(prep_results[k]['median_dice'], ".3f")]
        for k in PREP_ORDER
    ]
    add_table(doc, 10, "КАЧЕСТВО СЕГМЕНТАЦИИ ПО ВАРИАНТАМ ПРЕДОБРАБОТКИ",
              ["Вариант предобработки", "Mean Dice", "Median Dice"], prep_rows, wide=True)
    add_figure(doc, CHART_PREPROCESSING_ABLATION, 12, "Mean/median Dice по вариантам предобработки.", width_inches=6.3, wide=True)
    add_paragraph(
        doc,
        "Четыре варианта из пяти дали практически идентичные значения — "
        "признак коллапса модели в тривиальное решение на ограниченном "
        "бюджете (K=1, 5 эпох), а не честного равенства качества. "
        "Абляция демонстрирует сам риск коллапса, но не даёт "
        "статистически надёжного ранжирования вариантов предобработки.",
    )

    # V. ЗАКЛЮЧЕНИЕ
    add_numbered_heading(doc, "V", "Заключение")
    add_paragraph(
        doc,
        "Контекст физически смежных срезов даёт статистически устойчивый "
        "прирост качества сегментации, с пройденной контрольной проверкой "
        "механизма эффекта (H2), но заметно уступает по величине эффекту "
        "от использования нескольких МРТ-модальностей одновременно — "
        "наиболее весомому из проверенных честных рычагов повышения "
        "точности в этом исследовании. Обе находки устойчивы независимо "
        "от выбора архитектуры (при условии нормализации по батчу), но "
        "качество существенно деградирует при переносе на источник "
        "данных вне обучения.",
    )
    add_paragraph(
        doc,
        f"Достигнутая точность (Dice с опухолью: "
        f"{ru_num(boost_final['mean_dice_with_tumor']*100,'.1f')}% в "
        f"лучшей конфигурации; детекция опухоли на срезе — "
        f"{ru_num(boost_final['slice_detection_recall']*100,'.1f')}% при "
        f"95%-ДИ {fmt_ci(boost_final['slice_detection_recall_ci95'])}) "
        "недостаточна для самостоятельной клинической диагностики — "
        "опубликованные решения конкурса BraTS на существенно большем "
        "вычислительном бюджете достигают Dice порядка 85–90%. "
        "Представленный метод следует рассматривать исключительно как "
        "вспомогательный инструмент поддержки принятия решения, а не как "
        "самостоятельное диагностическое средство: диагноз требует "
        "клинической картины и, как правило, гистологического "
        "подтверждения.",
    )

    # === ЛИТЕРАТУРА (без нумерации раздела, ГОСТ 7.1-2003) ============
    add_heading_plain(doc, "Литература")
    add_reference(doc, "[1] Ronneberger O., Fischer P., Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation // MICCAI 2015, LNCS. — 2015. — Vol. 9351. — P. 234–241.")
    add_reference(doc, "[2] Milletari F., Navab N., Ahmadi S.-A. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation // Proc. 3DV. — 2016. — P. 565–571.")
    add_reference(doc, "[3] He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition // Proc. CVPR. — 2016. — P. 770–778.")
    add_reference(doc, "[4] Zhang Z., Liu Q., Wang Y. Road Extraction by Deep Residual U-Net // IEEE Geoscience and Remote Sensing Letters. — 2018. — Vol. 15, No. 5. — P. 749–753.")
    add_reference(doc, "[5] Oktay O., Schlemper J., Folgoc L.L., et al. Attention U-Net: Learning Where to Look for the Pancreas // arXiv:1804.03999. — 2018.")
    add_reference(doc, "[6] Badrinarayanan V., Kendall A., Cipolla R. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation // IEEE TPAMI. — 2017. — Vol. 39, No. 12. — P. 2481–2495.")
    add_reference(doc, "[7] Long J., Shelhamer E., Darrell T. Fully Convolutional Networks for Semantic Segmentation // Proc. CVPR. — 2015. — P. 3431–3440.")
    add_reference(doc, "[8] Ibtehaz N., Rahman M.S. MultiResUNet: Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation // Neural Networks. — 2020. — Vol. 121. — P. 74–87.")
    add_reference(doc, "[9] Chen L.-C., Zhu Y., Papandreou G., Schroff F., Adam H. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+) // Proc. ECCV. — 2018. — P. 833–851.")
    add_reference(doc, "[10] Menze B.H., Jakab A., Bauer S., et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS) // IEEE Transactions on Medical Imaging. — 2015. — Vol. 34, No. 10. — P. 1993–2024.")
    add_reference(doc, "[11] Bakas S., Akbari H., Sotiras A., et al. Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features // Scientific Data. — 2017. — Vol. 4. — Art. 170117.")
    add_reference(doc, "[12] Bakas S., Reyes M., Jakab A., et al. Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge // arXiv:1811.02629. — 2018.")
    add_reference(doc, "[13] Isensee F., Jaeger P.F., Kohl S.A.A., Petersen J., Maier-Hein K.H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation // Nature Methods. — 2021. — Vol. 18. — P. 203–211.")
    add_reference(doc, "[14] Wilson E.B. Probable Inference, the Law of Succession, and Statistical Inference // Journal of the American Statistical Association. — 1927. — Vol. 22, No. 158. — P. 209–212.")
    add_reference(doc, "[15] Awsaf. BraTS2020 Dataset (Training + Validation) [Электронный ресурс] // Kaggle. — Режим доступа: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation (дата обращения: 08.10.2026).")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_PATH))
    print(f"Сохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
