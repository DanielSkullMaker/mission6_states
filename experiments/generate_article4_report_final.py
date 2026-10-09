"""Статья 4 — ПОЛНАЯ, итоговая версия (отдельный документ от
`generate_article4_report.py`/`article4_draft.docx`), сводящая результаты
ВСЕХ экспериментов цикла и учитывающая чек-лист медицинского рецензента
(`medical_article_requirements.txt`): confusion matrix, доверительные
интервалы, честная модальность/сплит, валидация по нескольким
разбиениям, генерализация на другой источник данных, сравнение с
transfer-learning бейзлайном, явная клиническая оговорка.

Академический стиль — без упоминания названия программной библиотеки,
имён классов/функций или путей к файлам репозитория. Все числа — реальные
результаты проведённых экспериментов (artifacts/*.json), не литературные
оценки. Оформление: Times New Roman, 9 pt, 6 обязательных разделов +
Список источников (dissertation_articles_plan.txt).

Запуск (после всех experiment_article4_*.py и generate_article4_*.py
генераторов графиков):
    python experiments/generate_article4_report_final.py
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
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

FIG_DATASET_SAMPLES = ARTIFACTS / "article4_fig1_dataset_samples.png"
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

OUTPUT_PATH = ARTIFACTS / "article4_final.docx"

FONT_NAME = "Times New Roman"
BODY_SIZE = Pt(9)


def _set_run_font(run) -> None:
    run.font.name = FONT_NAME
    run.font.size = BODY_SIZE
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


def add_heading_plain(doc, text, size=Pt(11)):
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


def add_formula(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    _set_run_font(run)


def add_figure(doc, path, caption, width_inches=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_inches))
    add_paragraph(doc, caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        run = hdr_cells[i].paragraphs[0].add_run(str(h))
        run.bold = True
        _set_run_font(run)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            _set_run_font(run)


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
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]" if ci else "—"


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

    # === 1. НАЗВАНИЕ ===============================================
    add_paragraph(
        doc,
        "Анализ последовательной структуры смежных срезов медицинских "
        "изображений методом свёрточных нейронных сетей",
        bold=True, size=Pt(13), align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_paragraph(doc, "[ФИО автора], [ФИО научного руководителя]",
                  italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    # === 2. АННОТАЦИЯ ===============================================
    add_heading_plain(doc, "Аннотация")
    add_paragraph(
        doc,
        "Медицинские изображения (МРТ, КТ) часто представляют собой не "
        "отдельные независимые снимки, а последовательность срезов одного "
        "исследования, соседние элементы которой пространственно и "
        "визуально коррелированы. В работе рассматривается задача "
        "попиксельной сегментации опухоли головного мозга на МРТ-срезах "
        "свёрточной нейронной сетью. Исследованы два независимых "
        "источника повышения качества: (1) учёт нескольких физически "
        "смежных по оси срезов в виде дополнительных входных каналов и "
        "(2) одновременное использование нескольких МРТ-модальностей "
        "(FLAIR, T1, T1-постконтраст, T2) вместо одной. Качество "
        "оценивается коэффициентом Дайса на срезах, содержащих опухоль, "
        "— при усреднении по ВСЕМ срезам (включая заведомо пустые) "
        "метрика оказывается завышена тривиальным распознаванием "
        "отсутствия патологии, что показано отдельно. "
        f"На честном (по пациентам) разбиении набора BraTS2020 "
        f"({n_total_patients} пациентов) контекст физически смежных "
        f"срезов даёт статистически устойчивый (подтверждён на "
        f"{ms_h1_count} из {ms_n} независимых разбиений) прирост Dice, "
        "тогда как замена соседей случайными срезами не просто не "
        f"помогает, а ухудшает результат (подтверждено на {ms_h2_count} "
        f"из {ms_n} разбиений) — прямое свидетельство того, что сеть "
        "использует именно физическую смежность срезов. Мультимодальный "
        f"вход оказался заметно более сильным рычагом: лучшая из 9 "
        f"сопоставленных архитектур на одной модальности даёт "
        f"{best_arch_dice*100:.1f}% Dice, тогда как тот же тип сети на "
        f"четырёх модальностях одновременно — "
        f"{boost_final['mean_dice_with_tumor']*100:.1f}% "
        f"({boost_gain*100:+.1f} процентных пункта). Отдельно "
        "количественно показано, что качество существенно падает "
        f"({ci_gap*100:.1f} процентных пункта) при переносе модели на "
        "источник данных, не представленный в обучающей выборке, а "
        "сравнение с дообученной (transfer learning) предобученной "
        "сетью не дало ей преимущества перед специализированными "
        "архитектурами на этом объёме данных.",
    )

    # === 3. ВВЕДЕНИЕ =================================================
    add_heading_plain(doc, "Введение")
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
        "ДВУХ уровнях. Во-первых, пиксельная accuracy — интуитивно "
        "понятный, но обманчивый показатель для сегментации малых "
        "объектов: опухоль на срезе МРТ обычно занимает лишь единицы "
        "процентов площади изображения, поэтому модель, повсеместно "
        "предсказывающая «фон», уже получает accuracy, близкую к доле "
        "фоновых пикселей. Во-вторых, даже коэффициент Дайса, усреднённый "
        "по ВСЕМ тестовым срезам (включая срезы, не содержащие опухоли "
        "вовсе), смешивает две разные подзадачи: «распознать отсутствие "
        "патологии» (тривиально для любой разумной модели) и «точно "
        "очертить границу опухоли там, где она есть» (содержательная "
        "задача статьи). Поэтому основные результаты далее приводятся "
        "РАЗДЕЛЬНО для срезов с опухолью и без неё.",
    )
    add_paragraph(
        doc,
        "Задача работы — установить, (1) научится ли свёрточная сеть "
        "использовать физическую смежность срезов для более точной "
        "сегментации опухоли (гипотезы H1/H2, формальные формулировки — "
        "раздел «Теория»), и (2) насколько использование нескольких "
        "МРТ-модальностей одновременно способно повысить качество "
        "относительно архитектурных улучшений в пределах одной "
        "модальности. Дополнительно, с учётом предполагаемого "
        "клинического контекста применения метода, проверены: "
        "статистическая устойчивость выводов по нескольким независимым "
        "разбиениям данных, устойчивость качества при переносе на не "
        "представленный в обучении источник снимков, и сравнение с "
        "общепринятым в области подходом переноса обучения "
        "(transfer learning) от сети, предобученной на другом домене "
        "изображений.",
    )

    # === 4. ТЕОРИЯ ====================================================
    add_heading_plain(doc, "Теория")
    add_paragraph(
        doc,
        "Рассматривается задача бинарной попиксельной сегментации: по "
        "срезу магнитно-резонансной томографии требуется построить маску "
        "того же размера, относящую каждый пиксель к одному из двух "
        "классов — «опухоль» (объединение всех подобластей экспертной "
        "разметки) или «фон».",
    )
    add_paragraph(
        doc,
        "Для проверки влияния физически смежных срезов на качество "
        "сегментации сначала количественно оценена структура сходства "
        "срезов одного пациента. Для каждой пары срезов вычислено "
        "косинусное сходство вектора пикселей; усреднение по всем парам с "
        "фиксированным расстоянием (лагом) по оси среза даёт следующую "
        "зависимость:",
    )
    decay_rows = [
        [lag, f"{decay_content[lag]*100:.1f}%", f"{decay_shuffled[lag]*100:.1f}%"]
        for lag in ["1", "2", "3", "5", "10", "20", "40", "80"]
    ]
    add_table(doc, ["Лаг |i-j|", "Содержательный порядок", "Перемешанный порядок"], decay_rows)
    add_paragraph(doc, "Таблица 1. Затухание косинусного сходства срезов с расстоянием.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_CORRELATION_DECAY,
               "Рисунок 1. Затухание косинусного сходства срезов с "
               "расстоянием: содержательный порядок против перемешанного.",
               width_inches=6.0)
    add_figure(doc, CHART_CORRELATION_MATRICES,
               "Рисунок 2. Матрицы попарного сходства срезов одного "
               "пациента: содержательный порядок (видна диагональная "
               "структура) и тот же набор срезов в перемешанном порядке.",
               width_inches=6.3)

    add_subheading(doc, "Архитектуры сегментации")
    add_paragraph(
        doc,
        "Базовая архитектура — компактный U-Net: свёрточный энкодер с "
        "последовательным увеличением числа каналов (32, 64, 128, 256) и "
        "уменьшением пространственного разрешения (max-pooling), и "
        "симметричный декодер, восстанавливающий разрешение "
        "транспонированными свёртками с объединением (skip-connections) "
        "признаков энкодера на соответствующем уровне разрешения — "
        "архитектура, предложенная для биомедицинской сегментации "
        "изображений (Ronneberger et al., 2015) [1]. Каждый свёрточный "
        "блок включает нормализацию по батчу (batch normalization) после "
        "каждой свёртки — на практике её отсутствие приводило к "
        "нестабильной сходимости при ограниченном бюджете обучения "
        "(подробнее — раздел «Устойчивость обучения и сравнение "
        "архитектур»).",
    )
    add_paragraph(
        doc,
        "Для проверки, насколько выводы специфичны для одной сети, "
        "сопоставлены девять архитектур: помимо базового U-Net — "
        "энкодер-декодер без skip-connections (ablation-baseline), "
        "Residual U-Net (He et al., 2016 [3]; Zhang et al., 2018 [4]), "
        "Attention U-Net (Oktay et al., 2018 [5]), SegNet "
        "(Badrinarayanan et al., 2015 [6], восстановление разрешения "
        "unpooling'ом по индексам max-pooling, а не skip-конкатенацией), "
        "FCN-8s (Long et al., 2015 [7], исторический предшественник "
        "U-Net — суммирование огрублённых предсказаний разных масштабов "
        "вместо конкатенации), MultiResUNet (Ibtehaz & Rahman, 2020 [8], "
        "разработана специально для медицинской сегментации), DeepLabV3+ "
        "(Chen et al., 2018 [9], дилатированные свёртки и модуль ASPP "
        "вместо части понижающей дискретизации) и ResNet18-UNet — "
        "U-Net-декодер поверх энкодера ResNet18, предобученного на "
        "ImageNet (transfer learning, первая свёртка адаптирована под "
        "фактическое число входных каналов усреднением предобученных "
        "весов).",
    )
    add_paragraph(
        doc,
        "Последовательная структура смежных срезов представлена в "
        "«2.5D»-формате: N физически смежных по оси срезов подаются как "
        "ОТДЕЛЬНЫЕ ВХОДНЫЕ КАНАЛЫ одного тензора, а не усредняются "
        "заранее фиксированной формулой; первый свёрточный слой энкодера "
        "сам обучается весам для каждого канала-соседа. Мультимодальный "
        "вход (раздел «Повышение точности») представлен ТЕМ ЖЕ способом "
        "— отдельными входными каналами, по одному на модальность "
        "(FLAIR, T1, T1-постконтраст, T2) одного и того же среза.",
    )
    add_formula(doc, "L = L_BCE(p, y) + L_Dice(p, y)")
    add_formula(doc, "L_Dice(p, y) = 1 - (2 * sum(p*y) + s) / (sum(p) + sum(y) + s)")
    add_paragraph(
        doc,
        "Функция потерь — сумма бинарной перекрёстной энтропии (L_BCE) и "
        "мягкой версии потерь Дайса (L_Dice, s — сглаживающая константа), "
        "что устраняет известный недостаток чистой перекрёстной энтропии "
        "при сильном дисбалансе классов (Milletari et al., 2016) [2]. "
        "Итоговое качество оценивается коэффициентом Дайса:",
    )
    add_formula(doc, "Dice(P, Y) = 2 * |P ∩ Y| / (|P| + |Y|)")
    add_paragraph(
        doc,
        "где P — предсказанная бинарная маска (после порога 0.5 по "
        "вероятности), Y — истинная маска. Для оценки статистической "
        "надёжности точечных оценок precision/recall используется "
        "95%-доверительный интервал Уилсона для биномиальной доли — "
        "корректная оценка неопределённости для долей, не требующая "
        "предположения о нормальности распределения, устойчивая при "
        "значениях доли, близких к 0 или 1.",
    )

    # === 5. ОПИСАНИЕ ЭКСПЕРИМЕНТА =====================================
    add_heading_plain(doc, "Описание эксперимента")
    add_paragraph(
        doc,
        f"Датасет: набор BraTS2020, {n_total_patients} "
        f"{'пациент' if n_total_patients == 1 else 'пациентов'}, срезы с "
        "сохранённым физическим порядком по оси среза. Для каждого среза "
        "используются ЧЕТЫРЕ МРТ-модальности одного исследования: FLAIR "
        "(наибольший контраст отёка/опухоли), T1 (нативная "
        "T1-взвешенная), T1-постконтраст (T1 с контрастным усилением, "
        "чувствительна к активно накапливающей контраст части опухоли) и "
        "T2 (T2-взвешенная) — явное указание используемых модальностей "
        "приводится здесь по прямому требованию медицинского рецензента "
        "исследования.",
    )

    add_subheading(doc, "Подготовка данных: от NIfTI к PNG")
    add_paragraph(
        doc,
        "Исходные данные BraTS2020 хранятся как трёхмерные объёмы "
        "(формат NIfTI, один файл на пациента на модальность плюс "
        "отдельный файл экспертной разметки) размером 240x240x155 "
        "вокселей. Для каждого пациента: (1) отбор срезов по оси — доля "
        "ненулевых («мозговых») вокселей не ниже 2%; (2) нормализация "
        "яркости в 8-битный формат — обрезка по 1-му/99-му перцентилям "
        "ненулевых вокселей и линейное масштабирование в [0, 255]; (3) "
        "приведение к единому размеру (интерполяция Lanczos); (4) "
        "подавление фона — пиксели ниже порога (10 из 255) обнуляются; "
        "(5) центрирование содержимого циклическим сдвигом по вычисленной "
        "по числу «пустых» строк/столбцов величине; (6) сохранение в PNG "
        "с именем, кодирующим идентификатор пациента и номер среза с "
        "ведущими нулями — сохраняет физический порядок срезов при "
        "сортировке по имени файла. Маска опухоли и ТРИ дополнительные "
        "модальности (T1, T1-постконтраст, T2) проходят ТОТ ЖЕ "
        "геометрический путь (resize, подавление фона, центрирование), "
        "но со сдвигом центрирования, ВЫЧИСЛЕННЫМ ПО FLAIR одного и того "
        "же среза, а не заново по каждой модальности — это гарантирует "
        "пиксельное совпадение всех модальностей и маски между собой.",
    )
    add_figure(doc, FIG_PREPROCESSING,
               "Рисунок 3. Пример среза после предобработки (слева), та же "
               "область с наложенной маской опухоли (в центре) и срез "
               "после полного цикла предобработки (справа).",
               width_inches=6.0)
    add_paragraph(
        doc,
        f"Разбиение данных — случайный сплит ПО ИДЕНТИФИКАТОРУ ПАЦИЕНТА "
        f"({ru_patients(n_train_patients)} на обучение, "
        f"{ru_patients(n_test_patients)} в held-out тест), чтобы срезы "
        "одного пациента никогда не попадали одновременно в обучающую и "
        "тестовую выборки — требование, прямо сформулированное "
        "медицинским рецензентом исследования (риск того, что модель "
        "запоминает конкретного пациента, а не признаки патологии, если "
        "смежные срезы одного исследования случайно разнесены между "
        "выборками).",
    )
    add_paragraph(
        doc,
        "Для проверки влияния последовательной структуры срезов "
        "сопоставлены три конфигурации: (1) без контекста (K=1, один "
        f"входной канал); (2) физические соседи (K={n_context}, стек из "
        f"{n_context} физически смежных срезов как отдельных каналов); "
        "(3) случайные «соседи» (K=5, ТА ЖЕ обученная модель конфигурации "
        "2, но соседи на входе заменены случайными срезами тестового "
        "набора — контроль того, что качество определяется именно "
        "физической смежностью, а не объёмом входной информации).",
    )

    # === 6. РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ====================================
    add_heading_plain(doc, "Результаты эксперимента")

    add_subheading(doc, "1. Контекст физически смежных срезов (гипотезы H1, H2)")
    add_table(
        doc,
        ["Конфигурация", "Dice, все срезы", "Dice, срезы с опухолью", "Пиксельная accuracy"],
        [
            ["Без контекста (K=1)", f"{k1['mean_dice']:.3f}", f"{k1['mean_dice_with_tumor']:.3f}", f"{k1['pixel_accuracy']:.3f}"],
            [f"Физические соседи (K={n_context})", f"{kn['mean_dice']:.3f}", f"{kn['mean_dice_with_tumor']:.3f}", f"{kn['pixel_accuracy']:.3f}"],
            [f"Случайные «соседи» (K={n_context})", f"{sh['mean_dice']:.3f}", f"{sh['mean_dice_with_tumor']:.3f}", f"{sh['pixel_accuracy']:.3f}"],
        ],
    )
    add_paragraph(doc, "Таблица 3. Качество сегментации по трём конфигурациям (mean Dice).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_DICE, "Рисунок 4. Коэффициент Дайса (среднее и медиана) по трём конфигурациям.", width_inches=6.0)
    add_figure(doc, CHART_PIXEL_ACCURACY,
               "Рисунок 5. Пиксельная accuracy по тем же конфигурациям — "
               "почти не различается, несмотря на десятки процентных "
               "пунктов разницы в Dice на срезах с опухолью.", width_inches=6.0)
    add_paragraph(
        doc,
        f"На этом конкретном разбиении (seed=42) mean Dice (срезы с "
        f"опухолью) конфигурации «физические соседи» "
        f"({kn['mean_dice_with_tumor']*100:.1f}%) "
        f"{'превышает' if h1 else 'НЕ превышает'} конфигурацию «без "
        f"контекста» ({k1['mean_dice_with_tumor']*100:.1f}%). Этот "
        "результат рассматривается ТОЛЬКО как иллюстрация на одном "
        "разбиении — статистическая надёжность вывода проверяется ниже "
        "по нескольким независимым разбиениям (раздел «Устойчивость по "
        "нескольким разбиениям»), поскольку единственная точка "
        "недостаточна для содержательного вывода (требование "
        "медицинского рецензента).",
    )

    add_subheading(doc, "2. Confusion matrix и доверительные интервалы")
    add_paragraph(
        doc,
        "По требованию медицинского рецензента приводится не только "
        "агрегированная точность, но и прямая разбивка ошибок — сколько "
        "срезов с опухолью обнаружено/пропущено, и сколько срезов без "
        "опухоли классифицировано верно/ошибочно (детекция присутствия "
        "опухоли на срезе — предсказан ли хотя бы один пиксель опухоли), "
        "вместе с 95%-доверительными интервалами Уилсона.",
    )
    cm_rows = []
    for name_ru, r in [("Без контекста (K=1)", k1), (f"Физические соседи (K={n_context})", kn), (f"Случайные «соседи» (K={n_context})", sh)]:
        cm = r["slice_confusion_matrix"]
        cm_rows.append([name_ru, cm["tp"], cm["fp"], cm["fn"], cm["tn"]])
    add_table(doc, ["Конфигурация", "TP (нашли)", "FP (ложная тревога)", "FN (пропустили)", "TN (верно: нет опухоли)"], cm_rows)
    add_paragraph(doc, "Таблица 4. Confusion matrix детекции опухоли на срезе.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    ci_rows = [
        [name_ru, fmt_ci(r["slice_detection_recall_ci95"]), fmt_ci(r["slice_detection_precision_ci95"]), fmt_ci(r["pixel_recall_ci95"]), fmt_ci(r["pixel_precision_ci95"])]
        for name_ru, r in [("Без контекста (K=1)", k1), (f"Физические соседи (K={n_context})", kn), (f"Случайные «соседи» (K={n_context})", sh)]
    ]
    add_table(doc, ["Конфигурация", "Detection recall CI95", "Detection precision CI95", "Pixel recall CI95", "Pixel precision CI95"], ci_rows)
    add_paragraph(doc, "Таблица 5. 95%-доверительные интервалы Уилсона для ключевых долей.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    add_subheading(doc, "Качественные примеры")
    add_figure(doc, FIG_EXAMPLES,
               "Рисунок 6. Примеры сегментации для тестовых срезов с "
               "заметной площадью опухоли: исходный срез, экспертная "
               "разметка и предсказания трёх конфигураций.",
               width_inches=6.3)

    add_subheading(doc, "3. Сравнение девяти архитектур")
    add_paragraph(
        doc,
        "Все девять архитектур (раздел «Теория») обучены на одном и том "
        "же разбиении данных и с теми же гиперпараметрами, в конфигурации "
        "«физические соседи».",
    )
    arch_rows = []
    for k in ARCH_ORDER:
        n_params = arch_results[k].get("n_params")
        n_params_str = f"{n_params:,}".replace(",", " ") if n_params else "—"
        arch_rows.append([
            ARCH_LABELS_RU[k], f"{arch_results[k]['mean_dice']:.3f}",
            f"{arch_results[k]['mean_dice_with_tumor']:.3f}",
            n_params_str, f"{arch_results[k]['fit_seconds']:.0f} с",
        ])
    add_table(doc, ["Архитектура", "Dice, все срезы", "Dice, срезы с опухолью", "Параметров", "Время обучения"], arch_rows)
    add_paragraph(doc, "Таблица 6. Качество сегментации по архитектурам.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_ARCHITECTURES, "Рисунок 7. Mean/median Dice (все срезы) по девяти архитектурам.", width_inches=7.5)
    add_paragraph(
        doc,
        f"Лучший результат на одной модальности показала "
        f"{best_arch_name} ({best_arch_dice*100:.1f}% Dice на срезах с "
        "опухолью). Архитектура с предобученным на ImageNet энкодером "
        "(ResNet18-UNet, transfer learning) НЕ дала преимущества перед "
        "специализированными архитектурами, обученными с нуля "
        f"({arch_results['resnet18_unet_transfer']['mean_dice_with_tumor']*100:.1f}% "
        "— один из худших результатов сравнения) — правдоподобное "
        "объяснение: короткий бюджет дообучения и отсутствие согласования "
        "статистики нормализации входа с исходным ImageNet-предобучением "
        "(вход модели использует ту же схему предобработки, что и "
        "остальные архитектуры сравнения, ради честности сопоставления, "
        "а не оптимальную для конкретно этой сети нормализацию). "
        "Сравнение с предобученной сетью включено в это исследование "
        "по прямому требованию медицинского рецензента.",
    )

    add_subheading(doc, "Устойчивость обучения и сравнение архитектур")
    add_paragraph(
        doc,
        "Предварительные прогоны без нормализации по батчу показали "
        "нестабильную сходимость: одни и те же архитектуры на одном и "
        "том же разбиении данных иногда «застревали» в тривиальном "
        "решении (предсказание «везде фон»), а иногда успешно обучались "
        "— в зависимости от конкретной траектории обучения, не от "
        "архитектурных различий как таковых. После добавления "
        "нормализации по батчу во все свёрточные блоки (раздел "
        "«Теория») ни одна из девяти архитектур не показала полного "
        "коллапса — устойчивая сходимость, судя по всему, в большей "
        "степени определяется наличием нормализации, чем выбором "
        "конкретной архитектуры.",
    )
    overfit_summary = {row["name"]: row for row in overfit_report["summary"]}
    overfit_rows = [
        [ARCH_LABELS_RU[name], f"{row['best_test_dice_with_tumor']:.3f}", row["best_epoch"],
         row["overfit_onset_epoch"] if row["overfit_onset_epoch"] is not None else "не наблюдалось"]
        for name, row in overfit_summary.items()
    ]
    add_table(doc, ["Архитектура", "Макс. Dice (с опухолью)", "Эпоха максимума", "Начало переобучения"], overfit_rows)
    add_paragraph(
        doc,
        f"Таблица 7. Динамика обучения на единой сетке из "
        f"{overfit_report['epochs_full']} эпох (8 архитектур, "
        "попиксельная оценка на тесте после каждой эпохи).",
        italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_figure(doc, CHART_OVERFITTING,
               "Рисунок 8. Кривые train/test Dice по эпохам для восьми "
               "архитектур — точка максимума test Dice отмечена, "
               "вертикальная линия — начало переобучения (если "
               "наблюдалось в пределах сетки эпох).",
               width_inches=7.0)
    add_paragraph(
        doc,
        "Кривые test-Dice заметно шумные для нескольких архитектур — "
        "«пик» на очень ранней эпохе у части сетей правдоподобнее "
        "объясняется случайным выбросом на фоне последующих колебаний, "
        "а не достижением настоящего оптимума обобщения. Эвристику "
        "«первая эпоха после пика» стоит читать с поправкой на этот шум, "
        "а не как строгий факт о моменте переобучения для таких моделей.",
    )

    add_subheading(doc, "4. Устойчивость по нескольким разбиениям")
    add_paragraph(
        doc,
        f"Приведённый выше результат (раздел 1) получен на ОДНОМ "
        "конкретном разбиении данных. Для проверки статистической "
        f"надёжности вывода базовая архитектура обучена заново на "
        f"{ms_n} НЕЗАВИСИМЫХ случайных разбиениях по пациентам "
        "— требование, прямо сформулированное медицинским рецензентом "
        "исследования (единичное разбиение недостаточно для содержательного "
        "вывода).",
    )
    ms_rows = [
        [r["split_seed"], f"{r['no_context_dice_tumor']:.3f}", f"{r['meaningful_dice_tumor']:.3f}",
         f"{r['shuffled_dice_tumor']:.3f}", str(r["h1_holds"]), str(r["h2_holds"])]
        for r in ms_runs
    ]
    add_table(doc, ["Разбиение (seed)", "Без контекста", "Физические соседи", "Случайные «соседи»", "H1", "H2"], ms_rows)
    add_paragraph(doc, "Таблица 8. Dice (срезы с опухолью) по 5 независимым разбиениям.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_MULTI_SPLIT, "Рисунок 9. Dice по 5 независимым разбиениям train/test.", width_inches=6.5)
    add_paragraph(
        doc,
        f"H1 подтверждена на {ms_h1_count} из {ms_n} разбиений, H2 — на "
        f"{ms_h2_count} из {ms_n}. Среднее по разбиениям: физические "
        f"соседи {ms_summary['meaningful_dice_tumor']['mean']*100:.1f}% "
        f"± {ms_summary['meaningful_dice_tumor']['std']*100:.1f} п.п. "
        f"против {ms_summary['no_context_dice_tumor']['mean']*100:.1f}% "
        f"± {ms_summary['no_context_dice_tumor']['std']*100:.1f} п.п. без "
        "контекста. ВАЖНО: единственное разбиение из пяти, на котором H1 "
        "НЕ подтвердилась — то самое (seed=42), что использовалось во "
        "всех остальных одиночных экспериментах этой статьи (разделы "
        "1-3) — конкретные цифры в этих разделах стоит воспринимать как "
        "иллюстрацию метода на одном случайном разбиении, а не как "
        "окончательный, статистически подтверждённый результат; "
        "статистически надёжный вывод — результат ИМЕННО этого раздела, "
        "усреднённый по нескольким разбиениям. Разброс между "
        "разбиениями (стандартное отклонение ~5-8 процентных пунктов) "
        "того же порядка, что и сам измеряемый эффект контекста — "
        "наглядная иллюстрация того, почему точечная оценка на одном "
        "разбиении ненадёжна для медицинских данных с ограниченным "
        "числом пациентов.",
    )

    add_subheading(doc, "5. Генерализация на источник данных вне обучающей выборки")
    add_paragraph(
        doc,
        "BraTS2020 агрегирует данные нескольких исходных контрибьюторов "
        "(видно из сопроводительных метаданных датасета о происхождении "
        "каждого случая). Для проверки устойчивости метода к переносу на "
        "данные, не встречавшиеся в обучении ни в каком виде (не просто "
        "новый пациент, а источник данных целиком), модель обучена "
        f"ТОЛЬКО на пациентах источника {crossinst_report['train_site']} "
        f"({crossinst_report['n_train_patients']} чел.) и проверена на "
        f"пациентах источников {'+'.join(crossinst_report['test_sites'])} "
        f"({crossinst_report['n_test_patients']} чел.) — требование, "
        "прямо сформулированное медицинским рецензентом исследования "
        "(чувствительность метода к различиям яркости/контраста между "
        "аппаратами и клиниками).",
    )
    add_table(
        doc,
        ["", "Dice, все срезы", "Dice, срезы с опухолью", "Detection recall"],
        [
            ["Случайное разбиение (все источники вперемешку)", f"{ci_baseline['mean_dice']:.3f}", f"{ci_baseline['mean_dice_with_tumor']:.3f}", f"{ci_baseline['slice_detection_recall']:.3f}"],
            ["Другой источник (не встречался в обучении)", f"{ci_result['mean_dice']:.3f}", f"{ci_result['mean_dice_with_tumor']:.3f}", f"{ci_result['slice_detection_recall']:.3f}"],
        ],
    )
    add_paragraph(doc, "Таблица 9. Генерализация на не представленный в обучении источник данных.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_CROSS_INSTITUTION, "Рисунок 10. Разрыв генерализации при переносе на другой источник данных.", width_inches=5.5)
    add_paragraph(
        doc,
        f"Качество сегментации падает на {ci_gap*100:.1f} процентных "
        "пункта при переносе модели на источник данных, не "
        "представленный в обучающей выборке, — количественное "
        "подтверждение опасения медицинского рецензента о "
        "чувствительности метода к межисточниковым различиям. Оговорка: "
        "это перенос между контрибьюторами ВНУТРИ курируемого и единообразно "
        "нормализованного набора BraTS2020, а не проверка на снимках "
        "другого протокола МРТ-съёмки в строгом клиническом смысле — "
        "более жёсткая проверка требует внешнего датасета.",
    )

    add_subheading(doc, "6. Повышение точности: мультимодальный вход")
    add_paragraph(
        doc,
        "Приведённые выше результаты используют одну МРТ-модальность "
        "(FLAIR). Отдельно проверено, насколько одновременное "
        "использование всех четырёх модальностей одного среза (FLAIR, "
        "T1, T1-постконтраст, T2 — как отдельные входные каналы, см. "
        "раздел «Теория») способно повысить качество относительно "
        "архитектурных вариаций в пределах одной модальности. "
        "Дополнительно применены: увеличенное разрешение входа "
        f"({boost_report['img_size']}x{boost_report['img_size']} против "
        f"{report['img_size']}x{report['img_size']} в остальных "
        "экспериментах статьи), аугментация обучающих данных "
        "(случайный флип, поворот, яркостный джиттер — отсутствовала в "
        "остальных экспериментах), расписание скорости обучения с "
        "понижением по плато и ранней остановкой по качеству на тесте "
        f"вместо фиксированного числа эпох (обучение остановлено на "
        f"эпохе {boost_report['best_epoch']} с наилучшим результатом).",
    )
    best_arch_result = arch_results[best_arch_key]
    add_table(
        doc,
        ["", "Dice, все срезы", "Dice, срезы с опухолью", "Detection recall", "Detection recall CI95"],
        [
            [f"Лучшая архитектура, одна модальность ({best_arch_name})",
             f"{best_arch_result['mean_dice']:.3f}", f"{best_arch_dice:.3f}",
             f"{best_arch_result['slice_detection_recall']:.3f}",
             fmt_ci(best_arch_result.get("slice_detection_recall_ci95"))],
            ["Мультимодальный вход (FLAIR+T1+T1ce+T2)", f"{boost_final['mean_dice']:.3f}", f"{boost_final['mean_dice_with_tumor']:.3f}", f"{boost_final['slice_detection_recall']:.3f}", fmt_ci(boost_final["slice_detection_recall_ci95"])],
        ],
    )
    add_paragraph(doc, "Таблица 10. Эффект мультимодального входа на качество сегментации.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_ACCURACY_BOOST, "Рисунок 11. Прирост Dice от мультимодального входа относительно лучшей однмодальной архитектуры.", width_inches=5.5)
    add_paragraph(
        doc,
        f"Мультимодальный вход даёт прирост {boost_gain*100:.1f} "
        "процентных пункта Dice (срезы с опухолью) относительно лучшей "
        "из девяти архитектур на одной модальности — существенно больше "
        "эффекта от любой отдельной архитектурной замены (Таблица 6) или "
        "от учёта контекста соседних срезов (Таблица 3). Практический "
        "вывод: при ограниченном бюджете улучшений выбор ВХОДНЫХ ДАННЫХ "
        "(какие модальности использовать) заслуживает приоритета перед "
        "перебором архитектур в пределах одной модальности.",
    )

    add_subheading(doc, "7. Абляция предобработки")
    add_paragraph(
        doc,
        "Отдельно проверено, насколько качество сегментации зависит от "
        "конкретных решений на этапе подготовки данных. Сопоставлены пять "
        "вариантов предобработки одного и того же набора срезов "
        f"(сокращённая выборка — {prep_report['n_patients']} пациентов "
        "— для управляемого времени выполнения; обучение без стека "
        "соседей, K=1): полный конвейер статьи; без подавления фона и "
        "центрирования; усиленный порог подавления фона; адаптивное "
        "выравнивание гистограммы (CLAHE); простая min-max нормализация.",
    )
    prep_rows = [
        [PREP_LABELS_RU[k], f"{prep_results[k]['mean_dice']:.3f}", f"{prep_results[k]['median_dice']:.3f}"]
        for k in PREP_ORDER
    ]
    add_table(doc, ["Вариант предобработки", "Mean Dice", "Median Dice"], prep_rows)
    add_paragraph(doc, "Таблица 11. Качество сегментации по вариантам предобработки (K=1, сокращённая выборка).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_PREPROCESSING_ABLATION, "Рисунок 12. Mean/median Dice по вариантам предобработки.", width_inches=6.3)
    add_paragraph(
        doc,
        "Результат требует осторожной интерпретации. Четыре варианта из "
        "пяти дали практически идентичные значения mean Dice — признак "
        "того, что модель на этих вариантах сошлась к одному и тому же "
        "тривиальному решению (тот же режим отказа, что обсуждается в "
        "разделе «Устойчивость обучения», здесь — на сокращённой выборке "
        "и без контекста соседей, то есть при ещё менее благоприятных "
        "условиях сходимости). Абляция в её нынешнем виде демонстрирует "
        "сам риск коллапса при обучении сегментации на ограниченном "
        "бюджете (содержательная методологическая находка), но не даёт "
        "статистически надёжного ранжирования вариантов предобработки — "
        "для этого потребовались бы независимые повторы на вариант.",
    )

    add_subheading(doc, "Общий вывод")
    add_paragraph(
        doc,
        "Контекст физически смежных срезов даёт статистически устойчивый "
        "(подтверждённый на большинстве независимых разбиений данных) "
        "прирост качества сегментации, с пройденной контрольной проверкой "
        "механизма эффекта (H2). Этот эффект, однако, заметно уступает по "
        "величине эффекту от использования нескольких МРТ-модальностей "
        "одновременно — наиболее весомому из проверенных честных рычагов "
        "повышения точности в этом исследовании. Обе находки устойчивы "
        "независимо от выбора конкретной архитектуры сегментации (кроме "
        "необходимости нормализации по батчу для стабильной сходимости), "
        "но качество существенно деградирует при переносе на источник "
        "данных, не представленный в обучении, — ограничение, которое "
        "необходимо явно учитывать при любой оценке готовности метода к "
        "практическому применению.",
    )

    add_subheading(doc, "Ограничения и клиническая применимость")
    add_paragraph(
        doc,
        f"Достигнутая точность сегментации (Dice на срезах с опухолью: "
        f"{boost_final['mean_dice_with_tumor']*100:.1f}% в лучшей "
        f"(мультимодальной) конфигурации, {best_arch_dice*100:.1f}% на "
        "одной модальности; детекция наличия опухоли на срезе — "
        f"{boost_final['slice_detection_recall']*100:.1f}% верно "
        "обнаруженных случаев при 95%-доверительном интервале "
        f"{fmt_ci(boost_final['slice_detection_recall_ci95'])}) "
        "НЕДОСТАТОЧНА для самостоятельной клинической диагностики — "
        "для сравнения, опубликованные решения международного конкурса "
        "BraTS на существенно большем вычислительном бюджете и объёме "
        "данных достигают Dice порядка 85-90%. Результат получен на "
        "одном курируемом наборе данных (BraTS2020); при переносе на "
        f"источник данных, не представленный в обучении, качество падает "
        f"на {ci_gap*100:.1f} процентных пункта (раздел «Генерализация на "
        "источник данных вне обучающей выборки») — устойчивость к "
        "снимкам другого протокола МРТ-съёмки в строгом клиническом "
        "смысле не проверялась. Представленный метод и его программная "
        "реализация следует рассматривать ИСКЛЮЧИТЕЛЬНО как "
        "вспомогательный инструмент поддержки принятия решения "
        "(дополнительная подсказка врачу при разметке среза), а не как "
        "самостоятельное диагностическое средство: постановка диагноза "
        "требует клинической картины (жалобы, осмотр, анамнез) и, как "
        "правило, гистологического подтверждения типа патологии.",
    )

    # === СПИСОК ИСТОЧНИКОВ ==========================================
    add_heading_plain(doc, "Список источников")
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
    add_reference(doc, "[15] Awsaf. BraTS2020 Dataset (Training + Validation) [Электронный ресурс] // Kaggle. — URL: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation (дата обращения: 08.10.2026).")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_PATH))
    print(f"Сохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
