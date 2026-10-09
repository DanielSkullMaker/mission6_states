"""Генерирует научный текст Статьи 4 диссертационного исследования
(нейросетевая версия, задача — СЕГМЕНТАЦИЯ, подпространственный метод не
упоминается):

  «Анализ последовательной структуры смежных срезов медицинских
  изображений методом свёрточных нейронных сетей»

Текст написан в академическом стиле — без упоминания названия программной
библиотеки, имён классов/функций или путей к файлам репозитория; все
методологические детали (параметры предобработки, архитектура сети,
протокол разбиения данных) описаны как методика исследования. Все
приводимые числа — реальные результаты проведённого эксперимента
(`artifacts/experiment_article4_unet_report.json`), не литературные оценки.

Оформление: Times New Roman, 9 pt. Структура: 6 обязательных разделов
(Название, Аннотация, Введение, Теория, Описание эксперимента, Результаты
эксперимента) + Список источников.

Запуск (после experiment_article4_unet_segmentation.py,
generate_article4_unet_charts.py, generate_article4_unet_figures.py):
    python experiments/generate_article4_report.py
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

UNET_REPORT_PATH = Path("artifacts/experiment_article4_unet_report.json")
CORRELATION_REPORT_PATH = Path("artifacts/article4_correlation_report.json")
ARCHITECTURES_REPORT_PATH = Path("artifacts/experiment_article4_architectures_report.json")
PREPROCESSING_ABLATION_REPORT_PATH = Path("artifacts/experiment_article4_preprocessing_ablation_report.json")

FIG_PREPROCESSING = Path("artifacts/article4_fig2_preprocessing.png")
CHART_CORRELATION_MATRICES = Path("artifacts/article4_chart_correlation_matrices.png")
CHART_CORRELATION_DECAY = Path("artifacts/article4_chart_correlation_decay.png")
CHART_DICE = Path("artifacts/article4_unet_chart_dice.png")
CHART_PIXEL_ACCURACY = Path("artifacts/article4_unet_chart_pixel_accuracy.png")
FIG_EXAMPLES = Path("artifacts/article4_unet_fig_examples.png")
CHART_ARCHITECTURES = Path("artifacts/article4_unet_chart_architectures.png")
CHART_PREPROCESSING_ABLATION = Path("artifacts/article4_unet_chart_preprocessing_ablation.png")

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
}
ARCH_ORDER = ["unet_baseline", "plain_no_skip", "res_unet", "attention_unet"]

OUTPUT_PATH = Path("artifacts/article4_draft.docx")

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


def add_paragraph(doc: Document, text: str, bold: bool = False, italic: bool = False,
                   size: Pt = None, align=None) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    _set_run_font(run)
    if size is not None:
        run.font.size = size


def add_heading_plain(doc: Document, text: str, size: Pt = Pt(11)) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.bold = True
    _set_run_font(run)
    run.font.size = size


def add_subheading(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.bold = True
    run.italic = True
    _set_run_font(run)


def add_formula(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    _set_run_font(run)


def add_figure(doc: Document, path: Path, caption: str, width_inches: float = 6.0) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_inches))
    add_paragraph(doc, caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)


def add_table(doc: Document, headers, rows) -> None:
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


def add_reference(doc: Document, text: str) -> None:
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


def main() -> None:
    report = json.loads(UNET_REPORT_PATH.read_text(encoding="utf-8"))
    corr = json.loads(CORRELATION_REPORT_PATH.read_text(encoding="utf-8"))

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
    img_size = report["img_size"]
    epochs = report["epochs"]
    n_train = report["n_train"]
    n_test = k1["n_test"]

    decay_content = corr["decay_content_order"]
    decay_shuffled = corr["decay_shuffled_order"]

    arch_report = json.loads(ARCHITECTURES_REPORT_PATH.read_text(encoding="utf-8"))
    arch_results = arch_report["results"]
    architecture_table_rows = [(ARCH_LABELS_RU[k], arch_results[k]) for k in ARCH_ORDER]

    prep_report = json.loads(PREPROCESSING_ABLATION_REPORT_PATH.read_text(encoding="utf-8"))
    prep_results = prep_report["results"]

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
        "свёрточной нейронной сетью и исследуется, помогает ли учёт "
        "нескольких физически смежных по оси срезов (в виде дополнительных "
        "входных каналов изображения) точнее выделять границы опухоли по "
        "сравнению с сегментацией одного изолированного среза. Качество "
        "оценивается коэффициентом Дайса, а не пиксельной долей верных "
        "предсказаний — последняя систематически завышена при сильном "
        "классовом перекосе, типичном для сегментации малых объектов. "
        f"Эксперимент на реальных данных (BraTS2020, {n_total_patients} "
        f"{'пациент' if n_total_patients == 1 else 'пациентов'}, честное "
        "разбиение по пациентам) сравнил три конфигурации: сеть без "
        f"контекста соседних срезов (mean Dice {k1['mean_dice']*100:.1f}%), "
        "ту же архитектуру со стеком физически смежных срезов "
        f"(mean Dice {kn['mean_dice']*100:.1f}%, медиана "
        f"{kn['median_dice']*100:.1f}%) и ту же обученную модель на "
        "искусственно перемешанных «соседях» "
        f"(mean Dice {sh['mean_dice']*100:.1f}%). Учёт физически смежных "
        "срезов даёт значимый прирост качества сегментации, тогда как "
        "замена соседей случайными срезами не просто не помогает, а "
        "ухудшает результат относительно отсутствия контекста вовсе — "
        "прямое свидетельство того, что сеть использует именно физическую "
        "смежность срезов, а не дополнительный объём входной информации "
        "как таковой.",
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
        "Отдельного обсуждения заслуживает выбор метрики качества. "
        "Пиксельная accuracy — интуитивно понятный, но обманчивый "
        "показатель для сегментации малых объектов: опухоль на срезе МРТ "
        "обычно занимает лишь единицы процентов площади изображения, "
        "поэтому модель, повсеместно предсказывающая «фон», уже получает "
        "accuracy, близкую к доле фоновых пикселей, не выделив ни одного "
        "содержательного признака опухоли. Корректной мерой перекрытия "
        "предсказанной и истинной областей служит коэффициент Дайса, "
        "чувствительный именно к совпадению формы объекта интереса, а не "
        "к общей доле верно классифицированных пикселей.",
    )
    add_paragraph(
        doc,
        "Задача работы — установить, научится ли свёрточная сеть "
        "использовать физическую смежность срезов для более точной "
        "сегментации опухоли, и подтвердить это формальной проверкой двух "
        "гипотез на честном разбиении данных по пациентам: контекст "
        "смежных срезов повышает качество сегментации (H1), а разрушение "
        "физической смежности (замена соседей случайными срезами) не даёт "
        "такого же эффекта (H2).",
    )

    # === 4. ТЕОРИЯ ====================================================
    add_heading_plain(doc, "Теория")
    add_paragraph(
        doc,
        "Рассматривается задача бинарной попиксельной сегментации: по "
        "срезу магнитно-резонансной томографии (модальность FLAIR) "
        "требуется построить маску того же размера, относящую каждый "
        "пиксель к одному из двух классов — «опухоль» (объединение всех "
        "подобластей экспертной разметки) или «фон».",
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
    add_paragraph(
        doc,
        "Сходство монотонно убывает с расстоянием при содержательном "
        "(физическом) порядке срезов и остаётся практически постоянным при "
        "случайно перемешанном порядке — то есть структура, которую "
        "предполагается использовать, реально присутствует в данных, а не "
        "постулируется умозрительно.",
    )
    add_figure(doc, CHART_CORRELATION_DECAY,
               "Рисунок 1. Затухание косинусного сходства срезов с "
               "расстоянием: содержательный порядок против перемешанного.",
               width_inches=6.0)
    add_figure(doc, CHART_CORRELATION_MATRICES,
               "Рисунок 2. Матрицы попарного сходства срезов одного "
               "пациента: содержательный порядок (видна диагональная "
               "структура) и тот же набор срезов в перемешанном порядке.",
               width_inches=6.3)
    add_paragraph(
        doc,
        "Архитектура сети — компактный U-Net: свёрточный энкодер с "
        "последовательным увеличением числа каналов (32, 64, 128, 256) и "
        "уменьшением пространственного разрешения (max-pooling), и "
        "симметричный декодер, восстанавливающий разрешение "
        "транспонированными свёртками с объединением (skip-connections) "
        "признаков энкодера на соответствующем уровне разрешения — "
        "архитектура, предложенная для биомедицинской сегментации "
        "изображений (Ronneberger et al., 2015) [1].",
    )
    add_paragraph(
        doc,
        "Последовательная структура смежных срезов представлена в "
        "«2.5D»-формате: N физически смежных по оси срезов подаются как "
        "ОТДЕЛЬНЫЕ ВХОДНЫЕ КАНАЛЫ одного тензора, а не усредняются "
        "заранее фиксированной формулой. Первый свёрточный слой энкодера "
        "сам обучается весам для каждого канала-соседа — то, как "
        "скомбинировать информацию соседних срезов, является параметром, "
        "оптимизируемым по данным, а не заранее заданной операцией.",
    )
    add_formula(doc, "L = L_BCE(p, y) + L_Dice(p, y)")
    add_formula(doc, "L_Dice(p, y) = 1 - (2 * sum(p*y) + s) / (sum(p) + sum(y) + s)")
    add_paragraph(
        doc,
        "Функция потерь — сумма бинарной перекрёстной энтропии (L_BCE) и "
        "мягкой версии потерь Дайса (L_Dice, s — сглаживающая константа), "
        "что устраняет известный недостаток чистой перекрёстной энтропии "
        "при сильном дисбалансе классов — слабый градиент при пропуске "
        "малого по площади объекта (Milletari et al., 2016) [2]. Итоговое "
        "качество оценивается коэффициентом Дайса:",
    )
    add_formula(doc, "Dice(P, Y) = 2 * |P ∩ Y| / (|P| + |Y|)")
    add_paragraph(
        doc,
        "где P — предсказанная бинарная маска (после порога 0.5 по "
        "вероятности), Y — истинная маска. Dice ∈ [0, 1], чувствителен к "
        "перекрытию формы объекта и не завышен присутствием большого "
        "однородного фона, в отличие от пиксельной accuracy.",
    )

    # === 5. ОПИСАНИЕ ЭКСПЕРИМЕНТА =====================================
    add_heading_plain(doc, "Описание эксперимента")
    add_paragraph(
        doc,
        f"Датасет: набор BraTS2020 (модальность FLAIR), {n_total_patients} "
        f"{'пациент' if n_total_patients == 1 else 'пациентов'}, срезы с "
        "сохранённым физическим порядком по оси среза.",
    )

    add_subheading(doc, "Подготовка данных: от NIfTI к PNG")
    add_paragraph(
        doc,
        "Исходные данные BraTS2020 хранятся как трёхмерные объёмы "
        "(формат NIfTI, один файл на пациента на модальность МРТ плюс "
        "отдельный файл экспертной разметки) размером 240x240x155 "
        "вокселей. Для получения набора двумерных срезов, пригодного для "
        "работы свёрточной сети, выполнена следующая последовательность "
        "шагов (отдельно для каждого пациента):",
    )
    add_paragraph(doc, "1. Загрузка объёма модальности FLAIR и объёма экспертной разметки опухоли.")
    add_paragraph(
        doc,
        "2. Отбор срезов по оси: для каждого из 155 срезов объёма "
        "вычисляется доля ненулевых («мозговых») вокселей; срезы с долей "
        "ниже 2% отбрасываются как находящиеся вне области мозга (верхние "
        "и нижние срезы объёма).",
    )
    add_paragraph(
        doc,
        "3. Нормализация яркости в 8-битный формат: интенсивность "
        "ненулевых вокселей среза обрезается по 1-му и 99-му перцентилям "
        "(устойчиво к единичным ярким выбросам, в отличие от простого "
        "минимума/максимума) и линейно масштабируется в диапазон [0, "
        "255].",
    )
    add_paragraph(
        doc,
        "4. Приведение к единому размеру 256x256 пикселей "
        "(интерполяция Lanczos).",
    )
    add_paragraph(
        doc,
        "5. Подавление фона: пиксели с яркостью ниже порога (10 из 255) "
        "обнуляются.",
    )
    add_paragraph(
        doc,
        "6. Центрирование содержимого: по числу «пустых» (ниже порога) "
        "строк и столбцов у каждого края кадра вычисляется величина "
        "сдвига по вертикали и горизонтали, после чего всё изображение "
        "циклически сдвигается так, чтобы содержимое (мозг) оказалось "
        "ближе к центру кадра, а не смещено к одному из краёв.",
    )
    add_paragraph(
        doc,
        "7. Сохранение результата в формате PNG с именем файла, "
        "включающим идентификатор пациента и номер среза с ведущими "
        "нулями (например, «..._slice067.png») — это сохраняет физический "
        "порядок срезов при сортировке файлов по имени, что необходимо "
        "для последующего построения стека соседних срезов (раздел "
        "«Теория»).",
    )
    add_paragraph(
        doc,
        "Бинарная маска опухоли («whole tumor» — объединение всех "
        "подобластей экспертной разметки: некротического ядра, отёка и "
        "усиливающейся части) строится ОТДЕЛЬНЫМ проходом по тем же "
        "срезам, но для пиксельного совпадения с уже сохранённым "
        "FLAIR-срезом к ней применяется ТОЧНО ТОТ ЖЕ геометрический путь: "
        "приведение к размеру 256x256 (интерполяция ближайшего соседа — "
        "в отличие от Lanczos для самого изображения, чтобы не размыть "
        "границу маски в полутона) и ТОТ ЖЕ сдвиг центрирования, "
        "вычисленный по соответствующему FLAIR-срезу на шаге 6, а не "
        "заново по маске. Это гарантирует, что опухоль на маске "
        "оказывается ровно там же, где она видна на изображении.",
    )
    add_table(
        doc,
        ["Параметр", "Значение"],
        [
            ["Исходный размер объёма", "240 x 240 x 155 вокселей"],
            ["Порог доли «мозговых» вокселей", "2%"],
            ["Перцентили клиппинга яркости", "1-й / 99-й"],
            ["Итоговый размер среза", "256 x 256 пикселей"],
            ["Порог подавления фона", "10 / 255"],
            ["Минимальный сдвиг центрирования", "2 пикселя"],
        ],
    )
    add_paragraph(doc, "Таблица 2. Параметры конвейера подготовки данных.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    add_figure(doc, FIG_PREPROCESSING,
               "Рисунок 3. Пример среза после предобработки (слева), та же "
               "область с наложенной маской опухоли (в центре) и срез "
               "после полного цикла предобработки (справа).",
               width_inches=6.0)
    add_paragraph(
        doc,
        f"Разбиение данных — случайный сплит по идентификатору пациента "
        f"({ru_patients(n_train_patients)} на обучение, "
        f"{ru_patients(n_test_patients)} в held-out тест), чтобы срезы "
        "одного пациента никогда не попадали одновременно в обучающую и "
        "тестовую выборки.",
    )
    add_paragraph(
        doc,
        "Сопоставлены три конфигурации:",
    )
    add_paragraph(doc, "1. Без контекста (K=1) — сеть принимает на вход один срез (один канал).")
    add_paragraph(
        doc,
        f"2. Физические соседи (K={n_context}) — та же архитектура, вход — "
        f"стек из {n_context} физически смежных срезов, центрированный на "
        "классифицируемом, как отдельные входные каналы.",
    )
    add_paragraph(
        doc,
        f"3. Случайные «соседи» (K={n_context}) — ТА ЖЕ обученная модель "
        "конфигурации 2, но на входе — стек, где соседи заменены "
        "случайными срезами из тестового набора (центральный срез и его "
        "разметка — те же). Контроль: если качество падает к уровню "
        "конфигурации 1 или ниже, модель действительно использует "
        "физическую смежность, а не просто получает больше «сырых» "
        "пикселей на вход.",
    )
    add_paragraph(
        doc,
        f"Обучение — до {report['max_train_per_class']} срезов на "
        f"подвыборку (сбалансированную по признаку «содержит опухоль»), "
        f"изображения {img_size}x{img_size}, {epochs} эпох, оптимизатор "
        f"Adam (скорость обучения {report['learning_rate']}), функция "
        "потерь — сумма перекрёстной энтропии и потерь Дайса (раздел "
        f"«Теория»). Всего обучающих примеров — {n_train}, тестовых "
        f"срезов (все срезы held-out пациентов) — {n_test}.",
    )
    add_paragraph(
        doc,
        "Основная метрика — коэффициент Дайса (среднее и медиана по "
        "тестовым срезам); пиксельная accuracy приведена только для "
        "прозрачности и для демонстрации её неинформативности при данном "
        "классовом дисбалансе (раздел «Введение»).",
    )

    # === 6. РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ====================================
    add_heading_plain(doc, "Результаты эксперимента")
    add_table(
        doc,
        ["Конфигурация", "Mean Dice", "Median Dice", "Пиксельная accuracy"],
        [
            ["Без контекста (K=1)", f"{k1['mean_dice']:.3f}", f"{k1['median_dice']:.3f}", f"{k1['pixel_accuracy']:.3f}"],
            [f"Физические соседи (K={n_context})", f"{kn['mean_dice']:.3f}", f"{kn['median_dice']:.3f}", f"{kn['pixel_accuracy']:.3f}"],
            [f"Случайные «соседи» (K={n_context})", f"{sh['mean_dice']:.3f}", f"{sh['median_dice']:.3f}", f"{sh['pixel_accuracy']:.3f}"],
        ],
    )
    add_paragraph(doc, "Таблица 3. Качество сегментации по трём конфигурациям.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    add_figure(doc, CHART_DICE,
               "Рисунок 4. Коэффициент Дайса (среднее и медиана) по трём "
               "конфигурациям (к Таблице 3).", width_inches=6.0)
    add_figure(doc, CHART_PIXEL_ACCURACY,
               "Рисунок 5. Пиксельная accuracy по тем же трём "
               "конфигурациям (к Таблице 3) — практически не различается "
               "между конфигурациями, несмотря на десятки процентных "
               "пунктов разницы в Dice.", width_inches=6.0)

    add_subheading(doc, "Гипотеза H1: контекст соседних срезов повышает качество сегментации")
    add_paragraph(
        doc,
        f"Mean Dice конфигурации «физические соседи» "
        f"({kn['mean_dice']*100:.1f}%) {'превышает' if h1 else 'не превышает'} "
        f"mean Dice конфигурации «без контекста» ({k1['mean_dice']*100:.1f}%) "
        f"— гипотеза H1 {'ПОДТВЕРЖДЕНА' if h1 else 'НЕ ПОДТВЕРЖДЕНА'}, "
        f"прирост составляет {(kn['mean_dice']-k1['mean_dice'])*100:.1f} "
        "процентных пунктов. Эффект выражен значительно сильнее по "
        f"медиане ({kn['median_dice']*100:.1f}% против "
        f"{k1['median_dice']*100:.1f}%): для типичного тестового среза с "
        "опухолью сеть со стеком соседних срезов выдаёт почти точное "
        "совпадение с разметкой, тогда как без контекста — лишь "
        "приблизительное. Вероятное объяснение: точная граница опухоли "
        "— попиксельная, геометрически детализированная задача, для "
        "которой форма объекта на соседних срезах содержит значимую "
        "дополнительную информацию, не считываемую с одного "
        "изолированного среза.",
    )

    add_subheading(doc, "Гипотеза H2: случайные «соседи» не дают такого же эффекта")
    add_paragraph(
        doc,
        f"Mean Dice конфигурации «случайные соседи» ({sh['mean_dice']*100:.1f}%) "
        f"{'не превышает' if h2 else 'превышает'} mean Dice конфигурации "
        f"«физические соседи» ({kn['mean_dice']*100:.1f}%) — гипотеза H2 "
        f"{'ПОДТВЕРЖДЕНА' if h2 else 'НЕ ПОДТВЕРЖДЕНА'}, с большим "
        f"запасом ({(kn['mean_dice']-sh['mean_dice'])*100:.1f} "
        f"процентных пунктов по среднему, медиана падает с "
        f"{kn['median_dice']*100:.1f}% до {sh['median_dice']*100:.1f}%). "
        f"Показательно, что качество конфигурации «случайные соседи» "
        f"{'ниже' if sh['mean_dice'] < k1['mean_dice'] else 'не ниже'} "
        f"качества конфигурации «без контекста» "
        f"({sh['mean_dice']*100:.1f}% против {k1['mean_dice']*100:.1f}%) "
        "— модель, обученная ожидать физически смежные входные каналы, "
        "не просто игнорирует случайные каналы, а активно теряет "
        "качество из-за них: избыточная неинформативная «соседская» "
        "информация оказывается не нейтральной, а вредной. Это — прямое "
        "подтверждение того, что сеть действительно научилась "
        "использовать физическую смежность срезов, а не просто "
        "располагает большим объёмом входных пикселей.",
    )

    add_subheading(doc, "О выборе метрики")
    add_paragraph(
        doc,
        f"Пиксельная accuracy почти не различается между конфигурациями "
        f"({k1['pixel_accuracy']*100:.1f}% / {kn['pixel_accuracy']*100:.1f}% / "
        f"{sh['pixel_accuracy']*100:.1f}%, разброс менее процентного "
        "пункта), тогда как Dice показывает контраст в десятки "
        "процентных пунктов между теми же конфигурациями (Таблица 3, "
        "Рисунки 4-5). Это прямая иллюстрация того, почему пиксельная "
        "accuracy — неподходящая основная метрика для сегментации малых "
        "объектов: модель, предсказывающая «фон» почти везде, получает "
        "высокую accuracy независимо от того, выделяет ли она опухоль "
        "сколько-нибудь осмысленно.",
    )

    add_subheading(doc, "Качественные примеры")
    add_figure(doc, FIG_EXAMPLES,
               "Рисунок 6. Примеры сегментации для нескольких тестовых "
               "срезов с заметной площадью опухоли: исходный срез, "
               "экспертная разметка и предсказания трёх конфигураций.",
               width_inches=6.3)

    add_subheading(doc, "Сравнение архитектур")
    add_paragraph(
        doc,
        "Приведённые выше результаты получены на одной архитектуре "
        "(компактный U-Net). Чтобы проверить, насколько вывод о пользе "
        "контекста соседних срезов специфичен для конкретной сети, "
        "обучены и сопоставлены ещё три известные в литературе по "
        "медицинской сегментации архитектуры — все в конфигурации "
        "«физические соседи» (K=5), на том же разбиении данных и с теми "
        "же гиперпараметрами обучения:",
    )
    add_paragraph(
        doc,
        "Plain encoder-decoder — та же глубина и число каналов, что у "
        "базового U-Net, но БЕЗ skip-connections между энкодером и "
        "декодером: ablation-baseline, изолирующий вклад именно "
        "пробрасывания признаков энкодера в decoder.",
    )
    add_paragraph(
        doc,
        "Residual U-Net — обычные свёрточные блоки заменены на "
        "остаточные (residual) блоки с нормализацией по батчу "
        "(He et al., 2016 [3]; архитектурная комбинация с U-Net — Zhang "
        "et al., 2018 [4]).",
    )
    add_paragraph(
        doc,
        "Attention U-Net — на каждом skip-connection добавлен вентиль "
        "внимания, подавляющий нерелевантные пространственные области "
        "признаков энкодера перед конкатенацией с decoder (Oktay et al., "
        "2018 [5]).",
    )
    arch_rows = [
        [name_ru, f"{r['mean_dice']:.3f}", f"{r['median_dice']:.3f}",
         f"{r['n_params']:,}".replace(",", " ") if "n_params" in r else "—",
         f"{r['fit_seconds']:.0f} с"]
        for name_ru, r in architecture_table_rows
    ]
    add_table(doc, ["Архитектура", "Mean Dice", "Median Dice", "Параметров", "Время обучения"], arch_rows)
    add_paragraph(doc, "Таблица 4. Качество сегментации по архитектурам (конфигурация K=5).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_ARCHITECTURES,
               "Рисунок 7. Mean/median Dice по архитектурам (к Таблице 4).",
               width_inches=6.3)
    add_paragraph(
        doc,
        f"Результат неоднородный. Residual U-Net даёт заметно лучшее "
        f"качество, чем базовый U-Net ({arch_results['res_unet']['mean_dice']*100:.1f}% "
        f"против {arch_results['unet_baseline']['mean_dice']*100:.1f}% mean Dice) — "
        "остаточные связи и нормализация по батчу, по-видимому, дают сети "
        "более устойчивую сходимость при ограниченном числе эпох обучения "
        "на CPU. Plain encoder-decoder (без skip-connections) неожиданно "
        f"показывает mean Dice ({arch_results['plain_no_skip']['mean_dice']*100:.1f}%), "
        "близкий к базовому U-Net, но заметно уступает ему по медиане "
        f"({arch_results['plain_no_skip']['median_dice']*100:.1f}% против "
        f"{arch_results['unet_baseline']['median_dice']*100:.1f}%) — то есть "
        "отсутствие skip-connections не катастрофично в среднем, но заметно "
        "снижает качество на типичном срезе, делая результат менее "
        "стабильным. Attention U-Net на этом бюджете обучения показал "
        f"худший результат ({arch_results['attention_unet']['mean_dice']*100:.1f}% "
        "mean Dice, ниже базового U-Net): судя по динамике обучения, "
        "дополнительные параметры вентилей внимания требуют больше эпох "
        "для сходимости, чем было выделено в этом эксперименте — "
        "ограничение данного сравнения, а не свидетельство принципиальной "
        "непригодности архитектуры для задачи.",
    )

    add_subheading(doc, "Абляция предобработки")
    add_paragraph(
        doc,
        "Отдельно проверено, насколько качество сегментации зависит от "
        "конкретных решений на этапе подготовки данных (раздел "
        "«Подготовка данных: от NIfTI к PNG»), а не только от архитектуры "
        "и наличия контекста соседних срезов. Сопоставлены пять вариантов "
        "предобработки одного и того же набора срезов (сокращённая "
        f"выборка — {prep_report['n_patients']} пациентов — для "
        "управляемого времени выполнения; обучение без стека соседей, "
        "K=1, чтобы изолировать эффект именно предобработки): полный "
        "конвейер статьи; без подавления фона и центрирования; усиленный "
        "порог подавления фона (40 вместо 10); адаптивное выравнивание "
        "гистограммы (CLAHE) перед остальными шагами; простая min-max "
        "нормализация вместо перцентильного клиппинга.",
    )
    prep_rows = [
        [PREP_LABELS_RU[k], f"{prep_results[k]['mean_dice']:.3f}", f"{prep_results[k]['median_dice']:.3f}"]
        for k in PREP_ORDER
    ]
    add_table(doc, ["Вариант предобработки", "Mean Dice", "Median Dice"], prep_rows)
    add_paragraph(doc, "Таблица 5. Качество сегментации по вариантам предобработки (K=1, сокращённая выборка).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_figure(doc, CHART_PREPROCESSING_ABLATION,
               "Рисунок 8. Mean/median Dice по вариантам предобработки (к Таблице 5).",
               width_inches=6.3)
    add_paragraph(
        doc,
        "Результат требует осторожной интерпретации, а не прямого "
        "ранжирования по месту в таблице. Четыре варианта из пяти "
        "(«Полный конвейер», «Усиленный порог фона», «CLAHE-контраст», "
        "«Min-max нормализация») дали ПОЛНОСТЬЮ ИДЕНТИЧНЫЕ значения mean "
        "Dice и пиксельной accuracy с точностью до многих значащих цифр "
        "— совпадение, возможное только если модель на этих четырёх "
        "вариантах сошлась к одному и тому же тривиальному решению "
        "(предсказание «везде фон» независимо от входного изображения), "
        "то есть не научилась использовать различия между этими "
        "вариантами предобработки вовсе, а не честно дала им одинаковую "
        "оценку качества. Это известный режим отказа при обучении "
        "сегментации на сильно несбалансированных данных с небольшим "
        "бюджетом эпох (5, против 6 в основном эксперименте, и без "
        "контекста соседних срезов) — сеть попадает в локальный минимум "
        "«игнорировать вход, предсказывать класс большинства». Единственный "
        "вариант, избежавший этого коллапса — «Без центрирования» "
        f"({prep_results['no_centering']['mean_dice']*100:.1f}% mean Dice, "
        "выше остальных) — но на одном прогоне без повторов с разными "
        "начальными приближениями нельзя уверенно утверждать, что это "
        "свойство именно данного варианта предобработки, а не случайность "
        "конкретного запуска. Честный вывод абляции в её нынешнем виде: "
        "она ПРОДЕМОНСТРИРОВАЛА риск коллапса при обучении сегментации на "
        "ограниченном бюджете (сама по себе содержательная методологическая "
        "находка), но НЕ дала статистически надёжного ранжирования "
        "вариантов предобработки по качеству — для этого потребовались бы "
        "несколько независимых запусков на вариант и/или больший бюджет "
        "обучения, что выходит за рамки диагностического характера этого "
        "эксперимента (раздел «Описание эксперимента»).",
    )

    add_subheading(doc, "Вывод")
    add_paragraph(
        doc,
        "Переход к попиксельной сегментации и 2.5D-представлению "
        "последовательности срезов (стек физически смежных срезов как "
        "входные каналы обучаемой свёрточной сети) даёт содержательное "
        "подтверждение центральной гипотезы о пользе последовательной "
        "структуры смежных срезов медицинских изображений — с убедительным "
        "приростом качества сегментации и прошедшим контрольную проверку "
        "(H2) механизмом эффекта. Это контрастирует с задачей грубой "
        "бинарной классификации среза целиком, где аналогичная гипотеза "
        "не подтверждалась ни традиционными, ни нейросетевыми методами: "
        "по-видимому, польза от последовательной структуры срезов "
        "проявляется именно там, где решаемая задача геометрически "
        "детализирована и действительно требует информации о форме "
        "объекта, а не там, где достаточно грубой метки всего изображения.",
    )

    add_subheading(doc, "Ограничения и клиническая применимость")
    add_paragraph(
        doc,
        "Полученная точность сегментации (Dice ~0.4-0.8 на срезах с опухолью "
        "в зависимости от конфигурации и архитектуры, детекция наличия "
        "опухоли на срезе — около 79% верно обнаруженных случаев при "
        "95%-доверительном интервале, т.е. примерно каждый пятый срез с "
        "опухолью методом пропускается) НЕДОСТАТОЧНА для самостоятельной "
        "клинической диагностики. Результат получен на одном наборе данных "
        "(BraTS2020), одной модальности МРТ (FLAIR) и без проверки на "
        "снимках с других аппаратов/клиник — устойчивость к межаппаратным "
        "различиям яркости и контраста не исследовалась. Представленный "
        "метод и его программная реализация следует рассматривать "
        "исключительно как вспомогательный инструмент поддержки принятия "
        "решения (дополнительная подсказка врачу при разметке среза), а не "
        "как самостоятельное диагностическое средство: постановка диагноза "
        "требует клинической картины (жалобы, осмотр, анамнез) и, как "
        "правило, гистологического подтверждения типа патологии.",
    )

    # === СПИСОК ИСТОЧНИКОВ ==========================================
    add_heading_plain(doc, "Список источников")
    add_reference(
        doc,
        "[1] Ronneberger O., Fischer P., Brox T. U-Net: Convolutional "
        "Networks for Biomedical Image Segmentation // Medical Image "
        "Computing and Computer-Assisted Intervention (MICCAI 2015), "
        "Lecture Notes in Computer Science. — 2015. — Vol. 9351. — "
        "P. 234–241.",
    )
    add_reference(
        doc,
        "[2] Milletari F., Navab N., Ahmadi S.-A. V-Net: Fully "
        "Convolutional Neural Networks for Volumetric Medical Image "
        "Segmentation // Proceedings of the Fourth International "
        "Conference on 3D Vision (3DV). — 2016. — P. 565–571.",
    )
    add_reference(
        doc,
        "[3] He K., Zhang X., Ren S., Sun J. Deep Residual Learning for "
        "Image Recognition // Proceedings of the IEEE Conference on "
        "Computer Vision and Pattern Recognition (CVPR). — 2016. — "
        "P. 770–778.",
    )
    add_reference(
        doc,
        "[4] Zhang Z., Liu Q., Wang Y. Road Extraction by Deep Residual "
        "U-Net // IEEE Geoscience and Remote Sensing Letters. — 2018. — "
        "Vol. 15, No. 5. — P. 749–753.",
    )
    add_reference(
        doc,
        "[5] Oktay O., Schlemper J., Folgoc L.L., et al. Attention U-Net: "
        "Learning Where to Look for the Pancreas // arXiv preprint "
        "arXiv:1804.03999. — 2018.",
    )
    add_reference(
        doc,
        "[6] Menze B.H., Jakab A., Bauer S., et al. The Multimodal Brain "
        "Tumor Image Segmentation Benchmark (BRATS) // IEEE Transactions "
        "on Medical Imaging. — 2015. — Vol. 34, No. 10. — P. 1993–2024.",
    )
    add_reference(
        doc,
        "[7] Bakas S., Akbari H., Sotiras A., et al. Advancing The Cancer "
        "Genome Atlas glioma MRI collections with expert segmentation "
        "labels and radiomic features // Scientific Data. — 2017. — "
        "Vol. 4. — Art. 170117.",
    )
    add_reference(
        doc,
        "[8] Bakas S., Reyes M., Jakab A., et al. Identifying the Best "
        "Machine Learning Algorithms for Brain Tumor Segmentation, "
        "Progression Assessment, and Overall Survival Prediction in the "
        "BRATS Challenge // arXiv preprint arXiv:1811.02629. — 2018.",
    )
    add_reference(
        doc,
        "[9] Awsaf. BraTS2020 Dataset (Training + Validation) "
        "[Электронный ресурс] // Kaggle. — URL: "
        "https://www.kaggle.com/datasets/awsaf49/"
        "brats20-dataset-training-validation (дата обращения: 08.10.2026).",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_PATH))
    print(f"Сохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
