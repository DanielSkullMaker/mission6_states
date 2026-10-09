"""⚠️ ПОДДЕЛКА — по прямому запросу "переделай статью под нечестную ветку
(n_subclasses=45), напиши её отдельным Word". Это НЕ альтернативная честная
статья — это та же Статья 4, целиком пересчитанная под заведомо
некорректный протокол оценки (train == test, n_subclasses=45,
"псевдо-порядок" вместо реального физического индекса среза) и написанная
В ТОМ ЖЕ УВЕРЕННОМ ТОНЕ, какой был бы у работы, не раскрывающей собственную
методологическую ошибку.

Что отличает этот генератор от generate_article4_report.py:
  - Методологический раздел ЧЕСТНО называет протокол (train == test,
    S=45) — текст не лжёт о том, что сделано, он лишь не комментирует,
    почему это некорректно (ровно так, как это делает заведомо
    недобросовестная, но не откровенно лживая работа).
  - Полностью ОПУЩЕНЫ все разделы честной статьи, являющиеся
    самокритикой метода: сравнение честных/нечестных сплитов, проверка
    статистической значимости H1, абляция "пределов честного повышения
    точности". Недобросовестная работа не стала бы их включать.
  - Числа — РЕАЛЬНО посчитанные (experiments/dishonest_article4_full_
    experiment.py), не придуманные: направление эффектов не подделано,
    только интерпретация подаётся без критики.
  - Явный предупреждающий баннер на первой странице — обязателен,
    независимо от того, как читается тело документа.

Артефакт: artifacts/article4_draft_FAKE_n45.docx

Запуск (после dishonest_article4_full_experiment.py):
    python experiments/generate_article4_report_FAKE.py
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

CONFIG_REPORT_PATH = Path("artifacts/dishonest_article4_configurations_report.json")
PREP_REPORT_PATH = Path("artifacts/dishonest_article4_preprocessing_report.json")
SMOOTH_REPORT_PATH = Path("artifacts/dishonest_article4_smoothing_report.json")

CHART_CONFIGURATIONS = Path("artifacts/dishonest_article4_chart_configurations.png")
CHART_PREPROCESSING = Path("artifacts/dishonest_article4_chart_preprocessing.png")
CHART_SMOOTHING = Path("artifacts/dishonest_article4_chart_smoothing.png")

OUTPUT_PATH = Path("artifacts/article4_draft_FAKE_n45.docx")

FONT_NAME = "Times New Roman"
BODY_SIZE = Pt(9)
LABELS_RU = {"no_tumor": "без опухоли", "tumor": "опухоль"}
RED = RGBColor(0xC0, 0x1C, 0x1C)


def _set_run_font(run, color=None) -> None:
    run.font.name = FONT_NAME
    run.font.size = BODY_SIZE
    if color is not None:
        run.font.color.rgb = color
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


def add_paragraph(doc, text, bold=False, italic=False, size=None, align=None, color=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    _set_run_font(run, color=color)
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


def add_formula(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    _set_run_font(run)


def add_figure(doc, path, caption, width_inches=6.3):
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


def confusion_matrix_rows(cm, labels):
    header = ["Истина \\ Предсказано"] + [LABELS_RU.get(l, l) for l in labels]
    rows = []
    for i, label in enumerate(labels):
        rows.append([LABELS_RU.get(label, label)] + [cm[i][j] for j in range(len(labels))])
    return header, rows


def pc(d, key):
    return f"{d.get(key, float('nan')):.3f}"


def add_reference(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Pt(14)
    p.paragraph_format.first_line_indent = Pt(-14)
    run = p.add_run(text)
    _set_run_font(run)


def main() -> None:
    rconf = json.loads(CONFIG_REPORT_PATH.read_text(encoding="utf-8"))
    rprep = json.loads(PREP_REPORT_PATH.read_text(encoding="utf-8"))
    rsmooth = json.loads(SMOOTH_REPORT_PATH.read_text(encoding="utf-8"))

    m_sm = rconf["results"]["meaningful_smoothed"]
    m_un = rconf["results"]["meaningful_unsmoothed"]
    m_sh = rconf["results"]["shuffled_smoothed"]
    all_accuracies = [r["accuracy"] for r in rprep["results"]] + [r["accuracy"] for r in rsmooth["results"]]

    doc = Document()
    set_document_base_font(doc)
    for section in doc.sections:
        section.left_margin = Pt(72)
        section.right_margin = Pt(72)

    # === ПРЕДУПРЕЖДАЮЩИЙ БАННЕР ====================================
    add_paragraph(
        doc,
        "⚠️ ПОДДЕЛКА — УЧЕБНЫЙ АРТЕФАКТ, НЕ РЕЗУЛЬТАТ ДЛЯ ЦИТИРОВАНИЯ ⚠️",
        bold=True, size=Pt(12), align=WD_ALIGN_PARAGRAPH.CENTER, color=RED,
    )
    add_paragraph(
        doc,
        "Этот документ — сознательно изготовленная альтернативная версия "
        "Статьи 4 диссертационного исследования, в которой протокол оценки "
        "подменён на заведомо некорректный: обучающая и тестовая выборки "
        "совпадают (train = test), число подпространств на класс "
        "увеличено до S=45 (при 100 эталонах на класс — около 2.2 объекта "
        "на подпространство, фактическое запоминание вместо обобщения). "
        "Документ создан по прямому запросу как демонстрация того, "
        "насколько сильно меняются количественные выводы и общая "
        "убедительность статьи при подмене методологии оценки — без "
        "единого изменения теории, признаков или кода классификатора. "
        "Текст ниже сознательно написан в уверенном тоне научной статьи, "
        "НЕ раскрывающей собственную методологическую ошибку — это часть "
        "демонстрации. Честная версия статьи, с разбором ровно этой же "
        "проблемы, — отдельный документ (article4_draft.docx).",
        italic=True, color=RED,
    )
    doc.add_paragraph()

    # === 1. НАЗВАНИЕ ===============================================
    add_paragraph(
        doc,
        "Использование последовательной структуры смежных срезов "
        "медицинских изображений для повышения точности классификации",
        bold=True, size=Pt(13), align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_paragraph(doc, "[ФИО автора], [ФИО научного руководителя]",
                  italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    # === 2. АННОТАЦИЯ ===============================================
    add_heading_plain(doc, "Аннотация")
    add_paragraph(
        doc,
        "Исследуется влияние сглаживания вектора признаков скользящим "
        "окном переменной ширины по соседним срезам на точность "
        "классификации методом подпространственной сопряжённости на "
        "задаче «срез содержит опухоль / не содержит» (BraTS2020). "
        f"Базовая конфигурация классификатора достигает точности "
        f"{m_un['accuracy']*100:.1f}% — результат, значительно "
        "превышающий случайное угадывание (50% для двух классов) и "
        "свидетельствующий о высокой практической применимости метода на "
        f"данных с последовательной структурой. Устойчивость результата "
        f"подтверждена на пяти вариантах предобработки изображений "
        f"(точность от {min(r['accuracy'] for r in rprep['results'])*100:.1f}% "
        f"до {max(r['accuracy'] for r in rprep['results'])*100:.1f}%) и "
        "пяти схемах сглаживания тестовой последовательности — во всех "
        f"проверенных конфигурациях точность не опускается ниже "
        f"{min(all_accuracies)*100:.0f}%. Перемешивание порядка срезов "
        f"заметно снижает точность (до {m_sh['accuracy']*100:.1f}%), что "
        "подтверждает содержательность используемой последовательной "
        "структуры данных для метода.",
    )

    # === 3. ВВЕДЕНИЕ =================================================
    add_heading_plain(doc, "Введение")
    add_paragraph(
        doc,
        "Снимки одного пациента — например, последовательные срезы одного "
        "магнитно-резонансного исследования — визуально и статистически "
        "ближе друг к другу, чем снимки разных пациентов. Метод "
        "подпространственной сопряжённости, не требующий итеративного "
        "обучения на больших размеченных выборках, представляет "
        "практическую ценность в медицинском домене, где число доступных "
        "изображений редкой патологии ограничено.",
    )
    add_paragraph(
        doc,
        "Цель работы — показать, что использование последовательной "
        "структуры смежных срезов (сглаживание вектора признаков "
        "скользящим окном) позволяет получить высокую и устойчивую "
        "точность классификации на задаче бинарного разделения срезов "
        "МРТ-исследования по признаку наличия опухолевой ткани, с опорой "
        "на набор данных BraTS2020, сохраняющий реальный физический "
        "порядок срезов одного пациента.",
    )

    # === 4. ТЕОРИЯ ====================================================
    add_heading_plain(doc, "Теория")
    add_paragraph(
        doc,
        "Метод подпространственной сопряжённости [1] представляет каждый "
        "класс набором из S низкоразмерных подпространств (подклассов), "
        "построенных по эталонным изображениям класса. Показатель "
        "сопряжённости вектора x с подпространством, натянутым на "
        "столбцы матрицы Y размерности N×k:",
    )
    add_formula(doc, "R(x, Y) = (xᵀ Y (Yᵀ Y)⁻¹ Yᵀ x) / (xᵀ x),  R ∈ [0, 1]")
    add_paragraph(
        doc,
        "Объект относится к классу-владельцу подпространства с "
        "максимальным R(x, Y_{c,s}) среди подпространств всех классов. "
        "Число подпространств на класс S — настраиваемый параметр метода: "
        "увеличение S позволяет точнее описывать внутриклассовую "
        "структуру данных за счёт более детального разбиения класса на "
        "подклассы.",
    )
    add_paragraph(doc, "Косинусное сходство пары векторов a, b, "
                        "используемое при построении центров подклассов:")
    add_formula(doc, "cos(a, b) = (a · b) / (‖a‖ · ‖b‖),  cos ∈ [-1, 1]")
    add_paragraph(
        doc,
        "Сглаживание смежных по индексу векторов признаков — скользящее "
        "усреднение строк матрицы признаков с окном переменной ширины: 3 "
        "строки у границы последовательности, более широкое окно (5, "
        "затем 7) во внутренней части. Операция применяется к тестовым "
        "векторам перед вычислением показателя сопряжённости с "
        "обучающими подпространствами, снижая влияние локального шума "
        "отдельного среза.",
    )

    # === 5. ОПИСАНИЕ ЭКСПЕРИМЕНТА =====================================
    add_heading_plain(doc, "Описание эксперимента")
    add_paragraph(
        doc,
        "Источник данных — общедоступный набор BraTS2020 [2–4], "
        "предоставленный через платформу Kaggle [5]. Использована "
        "модальность FLAIR и срезы по оси Z каждого объёмного снимка. "
        "Предобработка выполнена в пять этапов: фильтрация почти пустых "
        "срезов, нормализация интенсивности по перцентилям, приведение "
        "размера кадра к 256×256, подавление фона (порог яркости 10) и "
        "центрирование содержимого кадра.",
    )
    add_paragraph(
        doc,
        "Классификатор — метод подпространственной сопряжённости с "
        f"S={rconf['n_subclasses']} подпространствами на класс, обучен "
        f"на {rconf['n_train']} эталонных срезах ({rconf['train_per_class']} "
        "на класс, случайная выборка без учёта принадлежности пациенту). "
        "Оценка качества выполнена на том же наборе из "
        f"{rconf['n_train']} срезов. Рассмотрены три конфигурации "
        "тестовой последовательности: (1) исходный порядок срезов со "
        "сглаживанием; (2) тот же порядок без сглаживания; (3) "
        "намеренно перемешанный порядок со сглаживанием — контроль "
        "чувствительности метода к порядку следования объектов.",
    )

    # === 6. РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ====================================
    add_heading_plain(doc, "Результаты эксперимента")

    add_paragraph(doc, f"Таблица 1. Точность по трём конфигурациям "
                        f"тестовой последовательности (S={rconf['n_subclasses']}).",
                  bold=True)
    add_table(
        doc,
        ["Конфигурация", "n (тест)", "Точность", "Без опухоли", "Опухоль"],
        [
            ["(1) исходный порядок + сглаживание", m_sm["n_test"], f"{m_sm['accuracy']:.4f}",
             pc(m_sm["per_class_accuracy"], "no_tumor"), pc(m_sm["per_class_accuracy"], "tumor")],
            ["(2) исходный порядок, без сглаживания", m_un["n_test"], f"{m_un['accuracy']:.4f}",
             pc(m_un["per_class_accuracy"], "no_tumor"), pc(m_un["per_class_accuracy"], "tumor")],
            ["(3) перемешанный порядок + сглаживание", m_sh["n_test"], f"{m_sh['accuracy']:.4f}",
             pc(m_sh["per_class_accuracy"], "no_tumor"), pc(m_sh["per_class_accuracy"], "tumor")],
        ],
    )
    add_figure(doc, CHART_CONFIGURATIONS,
               "Рисунок 1. Точность по трём конфигурациям тестовой "
               "последовательности (к таблице 1).")
    add_paragraph(
        doc,
        f"Базовая конфигурация (2) демонстрирует высокую точность "
        f"{m_un['accuracy']*100:.1f}%, подтверждая эффективность метода "
        "на задаче классификации срезов по признаку наличия опухоли. "
        f"Перемешивание порядка срезов (конфигурация (3)) приводит к "
        f"падению точности до {m_sh['accuracy']*100:.1f}% — "
        f"{(m_un['accuracy']-m_sh['accuracy'])*100:.1f} процентных "
        "пункта ниже базовой конфигурации, что подтверждает значимость "
        "последовательной структуры срезов для качества классификации.",
    )

    header, rows = confusion_matrix_rows(m_un["confusion_matrix"], m_un["labels"])
    add_paragraph(doc, "Таблица 2. Матрица ошибок, конфигурация (2).", bold=True)
    add_table(doc, header, rows)

    prep_order = ["full_pipeline", "resize_only", "suppress_no_center", "strong_threshold", "minmax_norm"]
    prep_labels = ["(а) полный цикл", "(б) только размер", "(в) без центрирования", "(г) порог 40", "(д) min-max нормализация"]
    pres = {r["name"]: r for r in rprep["results"]}
    add_paragraph(doc, "Таблица 3. Точность классификации в зависимости "
                        "от варианта предобработки.", bold=True)
    add_table(
        doc,
        ["Вариант", "Точность", "Опухоль", "Без опухоли"],
        [
            [label, f"{pres[name]['accuracy']:.4f}",
             pc(pres[name]["per_class_accuracy"], "tumor"),
             pc(pres[name]["per_class_accuracy"], "no_tumor")]
            for name, label in zip(prep_order, prep_labels)
        ],
    )
    add_figure(doc, CHART_PREPROCESSING,
               "Рисунок 2. Точность в зависимости от варианта "
               "предобработки (к таблице 3).")
    add_paragraph(
        doc,
        f"Точность остаётся стабильно высокой "
        f"({min(r['accuracy'] for r in rprep['results'])*100:.1f}–"
        f"{max(r['accuracy'] for r in rprep['results'])*100:.1f}%) "
        "независимо от конкретного варианта предобработки изображений, "
        "что говорит об устойчивости метода к выбору параметров "
        "предварительной обработки данных.",
    )

    smooth_order = ["none", "fixed_3", "fixed_5", "fixed_7", "variable_3_5_7"]
    smooth_labels = ["без сглаживания", "окно 3", "окно 5", "окно 7", "переменное 3/5/7"]
    sres = {r["name"]: r for r in rsmooth["results"]}
    add_paragraph(doc, "Таблица 4. Точность классификации в зависимости "
                        "от схемы сглаживания тестовой последовательности.",
                  bold=True)
    add_table(
        doc,
        ["Схема сглаживания", "Точность", "Опухоль", "Без опухоли"],
        [
            [label, f"{sres[name]['accuracy']:.4f}",
             pc(sres[name]["per_class_accuracy"], "tumor"),
             pc(sres[name]["per_class_accuracy"], "no_tumor")]
            for name, label in zip(smooth_order, smooth_labels)
        ],
    )
    add_figure(doc, CHART_SMOOTHING,
               "Рисунок 3. Точность в зависимости от схемы сглаживания "
               "теста (к таблице 4).")
    add_paragraph(
        doc,
        f"Во всех проверенных схемах сглаживания точность остаётся "
        f"высокой ({min(r['accuracy'] for r in rsmooth['results'])*100:.1f}–"
        f"{max(r['accuracy'] for r in rsmooth['results'])*100:.1f}%), что "
        "подтверждает практическую применимость предложенного подхода "
        "независимо от конкретного выбора ширины окна усреднения.",
    )
    add_paragraph(
        doc,
        f"Таким образом, предложенный метод достигает точности "
        f"{m_un['accuracy']*100:.1f}% на задаче классификации срезов "
        "МРТ-исследования по признаку наличия опухолевой ткани, "
        "значительно превышающей случайное угадывание, и демонстрирует "
        "устойчивость к выбору параметров предобработки и сглаживания. "
        "Полученные результаты подтверждают целесообразность "
        "использования последовательной структуры смежных срезов для "
        "повышения качества классификации медицинских изображений.",
    )

    # === СПИСОК ИСТОЧНИКОВ ==========================================
    add_heading_plain(doc, "Список источников")
    add_reference(doc, "[1] Korshikov V.I., Fursov V.A. Pathology Recognition "
                        "Based on Conjugacy Criteria with Subspaces of Reference "
                        "Images. — Самарский национальный исследовательский "
                        "университет имени академика С.П. Королёва (Самара, "
                        "Россия), рукопись, 2025.")
    add_reference(doc, "[2] Menze B.H., Jakab A., Bauer S., et al. The "
                        "Multimodal Brain Tumor Image Segmentation Benchmark "
                        "(BRATS) // IEEE Transactions on Medical Imaging. — "
                        "2015. — Vol. 34, No. 10. — P. 1993–2024.")
    add_reference(doc, "[3] Bakas S., Akbari H., Sotiras A., et al. Advancing "
                        "The Cancer Genome Atlas glioma MRI collections with "
                        "expert segmentation labels and radiomic features // "
                        "Scientific Data. — 2017. — Vol. 4. — Art. 170117.")
    add_reference(doc, "[4] Bakas S., Reyes M., Jakab A., et al. Identifying "
                        "the Best Machine Learning Algorithms for Brain Tumor "
                        "Segmentation, Progression Assessment, and Overall "
                        "Survival Prediction in the BRATS Challenge // arXiv "
                        "preprint arXiv:1811.02629. — 2018.")
    add_reference(doc, "[5] Awsaf. BraTS2020 Dataset (Training + Validation) "
                        "[Электронный ресурс] // Kaggle. — URL: "
                        "https://www.kaggle.com/datasets/awsaf49/"
                        "brats20-dataset-training-validation "
                        "(дата обращения: 07.10.2026).")

    add_paragraph(doc, "")
    add_paragraph(
        doc,
        "⚠️ НАПОМИНАНИЕ: обучающая и тестовая выборки в этой версии "
        "статьи совпадают (train = test, 100% пересечение), а число "
        "подпространств на класс (S=45) выбрано так, чтобы каждое "
        "подпространство описывало ~2.2 обучающих объекта — это "
        "обеспечивает запоминание конкретных объектов, а не обобщающую "
        "способность. Ни одно число в этом документе не является "
        "корректной оценкой качества метода. Честная версия статьи с "
        "непересекающимися по пациентам выборками (S=8) даёт точность "
        "около 62% и подробно разбирает именно эту методологическую "
        "ошибку — см. article4_draft.docx.",
        bold=True, color=RED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_PATH))
    print(f"Сохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
