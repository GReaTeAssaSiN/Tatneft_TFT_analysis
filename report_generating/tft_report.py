"""
Формирование DOCX-отчёта по TFT-анализу данных.
Описывает таксономию переменных, предобработку и архитектуру TFT-модели.

Запускать из корня проекта: python report_generating/tft_report.py
Результат: reports/tft_report.docx
"""

import os
import sys

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_utils import (
    EXCLUDED_COLS,
    LOG_COLS,
    NO_ZSCORE_COLS,
    STATIC_CATS,
    STATIC_REALS,
    TARGET_COLS,
    TIME_VARYING_KNOWN_CATS,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
)

os.makedirs("reports", exist_ok=True)

# ── Проверка наличия данных ──────────────────────────────────
_src_prepared = "data/prepared_data.csv"
_src_merged   = "data/merged_data.csv"

if not os.path.exists(_src_prepared) or not os.path.exists(_src_merged):
    missing = [p for p in (_src_prepared, _src_merged) if not os.path.exists(p)]
    print(f"[ОШИБКА] Файлы не найдены: {missing}")
    print("  Запустите: python explore_data.py && python eda/eda_preprocessing.py")
    sys.exit(1)

df     = pd.read_csv(_src_prepared, parse_dates=["timestamp"])
merged = pd.read_csv(_src_merged,   parse_dates=["timestamp"])

# Периоды сплита (из данных)
TRAIN_END_DATE = "31 октября 2023"
VAL_END_DATE   = "30 ноября 2023"
TEST_END_DATE  = "31 декабря 2023"

# ── Вспомогательные функции ──────────────────────────────────
def cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def h1(doc, text):
    p = doc.add_heading(text, level=1)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x39, 0x64)
    return p


def h2(doc, text):
    p = doc.add_heading(text, level=2)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    return p


def h3(doc, text):
    p = doc.add_heading(text, level=3)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x37, 0x86, 0x8C)
    return p


def add_table(doc, headers, rows, col_widths=None, header_color="1F3964"):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    hdr = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        cell_bg(hdr[i], header_color)
        for run in hdr[i].paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9)

    for ri, row in enumerate(rows):
        cells = tbl.rows[ri + 1].cells
        bg = "F2F2F2" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            cell_bg(cells[ci], bg)
            for run in cells[ci].paragraphs[0].runs:
                run.font.size = Pt(9)

    if col_widths:
        for row in tbl.rows:
            for ci, w in enumerate(col_widths):
                row.cells[ci].width = Cm(w)
    return tbl


def para(doc, text, bold=False, italic=False, size=10, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    return p


# ============================================================
# Документ
# ============================================================
doc = Document()

for section in doc.sections:
    section.top_margin    = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2)

# ── Титул ────────────────────────────────────────────────────
title = doc.add_heading("Анализ данных и предобработка для TFT-модели", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.color.rgb = RGBColor(0x1F, 0x39, 0x64)
    run.font.size = Pt(18)

subtitle = doc.add_paragraph("Прогнозирование продаж топлива сети АЗС Татнефть")
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in subtitle.runs:
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

doc.add_paragraph(
    f"Temporal Fusion Transformer (Lim et al., 2020)  |  "
    f"Период: 2023 год  |  Станций: {merged['station_id'].nunique()}"
).alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_page_break()

# ============================================================
# 1. ИСТОЧНИКИ ДАННЫХ
# ============================================================
h1(doc, "1. Источники данных")

para(doc, "В работе используются два исходных файла для 5 АЗС Татнефть:", bold=True)

add_table(
    doc,
    headers=["Файл", "Строк", "Колонок", "Содержимое"],
    rows=[
        ["5stations_metadata.csv", "5",  "32",
         "Паспорт АЗС: тип дороги, услуги, количество колонок, базовые цены"],
        ["5stations_data.csv",     "43 800", "72",
         "Почасовые данные 2023 г.: погода, трафик, продажи, цены, акции, реклама"],
        [f"merged_data.csv (LEFT JOIN по station_id)",
         str(merged.shape[0]), str(merged.shape[1]),
         "Объединённый датасет: все временны́е данные + статические характеристики АЗС"],
        [f"prepared_data.csv (после предобработки)",
         str(df.shape[0]), str(df.shape[1]),
         "Нормализованный датасет для подачи в TimeSeriesDataSet"],
    ],
    col_widths=[5.5, 2, 2.5, 8],
)

doc.add_paragraph()
para(
    doc,
    f"LEFT JOIN добавляет к каждой из 43 800 строк временного ряда "
    f"{merged.shape[1] - 72} статических признаков из паспорта АЗС. "
    "Все строки временного ряда сохраняются (пропущенных периодов нет).",
)

# ============================================================
# 2. ИСКЛЮЧЁННЫЕ ПЕРЕМЕННЫЕ
# ============================================================
h1(doc, "2. Исключённые переменные (не подаются в модель)")

para(
    doc,
    f"Из 89 колонок объединённого датасета {len(EXCLUDED_COLS)} исключены как избыточные (см. таблицу ниже). "
    f"Остаётся 89 − {len(EXCLUDED_COLS)} = {89 - len(EXCLUDED_COLS)} колонок. "
    f"Препроцессинг добавляет: 1 is_shop_open + 7 _enc + 8 sin/cos + 1 time_idx + 12 _orig = 29 новых колонок. "
    f"Итого в prepared_data.csv: {89 - len(EXCLUDED_COLS) + 29} колонок, "
    f"из которых {80 + 1 + 8} = 89 — прямые входы TimeSeriesDataSet "
    f"(80 из merged + 1 is_shop_open + 8 sin/cos).",
)

add_table(
    doc,
    headers=["Переменная", "Причина исключения"],
    rows=[
        ["station_name",     "Служебная строка; идентификатор станции — station_id (целое число)"],
        ["total_pumps",      "= sum(num_pumps_AI92, …, num_pumps_DT_WINTER) — дублирует детальные колонки"],
        ["total_fuel_sales", "= sum(sales_AI92, …, sales_DT_WINTER) — дублирует целевые переменные"],
        ["shop_total_revenue","= sum(shop_напитки, …, shop_табак) — дублирует целевые переменные магазина"],
        ["total_traffic",    "Содержит скрытую категорию ТС: total > sum(traffic_*). Несогласован с детальными"],
        ["quarter",          "Однозначно выводится из month; избыточен для модели"],
        ["day_name",         "Строковая форма day_of_week; day_of_week уже в модели c sin/cos кодированием"],
    ],
    col_widths=[4.5, 14],
)

# ============================================================
# 3. ТАКСОНОМИЯ ПЕРЕМЕННЫХ TFT
# ============================================================
h1(doc, "3. Таксономия переменных TFT")

para(
    doc,
    "Temporal Fusion Transformer (Lim et al., 2020) принимает три типа входов. "
    "Разделение определяет, какую информацию модель может использовать при прогнозировании: "
    "статические — зашиты в веса; известные будущие — подаются в декодер; "
    "наблюдаемые прошлые — только в энкодер.",
    bold=True,
)

add_table(
    doc,
    headers=["Тип входа TFT", "Определение", "Кол-во"],
    rows=[
        ["static_categoricals",
         "Категориальные свойства АЗС, не меняющиеся во времени",
         str(len(STATIC_CATS))],
        ["static_reals",
         "Числовые свойства АЗС из паспорта (metadata), не нормализуются",
         str(len(STATIC_REALS))],
        ["time_varying_known_categoricals",
         "Категориальные признаки, известные заранее или задаваемые в what-if",
         str(len(TIME_VARYING_KNOWN_CATS))],
        ["time_varying_known_reals",
         "Числовые признаки, известные на весь горизонт прогноза. "
         "Сюда входят: время, погода, трафик, цены конкурентов, акции. "
         "Только эти переменные можно менять в сценарном what-if анализе.",
         str(len(TIME_VARYING_KNOWN_REALS))],
        ["time_varying_unknown_reals",
         "Наблюдаемые прошлые (энкодер видит их только за прошлое). "
         "Намеренно пусто: все ковариаты перенесены в KNOWN для поддержки what-if.",
         str(len(TIME_VARYING_UNKNOWN_REALS))],
        ["target",
         "12 целевых переменных. TFT использует их как авторегрессивные "
         "входы энкодера (лаговые значения) автоматически через target=TARGET_COLS.",
         str(len(TARGET_COLS))],
    ],
    col_widths=[5.5, 11, 2],
)

# ── 3.1 Статические ──────────────────────────────────────────
h2(doc, "3.1 Статические переменные")

para(
    doc,
    "Паспортные данные АЗС из 5stations_metadata.csv. "
    "Не изменяются во времени и не нормализуются (Z-score не применяется). "
    "TFT строит для них глобальный статический энкодер (контекстный вектор).",
)

add_table(
    doc,
    headers=["Переменная", "Тип TFT-входа", "Предобработка", "Описание"],
    rows=[
        ["road_type",          "static_categoricals", "LabelEncoder → road_type_enc",       "Тип дороги (федеральная / региональная)"],
        ["direction",          "static_categoricals", "LabelEncoder → direction_enc",        "Направление трафика (транзит / въезд / выезд)"],
        ["settlement_size",    "static_categoricals", "LabelEncoder → settlement_size_enc",  "Размер ближайшего населённого пункта"],
        ["distance_to_city_km","static_reals",        "—",                                    "Расстояние до ближайшего города, км"],
        ["shop_area_m2",       "static_reals",        "—",                                    "Площадь магазина, м²"],
        ["num_pumps_AI92/95/98/DT_* (7 к.)", "static_reals", "—",                           "Количество колонок по каждому виду топлива"],
        ["has_car_wash / cafe / shop / tire / hotel (5 к.)", "static_reals", "— (бинарные)", "Наличие сервисов (0/1)"],
        ["competitors_within_5km", "static_reals",   "—",                                    "Число конкурирующих АЗС в радиусе 5 км"],
        ["customer_loyalty_score", "static_reals",   "—",                                    "Оценка лояльности клиентов"],
        ["staff_quality_score",    "static_reals",   "—",                                    "Оценка качества работы персонала"],
        ["corporate_customer_ratio","static_reals",  "—",                                    "Доля корпоративных клиентов"],
        ["staff_engagement_score", "static_reals",   "—",                                    "Оценка вовлечённости персонала"],
        ["base_price_AI92/95/98/DT_* (7 к.)", "static_reals", "—",                          "Базовая цена АЗС по каждому виду топлива, руб/л. "
                                                                                              "Семантически отличается от текущей цены price_*: "
                                                                                              "это эталонная паспортная цена, не оперативная."],
    ],
    col_widths=[5, 4, 4.5, 5],
)

# ── 3.2 Известные будущие ────────────────────────────────────
h2(doc, "3.2 Известные будущие переменные")

para(
    doc,
    "Эти признаки задаются заранее (расписание, метеопрогноз, плановые акции) "
    "или устанавливаются пользователем в what-if сценарии. "
    "TFT подаёт их и в энкодер (ретроспектива), и в декодер (горизонт прогноза). "
    "Именно поэтому к ним применима сценарная аналитика: "
    "\"что если завтра снег?\" или \"что если включить акцию?\"",
)

add_table(
    doc,
    headers=["Переменная / Группа", "Тип TFT-входа", "Предобработка", "Обоснование включения"],
    rows=[
        # KNOWN CATS
        ["season",            "known_categoricals", "LabelEncoder → season_enc",         "Сезон известен заранее; what-if: изменение сезона"],
        ["weather_condition", "known_categoricals", "LabelEncoder → weather_condition_enc","Тип погоды (ясно/снег/дождь…): задаётся по прогнозу; what-if: \"что если метель?\""],
        ["ad_channel",        "known_categoricals", "NaN→'нет_рекламы'; LabelEncoder",   "Канал рекламы планируется заранее; what-if: выбор канала"],
        ["holiday_name",      "known_categoricals", "NaN→'нет_праздника'; LabelEncoder", "Государственные праздники известны заранее"],
        # KNOWN REALS — время
        ["hour, hour_sin, hour_cos",          "known_reals", "sin/cos(2π·h/24)",    "Час суток; циклическое кодирование устраняет разрыв 23→0"],
        ["day_of_week, dow_sin, dow_cos",     "known_reals", "sin/cos(2π·d/7)",     "День недели"],
        ["week_of_year, woy_sin, woy_cos",    "known_reals", "sin/cos(2π·w/52)",    "Номер недели (сезонность)"],
        ["month, month_sin, month_cos",       "known_reals", "sin/cos(2π·m/12)",    "Месяц (месячная сезонность)"],
        ["is_weekend",        "known_reals", "— (бинарная)",  "Выходной день"],
        ["is_holiday",        "known_reals", "— (бинарная)",  "Праздничный день"],
        ["is_rush_hour",      "known_reals", "— (бинарная)",  "Час пик"],
        ["is_night",          "known_reals", "— (бинарная)",  "Ночное время"],
        ["is_shop_open",      "known_reals", "— (бинарная)", "Магазин открыт (1: 05:00–21:00, 0: 22:00–04:00). Производный признак. "
                                                              "Стабилизирует нулевые продажи shop_*: модель явно знает «закрыто»."],
        # KNOWN REALS — акции
        ["promotion_fuel_active", "known_reals", "— (бинарная)", "Акция на топливо активна; what-if: включить/выключить"],
        ["promotion_shop_active", "known_reals", "— (бинарная)", "Акция в магазине; what-if"],
        ["promotion_cafe_active", "known_reals", "— (бинарная)", "Акция в кафе; what-if"],
        ["ad_active",             "known_reals", "— (бинарная)", "Реклама активна; what-if"],
        # KNOWN REALS — цены
        ["price_AI92/95/98/DT_* (7 к.)", "known_reals", "Без Z-score (std=0 per-station)",
         "Текущие цены топлива. В 2023 г. постоянны внутри станции (std=0) → Z-score не применяется. "
         "Декодер может получать гипотетические цены в what-if."],
        # KNOWN REALS — погода
        ["temperature",           "known_reals", "Z-score (train only)", "Температура воздуха, °C; задаётся по метеопрогнозу"],
        ["precipitation_mm",      "known_reals", "Z-score (train only)", "Осадки, мм"],
        ["visibility_km",         "known_reals", "Z-score (train only)", "Видимость, км"],
        ["wind_speed_ms",         "known_reals", "Z-score (train only)", "Скорость ветра, м/с"],
        ["is_snow / is_rain / is_fog (3 к.)", "known_reals", "— (бинарные)", "Погодные флаги"],
        # KNOWN REALS — трафик
        ["traffic_Passengers_cars",  "known_reals", "Z-score (train only)", "Поток легковых автомобилей"],
        ["traffic_Truck_short/Truck/Truck_long (3 к.)", "known_reals", "Z-score (train only)",
         "Грузовые потоки по классам тоннажа"],
        ["traffic_Transporter / traffic_Undefined (2 к.)", "known_reals", "Z-score (train only)",
         "Микроавтобусы и неклассифицированные ТС"],
        # KNOWN REALS — конкуренты
        ["competitor_price_AI92/95/DT (3 к.)", "known_reals", "Z-score (train only)",
         "Цены конкурентов; мониторинг. what-if: снижение/повышение конкурентной цены"],
    ],
    col_widths=[5, 3.5, 4, 6],
)

# ── 3.3 Наблюдаемые прошлые ──────────────────────────────────
h2(doc, "3.3 Наблюдаемые прошлые (time_varying_unknown_reals)")

para(
    doc,
    "В данном проекте TIME_VARYING_UNKNOWN_REALS = [] (пустой список). "
    "Это архитектурное решение, отличающее данную реализацию от стандартного примера TFT.",
)
para(
    doc,
    "Стандартный подход: погода и трафик — UNKNOWN (энкодер видит только прошлое, "
    "декодер — нет). Ограничение: what-if анализ невозможен (\"что если завтра снег?\" "
    "нельзя задать, потому что снег не в KNOWN).",
)
para(
    doc,
    "Выбранный подход: все ковариаты перенесены в KNOWN. Это означает, что модель "
    f"ожидает значения погоды и трафика на весь горизонт прогноза ({PREDICTION_LENGTH} ч). "
    "Источник: метеопрогноз + прогноз дорожного движения. "
    "Преимущество: полноценный сценарный what-if анализ по любой переменной.",
)
para(
    doc,
    "Целевые переменные (TARGET_COLS) добавляются TFT автоматически как авторегрессивные "
    "входы энкодера через параметр target=TARGET_COLS в TimeSeriesDataSet. "
    "Их указывать в UNKNOWN_REALS не нужно.",
    italic=True,
)

# ── 3.4 Целевые переменные ────────────────────────────────────
h2(doc, "3.4 Целевые переменные (12 штук)")

para(
    doc,
    "TFT прогнозирует 12 переменных совместно, используя взаимные корреляции "
    "(покупатели топлива = покупатели магазина). Модель выдаёт квантильные "
    "прогнозы [q2, q10, q25, q50, q75, q90, q98] для каждой цели.",
)

fuel_cols = [c for c in TARGET_COLS if c.startswith("sales_")]
shop_cols = [c for c in TARGET_COLS if c.startswith("shop_")]

skew_rows = []
for col in TARGET_COLS:
    orig_col = col + "_orig"
    if orig_col in df.columns:
        skew_before = df[orig_col].skew()
        skew_after  = df[col].skew()
        mn = df[orig_col].min()
        mx = df[orig_col].max()
        unit = "л/ч" if col.startswith("sales_") else "руб/ч"
        skew_rows.append([col, f"{mn:.0f}–{mx:.0f} {unit}",
                          f"{skew_before:.2f}", f"{skew_after:.2f}"])
    else:
        skew_rows.append([col, "—", "—", "—"])

add_table(
    doc,
    headers=["Переменная", "Диапазон", "Skew до log1p", "Skew после log1p"],
    rows=skew_rows,
    col_widths=[5.5, 4, 3.5, 3.5],
)

doc.add_paragraph()
para(
    doc,
    "log1p(x) = log(1+x) снижает правостороннюю асимметрию продаж (skew 2–4 → 0.2–1.5). "
    "Работает при x=0 (ночные часы без продаж). "
    "TorchNormalizer(method='robust') внутри TFT дополнительно нормализует по медиане и IQR.",
)

# ============================================================
# 4. ПРЕДОБРАБОТКА — ПОШАГОВОЕ ОПИСАНИЕ
# ============================================================
h1(doc, "4. Предобработка данных")

para(doc, "Файл: eda/eda_preprocessing.py. Порядок шагов строго фиксирован.", bold=True)

steps = [
    (
        "Шаг 1. Заполнение пропусков",
        "Два поля содержат пропуски:\n"
        "  - holiday_name (~97% строк — обычные дни без праздника)\n"
        "  - ad_channel (~70% строк — дни без рекламы)\n"
        "Заполняются семантическими значениями:\n"
        "  - holiday_name → 'нет_праздника'\n"
        "  - ad_channel → 'нет_рекламы'\n"
        "Удаление строк недопустимо: нарушило бы непрерывность временного ряда.",
    ),
    (
        "Шаг 2. Исключение избыточных колонок",
        f"Из датафрейма удаляются {len(EXCLUDED_COLS)} избыточных колонок (см. раздел 2):\n"
        "  - station_name, total_pumps, total_fuel_sales,\n"
        "  - shop_total_revenue, total_traffic, quarter, day_name.\n"
        "После исключения: 89 − 7 = 82 колонки.\n"
        "Winsorization (метод IQR) не применяется: статья TFT (Lim et al., 2020) "
        "не описывает этот метод предобработки; выбросы целевых переменных "
        "обрабатываются TorchNormalizer(method='robust') внутри модели.",
    ),
    (
        "Шаг 3. Label Encoding категориальных переменных",
        "Категориальные строковые переменные кодируются в целые числа:\n"
        "  - road_type, direction, settlement_size, weather_condition,\n"
        "  - ad_channel, season, holiday_name\n"
        "Создаются новые колонки с суффиксом _enc.\n"
        "Важно: LabelEncoder обучается (fit) на ВСЕХ данных (43 800 строк), "
        "а не только на train. Это технически необходимо: праздники (holiday_name) "
        "встречаются только в ноябре и декабре, которых нет в train (ноябрь/декабрь — val/test). "
        "Fit только на train вызвал бы KeyError. Аналогично и NaNLabelEncoder в prepare_dataset.py. "
        "Для категориального кодирования (не для обучения модели) это допустимо.",
    ),
    (
        "Шаг 4. Циклическое sin/cos кодирование",
        "Временные признаки hour, day_of_week, month, week_of_year имеют циклическую природу:\n"
        "  hour=23 и hour=0 — соседние часы, но числово далеки (разница 23).\n"
        "Решение: проекция на единичную окружность:\n"
        "  hour_sin = sin(2π × hour / 24)\n"
        "  hour_cos = cos(2π × hour / 24)\n"
        "Расстояние ||(23_sin,23_cos) − (0_sin,0_cos)||₂ минимально.\n"
        "Добавляются 8 новых колонок (4 признака × 2). Итого: 82 + 8 = 90.",
    ),
    (
        "Шаг 5. Монотонный индекс времени (time_idx)",
        "TFT требует целочисленный time_idx без пропусков для каждой группы (станции).\n"
        "Реализация: cumcount() после сортировки по (station_id, timestamp).\n"
        "  time_idx = 0, 1, 2, …, 8759 для каждой из 5 станций (8760 часов = 365 дней).",
    ),
    (
        "Шаг 6. Логарифмическое преобразование (log1p)",
        "Все 12 целевых переменных (TARGET_COLS) имеют правостороннюю асимметрию.\n"
        "log1p(x) = log(1+x):\n"
        "  - работает при x=0 (ночные нулевые продажи)\n"
        "  - снижает skew с 2–4 до 0.2–1.5\n"
        "  - улучшает сходимость нейронной сети\n"
        "shop_total_revenue ИСКЛЮЧЕНА из log1p и из модели (удалена на шаге 2).\n"
        "Оригинальные значения сохраняются в _orig для обратного преобразования в predict.py.",
    ),
    (
        "Шаг 7. Z-score нормализация per-station (без data leakage)",
        "Формула: (x − mean_train) / std_train\n"
        "Ключевое решение: mean и std вычисляются ТОЛЬКО по обучающей выборке (Jan–Oct 2023).\n"
        "Одни и те же статистики применяются к val и test — это стандартная ML-практика,\n"
        "исключающая data leakage из будущего в нормализацию обучающих данных.\n"
        "\n"
        "Исключения из нормализации:\n"
        "  - STATIC_REALS: паспортные данные АЗС (константны для каждой станции)\n"
        "  - Бинарные переменные (0/1): нормализация бессмысленна\n"
        "  - _enc колонки: целые числа для embedding\n"
        "  - _orig колонки: оригинальные значения до log1p\n"
        "  - TARGET_COLS: нормализует TorchNormalizer внутри TFT\n"
        f"  - NO_ZSCORE_COLS (price_*): константны per-station в 2023 г., std=0\n"
        "\n"
        "Скейлеры (mean, std) сохраняются в tft/scalers.pkl для обратного\n"
        "преобразования прогнозов в predict.py.",
    ),
    (
        "Шаг 8. Темпоральный сплит и сохранение",
        "Данные разбиваются строго по времени:\n"
        f"  Train : January–October 2023    (~83% строк)\n"
        f"  Val   : November 2023           (~8% строк)\n"
        f"  Test  : December 2023           (~8% строк)\n"
        "\n"
        "Случайная выборка недопустима: вызвала бы data leakage будущих значений в train.\n"
        "Результат: data/prepared_data.csv, data/train.csv, data/val.csv,\n"
        "           data/test.csv, tft/scalers.pkl",
    ),
]

for title_step, desc in steps:
    h3(doc, title_step)
    for line in desc.split("\n"):
        style = "List Bullet" if line.startswith("  -") else "Normal"
        p = doc.add_paragraph(line.lstrip("  ") if style == "List Bullet" else line,
                              style=style)
        for run in p.runs:
            run.font.size = Pt(10)

# ============================================================
# 5. ПАРАМЕТРЫ TFT-МОДЕЛИ
# ============================================================
h1(doc, "5. Параметры TFT-модели")

add_table(
    doc,
    headers=["Параметр", "CPU", "GPU", "Описание"],
    rows=[
        ["hidden_size",           "64",  "128", "Размер скрытого слоя"],
        ["attention_head_size",   "2",   "4",   "Число голов многоголового внимания"],
        ["hidden_continuous_size","32",  "64",  "Размер Variable Selection Network для числовых входов "
                                              "(= hidden_size // 2). "
                                              "Рекомендация статьи TFT: hidden_continuous ≤ hidden_size / 2. "
                                              "Создаёт лёгкое узкое место → регуляризация числовых признаков."],
        ["BATCH_SIZE",            "32",  "64",  "Размер мини-батча"],
        ["EPOCHS",                "50",  "80",  "Максимальное число эпох (EarlyStopping остановит раньше)"],
    ],
    col_widths=[5, 2, 2, 9.5],
)

doc.add_paragraph()
add_table(
    doc,
    headers=["Параметр", "Значение", "Описание"],
    rows=[
        ["ENCODER_LENGTH",    f"{ENCODER_LENGTH} ч ({ENCODER_LENGTH//24} сут.)",
         "Ретроспективное окно. Охватывает суточный и недельный циклы сезонности (7 дней)."],
        ["PREDICTION_LENGTH", f"{PREDICTION_LENGTH} ч ({PREDICTION_LENGTH//24} сут.)",
         "Горизонт прогноза: 1 сутки (24 часовых шага). Оптимален для оперативного планирования."],
        ["learning_rate",     "3e-4",       "Начальный LR. Снижается ReduceLROnPlateau(patience=5)"],
        ["dropout",           "0.15",       "Регуляризация (Dropout + VariationalDropout)"],
        ["gradient_clip",     "1.0",        "Обрезка градиентов (предотвращает взрыв градиентов)"],
        ["EarlyStopping",     "patience=12","Останавливается при отсутствии улучшения val_loss"],
        ["loss",              "QuantileLoss([0.02,0.1,0.25,0.5,0.75,0.9,0.98])",
         "7-квантильный прогноз: медиана + доверительные интервалы"],
        ["target_normalizer", "MultiNormalizer([TorchNormalizer(robust)] × 12)",
         "Нормализация целей по медиане+IQR (устойчива к выбросам). Применяется внутри TFT."],
        ["add_relative_time_idx", "True", "Относительная позиция шага в окне → known_real"],
        ["add_target_scales",     "True", "Масштаб каждой цели → static_real (location + scale)"],
        ["add_encoder_length",    "True", "Фактическая длина энкодера → static_real"],
    ],
    col_widths=[4.5, 5, 9],
)

# ============================================================
# 6. ИТОГОВЫЙ СОСТАВ ТЕНЗОРОВ TFT
# ============================================================
h1(doc, "6. Итоговый состав тензоров TFT")

para(
    doc,
    "Размеры тензоров, которые TFT получает на вход. Включают как исходные переменные, "
    "так и дополнительные признаки, добавляемые pytorch-forecasting автоматически.",
    bold=True,
)

# Вычисляем размеры
n_static_cats  = len(STATIC_CATS)           # 3
n_static_reals = len(STATIC_REALS)          # 26
n_target_scales = len(TARGET_COLS) * 2      # 12 × 2 = 24  (location + scale)
n_enc_length   = 1                           # encoder_length scalar
n_static_total = n_static_cats + n_static_reals + n_target_scales + n_enc_length

n_known_cats   = len(TIME_VARYING_KNOWN_CATS)   # 4
n_known_reals  = len(TIME_VARYING_KNOWN_REALS)  # 44 (включая is_shop_open)
n_rel_time     = 1                               # relative_time_idx
n_unknown      = len(TIME_VARYING_UNKNOWN_REALS) # 0
n_targets_enc  = len(TARGET_COLS)               # 12  (past observed)
n_enc_total    = n_known_cats + n_known_reals + n_rel_time + n_unknown + n_targets_enc

n_dec_total    = n_known_cats + n_known_reals + n_rel_time

h2(doc, f"6.1 Статические признаки — {n_static_total}")
add_table(
    doc,
    headers=["Компонент", "Кол-во", "Параметр датасета", "Описание"],
    rows=[
        ["static_categoricals",  str(n_static_cats),
         "static_categoricals",  "road_type_enc, direction_enc, settlement_size_enc"],
        ["static_reals (паспорт)", str(n_static_reals),
         "static_reals",          "26 паспортных признаков АЗС (расстояния, колонки, площадь, сервисы, оценки, базовые цены)"],
        ["target_scales",        str(n_target_scales),
         "add_target_scales=True","12 целей × 2: location (медиана) + scale (IQR) от TorchNormalizer(robust)"],
        ["encoder_length",       str(n_enc_length),
         "add_encoder_length=True","Фактическая длина энкодера в текущем сэмпле"],
        ["ИТОГО",                str(n_static_total), "—", ""],
    ],
    col_widths=[4.5, 2, 4.5, 7.5],
    header_color="1F3964",
)

doc.add_paragraph()
h2(doc, f"6.2 Признаки энкодера (прошлое) — {n_enc_total}")
add_table(
    doc,
    headers=["Компонент", "Кол-во", "Параметр датасета", "Описание"],
    rows=[
        ["known_categoricals",       str(n_known_cats),
         "time_varying_known_categoricals",
         "season_enc, weather_condition_enc, ad_channel_enc, holiday_name_enc"],
        ["known_reals",              str(n_known_reals),
         "time_varying_known_reals",
         "Время (12: raw+sin/cos×4), флаги (5: выходной/праздник/час-пик/ночь/магазин-открыт), "
         "акции+реклама (4), цены топлива (7), погода (7), трафик (6), цены конкурентов (3)"],
        ["relative_time_idx",        str(n_rel_time),
         "add_relative_time_idx=True",
         "Относительная позиция шага в энкодерном окне [0…1]"],
        ["unknown_reals",            str(n_unknown),
         "time_varying_unknown_reals = []",
         "Намеренно пусто. Все ковариаты перенесены в KNOWN для what-if."],
        ["targets (observed past)",  str(n_targets_enc),
         "target=TARGET_COLS",
         "12 целевых переменных — в энкодере видны как лаговые входы"],
        ["ИТОГО",                    str(n_enc_total), "—",
         "Полный набор признаков, доступных энкодеру за ретроспективный период"],
    ],
    col_widths=[4.5, 2, 4.5, 7.5],
    header_color="1F3964",
)

doc.add_paragraph()
h2(doc, f"6.3 Признаки декодера (будущее) — {n_dec_total}")
add_table(
    doc,
    headers=["Компонент", "Кол-во", "Параметр датасета", "Описание"],
    rows=[
        ["known_categoricals",  str(n_known_cats),
         "time_varying_known_categoricals",
         "season_enc, weather_condition_enc, ad_channel_enc, holiday_name_enc"],
        ["known_reals",         str(n_known_reals),
         "time_varying_known_reals",
         f"Те же {n_known_reals} признаков, что и в энкодере — задаются на весь горизонт прогноза"],
        ["relative_time_idx",   str(n_rel_time),
         "add_relative_time_idx=True",
         "Относительная позиция шага в декодерном окне [0…1]"],
        ["ИТОГО",               str(n_dec_total), "—",
         "Декодер видит ТОЛЬКО то, что можно знать заранее — нет unknown_reals и нет целевых"],
    ],
    col_widths=[4.5, 2, 4.5, 7.5],
    header_color="1F3964",
)

doc.add_paragraph()
para(
    doc,
    "Ключевое отличие: декодер получает меньше признаков, чем энкодер. "
    "В будущем нельзя наблюдать фактические продажи — поэтому 12 целевых переменных "
    "присутствуют только в энкодере (как прошлые наблюдения), но не в декодере. "
    "При этом благодаря переносу погоды, трафика и цен конкурентов в KNOWN, "
    "декодер располагает полным контекстом для сценарного прогнозирования.",
    italic=True,
)

# ============================================================
# 7. ИТОГОВАЯ СВОДКА
# ============================================================
h1(doc, "7. Итоговая сводка")

add_table(
    doc,
    headers=["TFT-роль", "Кол-во", "Примечание"],
    rows=[
        ["static_categoricals",            str(len(STATIC_CATS)),
         "road_type, direction, settlement_size"],
        ["static_reals",                   str(len(STATIC_REALS)),
         "26 паспортных признаков АЗС (total_pumps исключён как дубль)"],
        ["time_varying_known_categoricals", str(len(TIME_VARYING_KNOWN_CATS)),
         "season, weather_condition, ad_channel, holiday_name"],
        ["time_varying_known_reals",        str(len(TIME_VARYING_KNOWN_REALS)),
         "Время (12), флаги (5: выходной/праздник/час-пик/ночь/магазин-открыт), акции (4), "
         "цены топлива (7), погода (7), трафик (6), конкуренты (3)"],
        ["time_varying_unknown_reals",      str(len(TIME_VARYING_UNKNOWN_REALS)),
         "Пусто — все ковариаты перенесены в KNOWN для what-if анализа"],
        ["target (12 совместных прогнозов)", str(len(TARGET_COLS)),
         "7 видов топлива + 5 категорий магазина"],
        ["ИТОГО входных переменных",
         str(len(STATIC_CATS)+len(STATIC_REALS)+len(TIME_VARYING_KNOWN_CATS)
             +len(TIME_VARYING_KNOWN_REALS)+len(TIME_VARYING_UNKNOWN_REALS)+len(TARGET_COLS)),
         "Без учёта time_idx, station_id и автодобавляемых target_scales/enc_length/rel_time"],
    ],
    col_widths=[5.5, 2.5, 10.5],
    header_color="2E74B5",
)

doc.add_paragraph()
add_table(
    doc,
    headers=["Тензор", "Размер", "Содержимое"],
    rows=[
        ["static",        str(n_static_total),
         f"static_cats({n_static_cats}) + static_reals({n_static_reals}) "
         f"+ target_scales({n_target_scales}) + enc_length({n_enc_length})"],
        ["encoder (прошлое)", str(n_enc_total),
         f"known_cats({n_known_cats}) + known_reals({n_known_reals}) "
         f"+ rel_idx({n_rel_time}) + unknown({n_unknown}) + targets_past({n_targets_enc})"],
        ["decoder (будущее)", str(n_dec_total),
         f"known_cats({n_known_cats}) + known_reals({n_known_reals}) + rel_idx({n_rel_time})"],
    ],
    col_widths=[4, 2.5, 12],
    header_color="2E74B5",
)

# ============================================================
# Сохранение
# ============================================================
out_path = "reports/tft_report.docx"
doc.save(out_path)
print(f"Отчёт сохранён: {out_path}")

print("\n" + "=" * 60)
print("Готово.")
print(f"  Сохранено : {out_path}")
print("=" * 60)
