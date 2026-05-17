"""
Формирование DOCX-отчёта по TFT-анализу данных.
Описывает каждую переменную, её предобработку и роль в TFT-модели.

Запускать из корня проекта: python eda/tft_report.py
Результат: reports/tft_report.docx
"""

import os
import pickle
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
from utils.data_utils import CYCLICAL_FEATURES, LOG_COLS, TARGET_COLS

os.makedirs("reports", exist_ok=True)

# ── Загрузка данных ──────────────────────────────────────────
df = pd.read_csv("data/prepared_data.csv", parse_dates=["timestamp"])
merged = pd.read_csv("data/merged_data.csv", parse_dates=["timestamp"])
train_df = pd.read_csv("data/train.csv", parse_dates=["timestamp"])
val_df = pd.read_csv("data/val.csv", parse_dates=["timestamp"])
test_df = pd.read_csv("data/test.csv", parse_dates=["timestamp"])

with open("tft/scalers.pkl", "rb") as f:
    scaler_data = pickle.load(f)
scalers = scaler_data["scalers"]
log1p_cols = scaler_data["log1p_cols"]


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

    # Заголовок
    hdr = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        cell_bg(hdr[i], header_color)
        for run in hdr[i].paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9)

    # Данные
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

# Поля
for section in doc.sections:
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2)

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

para(doc, "В работе используются два исходных файла для 5 АЗС:", bold=True)

add_table(
    doc,
    headers=["Файл", "Строк", "Колонок", "Содержимое"],
    rows=[
        [
            "5stations_metadata.csv",
            "5",
            "32",
            "Паспорт АЗС: тип дороги, услуги, ценовые ориентиры, кол-во колонок",
        ],
        [
            "5stations_data.csv",
            "43 800",
            "72",
            "Почасовые данные 2023 г.: погода, трафик, продажи, цены, акции",
        ],
        [
            "merged_data.csv (JOIN)",
            str(merged.shape[0]),
            str(merged.shape[1]),
            "Объединённый датасет по station_id",
        ],
        [
            "prepared_data.csv (после предобработки)",
            str(df.shape[0]),
            str(df.shape[1]),
            "Нормализованный датасет для подачи в TFT",
        ],
    ],
    col_widths=[5.5, 2, 2.5, 8],
)

doc.add_paragraph()
para(
    doc,
    "Объединение выполняется через LEFT JOIN по полю station_id. "
    "Все 43 800 строк временного ряда сохраняются; к ним добавляются "
    f"{merged.shape[1] - 72} статических признаков из паспорта АЗС.",
)

# ============================================================
# 2. КЛАССИФИКАЦИЯ ПЕРЕМЕННЫХ
# ============================================================
h1(doc, "2. Классификация переменных для TFT-модели")

para(
    doc,
    "Согласно статье Lim et al. (2020), TFT принимает три типа входов:",
    bold=True,
)

add_table(
    doc,
    headers=["Тип входа TFT", "Определение", "Кол-во колонок"],
    rows=[
        [
            "static_categoricals",
            "Категориальные свойства АЗС, не меняющиеся во времени",
            "3",
        ],
        [
            "static_reals",
            "Числовые свойства АЗС из паспорта (metadata)",
            "27",
        ],
        [
            "time_varying_known_categoricals",
            "Категориальные признаки, известные заранее (сезон, праздник)",
            "4",
        ],
        [
            "time_varying_known_reals",
            "Числовые признаки, известные заранее (цены, акции, час суток)",
            "21 + 8 sin/cos",
        ],
        [
            "time_varying_unknown_reals",
            "Наблюдаемые прошлые (погода, трафик, shop_total_revenue, цены конкурентов) + TARGET_COLS",
            "19 + 12",
        ],
        [
            "target",
            "Целевые переменные: 7 видов топлива + 5 категорий магазина",
            "12",
        ],
    ],
    col_widths=[5, 10, 3.5],
)

# ── 2.1 Статические переменные ───────────────────────────────
h2(doc, "2.1 Статические переменные (не меняются во времени)")
para(
    doc,
    "Статические данные берутся из файла 5stations_metadata.csv и "
    "присоединяются к каждой строке временного ряда при JOIN.",
)

add_table(
    doc,
    headers=["Переменная", "Тип", "TFT-вход", "Предобработка"],
    rows=[
        ["road_type", "категориальная", "static_categoricals", "LabelEncoder -> road_type_enc"],
        ["direction", "категориальная", "static_categoricals", "LabelEncoder -> direction_enc"],
        ["settlement_size", "категориальная", "static_categoricals", "LabelEncoder -> settlement_size_enc"],
        ["distance_to_city_km", "вещественная", "static_reals", "Паспортные данные. Без изменений."],
        ["total_pumps", "вещественная", "static_reals", "Паспортные данные. Без изменений."],
        ["shop_area_m2", "вещественная", "static_reals", "Паспортные данные. Без изменений."],
        ["num_pumps_AI92/95/98", "вещественная", "static_reals", "Паспортные данные. Без изменений. (3 колонки)"],
        ["num_pumps_DT_*", "вещественная", "static_reals", "Паспортные данные. Без изменений. (4 колонки)"],
        ["has_car_wash/cafe/shop/...", "бинарная", "static_reals", "Паспортные данные. Бинарный 0/1."],
        ["competitors_within_5km", "вещественная", "static_reals", "Паспортные данные. Без изменений."],
        ["customer/staff/corporate scores", "вещественная", "static_reals", "Паспортные данные. Без изменений. (4 колонки)"],
        ["base_price_AI92/95/98/DT_*", "вещественная", "static_reals", "Паспортные данные. Без изменений. (7 колонок)"],
    ],
    col_widths=[5, 3, 4.5, 6],
)

# ── 2.2 Известные будущие переменные ─────────────────────────
h2(doc, "2.2 Известные будущие переменные")
para(
    doc,
    "Эти признаки можно знать заранее: расписание, цены, плановые акции. "
    "TFT подаёт их и в encoder (ретроспектива), и в decoder (прогнозный горизонт), "
    "что позволяет модели учитывать будущий контекст при построении прогноза.",
)

add_table(
    doc,
    headers=["Переменная", "Тип", "TFT-вход", "Предобработка"],
    rows=[
        ["season", "категориальная", "known_cats", "LabelEncoder -> season_enc"],
        ["day_name", "категориальная", "known_cats", "LabelEncoder -> day_name_enc"],
        ["ad_channel", "категориальная", "known_cats", "NaN -> 'нет_рекламы'. LabelEncoder"],
        ["holiday_name", "категориальная", "known_cats", "NaN -> 'нет_праздника'. LabelEncoder"],
        ["hour", "вещественная (цикл.)", "known_reals", "Z-score + hour_sin, hour_cos (period=24)"],
        ["day_of_week", "вещественная (цикл.)", "known_reals", "Z-score + dow_sin, dow_cos (period=7)"],
        ["month", "вещественная (цикл.)", "known_reals", "Z-score + month_sin, month_cos (period=12)"],
        ["week_of_year", "вещественная (цикл.)", "known_reals", "Z-score + woy_sin, woy_cos (period=52)"],
        ["quarter", "вещественная", "known_reals", "Z-score"],
        ["is_weekend / is_holiday / ...", "бинарная", "known_reals", "Без нормализации (0/1)"],
        ["promotion_fuel/shop/cafe_active", "бинарная", "known_reals", "Без нормализации (0/1)"],
        ["ad_active", "бинарная", "known_reals", "Без нормализации (0/1)"],
        ["price_AI92/95/98/DT_*", "вещественная", "known_reals", "Z-score per-station (7 колонок)"],
    ],
    col_widths=[5.5, 3.5, 3, 6.5],
)

# ── 2.3 Наблюдаемые прошлые ──────────────────────────────────
h2(doc, "2.3 Наблюдаемые прошлые (observed past inputs)")
para(
    doc,
    "Эти признаки доступны только за прошлое — их нельзя знать заранее. "
    "TFT использует их только в encoder (168 часов ретроспективы). "
    "Включают погоду, дорожный трафик, выручку магазина и цены конкурентов.",
)

add_table(
    doc,
    headers=["Группа", "Переменные", "Предобработка"],
    rows=[
        [
            "Погода (8 переменных)",
            "temperature, precipitation_mm, visibility_km, wind_speed_ms,\nis_snow, is_rain, is_fog, weather_condition",
            "Winsorization IQR + Z-score. weather_condition -> float через _enc",
        ],
        [
            "Трафик (7 переменных)",
            "traffic_Passengers_cars, traffic_Truck_short, traffic_Truck,\ntraffic_Truck_long, traffic_Transporter, traffic_Undefined, total_traffic",
            "Winsorization IQR + Z-score",
        ],
        [
            "Магазин (1 переменная)",
            "shop_total_revenue",
            "Z-score. log1p + _orig.\n"
            "Категории магазина (shop_напитки, shop_закуски, shop_автотовары,\n"
            "shop_кофе, shop_табак) перенесены в целевые переменные.",
        ],
        [
            "Цены конкурентов (3 переменных)",
            "competitor_price_AI92, competitor_price_AI95, competitor_price_DT",
            "Winsorization IQR + Z-score",
        ],
    ],
    col_widths=[4.5, 8, 6],
)

# ── 2.4 Целевые переменные ────────────────────────────────────
h2(doc, "2.4 Целевые переменные (выходы модели)")
para(
    doc,
    "12 переменных являются одновременно целью прогноза (выход модели) "
    "и наблюдаемыми прошлыми входами (encoder видит их лаговые значения). "
    "Это стандартная авторегрессионная схема TFT (Section 3, Lim et al., 2020):\n"
    "  - 7 видов топлива: sales_AI92/95/98, sales_DT_EURO/TANEKO/SUMMER/WINTER\n"
    "  - 5 категорий магазина: shop_напитки, shop_закуски, shop_автотовары, shop_кофе, shop_табак\n"
    "Модель прогнозирует продажи топлива и сопутствующих товаров совместно, "
    "используя взаимные корреляции между группами (покупатели топлива = покупатели магазина).",
)

# Вычисляем skew до/после
skew_rows = []
for col in TARGET_COLS:
    orig_col = col + "_orig"
    if orig_col in df.columns:
        skew_before = df[orig_col].skew()
        skew_after = df[col].skew()
        mn = df[orig_col].min()
        mx = df[orig_col].max()
        skew_rows.append([col, f"{mn:.0f} – {mx:.0f}", f"{skew_before:.2f}", f"{skew_after:.2f}"])
    else:
        skew_rows.append([col, "—", "—", "—"])

add_table(
    doc,
    headers=["Переменная", "Диапазон (л/ч)", "Skew до log1p", "Skew после log1p"],
    rows=skew_rows,
    col_widths=[5, 4, 4, 4],
)

doc.add_paragraph()
para(
    doc,
    "Дополнительная нормализация целевых переменных выполняется TorchNormalizer(method='robust') "
    "внутри TFT — этот нормализатор использует медиану и IQR вместо mean/std, "
    "что делает его устойчивым к оставшимся выбросам.",
)

# ============================================================
# 3. ПРЕДОБРАБОТКА: ПОШАГОВОЕ ОПИСАНИЕ
# ============================================================
h1(doc, "3. Предобработка данных")

steps = [
    (
        "Шаг 1. Заполнение пропусков",
        "Два поля содержат пропуски: holiday_name (42 720 из 43 800 строк) "
        "и ad_channel (30 609 строк). Отсутствие значения несёт смысл:\n"
        "  - holiday_name -> 'нет_праздника' (обычный рабочий день)\n"
        "  - ad_channel -> 'нет_рекламы' (реклама не размещалась)\n"
        "Удаление строк с NaN недопустимо — нарушает непрерывность временного ряда.",
    ),
    (
        "Шаг 2. Устранение выбросов (Winsorization)",
        "Метод IQR: значения, выходящие за [Q1 - 1.5*IQR, Q3 + 1.5*IQR], "
        "заменяются граничными значениями. Строки не удаляются.\n"
        "Применяется к интервальным (не бинарным) переменным временного ряда.\n"
        "Исключены: бинарные (0/1), категориальные, а также все статические "
        "переменные из metadata — они являются паспортными характеристиками АЗС "
        "и не могут быть выбросами.",
    ),
    (
        "Шаг 3. Label Encoding категориальных переменных",
        "Категориальные строковые переменные кодируются целыми числами (LabelEncoder).\n"
        "TFT строит Embedding-слои по целым числам — это эффективнее One-Hot Encoding.\n"
        "Создаются новые колонки с суффиксом _enc. Исходные строковые колонки сохранены.",
    ),
    (
        "Шаг 4. Циклическое sin/cos кодирование",
        "Временные признаки hour, day_of_week, month, week_of_year имеют циклическую природу:\n"
        "  - hour=23 и hour=0 — соседние часы, но числово далеки (разница 23)\n"
        "  - Решение: sin/cos на единичной окружности\n"
        "    hour_sin = sin(2π * hour / 24)\n"
        "    hour_cos = cos(2π * hour / 24)\n"
        "  => ||(23_sin, 23_cos) - (0_sin, 0_cos)||₂ минимален\n"
        "Добавляются 8 новых колонок (2 * 4 признака).",
    ),
    (
        "Шаг 5. Монотонный индекс времени (time_idx)",
        "TFT требует целочисленный, возрастающий time_idx без пропусков для каждой группы.\n"
        "Реализуется через cumcount() после сортировки по (station_id, timestamp):\n"
        "  time_idx = 0, 1, 2, ..., 8759 — для каждой из 5 станций",
    ),
    (
        "Шаг 6. Логарифмическое преобразование (log1p)",
        "Все целевые переменные (7 видов топлива + 5 категорий магазина) и shop_total_revenue "
        "имеют правостороннюю асимметрию (skew > 1). log1p(x) = log(1+x):\n"
        "  - работает при x=0 (ночные часы — продажи равны нулю)\n"
        "  - снижает skew и стабилизирует дисперсию\n"
        "  - улучшает сходимость нейросети\n"
        "Оригинальные значения сохраняются с суффиксом _orig для обратного преобразования.",
    ),
    (
        "Шаг 7. Z-score нормализация per-station",
        "Каждая станция нормализуется независимо: (x - mean_s) / std_s.\n"
        "Обоснование: трассовые АЗС имеют масштаб продаж в 3–5 раз выше городских.\n"
        "Исключены из нормализации:\n"
        "  - статические переменные из metadata (паспортные данные АЗС не меняются)\n"
        "  - бинарные переменные (0/1)\n"
        "  - _enc колонки (целые числа для embedding)\n"
        "  - _orig колонки (оригинальные значения до log1p)\n"
        "  - LOG_COLS (целевые переменные — нормализует TorchNormalizer внутри TFT)\n"
        "Скейлеры (mean, std) сохраняются в tft/scalers.pkl для обратного преобразования.",
    ),
    (
        "Шаг 8. Темпоральный сплит",
        "Данные разбиваются строго по времени (без рандомизации):\n"
        f"  Train : Jan – Oct 2023  ({len(train_df)} строк, {len(train_df)/len(df)*100:.1f}%)\n"
        f"  Val   : Nov 2023        ({len(val_df)} строк, {len(val_df)/len(df)*100:.1f}%)\n"
        f"  Test  : Dec 2023        ({len(test_df)} строк, {len(test_df)/len(df)*100:.1f}%)\n"
        "Случайная выборка недопустима — вызвала бы data leakage из будущего в прошлое.",
    ),
]

for title_step, desc in steps:
    h3(doc, title_step)
    for line in desc.split("\n"):
        p = doc.add_paragraph(line, style="List Bullet" if line.startswith("  -") else "Normal")
        for run in p.runs:
            run.font.size = Pt(10)

# ============================================================
# 4. ВХОДЫ И ВЫХОДЫ TFT
# ============================================================
h1(doc, "4. Итоговая схема входов и выходов TFT")

h2(doc, "4.1 Параметры модели")

add_table(
    doc,
    headers=["Параметр", "Значение", "Описание"],
    rows=[
        ["ENCODER_LENGTH", "168 ч", "Глубина ретроспективы (7 суток)"],
        ["PREDICTION_LENGTH", "24 ч", "Горизонт прогноза (1 сутки)"],
        ["hidden_size", "64 / 128", "CPU / GPU"],
        ["attention_head_size", "2 / 4", "CPU / GPU"],
        ["dropout", "0.1", "Регуляризация"],
        ["learning_rate", "1e-3", "Начальный LR (ReduceOnPlateau)"],
        ["gradient_clip", "0.1", "Обрезка градиентов"],
        ["EarlyStopping", "patience=5", "Останавливается при отсутствии улучшения"],
        ["target_normalizer", "TorchNormalizer(robust)", "Медиана + IQR, устойчив к выбросам"],
    ],
    col_widths=[5, 4, 9.5],
)

h2(doc, "4.2 Входные данные")

add_table(
    doc,
    headers=["TFT-вход", "Источник", "Колонки (после предобработки)"],
    rows=[
        [
            "group_ids",
            "5stations_data.csv",
            "station_id (строка)",
        ],
        [
            "static_categoricals",
            "metadata",
            "road_type_enc, direction_enc, settlement_size_enc",
        ],
        [
            "static_reals",
            "metadata",
            "distance_to_city_km, total_pumps, shop_area_m2, "
            "num_pumps_* (7), has_* (5), scores (4), base_price_* (7) — всего 27",
        ],
        [
            "known_categoricals",
            "data",
            "season_enc, day_name_enc, ad_channel_enc, holiday_name_enc",
        ],
        [
            "known_reals",
            "data",
            "hour (+sin/cos), day_of_week (+sin/cos), week_of_year (+sin/cos), "
            "month (+sin/cos), quarter, is_* (4), promotion_* (3), "
            "ad_active, price_* (7) — итого 29 колонок",
        ],
        [
            "unknown_reals",
            "data",
            "погода (8), трафик (7), магазин (1: shop_total_revenue), competitor_price (3) + TARGET_COLS (12) — итого 31",
        ],
    ],
    col_widths=[3.5, 3, 12],
)

h2(doc, "4.3 Выходные данные")
para(
    doc,
    "TFT выдаёт квантильные прогнозы (QuantileLoss) для каждой из 12 целевых переменных "
    "на 24 часа вперёд (7 видов топлива + 5 категорий магазина). "
    "Квантили по умолчанию: q2, q10, q25, q50, q75, q90, q98 — позволяют строить "
    "доверительные интервалы прогноза и оценивать неопределённость.",
)

fuel_cols = [c for c in TARGET_COLS if c.startswith("sales_")]
shop_cols = [c for c in TARGET_COLS if c.startswith("shop_")]

add_table(
    doc,
    headers=["Переменная", "Группа", "Единица", "Описание"],
    rows=(
        [[col, "Топливо", "л/ч", "Продажи топлива за 1 час"] for col in fuel_cols]
        + [[col, "Магазин", "руб/ч", "Выручка категории магазина за 1 час"] for col in shop_cols]
    ),
    col_widths=[5, 3, 2.5, 8],
)

para(
    doc,
    "\nОбратное преобразование прогноза (tft/predict.py): "
    "TorchNormalizer^-1 (mode='quantiles', автоматически) -> expm1(x) -> исходные единицы (литры/час для топлива, руб/ч для магазина).",
)

# ============================================================
# Сохранение
# ============================================================
out_path = "reports/tft_report.docx"
doc.save(out_path)
print(f"Отчёт сохранён: {out_path}")

print("\n" + "=" * 60)
print("Готово. Следующий шаг: python tft/prepare_dataset.py")
print("=" * 60)
