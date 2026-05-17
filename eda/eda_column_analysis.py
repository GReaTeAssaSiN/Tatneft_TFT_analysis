"""
Детальный анализ переменных merged_data.csv для TFT-модели.
Определяет TFT-роль каждой переменной и план предобработки
согласно статье Lim et al. (2020) «Temporal Fusion Transformers».

Запускать из корня проекта: python eda/eda_column_analysis.py
Результат: reports/column_analysis.txt + вывод в консоль
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_utils import CYCLICAL_FEATURES, FILL_MAP, LOG_COLS, TARGET_COLS

os.makedirs("reports", exist_ok=True)

df = pd.read_csv("data/merged_data.csv", parse_dates=["timestamp"])

# ============================================================
# TFT-классификация переменных (Lim et al., 2020, Section 3)
# ============================================================
#
# TFT принимает три типа входов:
#   (a) static covariates  — не меняются во времени (metadata)
#   (b) known future inputs — можно знать заранее (расписание, цены, акции)
#   (c) observed past inputs — известны только за прошлое (погода, трафик, продажи)
#
# Целевые переменные — особый подтип (c): они же входят как
# observed past (лаговые значения) и являются прогнозируемым выходом.

TFT_ROLES = {
    # ── Идентификаторы ──────────────────────────────────────────
    "station_id": {
        "role": "group_id",
        "tft_input": "group_ids",
        "preprocessing": "Строка. Ключ группировки для TimeSeriesDataSet.",
    },
    "station_name": {
        "role": "identifier",
        "tft_input": "не подаётся в модель",
        "preprocessing": "Строка. Только для читаемости, в модель не входит.",
    },
    "timestamp": {
        "role": "time_index_source",
        "tft_input": "не подаётся в модель",
        "preprocessing": "datetime. Используется для построения time_idx (cumcount).",
    },
    # ── Статические категориальные ───────────────────────────────
    "road_type": {
        "role": "static categorical",
        "tft_input": "static_categoricals",
        "preprocessing": "Label Encoding -> road_type_enc. TFT строит embedding-слой.",
    },
    "direction": {
        "role": "static categorical",
        "tft_input": "static_categoricals",
        "preprocessing": "Label Encoding -> direction_enc.",
    },
    "settlement_size": {
        "role": "static categorical",
        "tft_input": "static_categoricals",
        "preprocessing": "Label Encoding -> settlement_size_enc.",
    },
    # ── Статические вещественные ─────────────────────────────────
    "distance_to_city_km": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "total_pumps": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "shop_area_m2": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "num_pumps_AI92": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "num_pumps_AI95": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "num_pumps_AI98": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "num_pumps_DT_EURO": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "num_pumps_DT_TANEKO": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "num_pumps_DT_SUMMER": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "num_pumps_DT_WINTER": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "has_car_wash": {
        "role": "static real (binary)",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Бинарный 0/1. Не изменяется.",
    },
    "has_tire_service": {
        "role": "static real (binary)",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Бинарный 0/1. Не изменяется.",
    },
    "has_cafe": {
        "role": "static real (binary)",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Бинарный 0/1. Не изменяется.",
    },
    "has_hotel": {
        "role": "static real (binary)",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Бинарный 0/1. Не изменяется.",
    },
    "has_shop": {
        "role": "static real (binary)",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Бинарный 0/1. Не изменяется.",
    },
    "competitors_within_5km": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "customer_loyalty_score": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "staff_quality_score": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "staff_engagement_score": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "corporate_customer_ratio": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "base_price_AI92": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "base_price_AI95": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "base_price_AI98": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "base_price_DT_EURO": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "base_price_DT_TANEKO": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "base_price_DT_SUMMER": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    "base_price_DT_WINTER": {
        "role": "static real",
        "tft_input": "static_reals",
        "preprocessing": "Паспортные данные АЗС. Не изменяются.",
    },
    # ── Известные будущие категориальные ────────────────────────
    "season": {
        "role": "known future categorical",
        "tft_input": "time_varying_known_categoricals",
        "preprocessing": "Label Encoding -> season_enc.",
    },
    "day_name": {
        "role": "known future categorical",
        "tft_input": "time_varying_known_categoricals",
        "preprocessing": "Label Encoding -> day_name_enc.",
    },
    "ad_channel": {
        "role": "known future categorical",
        "tft_input": "time_varying_known_categoricals",
        "preprocessing": "NaN -> 'нет_рекламы'. Label Encoding -> ad_channel_enc.",
    },
    "holiday_name": {
        "role": "known future categorical",
        "tft_input": "time_varying_known_categoricals",
        "preprocessing": "NaN -> 'нет_праздника'. Label Encoding -> holiday_name_enc.",
    },
    # ── Известные будущие вещественные ──────────────────────────
    "hour": {
        "role": "known future real (cyclical)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": (
            "Z-score. + sin/cos: hour_sin=sin(2pi*hour/24), hour_cos=cos(2pi*hour/24)."
            " Циклическое кодирование устраняет разрыв 23:00->00:00."
        ),
    },
    "day_of_week": {
        "role": "known future real (cyclical)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": (
            "Z-score. + sin/cos: dow_sin=sin(2pi*dow/7), dow_cos=cos(2pi*dow/7)."
        ),
    },
    "week_of_year": {
        "role": "known future real (cyclical)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score. + sin/cos (period=52).",
    },
    "month": {
        "role": "known future real (cyclical)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score. + sin/cos (period=12).",
    },
    "quarter": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score.",
    },
    "is_weekend": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1. Не нормализуется.",
    },
    "is_holiday": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1. Не нормализуется.",
    },
    "is_rush_hour": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1. Не нормализуется.",
    },
    "is_night": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1. Не нормализуется.",
    },
    "promotion_fuel_active": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1. Акции планируются заранее.",
    },
    "promotion_shop_active": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1.",
    },
    "promotion_cafe_active": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1.",
    },
    "ad_active": {
        "role": "known future real (binary)",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Бинарный 0/1.",
    },
    "price_AI92": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score per-station. Цена устанавливается заранее.",
    },
    "price_AI95": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score per-station.",
    },
    "price_AI98": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score per-station.",
    },
    "price_DT_EURO": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score per-station.",
    },
    "price_DT_TANEKO": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score per-station.",
    },
    "price_DT_SUMMER": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score per-station.",
    },
    "price_DT_WINTER": {
        "role": "known future real",
        "tft_input": "time_varying_known_reals",
        "preprocessing": "Z-score per-station.",
    },
    # ── Наблюдаемые прошлые (не целевые) ────────────────────────
    "temperature": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "weather_condition": {
        "role": "observed past (categorical -> real)",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": (
            "Label Encoding -> weather_condition_enc (float)."
            " Используется как вещественное — TFT обрабатывает через prescaler."
        ),
    },
    "precipitation_mm": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "visibility_km": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station.",
    },
    "wind_speed_ms": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "is_snow": {
        "role": "observed past real (binary)",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Бинарный 0/1. Не нормализуется.",
    },
    "is_rain": {
        "role": "observed past real (binary)",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Бинарный 0/1.",
    },
    "is_fog": {
        "role": "observed past real (binary)",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Бинарный 0/1.",
    },
    "traffic_Passengers_cars": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "traffic_Truck_short": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "traffic_Truck": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "traffic_Truck_long": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "traffic_Transporter": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "traffic_Undefined": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station.",
    },
    "total_traffic": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": (
            "Z-score per-station. Является суммой компонент трафика."
            " VSN автоматически снизит вес при высокой коллинеарности."
        ),
    },
    "shop_напитки": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust) внутри TFT. Z-score не применяется.",
    },
    "shop_закуски": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "shop_автотовары": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "shop_кофе": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "shop_табак": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "shop_total_revenue": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig сохраняется. Z-score НЕ применяется (TorchNormalizer).",
    },
    "competitor_price_AI92": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "competitor_price_AI95": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    "competitor_price_DT": {
        "role": "observed past real",
        "tft_input": "time_varying_unknown_reals",
        "preprocessing": "Z-score per-station. Winsorization IQR.",
    },
    # ── Целевые переменные ───────────────────────────────────────
    "sales_AI92": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust) внутри TFT. Z-score не применяется.",
    },
    "sales_AI95": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "sales_AI98": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "sales_DT_EURO": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "sales_DT_TANEKO": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "sales_DT_SUMMER": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    "sales_DT_WINTER": {
        "role": "target + observed past real",
        "tft_input": "target + time_varying_unknown_reals",
        "preprocessing": "log1p -> _orig. TorchNormalizer(robust).",
    },
    # ── Исключены из модели ──────────────────────────────────────
    "total_fuel_sales": {
        "role": "excluded",
        "tft_input": "не подаётся в модель",
        "preprocessing": "Линейная сумма 7 целевых переменных. Избыточна для модели.",
    },
}

# ============================================================
# Построение Markdown-отчёта
# ============================================================
from collections import Counter

lines = []

TFT_GROUPS_ORDER = [
    "group_ids",
    "не подаётся в модель",
    "static_categoricals",
    "static_reals",
    "time_varying_known_categoricals",
    "time_varying_known_reals",
    "time_varying_unknown_reals",
    "target + time_varying_unknown_reals",
]

GROUP_LABELS = {
    "group_ids":                            "Идентификатор группы (`group_ids`)",
    "не подаётся в модель":                 "Не входят в модель",
    "static_categoricals":                  "Статические категориальные (`static_categoricals`)",
    "static_reals":                         "Статические вещественные (`static_reals`)",
    "time_varying_known_categoricals":      "Известные будущие категориальные (`known_cats`)",
    "time_varying_known_reals":             "Известные будущие вещественные (`known_reals`)",
    "time_varying_unknown_reals":           "Наблюдаемые прошлые вещественные (`unknown_reals`)",
    "target + time_varying_unknown_reals":  "Целевые переменные (`target + unknown_reals`)",
}

GROUP_DESCRIPTIONS = {
    "group_ids": (
        "Ключ группировки в `TimeSeriesDataSet`. TFT обучается отдельно по каждой станции."
    ),
    "не подаётся в модель": (
        "Читаемые идентификаторы или избыточные суммарные переменные."
    ),
    "static_categoricals": (
        "Характеристики АЗС, не меняющиеся во времени (тип дороги, направление, размер). "
        "TFT кодирует через Entity Embedding (learnable) — Section 4.1 статьи. "
        "Предобработка: `LabelEncoder` → целое число → embedding-слой."
    ),
    "static_reals": (
        "Числовые характеристики АЗС из паспорта (metadata). "
        "TFT использует через Static Variable Selection Network. "
        "Нормализация **не применяется** — паспортные данные не меняются, "
        "исключены из Z-score нормализации (`static_skip` в `eda_preprocessing.py`)."
    ),
    "time_varying_known_categoricals": (
        "Категориальные признаки, известные заранее (сезон, день недели, праздник). "
        "TFT подаёт в encoder **и** decoder — позволяет учесть будущий контекст. "
        "Предобработка: `NaN` → семантическое значение, `LabelEncoder` → embedding."
    ),
    "time_varying_known_reals": (
        "Числовые признаки, известные заранее (час, цены, акции, флаги дней). "
        "TFT подаёт в encoder **и** decoder. "
        "Предобработка: Z-score. Циклические (hour, dow, month, woy) → sin/cos. "
        "Бинарные (is_weekend, промо-флаги) — без нормализации."
    ),
    "time_varying_unknown_reals": (
        "Наблюдаемые переменные — доступны только в прошлом (погода, трафик). "
        "TFT использует **только в encoder** (168 часов ретроспективы). "
        "Предобработка: Winsorization IQR + Z-score per-station. "
        "`weather_condition`: LabelEncoder → float (через prescaler в TFT)."
    ),
    "target + time_varying_unknown_reals": (
        "12 переменных (7 топливо + 5 магазин) — **одновременно цель прогноза и observed past**. "
        "TFT статья (Section 3): target является частью observed inputs. "
        "Предобработка: log1p (skew устранён) + `TorchNormalizer(robust)` внутри TFT. "
        "Z-score в `eda_preprocessing.py` **не применяется**. "
        "`_orig`-колонки сохраняются для обратного преобразования прогнозов."
    ),
}

# ── Заголовок ─────────────────────────────────────────────────
lines.append("# Анализ переменных для TFT-модели")
lines.append("")
lines.append(f"**Файл:** `data/merged_data.csv`  ")
lines.append(f"**Строк:** {df.shape[0]}  ")
lines.append(f"**Колонок:** {df.shape[1]}  ")
lines.append(f"**Период:** {df['timestamp'].min().date()} — {df['timestamp'].max().date()}  ")
lines.append(f"**Станций:** {df['station_id'].nunique()}  ")
lines.append(f"**Часов/станцию:** {df.groupby('station_id').size().unique()[0]}")
lines.append("")

# ── Сводная таблица ───────────────────────────────────────────
lines.append("## Сводная таблица: TFT-роль → количество переменных")
lines.append("")
lines.append("| TFT-вход | Колонок (исходных) |")
lines.append("|---|---|")
role_counts = Counter(v["tft_input"] for v in TFT_ROLES.values())
for group_key in TFT_GROUPS_ORDER:
    cnt = role_counts.get(group_key, 0)
    if cnt:
        extra = " (+ 8 sin/cos → 28 в модели)" if group_key == "time_varying_known_reals" else ""
        lines.append(f"| {GROUP_LABELS[group_key]} | {cnt}{extra} |")
lines.append("")

# ── Описание по группам ───────────────────────────────────────
lines.append("## Описание групп переменных")
lines.append("")

for group_key in TFT_GROUPS_ORDER:
    group_cols = [col for col, meta in TFT_ROLES.items() if meta["tft_input"] == group_key]
    if not group_cols:
        continue

    lines.append(f"### {GROUP_LABELS[group_key]}")
    lines.append("")
    lines.append(GROUP_DESCRIPTIONS[group_key])
    lines.append("")
    lines.append(f"**Колонок:** {len(group_cols)}")
    lines.append("")

    # Таблица колонок
    lines.append("| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |")
    lines.append("|---|---|---|---|---|---|")

    for col in group_cols:
        if col not in df.columns:
            lines.append(f"| `{col}` | ? | ? | ? | отсутствует | ? |")
            continue
        dtype = str(df[col].dtype)
        n_null = int(df[col].isnull().sum())
        n_uniq = int(df[col].nunique())

        if df[col].dtype == object:
            vals = [str(v) for v in df[col].dropna().unique().tolist()]
            short = [v[:20] for v in vals[:3]]
            range_str = ", ".join(short) + (", ..." if n_uniq > 3 else "")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            range_str = f"{df[col].min().date()} — {df[col].max().date()}"
        else:
            range_str = f"[{df[col].min():.3g}, {df[col].max():.3g}]"

        prep = TFT_ROLES[col]["preprocessing"].replace("\n", " ")
        lines.append(f"| `{col}` | {dtype} | {n_uniq} | {n_null} | {range_str} | {prep} |")

    lines.append("")

# ── Циклическое кодирование ───────────────────────────────────
lines.append("## Циклическое кодирование (решение проблемы 23:00 → 00:00)")
lines.append("")
lines.append(
    "Числовой признак `hour=23` и `hour=0` далеки (разница 23), хотя физически соседние. "
    "То же для `day_of_week` (6→0) и `month` (12→1).\n\n"
    "**Решение:** sin/cos-кодирование проецирует признак на единичную окружность:\n"
    "```\n"
    "hour_sin = sin(2π * hour / 24)\n"
    "hour_cos = cos(2π * hour / 24)\n"
    "```\n"
    "Евклидово расстояние между (hour=23) и (hour=0) на окружности минимально."
)
lines.append("")
lines.append("| Признак | Period | Добавляемые колонки |")
lines.append("|---|---|---|")
for col, period in CYCLICAL_FEATURES.items():
    lines.append(f"| `{col}` | {period} | `{col}_sin`, `{col}_cos` |")
lines.append("")

# ── Коллинеарность ────────────────────────────────────────────
lines.append("## Коллинеарность и избыточность")
lines.append("")
lines.append(
    "| Переменная | Решение |\n"
    "|---|---|\n"
    "| `total_fuel_sales` | Исключена — линейная сумма 7 целевых, избыточна |\n"
    "| `total_traffic` | Оставлена — VSN автоматически снизит вес при высокой коллинеарности |\n"
    "| `base_price_*` vs `price_*` | Разные: `base_price` из паспорта (константа), `price` — текущая цена с акциями |\n"
    "| `is_snow/rain/fog` vs `weather_condition` | Частичное перекрытие, но `weather_condition` кодирует градацию (ясно/облачно/туман/дождь/снег) |"
)
lines.append("")

# ── Итог ─────────────────────────────────────────────────────
lines.append("## Итог: входы TFT-модели")
lines.append("")
lines.append("| Параметр | Значение |")
lines.append("|---|---|")
lines.append("| Encoder length | 168 ч (7 суток ретроспективы) |")
lines.append("| Decoder length | 24 ч (горизонт прогноза) |")
lines.append(f"| Целевых выходов | {len(TARGET_COLS)} |")
lines.append("")

role_summary = {
    "static_categoricals":             [],
    "static_reals":                    [],
    "time_varying_known_categoricals": [],
    "time_varying_known_reals":        [],
    "time_varying_unknown_reals":      [],
    "target":                          list(TARGET_COLS),
}
for col, meta in TFT_ROLES.items():
    inp = meta["tft_input"]
    if inp in role_summary:
        role_summary[inp].append(col)

lines.append("| TFT-вход | Колонок |")
lines.append("|---|---|")
for role, cols in role_summary.items():
    extra = " (20 base + 8 sin/cos)" if role == "time_varying_known_reals" else ""
    lines.append(f"| `{role}` | {len(cols)}{extra} |")

# ============================================================
# Сохранение
# ============================================================
report = "\n".join(lines)

out_path = "reports/column_analysis.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + "=" * 60)
print("Готово.")
print(f"  Сохранено : {out_path}")
print("  Следующий шаг: python eda/eda_preprocessing.py")
print("=" * 60)
