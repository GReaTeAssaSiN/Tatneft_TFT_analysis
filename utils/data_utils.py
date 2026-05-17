"""
Общие утилиты проекта FinalWorkDashboard.
Запускать из корня проекта.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# ============================================================
# Константы
# ============================================================

DATA_DIR = "SourceDataForWork/"

TARGET_COLS: List[str] = [
    # 7 видов топлива
    "sales_AI92",
    "sales_AI95",
    "sales_AI98",
    "sales_DT_EURO",
    "sales_DT_TANEKO",
    "sales_DT_SUMMER",
    "sales_DT_WINTER",
    # 5 категорий магазина
    "shop_напитки",
    "shop_закуски",
    "shop_автотовары",
    "shop_кофе",
    "shop_табак",
]

LOG_COLS: List[str] = TARGET_COLS + ["shop_total_revenue"]

FILL_MAP: Dict[str, str] = {
    "holiday_name": "нет_праздника",
    "ad_channel": "нет_рекламы",
}

# Статические вещественные переменные из metadata.
# Паспортные характеристики АЗС — не winsoriz-уются и не нормализуются.
STATIC_REALS: List[str] = [
    "distance_to_city_km",
    "total_pumps",
    "shop_area_m2",
    "num_pumps_AI92",
    "num_pumps_AI95",
    "num_pumps_AI98",
    "num_pumps_DT_EURO",
    "num_pumps_DT_TANEKO",
    "num_pumps_DT_SUMMER",
    "num_pumps_DT_WINTER",
    "has_car_wash",
    "has_tire_service",
    "has_cafe",
    "has_hotel",
    "has_shop",
    "competitors_within_5km",
    "customer_loyalty_score",
    "staff_quality_score",
    "corporate_customer_ratio",
    "staff_engagement_score",
    "base_price_AI92",
    "base_price_AI95",
    "base_price_AI98",
    "base_price_DT_EURO",
    "base_price_DT_TANEKO",
    "base_price_DT_SUMMER",
    "base_price_DT_WINTER",
]

# Циклические признаки: имя колонки -> период
CYCLICAL_FEATURES: Dict[str, int] = {
    "hour": 24,
    "day_of_week": 7,
    "month": 12,
    "week_of_year": 52,
}

# Границы сплитов (train / val / test = Jan–Oct / Nov / Dec 2023)
TRAIN_END = pd.Timestamp("2023-10-31 23:00:00")
VAL_END = pd.Timestamp("2023-11-30 23:00:00")
TEST_START = pd.Timestamp("2023-12-01 00:00:00")
TEST_END = pd.Timestamp("2023-12-31 23:00:00")

# Параметры окна TFT
ENCODER_LENGTH: int = 168   # ретроспектива: 7 суток (7 × 24 ч)
PREDICTION_LENGTH: int = 24  # горизонт прогноза: 24 часа

# Квантили QuantileLoss (pytorch-forecasting default)
QUANTILE_LEVELS = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
Q_MED = len(QUANTILE_LEVELS) // 2  # индекс медианы q50 = 3
Q_LO = 1                            # индекс q10
Q_HI = -2                           # индекс q90


# ============================================================
# Загрузка и объединение данных
# ============================================================


def load_and_merge(
    meta_file: str = DATA_DIR + "5stations_metadata.csv",
    data_file: str = DATA_DIR + "5stations_data.csv",
) -> pd.DataFrame:
    """Загружает metadata и временной ряд, возвращает объединённый DataFrame.

    LEFT JOIN по station_id: добавляет к временному ряду все колонки
    из metadata, которых нет в data.
    """
    meta = pd.read_csv(meta_file)
    data = pd.read_csv(data_file, parse_dates=["timestamp"])
    extra_cols = [c for c in meta.columns if c not in data.columns]
    df = data.merge(meta[["station_id"] + extra_cols], on="station_id", how="left")
    return df


# ============================================================
# Заполнение пропусков
# ============================================================


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в holiday_name и ad_channel семантическими значениями.

    Изменяет df inplace и возвращает его для цепочки вызовов.
    """
    for col, val in FILL_MAP.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ============================================================
# Циклическое кодирование временных признаков
# ============================================================


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет sin/cos кодирование для циклических временных признаков.

    Для каждой колонки из CYCLICAL_FEATURES создаёт две новые:
    {col}_sin = sin(2π * col / period)
    {col}_cos = cos(2π * col / period)

    Это позволяет TFT видеть, что hour=23 и hour=0 — соседние точки,
    а не отдалённые (проблема разрыва при делении суток на 0/24).
    """
    for col, period in CYCLICAL_FEATURES.items():
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


