"""
Общие утилиты проекта FinalWorkDashboard.
Запускать из корня проекта.
"""

from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

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


# ============================================================
# Группы колонок (для отчётов и дашборда)
# ============================================================


def get_column_groups() -> Dict[str, List[str]]:
    """Возвращает словарь {группа: [колонки]} для merged_data.csv."""
    return {
        "Идентификаторы": [
            "station_id",
            "station_name",
            "timestamp",
            "time_idx",
        ],
        "Статика (metadata)": [
            "road_type",
            "direction",
            "settlement_size",
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
        ],
        "Погода": [
            "temperature",
            "weather_condition",
            "precipitation_mm",
            "visibility_km",
            "wind_speed_ms",
            "is_snow",
            "is_rain",
            "is_fog",
        ],
        "Трафик": [
            "traffic_Passengers_cars",
            "traffic_Truck_short",
            "traffic_Truck",
            "traffic_Truck_long",
            "traffic_Transporter",
            "traffic_Undefined",
            "total_traffic",
        ],
        "Продажи (целевые)": TARGET_COLS + ["total_fuel_sales"],
        "Магазин": [
            "shop_напитки",
            "shop_закуски",
            "shop_автотовары",
            "shop_кофе",
            "shop_табак",
            "shop_total_revenue",
        ],
        "Акции и реклама": [
            "promotion_fuel_active",
            "promotion_shop_active",
            "promotion_cafe_active",
            "ad_active",
            "ad_channel",
        ],
        "Цены": [
            "competitor_price_AI92",
            "competitor_price_AI95",
            "competitor_price_DT",
            "price_AI92",
            "price_AI95",
            "price_AI98",
            "price_DT_EURO",
            "price_DT_TANEKO",
            "price_DT_SUMMER",
            "price_DT_WINTER",
        ],
        "Временные признаки": [
            "hour",
            "day_of_week",
            "week_of_year",
            "month",
            "quarter",
            "season",
            "is_weekend",
            "is_holiday",
            "holiday_name",
            "is_rush_hour",
            "is_night",
            "day_name",
        ],
    }


# ============================================================
# Обратное преобразование прогнозов TFT
# ============================================================


def inverse_transform_predictions(
    preds: np.ndarray,
    station_id: str,
    scalers_path: str = "tft/scalers.pkl",
) -> np.ndarray:
    """Обратное преобразование прогнозов TFT: z-score^-1 -> expm1.

    Args:
        preds: массив (n_steps, n_targets) в нормализованном пространстве.
        station_id: идентификатор станции (строка).
        scalers_path: путь к tft/scalers.pkl.

    Returns:
        Массив той же формы в исходных единицах (литры/час).
    """
    with open(scalers_path, "rb") as fh:
        scaler_data = pickle.load(fh)

    scalers: Dict = scaler_data["scalers"]
    log1p_cols_set = set(scaler_data["log1p_cols"])

    result = preds.copy().astype(float)
    station_scalers = scalers.get(str(station_id), {})

    for i, col in enumerate(TARGET_COLS):
        if col in station_scalers:
            mean, std = station_scalers[col]
            if std > 0:
                result[:, i] = result[:, i] * std + mean
        if col in log1p_cols_set:
            result[:, i] = np.expm1(result[:, i])

    return result


# ============================================================
# Загрузка scalers.pkl
# ============================================================


def load_scalers(
    scalers_path: str = "tft/scalers.pkl",
) -> Tuple[Dict, List[str]]:
    """Возвращает (scalers_dict, log1p_cols) из scalers.pkl."""
    with open(scalers_path, "rb") as fh:
        data = pickle.load(fh)
    return data["scalers"], data["log1p_cols"]
