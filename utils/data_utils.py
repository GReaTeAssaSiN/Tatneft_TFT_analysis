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

LOG_COLS: List[str] = TARGET_COLS  # только целевые; shop_total_revenue исключена из модели

FILL_MAP: Dict[str, str] = {
    "holiday_name": "нет_праздника",
    "ad_channel": "нет_рекламы",
}

# Избыточные колонки — исключаются из модели на этапе предобработки
EXCLUDED_COLS: List[str] = [
    "station_name",        # служебная строка (есть station_id), не признак модели
    "total_pumps",         # = sum(num_pumps_*), дублирует детальные колонки
    "total_fuel_sales",    # = sum(sales_*), дублирует целевые переменные
    "shop_total_revenue",  # = sum(shop_*), дублирует; не является целью модели
    "total_traffic",       # sum(traffic_*) + скрытая категория → несогласован
    "quarter",             # выводится из month, избыточен
    "day_name",            # дублирует day_of_week (_enc тоже), строковый
]

# Колонки с нулевым std per-station — исключаются из Z-score нормализации.
# Текущие цены в 2023 г. постоянны внутри каждой станции (std=0 → NaN при делении).
# TFT принимает их в сыром масштабе; TorchNormalizer обрабатывает цели отдельно.
NO_ZSCORE_COLS: List[str] = [
    "price_AI92", "price_AI95", "price_AI98",
    "price_DT_EURO", "price_DT_TANEKO", "price_DT_SUMMER", "price_DT_WINTER",
]

# Статические вещественные переменные из metadata.
# Паспортные характеристики АЗС — не нормализуются (зашиты в веса модели через NaNLabelEncoder).
STATIC_REALS: List[str] = [
    "distance_to_city_km",
    # total_pumps удалён: = sum(num_pumps_*), дублирует детальные колонки
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
    # base_price_* — базовая паспортная цена АЗС (семантически отличается от price_*)
    "base_price_AI92",
    "base_price_AI95",
    "base_price_AI98",
    "base_price_DT_EURO",
    "base_price_DT_TANEKO",
    "base_price_DT_SUMMER",
    "base_price_DT_WINTER",
]

# Статические категориальные (из metadata, не меняются во времени)
STATIC_CATS: List[str] = [
    "road_type_enc",
    "direction_enc",
    "settlement_size_enc",
]

# Известные будущие категориальные (можно знать заранее или задать в what-if)
TIME_VARYING_KNOWN_CATS: List[str] = [
    "season_enc",
    # day_name_enc удалён: day_of_week уже есть в KNOWN_REALS с sin/cos
    "weather_condition_enc",  # тип погоды — задаётся по прогнозу/what-if
    "ad_channel_enc",
    "holiday_name_enc",
]

# Известные будущие вещественные.
# Декодер TFT принимает эти переменные для всего горизонта прогноза.
# Всё, что можно задать в what-if сценарии, должно быть здесь.
TIME_VARYING_KNOWN_REALS: List[str] = [
    # Циклические признаки — raw + sin/cos (решает разрыв 23:00 → 00:00)
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "day_of_week_sin", "day_of_week_cos",
    "week_of_year", "week_of_year_sin", "week_of_year_cos",
    "month", "month_sin", "month_cos",
    # quarter удалён: выводится из month, избыточен
    # Бинарные флаги
    "is_weekend", "is_holiday", "is_rush_hour", "is_night",
    # Режим работы магазина (05:00–21:00 = 1, 22:00–04:00 = 0)
    # Стабилизирует нулевые продажи shop_* ночью: модель явно знает «закрыто»
    "is_shop_open",
    # Акции и реклама
    "promotion_fuel_active", "promotion_shop_active", "promotion_cafe_active",
    "ad_active",
    # Текущие цены топлива (устанавливаются заранее, постоянны внутри станции в 2023 г.)
    "price_AI92", "price_AI95", "price_AI98",
    "price_DT_EURO", "price_DT_TANEKO", "price_DT_SUMMER", "price_DT_WINTER",
    # Погода (задаётся по метеопрогнозу; what-if: снег/ясно/дождь)
    "temperature", "precipitation_mm", "visibility_km", "wind_speed_ms",
    "is_snow", "is_rain", "is_fog",
    # Трафик по типам ТС (прогнозируется службами дорожного движения)
    # total_traffic исключён: сумма + скрытая категория → несогласован с индивидуальными
    "traffic_Passengers_cars", "traffic_Truck_short", "traffic_Truck",
    "traffic_Truck_long", "traffic_Transporter", "traffic_Undefined",
    # Цены конкурентов (мониторинг; what-if: снижение/рост конкурентной цены)
    "competitor_price_AI92", "competitor_price_AI95", "competitor_price_DT",
]

# Наблюдаемые прошлые — пусто: все ковариаты перенесены в KNOWN для what-if анализа.
# Целевые переменные (авторегрессивные входы энкодера) передаются через target=TARGET_COLS
# в TimeSeriesDataSet и добавляются TFT автоматически.
TIME_VARYING_UNKNOWN_REALS: List[str] = []

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
# Encoder 168h (7 суток) — охватывает суточный и недельный циклы сезонности.
# Decoder 24h — горизонт оперативного прогноза: следующие сутки.
ENCODER_LENGTH: int = 168    # ретроспектива: 7 суток (7 × 24 ч)
PREDICTION_LENGTH: int = 24  # горизонт прогноза: 1 сутки (24 ч)

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


