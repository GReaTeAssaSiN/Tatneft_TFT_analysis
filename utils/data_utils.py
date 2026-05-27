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

DATA_DIR     = "SourceDataForWork/"
SRC_STATIC   = DATA_DIR + "gas_stations_static.csv"
SRC_TEMPORAL = DATA_DIR + "gas_stations_temporal_daily_2025.csv"

TARGET_COLS: List[str] = [
    # 5 видов топлива
    "sales_AI92",
    "sales_AI95",
    "sales_DT",
    "sales_DT_bio",
    "sales_AI100_bio",
    # 7 категорий магазина
    "shop_напитки_безалкогольные",
    "shop_кондитерка_снеки",
    "shop_мороженое",
    "shop_автотовары",
    "shop_кафе_вся_еда",
    "shop_кофе_все_горячие_напитки",
    "shop_табак",
]

LOG_COLS: List[str] = TARGET_COLS  # только целевые; total_fuel_sales исключена из модели

FILL_MAP: Dict[str, str] = {
    "holiday_name": "нет_праздника",
}

# Избыточные колонки — исключаются из модели на этапе предобработки
EXCLUDED_COLS: List[str] = [
    "station_name",        # служебная строка (есть station_id), не признак модели
    "total_pumps",         # = sum(num_pumps_*), дублирует детальные колонки
    "total_fuel_sales",    # = sum(sales_*), дублирует целевые переменные
    "quarter",             # выводится из month, избыточен
    # Уровень обслуживания дороги (A–F) — строковые категории, информация
    # уже представлена косвенно через счётчики трафика в KNOWN_REALS
    "traffic_uroven_jbslugi_1_poputn",
    "traffic_uroven_jbslugi_2_poputn",
    "traffic_uroven_jbslugi_1_wstrechn",
    "traffic_uroven_jbslugi_2_wstrechn",
]

# Колонки с нулевым std per-station — исключаются из Z-score нормализации.
# В данных 2025 г. цены меняются ежедневно (std > 0), поэтому список пуст.
# TorchNormalizer обрабатывает целевые переменные отдельно.
NO_ZSCORE_COLS: List[str] = []

# Статические вещественные переменные из metadata.
# Паспортные характеристики АЗС — не нормализуются.
STATIC_REALS: List[str] = [
    # Магазин
    "shop_area_m2",
    # Колонки по видам топлива (расширенная номенклатура)
    "num_pumps_AI92",
    "num_pumps_AI92_bio",
    "num_pumps_AI95",
    "num_pumps_AI95_bio",
    "num_pumps_AI100_bio",
    "num_pumps_DT",
    "num_pumps_DT_bio",
    "num_pumps_SUG",   # сжиженный углеводородный газ
    "num_pumps_KPG",   # компримированный природный газ
    "num_pumps_SPG",   # сжиженный природный газ
    # Услуги АЗС
    "has_car_wash",
    "has_tire_service",
    "has_cafe",
    "has_hotel",
    "has_shop",
    # Дополнительные услуги магазина
    "has_shop_молельная_комната",
    "has_shop_прачечная",
    "has_shop_электрозарядная_станция",
    "has_shop_подкачка_шин",
    # Конкурентная среда
    "competitors_wink",
]

# Статические категориальные (из metadata, не меняются во времени)
STATIC_CATS: List[str] = [
    "road_type_enc",
    "road_level_enc",           # уровень дороги — код (1, 5, …), не число
    "direction_enc",
    "settlement_size_enc",      # размер нас. пункта — код (1–6), не число
    "distance_to_city_km_enc",  # удалённость от города — код (0–6), не число
]

# Известные будущие категориальные (задаются заранее или в what-if)
TIME_VARYING_KNOWN_CATS: List[str] = [
    "season_enc",
    "holiday_name_enc",
    # weather_condition — перемещён в KNOWN_REALS: в данных 2025 г.
    # это бинарный флаг (0/1), не требует категориального эмбеддинга
]

# Известные будущие вещественные.
# Декодер TFT принимает эти переменные для всего горизонта прогноза.
# Всё, что задаётся в what-if сценарии, должно быть здесь.
TIME_VARYING_KNOWN_REALS: List[str] = [
    # Циклические признаки — raw + sin/cos (суточные данные, часы убраны)
    "day_of_week", "day_of_week_sin", "day_of_week_cos",
    "week_of_year", "week_of_year_sin", "week_of_year_cos",
    "month", "month_sin", "month_cos",
    # Бинарные флаги
    "is_weekend", "is_holiday",
    # Тип погоды (бинарный: 0 = ясно/облачно, 1 = осадки/плохая видимость)
    # В данных 2025 г. weather_condition = 0/1, а не строковая категория.
    "weather_condition",
    # Акции
    "promotion_fuel_active", "promotion_shop_active", "promotion_cafe_active",
    # Текущие цены топлива (меняются ежедневно; what-if: динамика ценообразования)
    "price_AI92", "price_AI95", "price_DT", "price_DT_bio", "price_AI100_bio",
    # Погода (по метеопрогнозу; what-if: температура, осадки)
    "temperature", "precipitation_mm",
    # Трафик по типам ТС — попутное направление, полосы 1 и 2
    # (прогнозируется службами дорожного движения; what-if: высокий/низкий трафик)
    "traffic_Passengers_cars_1_poputn", "traffic_Passengers_cars_2_poputn",
    "traffic_Truck_short_1_poputn",     "traffic_Truck_short_2_poputn",
    "traffic_Truck_1_poputn",           "traffic_Truck_2_poputn",
    "traffic_Truck_long_1_poputn",      "traffic_Truck_long_2_poputn",
    "traffic_Transporter_1_poputn",     "traffic_Transporter_2_poputn",
    "traffic_Undefined_1_poputn",       "traffic_Undefined_2_poputn",
    # Трафик — встречное направление, полосы 1 и 2
    "traffic_Passengers_cars_1_wstrechn", "traffic_Passengers_cars_2_wstrechn",
    "traffic_Truck_short_1_wstrechn",     "traffic_Truck_short_2_wstrechn",
    "traffic_Truck_1_wstrechn",           "traffic_Truck_2_wstrechn",
    "traffic_Truck_long_1_wstrechn",      "traffic_Truck_long_2_wstrechn",
    "traffic_Transporter_1_wstrechn",     "traffic_Transporter_2_wstrechn",
    "traffic_Undefined_1_wstrechn",       "traffic_Undefined_2_wstrechn",
    # Цены конкурентов (мониторинг; what-if: снижение/рост конкурентных цен)
    "competitor_price_AI92",       "competitor_price_AI95",       "competitor_price_DT",
    "competitor_price_AI92_brend", "competitor_price_AI95_brend", "competitor_price_DT_brend",
    "competitor_price_AI100",
]

# Наблюдаемые прошлые (только энкодер TFT).
# Значения известны за прошлые периоды, но недоступны для горизонта прогноза.
# Целевые переменные (авторегрессивные входы энкодера) передаются через
# target=TARGET_COLS в TimeSeriesDataSet и добавляются TFT автоматически.
TIME_VARYING_UNKNOWN_REALS: List[str] = [
    # Клиентские метрики — наблюдаемые, не прогнозируемые заранее
    "corporate_customer_ratio",
    "customer_loyalty_score",
    # Производные показатели трафика — результаты моделирования дорожного потока,
    # не задаются напрямую в what-if (в отличие от счётчиков ТС в KNOWN_REALS)
    "traffic_scorost_1_poputn",       "traffic_scorost_2_poputn",
    "traffic_scorost_1_wstrechn",     "traffic_scorost_2_wstrechn",
    "traffic_plotnost_1_poputn",      "traffic_plotnost_2_poputn",
    "traffic_plotnost_1_wstrechn",    "traffic_plotnost_2_wstrechn",
    "traffic_intensiv_priv_1_poputn", "traffic_intensiv_priv_2_poputn",
    "traffic_intensiv_priv_1_wstrechn","traffic_intensiv_priv_2_wstrechn",
    "traffic_intensiv_fiz_1_poputn",  "traffic_intensiv_fiz_2_poputn",
    "traffic_intensiv_fiz_1_wstrechn","traffic_intensiv_fiz_2_wstrechn",
]

# Циклические признаки: имя колонки -> период
# Данные суточные — поле hour отсутствует
CYCLICAL_FEATURES: Dict[str, int] = {
    "day_of_week":  7,
    "month":        12,
    # 2025 г.: Dec 29–31 → ISO-неделя 1 следующего года; max week_of_year = 52.
    # Период 52 даёт плавный переход: week 52 → angle ≈ 2π → week 1 (след. год)
    "week_of_year": 52,
}

# Границы сплитов (train / val / test = Jan–Oct / Nov / Dec 2025)
TRAIN_END  = pd.Timestamp("2025-10-31")
VAL_END    = pd.Timestamp("2025-11-30")
TEST_START = pd.Timestamp("2025-12-01")
TEST_END   = pd.Timestamp("2025-12-31")

# Параметры окна TFT (суточные данные)
# Encoder 30 дней — охватывает месячный и недельный циклы.
# Decoder 7 дней — горизонт оперативного прогноза: следующая неделя.
ENCODER_LENGTH:    int = 30   # ретроспектива: 1 месяц
PREDICTION_LENGTH: int = 7    # горизонт прогноза: 1 неделя

# Квантили QuantileLoss (pytorch-forecasting default)
QUANTILE_LEVELS = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
Q_MED = len(QUANTILE_LEVELS) // 2  # индекс медианы q50 = 3
Q_LO  = 1                           # индекс q10
Q_HI  = -2                          # индекс q90


# ============================================================
# Загрузка и объединение данных
# ============================================================


def load_and_merge(
    static_file:   str = SRC_STATIC,
    temporal_file: str = SRC_TEMPORAL,
) -> pd.DataFrame:
    """Загружает паспортные данные АЗС и временной ряд, возвращает объединённый DataFrame.

    Оба файла читаются с кодировкой utf-8-sig (UTF-8 с BOM).
    temporal LEFT JOIN static ON station_id: к временному ряду добавляются
    все колонки из static, кроме station_name (дублирует поле temporal).
    """
    static   = pd.read_csv(static_file,   encoding="utf-8-sig")
    temporal = pd.read_csv(temporal_file, encoding="utf-8-sig", parse_dates=["date"])
    static_for_merge = static.drop(columns=["station_name"], errors="ignore")
    extra_cols = [c for c in static_for_merge.columns if c != "station_id"]
    df = temporal.merge(static_for_merge, on="station_id", how="left")
    return df


# ============================================================
# Заполнение пропусков
# ============================================================


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в колонках из FILL_MAP семантическими значениями.

    На текущий момент: holiday_name -> 'нет_праздника'.
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

    Это позволяет TFT видеть, что day=6 (Вс) и day=0 (Пн) — соседние точки,
    а не отдалённые (проблема разрыва на числовой оси).
    """
    for col, period in CYCLICAL_FEATURES.items():
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df
