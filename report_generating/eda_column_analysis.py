"""
Классификация переменных merged_data.csv для TFT-модели.
Определяет роль каждой из 89 колонок согласно:
  — статье Lim et al. (2020) «Temporal Fusion Transformers
    for Interpretable Multi-horizon Time Series Forecasting»
  — требованию к сценарному (what-if) прогнозу

Запускать из корня проекта:
    python report_generating/eda_column_analysis.py
Результат: reports/column_analysis.md
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("reports", exist_ok=True)

# ── Загрузка данных для статистик ─────────────────────────────
SRC = "data/merged_data.csv"
if not os.path.exists(SRC):
    print(f"[ОШИБКА] Файл не найден: {SRC}")
    print("Сначала запустите: python explore_data.py")
    sys.exit(1)

df = pd.read_csv(SRC, parse_dates=["timestamp"])

# ══════════════════════════════════════════════════════════════
# КЛАССИФИКАЦИЯ ПЕРЕМЕННЫХ
# ══════════════════════════════════════════════════════════════
#
# Согласно TFT (Lim et al. 2020, Section 3) модель принимает:
#   (a) static covariates      — постоянные характеристики объекта
#   (b) time-varying known     — известны заранее (encoder + decoder)
#   (c) time-varying unknown   — известны только в прошлом (encoder only)
#   (d) targets                — прогнозируемые выходы (= observed past)
#
# Ключевое решение проекта:
#   Погода, трафик и цены конкурентов помещены в (b) KNOWN,
#   а не в (c) UNKNOWN. Это позволяет пользователю задавать их
#   при сценарном прогнозе (what-if), вводя прогноз погоды,
#   ожидаемый трафик и данные мониторинга конкурентов.
# ══════════════════════════════════════════════════════════════

# ── Технические идентификаторы (не входят в модель как признаки)
TECHNICAL = {
    "timestamp": {
        "reason": (
            "Индекс времени. Используется для построения time_idx = cumcount "
            "по станции. В модель как признак не подаётся."
        ),
    },
    "station_id": {
        "reason": (
            "Ключ группировки (group_id) в TimeSeriesDataSet. "
            "Идентифицирует станцию для разделения временных рядов. "
            "Не является признаком — TFT получает информацию о станции "
            "через статические переменные."
        ),
    },
}

# ── Исключены из модели (избыточные колонки) ──────────────────
EXCLUDED = {
    "station_name": {
        "reason": (
            "Строковое название АЗС. Дублирует station_id. "
            "Все характеристики станции уже представлены в STATIC_CATS и STATIC_REALS. "
            "Текстовая метка не несёт дополнительной предсказательной силы."
        ),
    },
    "total_pumps": {
        "reason": (
            "Точная сумма num_pumps_AI92 + num_pumps_AI95 + ... + num_pumps_DT_WINTER. "
            "Индивидуальные счётчики по каждому виду топлива несут больше информации "
            "(пропускная способность конкретного продукта), агрегат избыточен."
        ),
    },
    "total_fuel_sales": {
        "reason": (
            "Точная сумма 7 топливных целевых переменных. "
            "Если модель прогнозирует каждый вид топлива отдельно, "
            "агрегат выводится постфактум и не добавляет информации на входе."
        ),
    },
    "shop_total_revenue": {
        "reason": (
            "Точная сумма 5 категорий магазина (shop_напитки + ... + shop_табак). "
            "Аналогично total_fuel_sales — избыточный агрегат целевых переменных."
        ),
    },
    "total_traffic": {
        "reason": (
            "Агрегат транспортного потока. Все 6 типов ТС перенесены в KNOWN_REALS "
            "для сценарного анализа. Суммарное число транспортных средств выводится "
            "из компонент; при this подходе агрегат теряет самостоятельную ценность. "
            "Индивидуальные потоки по типам ТС информативнее для прогноза продаж "
            "конкретного топлива (грузовики → ДТ, легковые → АИ)."
        ),
    },
    "quarter": {
        "reason": (
            "Грубое укрупнение месяца (12 → 4 значения). "
            "Модель получает month в KNOWN_REALS и самостоятельно обнаружит "
            "квартальные паттерны. Квартал только снижает разрешение сигнала."
        ),
    },
    "day_name": {
        "reason": (
            "Строковая метка дня недели (Monday, Tuesday, ...). "
            "Дублирует day_of_week (0–6), который уже входит в KNOWN_REALS. "
            "В модель идёт только числовое/закодированное представление."
        ),
    },
}

# ── Статические категориальные ────────────────────────────────
STATIC_CATS = {
    "road_type": {
        "desc": "Тип дороги (федеральная трасса / региональная).",
        "preprocessing": (
            "Label Encoding → road_type_enc. "
            "TFT строит обучаемый embedding-слой (Entity Embedding, Section 4.1). "
            "Размер embedding подбирается автоматически."
        ),
        "what_if": "Нет (статическая характеристика станции, не меняется).",
    },
    "direction": {
        "desc": "Направление движения (транзит / въезд / выезд).",
        "preprocessing": "Label Encoding → direction_enc. Embedding TFT.",
        "what_if": "Нет.",
    },
    "settlement_size": {
        "desc": "Размер ближайшего населённого пункта (сельская / малый город / крупный).",
        "preprocessing": "Label Encoding → settlement_size_enc. Embedding TFT.",
        "what_if": "Нет.",
    },
}

# ── Статические вещественные ──────────────────────────────────
STATIC_REALS = {
    "distance_to_city_km": {
        "desc": "Расстояние до ближайшего крупного города, км.",
        "preprocessing": (
            "Без нормализации (паспортные данные исключены из Z-score). "
            "TFT использует через Static Variable Selection Network (VSN)."
        ),
        "what_if": (
            "Нет — физическое расстояние, константа станции. "
            "Для анализа влияния локации — сравнивать станции с разным значением."
        ),
    },
    "shop_area_m2": {
        "desc": "Площадь магазина, м².",
        "preprocessing": "Без нормализации.",
        "what_if": "Нет (инфраструктура).",
    },
    "num_pumps_AI92":     {"desc": "Число колонок АИ-92.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_AI95":     {"desc": "Число колонок АИ-95.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_AI98":     {"desc": "Число колонок АИ-98.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_DT_EURO":  {"desc": "Число колонок ДТ Евро+.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_DT_TANEKO":{"desc": "Число колонок ДТ ТАНЕКО.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_DT_SUMMER":{"desc": "Число колонок ДТ Летнее.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_DT_WINTER":{"desc": "Число колонок ДТ Зимнее.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "has_car_wash": {
        "desc": "Есть автомойка (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Нет (инфраструктура).",
    },
    "has_cafe":         {"desc": "Есть кафе (0/1).",         "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_shop":         {"desc": "Есть магазин (0/1).",       "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_tire_service": {"desc": "Есть шиномонтаж (0/1).",    "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_hotel":        {"desc": "Есть гостиница (0/1).",     "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "competitors_within_5km": {
        "desc": "Число конкурентов в радиусе 5 км.",
        "preprocessing": "Без нормализации.",
        "what_if": "Нет (константа среды).",
    },
    "corporate_customer_ratio": {
        "desc": "Доля корпоративных клиентов (0–1).",
        "preprocessing": "Без нормализации.",
        "what_if": "Нет.",
    },
    "staff_engagement_score": {
        "desc": "Оценка вовлечённости персонала (мотивация, инициативность).",
        "preprocessing": "Без нормализации.",
        "what_if": "Нет (агрегированная кадровая метрика, не почасовая).",
    },
    "staff_quality_score": {
        "desc": "Оценка качества работы персонала (скорость обслуживания, ошибки).",
        "preprocessing": "Без нормализации.",
        "what_if": "Нет.",
    },
    "customer_loyalty_score": {
        "desc": "Оценка лояльности клиентов (повторные визиты, NPS).",
        "preprocessing": "Без нормализации.",
        "what_if": "Нет.",
    },
    # Базовые цены топлива (паспорт станции) ─────────────────
    "base_price_AI92": {
        "desc": (
            "Базовая цена АИ-92 из паспорта станции, руб/л. "
            "Устанавливается при открытии / ценовой политике и фиксируется как ориентир. "
            "Семантически отличается от price_AI92 (текущая операционная цена): "
            "base_price — константа станции, price — актуальная цена на данный час."
        ),
        "preprocessing": "Без нормализации (статическая характеристика).",
        "what_if": "Нет.",
    },
    "base_price_AI95":     {"desc": "Базовая цена АИ-95, руб/л. Аналогично base_price_AI92.",     "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "base_price_AI98":     {"desc": "Базовая цена АИ-98, руб/л.",                                  "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "base_price_DT_EURO":  {"desc": "Базовая цена ДТ Евро+, руб/л.",                              "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "base_price_DT_TANEKO":{"desc": "Базовая цена ДТ ТАНЕКО, руб/л.",                             "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "base_price_DT_SUMMER":{"desc": "Базовая цена ДТ Летнее, руб/л.",                             "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "base_price_DT_WINTER":{"desc": "Базовая цена ДТ Зимнее, руб/л.",                             "preprocessing": "Без нормализации.", "what_if": "Нет."},
}

# ── Известные будущие категориальные ──────────────────────────
KNOWN_CATS = {
    "season": {
        "desc": "Сезон (winter / spring / summer / autumn).",
        "preprocessing": "Label Encoding → season_enc. NaN не ожидается.",
        "what_if": (
            "Да — задаётся автоматически из даты прогноза. "
            "Позволяет явно учитывать сезонные паттерны поведения."
        ),
    },
    "weather_condition": {
        "desc": "Тип погоды (ясно / облачно / дождь / снег / туман / метель).",
        "preprocessing": (
            "Label Encoding → weather_condition_enc. "
            "Перенесена из UNKNOWN в KNOWN: пользователь может задать "
            "тип погоды из прогноза метеосервиса для сценарного анализа."
        ),
        "what_if": (
            "Да — ключевой параметр сценария. "
            "Например: «что если завтра метель?»"
        ),
    },
    "ad_channel": {
        "desc": "Канал рекламы (ТВ / радио / билборд / email / нет_рекламы).",
        "preprocessing": "NaN → 'нет_рекламы'. Label Encoding → ad_channel_enc.",
        "what_if": "Да — планируется заранее маркетинговой службой.",
    },
    "holiday_name": {
        "desc": "Название праздника или 'нет_праздника'.",
        "preprocessing": "NaN → 'нет_праздника'. Label Encoding → holiday_name_enc.",
        "what_if": "Да — известно из производственного календаря.",
    },
}

# ── Известные будущие вещественные ───────────────────────────
KNOWN_REALS = {
    # Временные признаки ─────────────────────────────────────
    "hour": {
        "desc": "Час суток (0–23).",
        "preprocessing": (
            "Циклическое кодирование: hour_sin = sin(2π·h/24), "
            "hour_cos = cos(2π·h/24). Устраняет разрыв 23:00→00:00. "
            "Все три значения (raw + sin + cos) подаются в модель."
        ),
        "what_if": "Да — задаётся датой/часом прогноза.",
    },
    "day_of_week": {
        "desc": "День недели (0=Пн, …, 6=Вс).",
        "preprocessing": "Циклическое кодирование (period=7): day_of_week_sin, day_of_week_cos.",
        "what_if": "Да — задаётся датой прогноза.",
    },
    "week_of_year": {
        "desc": "Номер недели в году (1–52).",
        "preprocessing": "Циклическое кодирование (period=52): week_of_year_sin, week_of_year_cos.",
        "what_if": "Да — задаётся датой прогноза.",
    },
    "month": {
        "desc": "Месяц (1–12).",
        "preprocessing": "Циклическое кодирование (period=12): month_sin, month_cos.",
        "what_if": "Да — задаётся датой прогноза.",
    },
    "is_weekend": {
        "desc": "Выходной день (0/1). Производится из day_of_week, но сохраняется как явный сигнал.",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да — определяется датой прогноза.",
    },
    "is_holiday": {
        "desc": "Праздничный день (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да — из производственного календаря.",
    },
    "is_rush_hour": {
        "desc": "Час пик (0/1). Истинно при hour ∈ {7,8,9,17,18,19}.",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да — определяется часом прогноза.",
    },
    "is_night": {
        "desc": "Ночное время (0/1). Истинно при hour ∈ {0,1,2,3,4,5,22,23}.",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да — определяется часом прогноза.",
    },
    "is_shop_open": {
        "desc": (
            "Магазин открыт (0/1). Истинно при hour ∈ [5..21] (05:00–21:00), "
            "ложно при hour ∈ [22..4] (22:00–04:00). "
            "Производный бинарный признак — вычисляется в eda_preprocessing.py "
            "из колонки hour. В merged_data.csv отсутствует."
        ),
        "preprocessing": (
            "Бинарный (0/1). Без нормализации. "
            "Добавляется в eda_preprocessing.py до вычисления binary_cols, "
            "поэтому автоматически попадает в список бинарных и исключается из Z-score."
        ),
        "what_if": (
            "Да — полностью определяется часом прогноза. "
            "Стабилизирует нулевые продажи shop_* ночью: "
            "модель явно знает, что магазин закрыт."
        ),
    },
    # Акции и реклама ────────────────────────────────────────
    "promotion_fuel_active": {
        "desc": "Акция на топливо активна (0/1).",
        "preprocessing": "Бинарный. Без нормализации. Акции планируются заранее.",
        "what_if": "Да — ключевой параметр сценария.",
    },
    "promotion_shop_active": {
        "desc": "Акция в магазине активна (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да.",
    },
    "promotion_cafe_active": {
        "desc": "Акция в кафе активна (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да.",
    },
    "ad_active": {
        "desc": "Реклама активна (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да.",
    },
    # Текущие цены топлива ───────────────────────────────────
    "price_AI92": {
        "desc": "Текущая цена АИ-92, руб/л. Устанавливается коммерческой службой.",
        "preprocessing": (
            "В данных 2023 г. цены не менялись → std=0 per-station. "
            "Исключить из Z-score нормализации, подавать в исходном масштабе. "
            "Архитектурно остаётся в KNOWN: при реальном использовании цены меняются."
        ),
        "what_if": "Да — цена является управляемым параметром.",
    },
    "price_AI95":     {"desc": "Текущая цена АИ-95, руб/л.", "preprocessing": "Аналогично price_AI92.", "what_if": "Да."},
    "price_AI98":     {"desc": "Текущая цена АИ-98, руб/л.", "preprocessing": "Аналогично price_AI92.", "what_if": "Да."},
    "price_DT_EURO":  {"desc": "Текущая цена ДТ Евро+, руб/л.", "preprocessing": "Аналогично price_AI92.", "what_if": "Да."},
    "price_DT_TANEKO":{"desc": "Текущая цена ДТ ТАНЕКО, руб/л.", "preprocessing": "Аналогично price_AI92.", "what_if": "Да."},
    "price_DT_SUMMER":{"desc": "Текущая цена ДТ Летнее, руб/л.", "preprocessing": "Аналогично price_AI92.", "what_if": "Да."},
    "price_DT_WINTER":{"desc": "Текущая цена ДТ Зимнее, руб/л.", "preprocessing": "Аналогично price_AI92.", "what_if": "Да."},
    # Погода (перенесена из UNKNOWN) ─────────────────────────
    "temperature": {
        "desc": "Температура воздуха, °C.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": (
            "Да — пользователь вводит прогноз температуры из метеосервиса. "
            "Влияет на потребление топлива (зимний/летний режим двигателей), "
            "спрос на горячие напитки в кафе, трафик."
        ),
    },
    "precipitation_mm": {
        "desc": "Количество осадков, мм.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да — из прогноза погоды.",
    },
    "visibility_km": {
        "desc": "Видимость на дороге, км.",
        "preprocessing": "Z-score per-station.",
        "what_if": (
            "Да — при плохой видимости (туман, метель) водители чаще останавливаются "
            "на АЗС для отдыха, влияет на трафик и продажи кофе/кафе."
        ),
    },
    "wind_speed_ms": {
        "desc": "Скорость ветра, м/с.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да — из прогноза погоды.",
    },
    "is_snow": {
        "desc": "Идёт снег (0/1). Явный бинарный сигнал из weather_condition.",
        "preprocessing": (
            "Бинарный. Без нормализации. Дополняет weather_condition_enc: "
            "модель может реагировать на снег через простой порог, "
            "не раскодируя категорию."
        ),
        "what_if": "Да.",
    },
    "is_rain": {
        "desc": "Идёт дождь (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да.",
    },
    "is_fog": {
        "desc": "Туман (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да.",
    },
    # Трафик по типам ТС (перенесены из UNKNOWN) ─────────────
    "traffic_Passengers_cars": {
        "desc": "Поток легковых автомобилей через АЗС.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": (
            "Да — пользователь задаёт ожидаемый трафик по типам ТС. "
            "Легковые → преимущественно АИ-92/95/98. "
            "Для прогноза можно использовать исторический трафик за аналогичный день."
        ),
    },
    "traffic_Truck_short": {
        "desc": "Поток малотоннажных грузовиков (до 3,5 т).",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да — малотоннажные → смешанный спрос на АИ и ДТ.",
    },
    "traffic_Truck": {
        "desc": "Поток грузовиков среднего класса (3,5–12 т).",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да — средние грузовики → преимущественно ДТ.",
    },
    "traffic_Truck_long": {
        "desc": "Поток большегрузных автомобилей (>12 т, фуры).",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да — большегрузы → ДТ Евро+/ТАНЕКО, крупные заправки.",
    },
    "traffic_Transporter": {
        "desc": "Поток микроавтобусов и транспортёров (Газель и т.п.).",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да.",
    },
    "traffic_Undefined": {
        "desc": "Транспорт неопределённого типа.",
        "preprocessing": "Z-score per-station.",
        "what_if": (
            "Да — представляет фоновый трафик, не попавший в классификатор. "
            "При сценарном анализе можно задавать как остаточный поток."
        ),
    },
    # Цены конкурентов (перенесены из UNKNOWN) ───────────────
    "competitor_price_AI92": {
        "desc": "Средняя цена АИ-92 у конкурентов в радиусе 5 км, руб/л.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": (
            "Да — данные мониторинга конкурентов. "
            "Пользователь может задать «что если конкурент снизил цену на 2 руб»."
        ),
    },
    "competitor_price_AI95": {
        "desc": "Средняя цена АИ-95 у конкурентов, руб/л.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да.",
    },
    "competitor_price_DT": {
        "desc": "Средняя цена ДТ у конкурентов, руб/л.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2023).",
        "what_if": "Да.",
    },
}

# ── Целевые переменные (= observed past для энкодера) ─────────
TARGETS = {
    "sales_AI92": {
        "desc": "Продажи АИ-92, л/ч.",
        "preprocessing": (
            "log1p преобразование (устраняет правосторонний скос и нули). "
            "TorchNormalizer(method='robust') внутри TFT — нормализует цели отдельно. "
            "Z-score в eda_preprocessing.py НЕ применяется. "
            "Оригинал сохраняется в sales_AI92_orig для обратного преобразования."
        ),
    },
    "sales_AI95":     {"desc": "Продажи АИ-95, л/ч.",      "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_AI98":     {"desc": "Продажи АИ-98, л/ч.",      "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_DT_EURO":  {"desc": "Продажи ДТ Евро+, л/ч.",   "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_DT_TANEKO":{"desc": "Продажи ДТ ТАНЕКО, л/ч.",  "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_DT_SUMMER":{"desc": "Продажи ДТ Летнее, л/ч.",  "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_DT_WINTER":{"desc": "Продажи ДТ Зимнее, л/ч.",  "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_напитки":    {"desc": "Выручка магазина: напитки, руб/ч.",    "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_закуски":    {"desc": "Выручка магазина: закуски, руб/ч.",    "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_автотовары": {"desc": "Выручка магазина: автотовары, руб/ч.", "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_кофе":       {"desc": "Выручка магазина: кофе, руб/ч.",       "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_табак":      {"desc": "Выручка магазина: табак, руб/ч.",      "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
}

# ══════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ОТЧЁТА
# ══════════════════════════════════════════════════════════════

def stat_row(col: str) -> dict:
    """Возвращает словарь базовых статистик по колонке df."""
    if col not in df.columns:
        return {"dtype": "—", "n_uniq": "—", "n_null": "—", "range": "отсутствует"}
    s = df[col]
    n_null = int(s.isnull().sum())
    n_uniq = int(s.nunique())
    if pd.api.types.is_datetime64_any_dtype(s):
        rng = f"{s.min().date()} — {s.max().date()}"
    elif pd.api.types.is_numeric_dtype(s):
        rng = f"[{s.min():.4g}, {s.max():.4g}]"
    else:
        vals = [str(v)[:22] for v in s.dropna().unique()[:3]]
        rng = ", ".join(vals) + (", ..." if n_uniq > 3 else "")
    return {"dtype": str(s.dtype), "n_uniq": n_uniq, "n_null": n_null, "range": rng}


lines = []

# ── Заголовок ─────────────────────────────────────────────────
lines += [
    "# Классификация переменных для TFT-модели",
    "",
    "> **Источник:** Lim et al. (2020) «Temporal Fusion Transformers for Interpretable",
    "> Multi-horizon Time Series Forecasting», NeurIPS 2021.",
    "",
    f"**Файл:** `data/merged_data.csv` · {df.shape[0]} строк · {df.shape[1]} колонок  ",
    f"**Период:** {df['timestamp'].min().date()} — {df['timestamp'].max().date()}  ",
    f"**Станций:** {df['station_id'].nunique()} · Часов/станцию: {df.groupby('station_id').size().iloc[0]}",
    "",
]

# ── Ключевое решение проекта ──────────────────────────────────
lines += [
    "## Ключевое архитектурное решение: что можно менять в прогнозе",
    "",
    "В TFT (Section 3) только переменные **time-varying known** подаются в декодер,",
    "то есть модель «видит» их значения на горизонте прогноза.",
    "Именно эти переменные можно задавать в сценарном (what-if) анализе.",
    "",
    "**Принятое решение:** погода, трафик по типам ТС и цены конкурентов",
    "отнесены к **KNOWN** (а не к UNKNOWN, как принято по умолчанию),",
    "что позволяет пользователю моделировать сценарий на уровне каждого часа:",
    "",
    "| Что задаёт пользователь | Переменные |",
    "|---|---|",
    "| Прогноз погоды | temperature, precipitation_mm, visibility_km, wind_speed_ms, weather_condition, is_snow/rain/fog |",
    "| Ожидаемый трафик | traffic_Passengers_cars, traffic_Truck_short, traffic_Truck, traffic_Truck_long, traffic_Transporter, traffic_Undefined |",
    "| Мониторинг конкурентов | competitor_price_AI92, competitor_price_AI95, competitor_price_DT |",
    "| Акции и реклама | promotion_fuel_active, promotion_shop_active, promotion_cafe_active, ad_active, ad_channel |",
    "| Ценовая политика | price_AI92/95/98, price_DT_EURO/TANEKO/SUMMER/WINTER |",
    "| Календарь | дата, час → hour, day_of_week, is_weekend, is_holiday, holiday_name |",
    "",
    "> **Ограничение:** статические переменные (расположение, инфраструктура, оценки персонала)",
    "> вшиты в веса модели и не могут быть изменены без переобучения.",
    "> Для анализа влияния локации — сравнивать прогнозы разных станций.",
    "",
]

# ── Сводная таблица ───────────────────────────────────────────
n_tech  = len(TECHNICAL)
n_excl  = len(EXCLUDED)
n_sc    = len(STATIC_CATS)
n_sr    = len(STATIC_REALS)
n_kc    = len(KNOWN_CATS)
n_kr    = len(KNOWN_REALS)
n_tgt   = len(TARGETS)
n_total = n_tech + n_excl + n_sc + n_sr + n_kc + n_kr + n_tgt

# ── Производные колонки, добавляемые в препроцессинге ──────────
n_cyclical_extra = 8    # sin/cos: hour, day_of_week, week_of_year, month × 2
n_enc_extra      = n_sc + n_kc  # 3+4=7 _enc-колонок от LabelEncoder
n_orig_extra     = n_tgt         # 12 _orig (цели до log1p, нужны для inverse_transform)
n_time_idx       = 1             # time_idx (монотонный счётчик per-station)
n_is_shop_open   = 1             # is_shop_open (производный из hour; в merged_data отсутствует)

# Признаков, подаваемых в TimeSeriesDataSet (n_kr включает is_shop_open и raw циклику)
# После sin/cos расширения: n_tft_inputs = n_model_orig + 8
n_model_orig = n_sc + n_sr + n_kc + n_kr + n_tgt    # 81 (до sin/cos)
n_tft_inputs = n_model_orig + n_cyclical_extra        # 89 — реальные входы TFT

# Всего колонок в prepared_data.csv (модельные входы + служебные + _enc + _orig + time_idx)
# Formula: 89 (merged) − 7 (excluded) + 1 (is_shop_open) + 7 (_enc) + 8 (sin/cos) + 1 (time_idx) + 12 (_orig) = 111
n_prepared = df.shape[1] - n_excl + n_is_shop_open + n_enc_extra + n_cyclical_extra + n_time_idx + n_orig_extra

lines += [
    "## Сводная таблица",
    "",
    "| Категория | Колонок в merged_data.csv |",
    "|---|---|",
    f"| Технические идентификаторы (timestamp, station_id) | {n_tech} |",
    f"| **Исключены** (избыточные, не подаются в модель) | {n_excl} |",
    f"| STATIC_CATS | {n_sc} |",
    f"| STATIC_REALS | {n_sr} |",
    f"| TIME_VARYING_KNOWN_CATS | {n_kc} |",
    f"| TIME_VARYING_KNOWN_REALS | {n_kr} |",
    f"| TIME_VARYING_UNKNOWN_REALS | 0 |",
    f"| TARGET | {n_tgt} |",
    f"| **Итого** | **{n_total}** |",
    "",
    (f"Проверка по merged_data.csv: {n_tech} (тех.) + {n_excl} (искл.) + {n_sc} + {n_sr} + "
     f"{n_kc} + {n_kr - n_is_shop_open} (KNOWN_REALS из merged) + {n_tgt} = "
     f"**{n_total - n_is_shop_open}** из 89. "
     f"Плюс {n_is_shop_open} производный `is_shop_open` (в merged_data отсутствует)."),
    "",
    "### Как складываются колонки в prepared_data.csv",
    "",
    "```",
    f"  {df.shape[1]}  колонок в merged_data.csv",
    f"−  {n_excl}  исключённых (station_name, total_pumps, total_fuel_sales,",
    f"              shop_total_revenue, total_traffic, quarter, day_name)",
    f"─────",
    f"  {df.shape[1] - n_excl}  после исключения (включая 2 технических: timestamp, station_id)",
    f"+  {n_is_shop_open}  is_shop_open (производный бинарный: магазин открыт 05:00–21:00)",
    f"+  {n_enc_extra}  _enc (LabelEncoder для {n_enc_extra} категориальных признаков)",
    f"+  {n_cyclical_extra}  _sin/_cos (циклическое кодирование: hour, day_of_week, week_of_year, month)",
    f"+  {n_time_idx}  time_idx (монотонный счётчик часов per-station)",
    f"+ {n_orig_extra}  _orig (оригинальные значения {n_orig_extra} целевых до log1p)",
    f"─────",
    f" {n_prepared}  колонок итого в prepared_data.csv",
    f"       из них {n_tft_inputs} — прямые входы TimeSeriesDataSet",
    "```",
    "",
]


def section_header(title: str, tft_name: str, description: str) -> list:
    return [
        f"## {title} (`{tft_name}`)",
        "",
        description,
        "",
    ]


def table_rows(d: dict, extra_col: str | None = None) -> list:
    """Генерирует строки Markdown-таблицы для словаря переменных."""
    header = "| Переменная | Тип | Уник. | Пропуски | Диапазон / значения |"
    sep    = "|---|---|---|---|---|"
    if extra_col:
        header += f" {extra_col} |"
        sep    += "---|"
    rows = [header, sep]
    for col, meta in d.items():
        s = stat_row(col)
        row = (f"| `{col}` | {s['dtype']} | {s['n_uniq']} | {s['n_null']} "
               f"| {s['range']} |")
        if extra_col:
            val = meta.get(extra_col.strip().lower(), meta.get("desc", ""))
            # Экранируем переносы строк для Markdown-таблицы
            val = str(val).replace("\n", " ")
            row += f" {val} |"
        rows.append(row)
    return rows


# ── 1. Технические ────────────────────────────────────────────
lines += section_header(
    "Технические идентификаторы", "не признаки",
    "Используются для построения датасета, в модель как признаки не подаются.",
)
lines += table_rows(TECHNICAL, "Reason")
lines.append("")

# ── 2. Исключены ──────────────────────────────────────────────
lines += section_header(
    "Исключены из модели", "excluded",
    (
        "Переменные, которые дублируют информацию других колонок или являются "
        "производными от них. Оставлены в `merged_data.csv` для читаемости "
        "и EDA, но **не подаются в TFT**."
    ),
)
lines += table_rows(EXCLUDED, "Reason")
lines.append("")

# ── 3. STATIC_CATS ───────────────────────────────────────────
lines += section_header(
    "Статические категориальные", "static_categoricals",
    (
        "Неизменные характеристики АЗС. TFT кодирует через **Entity Embedding** "
        "(обучаемые векторы, Section 4.1 статьи). "
        "Предобработка: `LabelEncoder` → целое → embedding-слой. "
        "Информация используется **Static Variable Selection Network** для "
        "адаптации весов под каждую станцию."
    ),
)
lines += table_rows(STATIC_CATS, "Desc")
lines.append("")

# ── 4. STATIC_REALS ──────────────────────────────────────────
lines += section_header(
    "Статические вещественные", "static_reals",
    (
        "Числовые паспортные данные АЗС. Не меняются во времени — "
        "одно значение на всю историю станции. "
        "**Не нормализуются** (исключены из Z-score, нет временно́й вариации). "
        "TFT использует через Static Variable Selection Network. "
        "Переменная `distance_to_city_km` определяет транзитный / городской профиль."
    ),
)
lines += table_rows(STATIC_REALS, "Desc")
lines.append("")

# ── 5. KNOWN_CATS ─────────────────────────────────────────────
lines += section_header(
    "Известные будущие категориальные", "time_varying_known_categoricals",
    (
        "Категориальные переменные, доступные для **горизонта прогноза** (декодер). "
        "TFT подаёт их и в энкодер (прошлое), и в декодер (будущее). "
        "Предобработка: `NaN → семантическая метка`, `LabelEncoder` → embedding."
    ),
)
lines += table_rows(KNOWN_CATS, "What-if")
lines.append("")

# ── 6. KNOWN_REALS ────────────────────────────────────────────
lines += section_header(
    "Известные будущие вещественные", "time_varying_known_reals",
    (
        "Числовые переменные, доступные на горизонте прогноза (энкодер + декодер). "
        "**Это единственный класс, значения которого пользователь может задавать** "
        "для сценарного анализа. "
        "Содержит временны́е признаки, управляемые параметры (акции, цены) "
        "и перенесённые из UNKNOWN переменные среды (погода, трафик, конкуренты). "
        f"Исходных колонок: {n_kr}. После добавления sin/cos для "
        "hour / day_of_week / week_of_year / month — итого "
        f"{n_kr + n_cyclical_extra} в модели.\n\n"
        "**Реалистичность входов по горизонту прогноза:**\n\n"
        "| Горизонт | Погода | Трафик | Цены конкурентов | Характер входов |\n"
        "|---|---|---|---|---|\n"
        "| День (24 ч) | Точный прогноз | Плановый | Мониторинг | Фактические данные |\n\n"
        "Горизонт прогноза модели: **24 часа** (1 сутки). "
        "На этом горизонте KNOWN-переменные — реальные входы (метеопрогноз на сутки, "
        "плановый трафик, мониторинг цен конкурентов). "
        "Для сценарного анализа (what-if) пользователь может задавать любые значения."
    ),
)
lines += table_rows(KNOWN_REALS, "What-if")
lines.append("")

# ── 7. UNKNOWN_REALS (пояснительный раздел) ──────────────────
lines += [
    "## Наблюдаемые прошлые (`time_varying_unknown_reals`)",
    "",
    "В классическом TFT (Section 3 статьи) этот класс содержит переменные, "
    "которые **наблюдаются только в прошлом** — энкодер их видит, декодер нет. "
    "Типичный пример из статьи: прошлый спрос, прошлые продажи.",
    "",
    "**В данной модели этот список намеренно пуст.**",
    "",
    "Причина: все наблюдаемые переменные (погода, трафик, цены конкурентов) "
    "перенесены в `TIME_VARYING_KNOWN_REALS`, чтобы пользователь мог задавать "
    "их при сценарном анализе на горизонте день / неделя / месяц.",
    "",
    "| Вопрос | Ответ |",
    "|---|---|",
    "| Можно ли поместить переменную одновременно в KNOWN и UNKNOWN? | **Нет.** Архитектура TFT: каждая переменная принадлежит ровно одной группе. |",
    "| Что теряем, перенося погоду в KNOWN? | Ничего — энкодер получает прошлые значения из данных, декодер — заданные пользователем. |",
    "| Что теряем, если оставить погоду в UNKNOWN? | Возможность задавать погоду для прогноза. Энкодер видит прошлую погоду, но декодер игнорирует будущую. |",
    "| Можно ли добавить цели в unknown_reals явно? | **Нет.** `target=TARGET_COLS` в pytorch-forecasting уже делает это автоматически. Дублирование вызовет ошибку. |",
    "",
]

# ── 8. TARGETS ───────────────────────────────────────────────
lines += section_header(
    "Целевые переменные", "target + time_varying_unknown_reals",
    (
        "12 переменных одновременно являются **прогнозируемым выходом** и "
        "**авторегрессивным входом энкодера** (observed past, Section 3 статьи). "
        "Декодер их значений не видит — они неизвестны на горизонте прогноза. "
        "В pytorch-forecasting задаются через параметр `target=TARGET_COLS` — "
        "библиотека автоматически подаёт прошлые значения в энкодер. "
        "**Явно добавлять в `time_varying_unknown_reals` не нужно и нельзя** — "
        "это приведёт к дублированию входа.\n\n"
        "Предобработка: `log1p` → устранение правостороннего скоса и нулей. "
        "`TorchNormalizer(method='robust')` — робастная нормализация по медиане и IQR. "
        "Обратное преобразование: `expm1(TorchNormalizer.inverse_transform(pred))`.\n\n"
        "**Горизонты прогноза** подробно рассмотрены в разделе «Размеры окон» ниже."
    ),
)
lines += table_rows(TARGETS, "Desc")
lines.append("")

# ── 8. Циклическое кодирование ────────────────────────────────
lines += [
    "## Циклическое кодирование временных признаков",
    "",
    "**Проблема:** числовой признак `hour=23` и `hour=0` различаются на 23,",
    "хотя физически это соседние часы. То же для переходов Вс→Пн и Дек→Янв.",
    "",
    "**Решение (Section 3.3 статьи, стандартная практика):**",
    "проекция на единичную окружность через sin/cos:",
    "",
    "```",
    "hour_sin = sin(2π · hour / 24)",
    "hour_cos = cos(2π · hour / 24)",
    "```",
    "",
    "Евклидово расстояние между (hour=23) и (hour=0) на окружности минимально.",
    "Оба значения (sin и cos) нужны одновременно: только sin неоднозначен",
    "(час 3 и час 9 дают одинаковый sin).",
    "",
    "| Признак | Период | Добавляемые колонки |",
    "|---|---|---|",
    "| `hour` | 24 | `hour_sin`, `hour_cos` |",
    "| `day_of_week` | 7 | `day_of_week_sin`, `day_of_week_cos` |",
    "| `week_of_year` | 52 | `week_of_year_sin`, `week_of_year_cos` |",
    "| `month` | 12 | `month_sin`, `month_cos` |",
    "",
]

# ── 9. Предобработка — итоговый план ─────────────────────────
lines += [
    "## Итоговый план предобработки",
    "",
    "| Шаг | Что делаем | На каких колонках |",
    "|---|---|---|",
    f"| 1. Исключить | Убрать {n_excl} избыточных колонок из датасета до подачи в модель | station_name, total_pumps, total_fuel_sales, shop_total_revenue, total_traffic, quarter, day_name |",
    "| 2. Пропуски | `holiday_name → 'нет_праздника'`, `ad_channel → 'нет_рекламы'` | holiday_name, ad_channel |",
    "| 3. Label Encoding | `→ _enc`-колонки | STATIC_CATS (3) + KNOWN_CATS (4) |",
    "| 4. Циклическое кодирование | `→ _sin/_cos`-колонки (+8 новых) | hour, day_of_week, week_of_year, month |",
    "| 5. time_idx | Монотонный счётчик часов per-station через `cumcount` | Все станции |",
    "| 6. log1p | Логарифм(1+x). Оригинал → `_orig`-колонка | Все 12 TARGET_COLS |",
    "| 7. Z-score | Нормализация per-station: μ и σ → `tft/scalers.pkl` | Числовые KNOWN_REALS (кроме бинарных и price_*) |",
    "",
    "### О Winsorization",
    "",
    "**Статья Lim et al. (2020) Winsorization не упоминает.**  ",
    "Для нормализации признаков используется стандартный Z-score по обучающей выборке.  ",
    "Для целевых переменных применяется `TorchNormalizer(method='robust')` — ",
    "нормализация по медиане и IQR, которая сама по себе устойчива к выбросам.  ",
    "",
    "Поскольку данные не содержат артефактов измерения (синтетический датасет), "
    "а удалять строки нельзя (временной ряд должен быть непрерывным), "
    "**Winsorization не применяется**. "
    "Если в реальных данных возникнут технические выбросы (сбой датчиков и т.п.), "
    "следует добавить мягкое ограничение по перцентилю (например, 99.9-й) "
    "только для числовых признаков погоды и трафика.",
    "",
    "> **Особый случай — price_\\*:** в данных 2023 г. цены не менялись (std=0 per-station).",
    "> Z-score даст NaN. Исключить из Z-score, подавать в исходном масштабе.",
    "> Архитектурно они остаются в KNOWN_REALS как управляемый параметр.",
    "",
]

# ── 10. Размеры окон ─────────────────────────────────────────
lines += [
    "## Размеры окон энкодера и декодера",
    "",
    "### Почему длина энкодера важна",
    "",
    "Энкодер — это ретроспективное окно: TFT видит прошлые значения всех переменных "
    "за `encoder_length` часов перед точкой прогноза. "
    "Чтобы модель уловила сезонный паттерн, она должна **видеть хотя бы один полный цикл** "
    "этого паттерна в прошлом.",
    "",
    "| Паттерн | Период | Нужно энкодера | 168 ч хватает? |",
    "|---|---|---|---|",
    "| Суточный (час пик, ночь) | 24 ч | ≥ 48 ч | ✅ Да (7 циклов) |",
    "| Недельный (выходные vs будни) | 168 ч | ≥ 336 ч | ⚠️ Впритык (1 цикл) |",
    "| Месячный (начало/конец месяца) | 720 ч | ≥ 1440 ч | ❌ Нет |",
    "| Сезонный (зима/лето) | 2160+ ч | ≥ 4320 ч | ❌ Нет |",
    "",
    "> Сезонные паттерны при коротком энкодере модель частично восстанавливает "
    "через KNOWN-признаки (`month`, `season`, `is_holiday`), но это **экстраполяция "
    "по признакам**, а не обучение на реальной истории.",
    "",
    "### Рекомендуемые размеры окон по горизонту прогноза",
    "",
    "| Горизонт | decoder | Рекомендуемый encoder | Что охватывает encoder | Обучающих окон на станцию |",
    "|---|---|---|---|---|",
    "| День (24 ч) | 24 ч | 168 ч (7 дней) | Суточный + недельный цикл | ≈ 8568 |",
    "| Неделя (168 ч) | 168 ч | 336 ч (14 дней) | 2 недельных цикла | ≈ 8256 |",
    "| Месяц (720 ч) | 720 ч | 720 ч (30 дней) | Полный месячный цикл | ≈ 7320 |",
    "",
    "Обучающих окон: `8760 − encoder − decoder` (скользящие окна за 1 год, 1 станция).",
    "При 5 станциях — умножить на 5.",
    "",
    "### Выбранная конфигурация: ретроспектива 7 суток / горизонт 1 сутки",
    "",
    "TFT обучается с фиксированным `encoder_length` и `max_prediction_length`. "
    "Выбранные параметры оптимальны для задачи **оперативного суточного планирования**:",
    "",
    "```",
    "encoder_length       = 168  # 7 дней — охватывает суточный + недельный цикл",
    "max_prediction_length = 24  # 1 сутки — горизонт оперативного прогноза",
    "",
    "# Прогноз: 24 почасовых шага = следующие сутки",
    "```",
    "",
    "**Обоснование выбора:**",
    "- Суточный цикл (24 ч) полностью покрыт в 168-часовой ретроспективе (7 циклов).",
    "- Недельный цикл (рабочие дни vs выходные) покрыт ровно 1 раз — достаточно для TFT,",
    "  поскольку `day_of_week` с sin/cos кодированием явно передаётся в декодер.",
    "- Окна 168 + 24 = 192 ч → ~7 110 обучающих окон на станцию / **≈ 35 550 итого**.",
    "- Каждое окно в 7.5× короче 720-часового → обучение существенно быстрее.",
    "",
    "| Параметр | Значение | Обоснование |",
    "|---|---|---|",
    "| `encoder_length` | 168 ч (7 суток) | Суточный (7×) + недельный (1×) цикл |",
    "| `max_prediction_length` | 24 ч (1 сутки) | Горизонт оперативного планирования |",
    "| Обучающих окон (5 ст.) | ≈ 35 550 | 5 × (8760 − 168 − 24) |",
    "| Ускорение обучения | ~7.5× | Каждое окно 192 ч vs 1440 ч |",
    "",
]

# ── 11. Итог ─────────────────────────────────────────────────
lines += [
    "## Итог: состав входов TFT",
    "",
    "| Вход TFT | Колонок из merged_data | Колонок в prepared_data |",
    "|---|---|---|",
    f"| `static_categoricals` | {n_sc} | {n_sc} (_enc) |",
    f"| `static_reals` | {n_sr} | {n_sr} |",
    f"| `time_varying_known_categoricals` | {n_kc} | {n_kc} (_enc) |",
    f"| `time_varying_known_reals` | {n_kr - n_is_shop_open} из merged + {n_is_shop_open} произв. | {n_kr + n_cyclical_extra} (+ {n_cyclical_extra} sin/cos) |",
    "| `time_varying_unknown_reals` | 0 | 0 (все наблюдаемые перенесены в KNOWN; цели → autoregressive через `target=`) |",
    f"| `target` | {n_tgt} | {n_tgt} (log1p) |",
    f"| **Входов в TimeSeriesDataSet** | **{n_model_orig - n_is_shop_open} из merged + {n_is_shop_open} произв.** | **{n_tft_inputs}** |",
    f"| **Всего в prepared_data.csv** | — | **{n_prepared}** (+ _enc×{n_enc_extra} + _orig×{n_orig_extra} + time_idx) |",
    "",
    "| Параметр | Значение |",
    "|---|---|",
    "| `encoder_length` | 168 ч (7 суток) |",
    "| `max_prediction_length` | 24 ч (1 сутки) |",
    "| Горизонт прогноза | 24 шага = следующие сутки (почасово) |",
    "| Обучающих окон | ≈ 7 110 на станцию / ≈ 35 550 итого |",
    "",
]

# ── Сохранение ────────────────────────────────────────────────
report = "\n".join(lines)
out_path = "reports/column_analysis.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)

print("=" * 60)
print("Готово.")
print(f"  Сохранено : {out_path}")
print()
print(f"  {df.shape[1]} колонок в merged_data.csv")
print(f"  - {n_excl} исключённых (station_name, total_pumps, ...)")
print(f"  = {df.shape[1] - n_excl} после исключения")
print(f"  + {n_is_shop_open} is_shop_open (производный бинарный)")
print(f"  + {n_enc_extra} _enc (label encoding {n_enc_extra} категориальных)")
print(f"  + {n_cyclical_extra} sin/cos (циклические признаки)")
print(f"  + {n_time_idx} time_idx")
print(f"  + {n_orig_extra} _orig (цели до log1p)")
print(f"  = {n_prepared} итого в prepared_data.csv")
print(f"    из них {n_tft_inputs} -- входы TimeSeriesDataSet")
print("=" * 60)
