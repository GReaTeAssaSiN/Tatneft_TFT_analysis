"""
Классификация переменных merged_data.csv для TFT-модели.
Определяет роль каждой из 115 колонок согласно:
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

df = pd.read_csv(SRC, parse_dates=["date"])

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
#   Погода, трафик (счётчики ТС) и цены конкурентов помещены в (b) KNOWN,
#   что позволяет задавать их при сценарном прогнозе (what-if).
#   Производные метрики трафика (скорость, плотность, интенсивность)
#   помещены в (c) UNKNOWN — они наблюдаемые, но не прогнозируемые заранее.
#   Клиентские метрики (лояльность, корпоративная доля) — также UNKNOWN.
# ══════════════════════════════════════════════════════════════

# ── Технические идентификаторы (не входят в модель как признаки)
TECHNICAL = {
    "date": {
        "reason": (
            "Временная метка (суточная, 2025 г.). Используется для построения "
            "time_idx = cumcount по станции. "
            "Определяет сплиты train/val/test. В модель как признак не подаётся."
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

# ── Исключены из модели (избыточные / нерелевантные колонки) ──
EXCLUDED = {
    "station_name": {
        "reason": (
            "Строковое название АЗС. Дублирует station_id. "
            "Все характеристики станции представлены в STATIC_CATS и STATIC_REALS. "
            "Текстовая метка не несёт предсказательной силы."
        ),
    },
    "total_pumps": {
        "reason": (
            "Сумма всех num_pumps_*. Индивидуальные счётчики по каждому виду "
            "топлива несут больше информации (пропускная способность конкретного "
            "продукта), агрегат избыточен."
        ),
    },
    "total_fuel_sales": {
        "reason": (
            "Сумма всех sales_*. Если модель прогнозирует каждый вид топлива "
            "отдельно, агрегат выводится постфактум и дублирует целевые переменные."
        ),
    },
    "quarter": {
        "reason": (
            "Грубое укрупнение месяца (12 → 4 значения). Модель получает month "
            "в KNOWN_REALS с sin/cos и самостоятельно обнаружит квартальные паттерны. "
            "Квартал только снижает разрешение сигнала."
        ),
    },
    "traffic_uroven_jbslugi_1_poputn": {
        "reason": (
            "Уровень обслуживания дороги A–F (строковая категория). "
            "Семантически перекрывается со счётчиками трафика в KNOWN_REALS. "
            "Label Encoding потребовал бы 4 дополнительных категориальных признака, "
            "при этом информация уже косвенно представлена через интенсивность потока. "
            "Исключены все 4 варианта (полоса 1/2 × попутн./встречн.)."
        ),
    },
    "traffic_uroven_jbslugi_2_poputn":   {"reason": "Аналогично traffic_uroven_jbslugi_1_poputn."},
    "traffic_uroven_jbslugi_1_wstrechn": {"reason": "Аналогично traffic_uroven_jbslugi_1_poputn."},
    "traffic_uroven_jbslugi_2_wstrechn": {"reason": "Аналогично traffic_uroven_jbslugi_1_poputn."},
}

# ── Статические категориальные ────────────────────────────────
# ⚠️  ВАЖНО: settlement_size и distance_to_city_km — КОДЫ КАТЕГОРИЙ,
#            а не числовые значения. Обрабатываются Label Encoding,
#            не Z-score нормализацией.
STATIC_CATS = {
    "road_type": {
        "desc": "Тип дороги: федеральная трасса / улица в населённом пункте / др.",
        "encoding": "Label Encoding → road_type_enc.",
        "values": "Федеральная трасса, Улица в населенном пункте",
        "what_if": "Нет (статическая характеристика станции).",
    },
    "road_level": {
        "desc": (
            "Уровень дороги: 1=федеральная, 2=региональная, 3=локальная, "
            "4=магистральная улица, 5=районная улица."
        ),
        "encoding": "Label Encoding → road_level_enc. Значения 1 и 5 — коды, не числа.",
        "values": "1, 5",
        "what_if": "Нет.",
    },
    "direction": {
        "desc": "Направление движения: транзит-трассовая / городская / въезд с двух сторон / др.",
        "encoding": "Label Encoding → direction_enc. Embedding TFT.",
        "values": "транзит-трассовая АЗС, городская",
        "what_if": "Нет.",
    },
    "settlement_size": {
        "desc": (
            "Размер ближайшего населённого пункта (КОД КАТЕГОРИИ, не число жителей): "
            "1=город >1 млн, 2=500k–1 млн, 3=250–500k, 4=100–250k, "
            "5=<100k, 6=вне населённого пункта."
        ),
        "encoding": (
            "Label Encoding → settlement_size_enc. "
            "❗ Несмотря на числовой вид (3, 6), значения являются категориальными кодами. "
            "Z-score нормализация неприменима — семантика интервалов нарушена."
        ),
        "values": "3 (= 250–500k), 6 (= вне нас. пункта)",
        "what_if": "Нет (паспортная характеристика станции).",
    },
    "distance_to_city_km": {
        "desc": (
            "Удалённость от города (КОД КАТЕГОРИИ, не реальное расстояние в км): "
            "0=в черте города, 1=менее 20 км, 2=20–50 км, 3=50–80 км, "
            "4=80–110 км, 5=110–200 км, 6=свыше 200 км."
        ),
        "encoding": (
            "Label Encoding → distance_to_city_km_enc. "
            "❗ Несмотря на числовой вид (0, 1, 2...), значения являются порядковыми кодами. "
            "Z-score нормализация неприменима — арифметика на кодах теряет семантику диапазонов."
        ),
        "values": "0 (= в черте города), 1 (= < 20 км)",
        "what_if": "Нет (паспортная характеристика станции).",
    },
}

# ── Статические вещественные ──────────────────────────────────
STATIC_REALS = {
    "shop_area_m2": {
        "desc": "Площадь магазина, м². Реальное числовое значение.",
        "preprocessing": "Без нормализации (паспортные данные статические).",
        "what_if": "Нет (инфраструктура).",
    },
    # Число колонок по видам топлива
    "num_pumps_AI92":     {"desc": "Число колонок АИ-92.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_AI92_bio": {"desc": "Число колонок АИ-92 Bio.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_AI95":     {"desc": "Число колонок АИ-95.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_AI95_bio": {"desc": "Число колонок АИ-95 Bio.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_AI100_bio":{"desc": "Число колонок АИ-100 Bio.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_DT":       {"desc": "Число колонок ДТ.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_DT_bio":   {"desc": "Число колонок ДТ Bio.", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_SUG":      {"desc": "Число колонок СУГ (сжиженный углеводородный газ).", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_KPG":      {"desc": "Число колонок КПГ (компримированный природный газ).", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    "num_pumps_SPG":      {"desc": "Число колонок СПГ (сжиженный природный газ).", "preprocessing": "Без нормализации.", "what_if": "Нет."},
    # Услуги АЗС (бинарные)
    "has_car_wash":      {"desc": "Есть автомойка (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_tire_service":  {"desc": "Есть шиномонтаж (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_cafe":          {"desc": "Есть кафе (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_hotel":         {"desc": "Есть гостиница (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_shop":          {"desc": "Есть магазин (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    # Дополнительные услуги магазина (бинарные)
    "has_shop_молельная_комната":        {"desc": "Есть молельная комната (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_shop_прачечная":                {"desc": "Есть прачечная (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_shop_электрозарядная_станция":  {"desc": "Есть электрозарядная станция (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    "has_shop_подкачка_шин":             {"desc": "Есть подкачка шин (0/1).", "preprocessing": "Бинарный. Без нормализации.", "what_if": "Нет."},
    # Конкурентная среда
    "competitors_wink": {
        "desc": "Число конкурентов поблизости по данным Wink.",
        "preprocessing": "Без нормализации. Счётное число — семантически числовое.",
        "what_if": "Нет (константа среды).",
    },
}

# ── Известные будущие категориальные ──────────────────────────
KNOWN_CATS = {
    "season": {
        "desc": "Сезон: winter / spring / summer / autumn.",
        "preprocessing": "Label Encoding → season_enc. Определяется из даты.",
        "what_if": "Да — задаётся автоматически из даты прогноза.",
    },
    "holiday_name": {
        "desc": "Название праздника или 'нет_праздника' (заполняется из NaN).",
        "preprocessing": "NaN → 'нет_праздника'. Label Encoding → holiday_name_enc.",
        "what_if": "Да — известно из производственного календаря.",
    },
    # weather_condition (0/1 binary) перенесена в KNOWN_REALS:
    # в данных 2025 г. это бинарный флаг, не многоклассовая категория.
    # ad_channel отсутствует в данных 2025 г.
}

# ── Известные будущие вещественные ───────────────────────────
KNOWN_REALS = {
    # Циклические временные признаки (суточные данные, час отсутствует)
    "day_of_week": {
        "desc": "День недели (0=Пн ... 6=Вс).",
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
    # Бинарные флаги
    "is_weekend": {
        "desc": "Выходной день (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да — определяется датой прогноза.",
    },
    "is_holiday": {
        "desc": "Праздничный день (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да — из производственного календаря.",
    },
    # Погода
    "weather_condition": {
        "desc": "Тип погоды: 0 = ясно/облачно, 1 = осадки/плохая видимость. Бинарный флаг.",
        "preprocessing": "Бинарный (0/1). Без нормализации. Не требует Label Encoding — уже числовой.",
        "what_if": "Да — задаётся из метеопрогноза (осадки/без осадков).",
    },
    "temperature": {
        "desc": "Температура воздуха, °C.",
        "preprocessing": "Z-score per-station (по статистике train, Jan–Oct 2025).",
        "what_if": "Да — из прогноза погоды. Влияет на потребление топлива и продажи кофе.",
    },
    "precipitation_mm": {
        "desc": "Количество осадков, мм.",
        "preprocessing": "Z-score per-station.",
        "what_if": "Да — из прогноза погоды.",
    },
    # Акции
    "promotion_fuel_active": {
        "desc": "Акция на топливо активна (0/1).",
        "preprocessing": "Бинарный. Без нормализации.",
        "what_if": "Да — ключевой параметр сценария, планируется заранее.",
    },
    "promotion_shop_active": {"desc": "Акция в магазине активна (0/1).", "preprocessing": "Бинарный.", "what_if": "Да."},
    "promotion_cafe_active":  {"desc": "Акция в кафе активна (0/1).", "preprocessing": "Бинарный.", "what_if": "Да."},
    # Текущие цены топлива
    "price_AI92": {
        "desc": "Текущая цена АИ-92, руб/л. В 2025 г. меняется ежедневно.",
        "preprocessing": "Z-score per-station (цены варьируются, std > 0).",
        "what_if": "Да — управляемый параметр ценообразования.",
    },
    "price_AI95":    {"desc": "Текущая цена АИ-95, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "price_DT":      {"desc": "Текущая цена ДТ, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "price_DT_bio":  {"desc": "Текущая цена ДТ Bio, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "price_AI100_bio":{"desc": "Текущая цена АИ-100 Bio, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    # Трафик — счётчики ТС, попутное направление
    "traffic_Passengers_cars_1_poputn": {
        "desc": "Легковые авто, полоса 1, попутное направление (съезд на АЗС), авт/день.",
        "preprocessing": "Z-score per-station.",
        "what_if": "Да — пользователь задаёт ожидаемый трафик. Легковые → АИ-92/95.",
    },
    "traffic_Passengers_cars_2_poputn": {"desc": "Легковые авто, полоса 2, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_short_1_poputn":     {"desc": "Малотоннажные грузовики (до 3.5 т), полоса 1, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_short_2_poputn":     {"desc": "Малотоннажные грузовики, полоса 2, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_1_poputn":           {"desc": "Грузовики среднего класса (3.5–12 т), полоса 1, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_2_poputn":           {"desc": "Грузовики среднего класса, полоса 2, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_long_1_poputn":      {"desc": "Большегрузные авто (> 12 т), полоса 1, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да — большегрузы → ДТ."},
    "traffic_Truck_long_2_poputn":      {"desc": "Большегрузные авто, полоса 2, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Transporter_1_poputn":     {"desc": "Микроавтобусы/транспортёры, полоса 1, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Transporter_2_poputn":     {"desc": "Микроавтобусы/транспортёры, полоса 2, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Undefined_1_poputn":       {"desc": "Неопределённый тип ТС, полоса 1, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Undefined_2_poputn":       {"desc": "Неопределённый тип ТС, полоса 2, попутн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    # Трафик — счётчики ТС, встречное направление
    "traffic_Passengers_cars_1_wstrechn": {"desc": "Легковые авто, полоса 1, встречное направление, авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Passengers_cars_2_wstrechn": {"desc": "Легковые авто, полоса 2, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_short_1_wstrechn":     {"desc": "Малотоннажные грузовики, полоса 1, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_short_2_wstrechn":     {"desc": "Малотоннажные грузовики, полоса 2, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_1_wstrechn":           {"desc": "Грузовики среднего класса, полоса 1, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_2_wstrechn":           {"desc": "Грузовики среднего класса, полоса 2, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_long_1_wstrechn":      {"desc": "Большегрузные авто, полоса 1, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Truck_long_2_wstrechn":      {"desc": "Большегрузные авто, полоса 2, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Transporter_1_wstrechn":     {"desc": "Микроавтобусы/транспортёры, полоса 1, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Transporter_2_wstrechn":     {"desc": "Микроавтобусы/транспортёры, полоса 2, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Undefined_1_wstrechn":       {"desc": "Неопределённый тип ТС, полоса 1, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "traffic_Undefined_2_wstrechn":       {"desc": "Неопределённый тип ТС, полоса 2, встречн., авт/день.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    # Цены конкурентов
    "competitor_price_AI92": {
        "desc": "Цена АИ-92 у небрендовых конкурентов, руб/л.",
        "preprocessing": "Z-score per-station.",
        "what_if": "Да — что если конкурент снизил цену на X руб.?",
    },
    "competitor_price_AI95":       {"desc": "Цена АИ-95 у небрендовых конкурентов, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "competitor_price_DT":         {"desc": "Цена ДТ у небрендовых конкурентов, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "competitor_price_AI92_brend": {"desc": "Цена АИ-92 у брендовых конкурентов, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "competitor_price_AI95_brend": {"desc": "Цена АИ-95 у брендовых конкурентов, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "competitor_price_DT_brend":   {"desc": "Цена ДТ у брендовых конкурентов, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
    "competitor_price_AI100":      {"desc": "Цена АИ-100 у конкурентов, руб/л.", "preprocessing": "Z-score per-station.", "what_if": "Да."},
}

# ── Наблюдаемые прошлые вещественные (только энкодер) ────────
UNKNOWN_REALS = {
    # Клиентские метрики — наблюдаемые, но не прогнозируемые
    "corporate_customer_ratio": {
        "desc": "Доля корпоративных клиентов (0–1). Меняется ежедневно.",
        "preprocessing": "Z-score per-station.",
        "why_unknown": "Не прогнозируется заранее. Доступна только за прошлые периоды.",
    },
    "customer_loyalty_score": {
        "desc": "Оценка лояльности клиентов (0–1). Меняется ежедневно.",
        "preprocessing": "Z-score per-station.",
        "why_unknown": "Наблюдаемая метрика, не задаётся заранее.",
    },
    # Производные метрики трафика — результат моделирования потока
    "traffic_scorost_1_poputn": {
        "desc": "Средняя скорость потока, полоса 1, попутн., км/ч.",
        "preprocessing": "Z-score per-station.",
        "why_unknown": (
            "Производная от объёма потока и дорожных условий. "
            "Не задаётся в what-if независимо от счётчиков ТС."
        ),
    },
    "traffic_scorost_2_poputn":    {"desc": "Средняя скорость, полоса 2, попутн., км/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично scorost_1."},
    "traffic_scorost_1_wstrechn":  {"desc": "Средняя скорость, полоса 1, встречн., км/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично scorost_1."},
    "traffic_scorost_2_wstrechn":  {"desc": "Средняя скорость, полоса 2, встречн., км/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично scorost_1."},
    "traffic_plotnost_1_poputn":   {"desc": "Плотность потока, полоса 1, попутн., авт/км.", "preprocessing": "Z-score per-station.", "why_unknown": "Производная от объёма и скорости."},
    "traffic_plotnost_2_poputn":   {"desc": "Плотность потока, полоса 2, попутн., авт/км.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично plotnost_1."},
    "traffic_plotnost_1_wstrechn": {"desc": "Плотность потока, полоса 1, встречн., авт/км.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично plotnost_1."},
    "traffic_plotnost_2_wstrechn": {"desc": "Плотность потока, полоса 2, встречн., авт/км.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично plotnost_1."},
    "traffic_intensiv_priv_1_poputn":  {"desc": "Интенсивность приведённого потока, полоса 1, попутн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Расчётная метрика дорожного моделирования."},
    "traffic_intensiv_priv_2_poputn":  {"desc": "Интенсивность приведённого потока, полоса 2, попутн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично intensiv_priv_1."},
    "traffic_intensiv_priv_1_wstrechn":{"desc": "Интенсивность приведённого потока, полоса 1, встречн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично intensiv_priv_1."},
    "traffic_intensiv_priv_2_wstrechn":{"desc": "Интенсивность приведённого потока, полоса 2, встречн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично intensiv_priv_1."},
    "traffic_intensiv_fiz_1_poputn":   {"desc": "Интенсивность физического потока, полоса 1, попутн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Расчётная метрика."},
    "traffic_intensiv_fiz_2_poputn":   {"desc": "Интенсивность физического потока, полоса 2, попутн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично intensiv_fiz_1."},
    "traffic_intensiv_fiz_1_wstrechn": {"desc": "Интенсивность физического потока, полоса 1, встречн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично intensiv_fiz_1."},
    "traffic_intensiv_fiz_2_wstrechn": {"desc": "Интенсивность физического потока, полоса 2, встречн., авт/ч.", "preprocessing": "Z-score per-station.", "why_unknown": "Аналогично intensiv_fiz_1."},
}

# ── Целевые переменные (= observed past для энкодера) ─────────
TARGETS = {
    "sales_AI92": {
        "desc": "Продажи АИ-92, л/день.",
        "preprocessing": (
            "log1p (устраняет правосторонний скос и нули). "
            "TorchNormalizer(method='robust') внутри TFT — нормализует цели отдельно. "
            "Z-score в eda_preprocessing.py НЕ применяется. "
            "Оригинал → sales_AI92_orig для обратного преобразования."
        ),
    },
    "sales_AI95":    {"desc": "Продажи АИ-95, л/день.",     "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_DT":      {"desc": "Продажи ДТ, л/день.",        "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_DT_bio":  {"desc": "Продажи ДТ Bio, л/день.",    "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "sales_AI100_bio":{"desc": "Продажи АИ-100 Bio, л/день.", "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_напитки_безалкогольные":    {"desc": "Выручка: безалкогольные напитки, руб/день.",            "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_кондитерка_снеки":          {"desc": "Выручка: кондитерские изделия и снеки, руб/день.",      "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_мороженое":                 {"desc": "Выручка: мороженое, руб/день.",                         "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_автотовары":                {"desc": "Выручка: автотовары, руб/день.",                        "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_кафе_вся_еда":              {"desc": "Выручка: кафе (вся еда), руб/день.",                    "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_кофе_все_горячие_напитки":  {"desc": "Выручка: кофе и все горячие напитки, руб/день.",        "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
    "shop_табак":                     {"desc": "Выручка: табак, руб/день.",                             "preprocessing": "log1p + TorchNormalizer(robust). _orig сохраняется."},
}

# ══════════════════════════════════════════════════════════════
# ПОДСЧЁТ РАЗМЕРНОСТЕЙ
# ══════════════════════════════════════════════════════════════

n_tech   = len(TECHNICAL)
n_excl   = len(EXCLUDED)
n_sc     = len(STATIC_CATS)
n_sr     = len(STATIC_REALS)
n_kc     = len(KNOWN_CATS)
n_kr     = len(KNOWN_REALS)
n_unkr   = len(UNKNOWN_REALS)
n_tgt    = len(TARGETS)

# Производные колонки, добавляемые в препроцессинге
n_cyclical_extra = 6    # sin/cos: day_of_week, week_of_year, month × 2 (час убран — суточные данные)
n_enc_extra      = n_sc + n_kc   # _enc-колонки от LabelEncoder (STATIC_CATS + KNOWN_CATS)
n_orig_extra     = n_tgt          # _orig (цели до log1p)
n_time_idx       = 1              # time_idx (монотонный счётчик per-station)

# ══════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ОТЧЁТА
# ══════════════════════════════════════════════════════════════

def stat_row(col: str) -> dict:
    """Возвращает словарь базовых статистик по колонке df."""
    if col not in df.columns:
        return {"dtype": "—", "n_uniq": "—", "n_null": "—", "range": "отсутствует в merged_data"}
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
    f"**Период:** {df['date'].min().date()} — {df['date'].max().date()}  ",
    f"**Станций:** {df['station_id'].nunique()} · Дней/станцию: {df.groupby('station_id').size().iloc[0]}",
    "",
]

# ── Сценарный анализ (what-if) ───────────────────────────────
lines += [
    "## Сценарный анализ: что можно задавать в прогнозе",
    "",
    "Управлять прогнозом можно только через **time-varying known** переменные —",
    "те, значения которых известны заранее (или задаются вручную для сценария).",
    "",
    "| Что задаёт пользователь | Переменные |",
    "|---|---|",
    "| Прогноз погоды | temperature, precipitation_mm, weather_condition (0/1) |",
    "| Ожидаемый трафик (счётчики ТС) | traffic_*_1/2_poputn / traffic_*_1/2_wstrechn (24 колонки) |",
    "| Мониторинг конкурентов | competitor_price_AI92/AI95/DT + _brend + AI100 (7 колонок) |",
    "| Акции | promotion_fuel/shop/cafe_active |",
    "| Ценовая политика | price_AI92/AI95/DT/DT_bio/AI100_bio |",
    "| Календарь | дата → day_of_week, month, week_of_year, is_weekend, is_holiday, holiday_name |",
    "",
    "> **Нельзя задать в сценарии — time-varying unknown:** производные метрики трафика",
    "> (скорость, плотность, интенсивность) и клиентские метрики (лояльность, доля",
    "> корпоративных) — наблюдаемые данные прошлого, для горизонта прогноза недоступны.",
    "",
    "> **Нельзя задать в сценарии — статические:** характеристики АЗС (тип дороги,",
    "> инфраструктура) постоянны — для анализа влияния локации сравниваются прогнозы",
    "> разных станций.",
    "",
]

# ── Сводная таблица ───────────────────────────────────────────
n_total_raw   = n_tech + n_excl + n_sc + n_sr + n_kc + n_kr + n_unkr + n_tgt
n_model_vars  = n_sc + n_sr + n_kc + (n_kr + n_cyclical_extra) + n_unkr + n_tgt
n_prepared    = df.shape[1] - n_excl + n_enc_extra + n_cyclical_extra + n_time_idx + n_orig_extra
# Служебные в prepared_data, не являющиеся модельными признаками:
# 2 технических (date, station_id) + 1 time_idx
# + оригиналы категорий до _enc (n_sc STATIC_CATS + n_kc KNOWN_CATS = n_enc_extra оригиналов)
# + n_orig_extra _orig (до log1p)
n_aux = n_tech + n_time_idx + n_enc_extra + n_orig_extra   # 2+1+7+12 = 22

lines += [
    "## Сводная таблица",
    "",
    "| Категория | Колонок в merged_data.csv | Модельных признаков TFT |",
    "|---|---|---|",
    f"| Технические (date, station_id) | {n_tech} | — (индексация, не признаки) |",
    f"| **Исключены** (избыточные) | {n_excl} | ❌ удалены |",
    f"| STATIC_CATS | {n_sc} | ✅ {n_sc} (→ _enc, эмбеддинг TFT) |",
    f"| STATIC_REALS | {n_sr} | ✅ {n_sr} (без нормализации) |",
    f"| TIME_VARYING_KNOWN_CATS | {n_kc} | ✅ {n_kc} (→ _enc) |",
    f"| TIME_VARYING_KNOWN_REALS | {n_kr} из CSV + 6 sin/cos в препроц. | ✅ {n_kr + n_cyclical_extra} (47+6 sin/cos) |",
    f"| TIME_VARYING_UNKNOWN_REALS | {n_unkr} | ✅ {n_unkr} (Z-score) |",
    f"| TARGET | {n_tgt} | ✅ {n_tgt} (→ log1p) |",
    f"| **Итого в merged_data.csv** | **{n_total_raw}** | |",
    f"| — из них: модельных переменных TFT | — | **{n_model_vars}** |",
    "",
    f"> **Итого merged\\_data.csv = {n_total_raw}**: "
    f"2(tech) + 8(excl) + {n_sc}(SC) + {n_sr}(SR) + {n_kc}(KC) + {n_kr}(KR) + {n_unkr}(UR) + {n_tgt}(TGT) = **{n_total_raw}** ✓",
    f">",
    f"> **Модельных переменных TFT = {n_model_vars}**: "
    f"{n_sc}(SC\\_enc) + {n_sr}(SR) + {n_kc}(KC\\_enc) + {n_kr+n_cyclical_extra}(KR) + {n_unkr}(UR) + {n_tgt}(TGT) = **{n_model_vars}**",
    f">",
    f"> **Колонок в prepared\\_data.csv = {n_prepared}** (≠ {n_model_vars}!): "
    f"это НЕ количество признаков модели, а полное число столбцов файла. "
    f"Включает ещё {n_aux} служебных: 2 технических (date, station\\_id), "
    f"1 time\\_idx, {n_enc_extra} оригиналов до \\_enc, {n_orig_extra} \\_orig (до log1p).",
    "",
    "### Состав prepared_data.csv после препроцессинга",
    "",
    "```",
    f"  {df.shape[1]}  колонок в merged_data.csv",
    f"                 = 2(tech) + 8(excl) + 5(SC) + 21(SR) + 2(KC) + 47(KR) + 18(UR) + 12(TGT)",
    f"−  {n_excl}  исключённых ({', '.join(list(EXCLUDED.keys())[:4])}, ...)",
    f"─────",
    f"  {df.shape[1] - n_excl}  после исключения (2 технических + 105 модельно-релевантных)",
    f"+  {n_enc_extra}  _enc (LabelEncoder: {n_sc} STATIC_CATS + {n_kc} KNOWN_CATS; оригиналы сохраняются рядом)",
    f"+  {n_cyclical_extra}  _sin/_cos (day_of_week, week_of_year, month — час убран: суточные данные)",
    f"+  {n_time_idx}  time_idx (монотонный счётчик дней per-station, для TimeSeriesDataSet)",
    f"+ {n_orig_extra}  _orig (оригинальные значения {n_orig_extra} целевых до log1p, для expm1 при инференсе)",
    f"─────",
    f" {n_prepared}  колонок итого в prepared_data.csv",
    f"",
    f"  Из них TFT использует как признаки: {n_model_vars}",
    f"  Служебные (tech + time_idx + _enc-originals + _orig): {n_aux}",
    "```",
    "",
]


def section_header(title: str, tft_name: str, description: str) -> list:
    return [f"## {title} (`{tft_name}`)", "", description, ""]


def table_rows(d: dict, extra_col: str) -> list:
    header = f"| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | {extra_col} |"
    sep    =  "|---|---|---|---|---|---|"
    rows   = [header, sep]
    for col, meta in d.items():
        s   = stat_row(col)
        val = meta.get(extra_col.lower().strip(), meta.get("desc", meta.get("reason", "")))
        val = str(val).replace("\n", " ")
        rows.append(
            f"| `{col}` | {s['dtype']} | {s['n_uniq']} | {s['n_null']} "
            f"| {s['range']} | {val} |"
        )
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
        "Переменные, дублирующие информацию других колонок. "
        "Оставлены в `merged_data.csv` для EDA, **не подаются в TFT**."
    ),
)
lines += table_rows(EXCLUDED, "Reason")
lines.append("")

# ── 3. STATIC_CATS ───────────────────────────────────────────
lines += section_header(
    "Статические категориальные", "static_categoricals",
    (
        "Неизменные характеристики АЗС. Предобработка: LabelEncoder → целочисленный код.\n\n"
        "**⚠️ Важно: `settlement_size` и `distance_to_city_km` — коды категорий, "
        "не числовые значения.** Их нельзя нормализовать по Z-score — "
        "арифметика на кодах бессмысленна (расстояние между кодами 0 и 1 ≠ 20 км). "
        "Оба поля переведены в категориальный тип и кодируются через LabelEncoder."
    ),
)
lines += table_rows(STATIC_CATS, "Encoding")
lines.append("")

# ── 4. STATIC_REALS ──────────────────────────────────────────
lines += section_header(
    "Статические вещественные", "static_reals",
    (
        "Числовые паспортные данные АЗС. Не меняются во времени. "
        "**Не нормализуются** (исключены из Z-score): паспортные значения постоянны "
        "per-station, нормализация не добавляет информации."
    ),
)
lines += table_rows(STATIC_REALS, "Desc")
lines.append("")

# ── 5. KNOWN_CATS ─────────────────────────────────────────────
lines += section_header(
    "Известные будущие категориальные", "time_varying_known_categoricals",
    (
        "Категориальные переменные, известные заранее — доступны на горизонте прогноза. "
        "Предобработка: `LabelEncoder` → целочисленный код."
    ),
)
lines += table_rows(KNOWN_CATS, "What-if")
lines.append("")

# ── 6. KNOWN_REALS ────────────────────────────────────────────
lines += section_header(
    "Известные будущие вещественные", "time_varying_known_reals",
    (
        "Числовые переменные, известные заранее — доступны на горизонте прогноза. "
        "**Единственный класс, значения которого пользователь может задавать** "
        "для сценарного анализа.\n\n"
        f"Колонок из CSV: {n_kr}. "
        f"После циклического кодирования (day_of_week, week_of_year, month) — итого **{n_kr + n_cyclical_extra}**."
    ),
)
lines += table_rows(KNOWN_REALS, "What-if")
lines.append("")

# ── 7. UNKNOWN_REALS ──────────────────────────────────────────
lines += section_header(
    "Наблюдаемые прошлые вещественные", "time_varying_unknown_reals",
    (
        "Числовые переменные, **доступные только за прошлые периоды** — "
        "на горизонте прогноза не известны, задать в сценарии нельзя.\n\n"
        "Состав: клиентские метрики (`corporate_customer_ratio`, `customer_loyalty_score`) "
        "и 16 производных метрик трафика (скорость, плотность, интенсивность × 4 направления)."
    ),
)
lines += table_rows(UNKNOWN_REALS, "Why_unknown")
lines.append("")

# ── 8. TARGETS ───────────────────────────────────────────────
lines += section_header(
    "Целевые переменные", "targets",
    (
        f"{n_tgt} прогнозируемых переменных. Также используются как наблюдаемые значения "
        "прошлого при построении контекста.\n\n"
        "5 видов топлива (AI92 / AI95 / DT / DT_bio / AI100_bio) + "
        "7 категорий магазина (безалкогольные напитки, кондитерка_снеки, мороженое, "
        "автотовары, кафе_вся_еда, кофе_все_горячие_напитки, табак).\n\n"
        "Предобработка: `log1p` — логарифмирование устраняет правосторонний скос и "
        "стабилизирует нулевые значения. Оригинал сохраняется в `_orig` для "
        "обратного преобразования (`expm1`)."
    ),
)
lines += table_rows(TARGETS, "Desc")
lines.append("")

# ── 9. Циклическое кодирование ────────────────────────────────
lines += [
    "## Циклическое кодирование временных признаков",
    "",
    "**Проблема:** день_недели=6 (Вс) и день_недели=0 (Пн) различаются на 6,",
    "хотя физически это соседние дни. То же для перехода Декабрь→Январь.",
    "",
    "**Решение:** проекция на единичную окружность:",
    "",
    "```",
    "day_of_week_sin = sin(2π · day_of_week / 7)",
    "day_of_week_cos = cos(2π · day_of_week / 7)",
    "```",
    "",
    "Данные суточные — кодируются три признака: `day_of_week`, `week_of_year`, `month`.",
    "",
    "| Признак | Период | Добавляемые колонки |",
    "|---|---|---|",
    "| `day_of_week` | 7 | `day_of_week_sin`, `day_of_week_cos` |",
    "| `week_of_year` | 52 | `week_of_year_sin`, `week_of_year_cos` |",
    "| `month` | 12 | `month_sin`, `month_cos` |",
    "",
    f"Итого: +{n_cyclical_extra} новых колонок в prepared_data.csv.",
    "",
]

# ── 10. Итоговый план предобработки ──────────────────────────
lines += [
    "## Итоговый план предобработки",
    "",
    "| Шаг | Что делаем | На каких колонках |",
    "|---|---|---|",
    f"| 1. Исключить | Удалить {n_excl} избыточных колонок | station_name, total_pumps, total_fuel_sales, quarter, traffic_uroven_jbslugi_* (×4) |",
    "| 2. Пропуски | `holiday_name → 'нет_праздника'` | holiday_name |",
    f"| 3. Label Encoding | → _enc-колонки | STATIC_CATS ({n_sc}): road_type, road_level, direction, settlement_size, distance_to_city_km; KNOWN_CATS ({n_kc}): season, holiday_name |",
    "| 4. Циклическое кодирование | → _sin/_cos (+6 новых) | day_of_week (7), week_of_year (52), month (12) |",
    "| 5. time_idx | Монотонный счётчик дней per-station через cumcount | Все станции |",
    "| 6. log1p | log(1+x). Оригинал → _orig | Все 12 TARGET_COLS |",
    f"| 7. Z-score | Нормализация per-station: μ и σ → tft/scalers.pkl | KNOWN_REALS (кроме бинарных) + UNKNOWN_REALS (кроме бинарных) |",
    "",
    "### Что НЕ нормализуется (исключено из Z-score)",
    "",
    "| Группа | Причина |",
    "|---|---|",
    "| STATIC_CATS (_enc) | Категориальные коды → embedding, не числа |",
    "| STATIC_REALS | Паспортные константы, нет временно́й вариации |",
    "| Бинарные признаки (0/1) | Масштаб [0,1] уже нормализован |",
    "| TARGET_COLS | Нормализуются отдельно на уровне модели (после log1p) |",
    "| _enc и _orig колонки | Служебные |",
    "",
    "> **Примечание о price_\\*:** цены меняются ежедневно (std > 0), поэтому нормализуются по Z-score.",
    "",
]

# ── 11. Итог ─────────────────────────────────────────────────
# n_prepared уже вычислен выше в секции "Сводная таблица"

lines += [
    "## Итог: состав переменных",
    "",
    "| Категория TFT | Признаков в merged_data.csv | Признаков TFT (в модели) | Преобразование |",
    "|---|---|---|---|",
    f"| static_categoricals | {n_sc} | {n_sc} | LabelEncoder → _enc |",
    f"| static_reals | {n_sr} | {n_sr} | без нормализации |",
    f"| time_varying_known_categoricals | {n_kc} | {n_kc} | LabelEncoder → _enc |",
    f"| time_varying_known_reals | {n_kr} из CSV | {n_kr + n_cyclical_extra} (+6 sin/cos) | Z-score (кроме бинарных) |",
    f"| time_varying_unknown_reals | {n_unkr} | {n_unkr} | Z-score |",
    f"| targets | {n_tgt} | {n_tgt} | log1p, _orig сохраняется |",
    f"| **Итого (только TFT-признаки)** | **{n_model_vars - n_cyclical_extra}** из CSV | **{n_model_vars}** в модели | |",
    "",
    "> ⚠️ Не путайте три числа:",
    f"> - **{n_total_raw}** — колонок в `merged_data.csv` (включая 2 tech + 8 excl)",
    f"> - **{n_model_vars}** — признаков, которые TFT получает как входы",
    f"> - **{n_prepared}** — колонок в `prepared_data.csv` = {n_model_vars} модельных + {n_aux} служебных (tech/time_idx/_enc-originals/_orig)",
    "",
]

# ── 13. Заметки о данных ─────────────────────────────────────
# Определяем числовые колонки с одним уникальным значением — для информации
already_excluded = set(EXCLUDED.keys())
numeric_cols = df.select_dtypes(include="number").columns.tolist()
const_cols = [
    c for c in numeric_cols
    if df[c].nunique() == 1 and c not in already_excluded
]

lines += [
    "## Примечания о текущем датасете",
    "",
    f"В текущем наборе данных (5 АЗС, 2025 г.) **{len(const_cols)}** числовых колонок "
    "принимают одно уникальное значение (константа). Причины: одностороннее движение "
    "около АЗС (все встречные колонки = 0), отсутствие брендовых конкурентов, "
    "одинаковая базовая инфраструктура.",
    "",
    "Колонки **остаются в своих категориях** — их константность отражает реальное состояние "
    "сети, а не ошибку данных. При Z-score нормализации `std = 0` будет обработано "
    "в `eda_preprocessing.py` (пропуск нормализации для константных числовых признаков).",
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
print(f"  merged_data.csv : {df.shape[1]} колонок")
print(f"    = 2(tech) + {n_excl}(excl) + {n_sc}(SC) + {n_sr}(SR) + {n_kc}(KC) + {n_kr}(KR) + {n_unkr}(UR) + {n_tgt}(TGT)")
print()
print(f"  TFT-признаков   : {n_model_vars}")
print(f"    = {n_sc}(SC_enc) + {n_sr}(SR) + {n_kc}(KC_enc) + {n_kr+n_cyclical_extra}(KR) + {n_unkr}(UR) + {n_tgt}(TGT)")
print()
print(f"  prepared_data.csv: {n_prepared} колонок = {n_model_vars} модельных + {n_aux} служебных")
print(f"    {df.shape[1]} - {n_excl}(excl) + {n_enc_extra}(_enc) + {n_cyclical_extra}(sin/cos) + {n_time_idx}(time_idx) + {n_orig_extra}(_orig) = {n_prepared}")
print("=" * 60)
