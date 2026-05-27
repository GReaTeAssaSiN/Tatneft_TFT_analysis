"""
Генерация отчёта с описанием колонок merged_data.csv.
Запускать из корня проекта: python report_generating/column_descriptions.py
Результат: reports/merged_data_columns.txt
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Описания колонок ──────────────────────────────────────────
DESCRIPTIONS = {
    # Идентификаторы и время
    "timestamp":                  "Метка времени (почасовая, 2023 г.)",
    "station_id":                 "ID станции (0–4)",
    "station_name":               "Название станции",

    # Характеристики АЗС (статические)
    "road_type":                  "Тип дороги (трасса / региональная)",
    "direction":                  "Направление трассы",
    "settlement_size":            "Размер ближайшего населённого пункта",
    "distance_to_city_km":        "Расстояние до города, км",
    "total_pumps":                "Общее число колонок",
    "shop_area_m2":               "Площадь магазина, м²",
    "has_car_wash":               "Есть автомойка (0/1)",
    "has_cafe":                   "Есть кафе (0/1)",
    "has_shop":                   "Есть магазин (0/1)",
    "has_tire_service":           "Есть шиномонтаж (0/1)",
    "has_hotel":                  "Есть гостиница (0/1)",
    "competitors_within_5km":     "Число конкурентов в радиусе 5 км",
    "corporate_customer_ratio":   "Доля корпоративных клиентов",
    "staff_engagement_score":     "Оценка вовлечённости персонала",
    "staff_quality_score":        "Оценка качества работы персонала",
    "customer_loyalty_score":     "Оценка лояльности клиентов",
    "num_pumps_AI92":             "Число колонок АИ-92",
    "num_pumps_AI95":             "Число колонок АИ-95",
    "num_pumps_AI98":             "Число колонок АИ-98",
    "num_pumps_DT_EURO":          "Число колонок ДТ Евро+",
    "num_pumps_DT_TANEKO":        "Число колонок ДТ ТАНЕКО",
    "num_pumps_DT_SUMMER":        "Число колонок ДТ Летнее",
    "num_pumps_DT_WINTER":        "Число колонок ДТ Зимнее",
    "base_price_AI92":            "Базовая цена АИ-92 станции, руб/л",
    "base_price_AI95":            "Базовая цена АИ-95 станции, руб/л",
    "base_price_AI98":            "Базовая цена АИ-98 станции, руб/л",
    "base_price_DT_EURO":         "Базовая цена ДТ Евро+ станции, руб/л",
    "base_price_DT_TANEKO":       "Базовая цена ДТ ТАНЕКО станции, руб/л",
    "base_price_DT_SUMMER":       "Базовая цена ДТ Летнее станции, руб/л",
    "base_price_DT_WINTER":       "Базовая цена ДТ Зимнее станции, руб/л",

    # Погода
    "temperature":                "Температура воздуха, °C",
    "weather_condition":          "Тип погоды (ясно / облачно / снег и т.д.)",
    "precipitation_mm":           "Осадки, мм",
    "visibility_km":              "Видимость, км",
    "wind_speed_ms":              "Скорость ветра, м/с",
    "is_snow":                    "Идёт снег (0/1)",
    "is_rain":                    "Идёт дождь (0/1)",
    "is_fog":                     "Туман (0/1)",

    # Трафик
    "traffic_Passengers_cars":    "Поток легковых автомобилей",
    "traffic_Truck_short":        "Поток малотоннажных грузовиков",
    "traffic_Truck":              "Поток грузовиков среднего класса",
    "traffic_Truck_long":         "Поток большегрузных автомобилей",
    "traffic_Transporter":        "Поток микроавтобусов/транспортёров",
    "traffic_Undefined":          "Неопределённый тип ТС",
    "total_traffic":              "Суммарный поток всех ТС",

    # Продажи топлива (целевые)
    "sales_AI92":                 "Продажи АИ-92, л/ч",
    "sales_AI95":                 "Продажи АИ-95, л/ч",
    "sales_AI98":                 "Продажи АИ-98, л/ч",
    "sales_DT_EURO":              "Продажи ДТ Евро+, л/ч",
    "sales_DT_TANEKO":            "Продажи ДТ ТАНЕКО, л/ч",
    "sales_DT_SUMMER":            "Продажи ДТ Летнее, л/ч",
    "sales_DT_WINTER":            "Продажи ДТ Зимнее, л/ч",
    "total_fuel_sales":           "Суммарные продажи топлива, л/ч",

    # Продажи магазина (целевые)
    "shop_напитки":               "Выручка магазина: напитки, руб/ч",
    "shop_закуски":               "Выручка магазина: закуски, руб/ч",
    "shop_автотовары":            "Выручка магазина: автотовары, руб/ч",
    "shop_кофе":                  "Выручка магазина: кофе, руб/ч",
    "shop_табак":                 "Выручка магазина: табак, руб/ч",
    "shop_total_revenue":         "Суммарная выручка магазина, руб/ч",

    # Акции и реклама
    "promotion_fuel_active":      "Акция на топливо активна (0/1)",
    "promotion_shop_active":      "Акция в магазине активна (0/1)",
    "promotion_cafe_active":      "Акция в кафе активна (0/1)",
    "ad_active":                  "Реклама активна (0/1)",
    "ad_channel":                 "Канал рекламы (ТВ / радио / билборд и т.д.)",

    # Цены конкурентов
    "competitor_price_AI92":      "Цена АИ-92 у конкурентов, руб/л",
    "competitor_price_AI95":      "Цена АИ-95 у конкурентов, руб/л",
    "competitor_price_DT":        "Цена ДТ у конкурентов, руб/л",

    # Текущие цены топлива
    "price_AI92":                 "Текущая цена АИ-92, руб/л",
    "price_AI95":                 "Текущая цена АИ-95, руб/л",
    "price_AI98":                 "Текущая цена АИ-98, руб/л",
    "price_DT_EURO":              "Текущая цена ДТ Евро+, руб/л",
    "price_DT_TANEKO":            "Текущая цена ДТ ТАНЕКО, руб/л",
    "price_DT_SUMMER":            "Текущая цена ДТ Летнее, руб/л",
    "price_DT_WINTER":            "Текущая цена ДТ Зимнее, руб/л",

    # Временные признаки
    "hour":                       "Час суток (0–23)",
    "day_of_week":                "День недели (0=Пн … 6=Вс)",
    "week_of_year":               "Номер недели в году (1–52)",
    "month":                      "Месяц (1–12)",
    "quarter":                    "Квартал (1–4)",
    "season":                     "Сезон (весна / лето / осень / зима)",
    "is_weekend":                 "Выходной день (0/1)",
    "is_holiday":                 "Праздничный день (0/1)",
    "holiday_name":               'Название праздника или "нет_праздника"',
    "is_rush_hour":               "Час пик (0/1)",
    "is_night":                   "Ночное время (0/1)",
    "day_name":                   "Название дня недели",
}


def sample_values(series, n=3):
    """Возвращает строку с n примерами уникальных значений."""
    vals = series.dropna().unique()
    samples = [str(v) for v in vals[:n]]
    result = ", ".join(samples)
    if len(vals) > n:
        result += ", ..."
    return result


def _abort(reason: str, hint: str) -> None:
    """Выводит понятную диагностику и завершает работу без traceback."""
    print()
    print("  [ОШИБКА]", reason)
    print("  Что делать:", hint)
    print()
    sys.exit(1)


def main():
    src = "data/merged_data.csv"
    out = "reports/merged_data_columns.txt"

    print("=" * 60)
    print("Генерация описания колонок merged_data.csv")
    print("=" * 60)

    # ── Проверка наличия исходного файла ──────────────────────
    if not os.path.exists(src):
        _abort(
            f"Файл не найден: {src}",
            "Запустите из корня проекта:\n"
            "      python explore_data.py\n"
            "  Скрипт объединяет SourceDataForWork/5stations_data.csv\n"
            "  и SourceDataForWork/5stations_metadata.csv → data/merged_data.csv",
        )

    # ── Чтение CSV ────────────────────────────────────────────
    try:
        df = pd.read_csv(src, nrows=500)
    except Exception as exc:
        _abort(
            f"Не удалось прочитать {src}: {exc}",
            "Убедитесь, что файл не повреждён и не открыт другой программой.\n"
            "  При необходимости пересоздайте его: python explore_data.py",
        )

    if df.empty:
        _abort(
            f"Файл {src} пустой (0 строк).",
            "Пересоздайте файл: python explore_data.py",
        )

    print(f"  Загружено строк для примеров : 500")
    print(f"  Колонок всего               : {len(df.columns)}")

    col_w   = 30
    desc_w  = 42
    ex_w    = 45
    total_w = 5 + col_w + desc_w + ex_w

    lines = []
    lines.append("Описание колонок файла merged_data.csv (43 800 × 89)")
    lines.append("Источник: 5stations_data.csv JOIN 5stations_metadata.csv")
    lines.append("=" * total_w)
    lines.append(
        f"{'№':<5}"
        f"{'Колонка':<{col_w}}"
        f"{'Описание':<{desc_w}}"
        f"{'Примеры значений':<{ex_w}}"
    )
    lines.append("-" * total_w)

    for i, col in enumerate(df.columns, 1):
        desc     = DESCRIPTIONS.get(col, "—")
        examples = sample_values(df[col])
        lines.append(
            f"{i:<5}"
            f"{col:<{col_w}}"
            f"{desc:<{desc_w}}"
            f"{examples:<{ex_w}}"
        )

    lines.append("=" * total_w)
    lines.append(f"Всего колонок: {len(df.columns)}")

    # ── Запись результата ─────────────────────────────────────
    try:
        os.makedirs("reports", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as exc:
        _abort(
            f"Не удалось записать {out}: {exc}",
            "Проверьте права на запись в папку reports/.",
        )

    print(f"\n  Сохранено : {out}")
    print("=" * 60)
    print("Готово.")
    print("=" * 60)


if __name__ == "__main__":
    main()
