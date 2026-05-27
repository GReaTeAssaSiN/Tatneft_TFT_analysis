"""
Генерация отчёта с описанием колонок merged_data.csv.
Запускать из корня проекта: python report_generating/column_descriptions.py
Результат: reports/merged_data_columns.txt
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_utils import SRC_STATIC, SRC_TEMPORAL

# ── Описания колонок ──────────────────────────────────────────
DESCRIPTIONS = {
    # ── Идентификаторы и время ────────────────────────────────
    "date":                                "Дата наблюдения (ежедневно, 2025 г.)",
    "station_id":                          "ID станции",
    "station_name":                        "Название станции",

    # ── Временные признаки ────────────────────────────────────
    "day_of_week":                         "День недели (0=Пн ... 6=Вс)",
    "week_of_year":                        "Номер недели в году (1–52)",
    "month":                               "Месяц (1–12)",
    "quarter":                             "Квартал (1–4)",
    "season":                              "Сезон (winter / spring / summer / autumn)",
    "is_weekend":                          "Выходной день (0/1)",
    "is_holiday":                          "Праздничный день (0/1)",
    "holiday_name":                        "Название праздника или «нет_праздника»",

    # ── Погода ────────────────────────────────────────────────
    "temperature":                         "Температура воздуха, °C",
    "weather_condition":                   "Тип погоды (числовой код)",
    "precipitation_mm":                    "Осадки, мм",

    # ── Цены топлива (текущие) ────────────────────────────────
    "price_AI92":                          "Текущая цена АИ-92, руб/л",
    "price_AI95":                          "Текущая цена АИ-95, руб/л",
    "price_DT":                            "Текущая цена ДТ, руб/л",
    "price_DT_bio":                        "Текущая цена ДТ Bio, руб/л",
    "price_AI100_bio":                     "Текущая цена АИ-100 Bio, руб/л",

    # ── Цены конкурентов ──────────────────────────────────────
    "competitor_price_AI92":               "Цена АИ-92 у небрендовых конкурентов, руб/л",
    "competitor_price_AI95":               "Цена АИ-95 у небрендовых конкурентов, руб/л",
    "competitor_price_DT":                 "Цена ДТ у небрендовых конкурентов, руб/л",
    "competitor_price_AI92_brend":         "Цена АИ-92 у брендовых конкурентов, руб/л",
    "competitor_price_AI95_brend":         "Цена АИ-95 у брендовых конкурентов, руб/л",
    "competitor_price_DT_brend":           "Цена ДТ у брендовых конкурентов, руб/л",
    "competitor_price_AI100":              "Цена АИ-100 у конкурентов, руб/л",

    # ── Продажи топлива (целевые) ─────────────────────────────
    "sales_AI92":                          "Продажи АИ-92, л/день",
    "sales_AI95":                          "Продажи АИ-95, л/день",
    "sales_DT":                            "Продажи ДТ, л/день",
    "sales_DT_bio":                        "Продажи ДТ Bio, л/день",
    "sales_AI100_bio":                     "Продажи АИ-100 Bio, л/день",
    "total_fuel_sales":                    "Суммарные продажи топлива, л/день",

    # ── Акции ─────────────────────────────────────────────────
    "promotion_fuel_active":               "Акция на топливо активна (0/1)",
    "promotion_shop_active":               "Акция в магазине активна (0/1)",
    "promotion_cafe_active":               "Акция в кафе активна (0/1)",

    # ── Клиентские метрики (из temporal) ─────────────────────
    "corporate_customer_ratio":            "Доля корпоративных клиентов",
    "customer_loyalty_score":              "Оценка лояльности клиентов",

    # ── Продажи магазина (целевые) ────────────────────────────
    "shop_напитки_безалкогольные":         "Выручка: безалкогольные напитки, руб/день",
    "shop_кондитерка_снеки":               "Выручка: кондитерские изделия и снеки, руб/день",
    "shop_мороженое":                      "Выручка: мороженое, руб/день",
    "shop_автотовары":                     "Выручка: автотовары, руб/день",
    "shop_кафе_вся_еда":                   "Выручка: кафе (вся еда), руб/день",
    "shop_кофе_все_горячие_напитки":       "Выручка: кофе и горячие напитки, руб/день",
    "shop_табак":                          "Выручка: табак, руб/день",

    # ── Трафик — попутное направление, полоса 1 ───────────────
    "traffic_Passengers_cars_1_poputn":    "Легковые авто, полоса 1, попутное направление, авт/день",
    "traffic_Truck_short_1_poputn":        "Малотоннажные грузовики, полоса 1, попутн., авт/день",
    "traffic_Truck_1_poputn":              "Грузовики среднего класса, полоса 1, попутн., авт/день",
    "traffic_Truck_long_1_poputn":         "Большегрузные авто, полоса 1, попутн., авт/день",
    "traffic_Transporter_1_poputn":        "Микроавтобусы/транспортёры, полоса 1, попутн., авт/день",
    "traffic_Undefined_1_poputn":          "Неопределённый тип ТС, полоса 1, попутн., авт/день",
    "traffic_scorost_1_poputn":            "Средняя скорость, полоса 1, попутн., км/ч",
    "traffic_plotnost_1_poputn":           "Плотность потока, полоса 1, попутн., авт/км",
    "traffic_intensiv_priv_1_poputn":      "Интенсивность приведённого потока, полоса 1, попутн., авт/ч",
    "traffic_intensiv_fiz_1_poputn":       "Интенсивность физического потока, полоса 1, попутн., авт/ч",
    "traffic_uroven_jbslugi_1_poputn":     "Уровень обслуживания дороги, полоса 1, попутн. (A–F)",

    # ── Трафик — попутное направление, полоса 2 ───────────────
    "traffic_Passengers_cars_2_poputn":    "Легковые авто, полоса 2, попутное направление, авт/день",
    "traffic_Truck_short_2_poputn":        "Малотоннажные грузовики, полоса 2, попутн., авт/день",
    "traffic_Truck_2_poputn":              "Грузовики среднего класса, полоса 2, попутн., авт/день",
    "traffic_Truck_long_2_poputn":         "Большегрузные авто, полоса 2, попутн., авт/день",
    "traffic_Transporter_2_poputn":        "Микроавтобусы/транспортёры, полоса 2, попутн., авт/день",
    "traffic_Undefined_2_poputn":          "Неопределённый тип ТС, полоса 2, попутн., авт/день",
    "traffic_scorost_2_poputn":            "Средняя скорость, полоса 2, попутн., км/ч",
    "traffic_plotnost_2_poputn":           "Плотность потока, полоса 2, попутн., авт/км",
    "traffic_intensiv_priv_2_poputn":      "Интенсивность приведённого потока, полоса 2, попутн., авт/ч",
    "traffic_intensiv_fiz_2_poputn":       "Интенсивность физического потока, полоса 2, попутн., авт/ч",
    "traffic_uroven_jbslugi_2_poputn":     "Уровень обслуживания дороги, полоса 2, попутн. (A–F)",

    # ── Трафик — встречное направление, полоса 1 ──────────────
    "traffic_Passengers_cars_1_wstrechn":  "Легковые авто, полоса 1, встречное направление, авт/день",
    "traffic_Truck_short_1_wstrechn":      "Малотоннажные грузовики, полоса 1, встречн., авт/день",
    "traffic_Truck_1_wstrechn":            "Грузовики среднего класса, полоса 1, встречн., авт/день",
    "traffic_Truck_long_1_wstrechn":       "Большегрузные авто, полоса 1, встречн., авт/день",
    "traffic_Transporter_1_wstrechn":      "Микроавтобусы/транспортёры, полоса 1, встречн., авт/день",
    "traffic_Undefined_1_wstrechn":        "Неопределённый тип ТС, полоса 1, встречн., авт/день",
    "traffic_scorost_1_wstrechn":          "Средняя скорость, полоса 1, встречн., км/ч",
    "traffic_plotnost_1_wstrechn":         "Плотность потока, полоса 1, встречн., авт/км",
    "traffic_intensiv_priv_1_wstrechn":    "Интенсивность приведённого потока, полоса 1, встречн., авт/ч",
    "traffic_intensiv_fiz_1_wstrechn":     "Интенсивность физического потока, полоса 1, встречн., авт/ч",
    "traffic_uroven_jbslugi_1_wstrechn":   "Уровень обслуживания дороги, полоса 1, встречн. (A–F)",

    # ── Трафик — встречное направление, полоса 2 ──────────────
    "traffic_Passengers_cars_2_wstrechn":  "Легковые авто, полоса 2, встречное направление, авт/день",
    "traffic_Truck_short_2_wstrechn":      "Малотоннажные грузовики, полоса 2, встречн., авт/день",
    "traffic_Truck_2_wstrechn":            "Грузовики среднего класса, полоса 2, встречн., авт/день",
    "traffic_Truck_long_2_wstrechn":       "Большегрузные авто, полоса 2, встречн., авт/день",
    "traffic_Transporter_2_wstrechn":      "Микроавтобусы/транспортёры, полоса 2, встречн., авт/день",
    "traffic_Undefined_2_wstrechn":        "Неопределённый тип ТС, полоса 2, встречн., авт/день",
    "traffic_scorost_2_wstrechn":          "Средняя скорость, полоса 2, встречн., км/ч",
    "traffic_plotnost_2_wstrechn":         "Плотность потока, полоса 2, встречн., авт/км",
    "traffic_intensiv_priv_2_wstrechn":    "Интенсивность приведённого потока, полоса 2, встречн., авт/ч",
    "traffic_intensiv_fiz_2_wstrechn":     "Интенсивность физического потока, полоса 2, встречн., авт/ч",
    "traffic_uroven_jbslugi_2_wstrechn":   "Уровень обслуживания дороги, полоса 2, встречн. (A–F)",

    # ── Статические характеристики АЗС (из static) ───────────
    "road_type":                           "Тип дороги (федеральная / региональная / местная)",
    "road_level":                          "Уровень дороги (числовой код)",
    "direction":                           "Направление трассы",
    "settlement_size":                     "Размер ближайшего населённого пункта (категория)",
    "distance_to_city_km":                 "Расстояние до ближайшего города, км",
    "num_pumps_AI92":                      "Число колонок АИ-92",
    "num_pumps_AI92_bio":                  "Число колонок АИ-92 Bio",
    "num_pumps_AI95":                      "Число колонок АИ-95",
    "num_pumps_AI95_bio":                  "Число колонок АИ-95 Bio",
    "num_pumps_AI100_bio":                 "Число колонок АИ-100 Bio",
    "num_pumps_DT":                        "Число колонок ДТ",
    "num_pumps_DT_bio":                    "Число колонок ДТ Bio",
    "num_pumps_SUG":                       "Число колонок СУГ (сжиженный углеводородный газ)",
    "num_pumps_KPG":                       "Число колонок КПГ (компримированный природный газ)",
    "num_pumps_SPG":                       "Число колонок СПГ (сжиженный природный газ)",
    "total_pumps":                         "Общее число активных колонок",
    "has_shop":                            "Есть магазин (0/1)",
    "shop_area_m2":                        "Площадь магазина, м²",
    "has_car_wash":                        "Есть автомойка (0/1)",
    "has_tire_service":                    "Есть шиномонтаж (0/1)",
    "has_cafe":                            "Есть кафе (0/1)",
    "has_hotel":                           "Есть гостиница (0/1)",
    "has_shop_молельная_комната":          "Есть молельная комната (0/1)",
    "has_shop_прачечная":                  "Есть прачечная (0/1)",
    "has_shop_электрозарядная_станция":    "Есть электрозарядная станция (0/1)",
    "has_shop_подкачка_шин":               "Есть подкачка шин (0/1)",
    "competitors_wink":                    "Число конкурентов поблизости (данные Wink)",
}


def sample_values(series, n=3):
    """Возвращает строку с n примерами уникальных значений."""
    vals = series.dropna().unique()
    samples = [str(v) for v in vals[:n]]
    result = ", ".join(samples)
    if len(vals) > n:
        result += ", ..."
    return result


def _clip(text: str, width: int) -> str:
    """Усекает строку до width символов, добавляя '...' если строка длиннее.

    Гарантирует фиксированную ширину колонки в отчёте независимо от
    длины содержимого (f-string :<N только добавляет пробелы, не усекает).
    """
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


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
            f"  Скрипт объединяет {SRC_STATIC}\n"
            f"  и {SRC_TEMPORAL} -> data/merged_data.csv",
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

    # Определяем размерности из файла
    full_df = pd.read_csv(src, usecols=[0])
    n_rows = len(full_df)

    # col_w  — под самое длинное имя колонки (~34 ASCII-символа для traffic_*)
    # desc_w — под самое длинное описание (~59 символов для интенсивности трафика)
    # ex_w   — под примеры значений
    col_w   = 42
    desc_w  = 62
    ex_w    = 40
    total_w = 5 + col_w + desc_w + ex_w

    lines = []
    lines.append(
        f"Описание колонок файла merged_data.csv "
        f"({n_rows} x {len(df.columns)})"
    )
    lines.append(
        "Источник: gas_stations_temporal_daily_2025.csv "
        "LEFT JOIN gas_stations_static.csv"
    )
    lines.append("=" * total_w)
    lines.append(
        f"{'№':<5}"
        f"{'Колонка':<{col_w}}"
        f"{'Описание':<{desc_w}}"
        f"{'Примеры значений':<{ex_w}}"
    )
    lines.append("-" * total_w)

    for i, col in enumerate(df.columns, 1):
        desc     = _clip(DESCRIPTIONS.get(col, "—"), desc_w)
        examples = _clip(sample_values(df[col]),     ex_w)
        lines.append(
            f"{i:<5}"
            f"{col:<{col_w}}"
            f"{desc:<{desc_w}}"
            f"{examples}"
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
