"""
Загрузка и объединение исходных CSV-файлов: паспортные данные АЗС + временные ряды.
Запускать из корня проекта: python explore_data.py
Результат: data/merged_data.csv
"""

import os
import sys

import pandas as pd

os.makedirs("data", exist_ok=True)

# ── Импорт утилит и констант ──────────────────────────────────
from utils.data_utils import (
    SRC_STATIC,
    SRC_TEMPORAL,
    FILL_MAP,
    load_and_merge,
    fill_missing,
)

print("=" * 60)
print("ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ИСХОДНЫХ ДАННЫХ")
print("=" * 60)

# ── 1. Проверка наличия файлов ────────────────────────────────
for path in (SRC_STATIC, SRC_TEMPORAL):
    if not os.path.exists(path):
        print(f"\n  [ОШИБКА] Файл не найден: {path}")
        print("  Убедитесь, что SourceDataForWork/ присутствует в корне проекта.")
        sys.exit(1)

# ── 2. Загрузка и объединение ─────────────────────────────────
print("\nЧтение и объединение файлов (кодировка: utf-8-sig)...")

static   = pd.read_csv(SRC_STATIC,   encoding="utf-8-sig")
temporal = pd.read_csv(SRC_TEMPORAL, encoding="utf-8-sig", parse_dates=["date"])

print(f"  gas_stations_static.csv              : {static.shape[0]} строк x {static.shape[1]} колонок")
print(f"  gas_stations_temporal_daily_2025.csv : {temporal.shape[0]} строк x {temporal.shape[1]} колонок")

df = load_and_merge(SRC_STATIC, SRC_TEMPORAL)

# ── 3. Паспортные данные АЗС ──────────────────────────────────
print("\n" + "=" * 60)
print("ПАСПОРТНЫЕ ДАННЫЕ АЗС (static)")
print("=" * 60)

print(f"  Станций : {len(static)}")
print(f"  Колонок : {static.shape[1]}")

pump_cols  = [c for c in static.columns if c.startswith("num_pumps_")]
svc_cols   = [c for c in static.columns if c.startswith("has_") and not c.startswith("has_shop_")]
extra_svc  = [c for c in static.columns if c.startswith("has_shop_")]

print(f"\n  Колонки типов топлива ({len(pump_cols)} шт.): {pump_cols}")
print(f"  Услуги АЗС ({len(svc_cols)} шт.)           : {svc_cols}")
print(f"  Доп. услуги магазина ({len(extra_svc)} шт.): {extra_svc}")

static_nulls = static.isnull().sum()
static_nulls = static_nulls[static_nulls > 0]
if static_nulls.empty:
    print("\n  Пропусков в static: нет")
else:
    print("\n  Пропуски в static:")
    for col, n in static_nulls.items():
        print(f"    {col:40s}: {n}")

# ── 4. Временные ряды ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ВРЕМЕННЫЕ РЯДЫ (temporal)")
print("=" * 60)

print(f"  Строк   : {len(temporal)}")
print(f"  Колонок : {temporal.shape[1]}")
print(f"  Период  : {temporal['date'].min().date()} -- {temporal['date'].max().date()}")
print(f"  Станций : {temporal['station_id'].nunique()}")
rows_per = temporal.groupby("station_id").size()
print(f"  Строк на станцию: {rows_per.iloc[0]} (min {rows_per.min()}, max {rows_per.max()})")

# Группы колонок
sales_fuel   = [c for c in temporal.columns if c.startswith("sales_")]
sales_shop   = [c for c in temporal.columns
                if c.startswith("shop_")
                and "active" not in c
                and "total" not in c]
traffic_cols = [c for c in temporal.columns if c.startswith("traffic_")]
price_cols   = [c for c in temporal.columns if c.startswith("price_") or c.startswith("competitor_price_")]
promo_cols   = [c for c in temporal.columns if "promotion" in c or c == "holiday_name"]

print(f"\n  Продажи топлива    ({len(sales_fuel)} шт.) : {sales_fuel}")
print(f"  Продажи магазина   ({len(sales_shop)} шт.) : {sales_shop}")
print(f"  Цены + конкуренты  ({len(price_cols)} шт.)")
print(f"  Трафик             ({len(traffic_cols)} шт.)")
print(f"  Акции / праздники  ({len(promo_cols)} шт.) : {promo_cols}")

# ── 5. Пропуски temporal ──────────────────────────────────────
print("\n" + "=" * 60)
print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ (temporal)")
print("=" * 60)

temp_nulls = temporal.isnull().sum()
temp_nulls = temp_nulls[temp_nulls > 0]
if temp_nulls.empty:
    print("  Пропусков: нет")
else:
    for col, n in temp_nulls.items():
        pct = n / len(temporal) * 100
        print(f"  {col:40s}: {n:5d} ({pct:.1f}%)")

# ── 6. Заполнение пропусков ───────────────────────────────────
print("\n" + "=" * 60)
print("ЗАПОЛНЕНИЕ ПРОПУСКОВ")
print("=" * 60)

before = df.isnull().sum().sum()
df = fill_missing(df)
after  = df.isnull().sum().sum()

for col, val in FILL_MAP.items():
    if col in df.columns:
        print(f"  {col:30s} -> '{val}'")

remaining = df.isnull().sum()
remaining = remaining[remaining > 0]
print(f"  Пропусков до  : {before}")
print(f"  Пропусков после: {after}")
if not remaining.empty:
    print("  Оставшиеся пропуски:")
    for col, n in remaining.items():
        print(f"    {col:40s}: {n}")

# ── 7. JOIN: итоговый DataFrame ───────────────────────────────
print("\n" + "=" * 60)
print("ОБЪЕДИНЕНИЕ ДАННЫХ")
print("=" * 60)

static_for_merge = static.drop(columns=["station_name"], errors="ignore")
extra_cols = [c for c in static_for_merge.columns if c != "station_id"]
unmatched = df[extra_cols[0]].isnull().sum() if extra_cols else 0

print(f"  Тип JOIN              : temporal LEFT JOIN static ON station_id")
print(f"  Добавлено из static   : {len(extra_cols)} колонок")
print(f"  Итоговый DataFrame    : {df.shape[0]} строк x {df.shape[1]} колонок")
print(f"  Незаматченных строк   : {unmatched}")

# ── 8. Краткая статистика по станциям ─────────────────────────
print("\n" + "=" * 60)
print("СТАТИСТИКА ПО СТАНЦИЯМ")
print("=" * 60)

for sid, grp in df.groupby("station_id", sort=True):
    fuel_total = grp[sales_fuel].sum().sum() if sales_fuel else 0
    fuel_mean  = fuel_total / len(grp) if len(grp) else 0
    print(f"\n  {sid}")
    print(f"    Строк       : {len(grp)}")
    print(f"    Период      : {grp['date'].min().date()} -- {grp['date'].max().date()}")
    if sales_fuel:
        print(f"    Топливо итого : {fuel_total:>14,.0f} л")
        print(f"    Ср./день      : {fuel_mean:>14,.1f} л")
    if sales_shop:
        shop_total = grp[sales_shop].sum().sum()
        print(f"    Магазин итого : {shop_total:>14,.0f}")

# ── 9. Итоговая структура ─────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЕ КОЛОНКИ MERGED_DATA")
print("=" * 60)

col_groups = {
    "Ключевые / временные": ["date", "station_id", "station_name"],
    "Продажи топлива":      sales_fuel,
    "Продажи магазина":     sales_shop,
    "Цены и конкуренты":    price_cols,
    "Трафик":               traffic_cols,
    "Погода":               [c for c in df.columns if c in
                              ("temperature", "weather_condition", "precipitation_mm",
                               "visibility_km", "wind_speed_ms")],
    "Акции / праздники":    promo_cols,
    "Статические (static)": extra_cols,
}

covered = set()
for group, cols in col_groups.items():
    cols_in_df = [c for c in cols if c in df.columns]
    covered.update(cols_in_df)
    if cols_in_df:
        print(f"\n  {group} ({len(cols_in_df)}):")
        print(f"    {cols_in_df}")

uncovered = [c for c in df.columns if c not in covered]
if uncovered:
    print(f"\n  Прочие / не распределены ({len(uncovered)}):")
    print(f"    {uncovered}")

print(f"\n  Итого в col_groups : {sum(len([c for c in v if c in df.columns]) for v in col_groups.values())}")
print(f"  Прочие             : {len(uncovered)}")
print(f"  Всего колонок      : {df.shape[1]}")

# ── 10. Сохранение ────────────────────────────────────────────
out = "data/merged_data.csv"
try:
    df.to_csv(out, index=False)
except OSError as exc:
    print(f"\n  [ОШИБКА] Не удалось сохранить {out}: {exc}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Готово.")
print(f"  Сохранено     : {out} ({df.shape[0]} x {df.shape[1]})")
print("  Следующий шаг : python eda/eda_preprocessing.py")
print("=" * 60)
