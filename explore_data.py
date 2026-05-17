"""
Загрузка и объединение исходных CSV-файлов для 5 АЗС.
Запускать из корня проекта: python explore_data.py
Результат: data/merged_data.csv
"""

import os

import pandas as pd

from utils.data_utils import TARGET_COLS, fill_missing, load_and_merge

os.makedirs("data", exist_ok=True)

# ============================================================
# 1. Загрузка и JOIN
# ============================================================
print("=" * 60)
print("ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ИСХОДНЫХ ДАННЫХ")
print("=" * 60)

meta = pd.read_csv("SourceDataForWork/5stations_metadata.csv")
data = pd.read_csv("SourceDataForWork/5stations_data.csv", parse_dates=["timestamp"])

print(f"  5stations_metadata.csv : {meta.shape[0]} строк x {meta.shape[1]} колонок")
print(f"  5stations_data.csv     : {data.shape[0]} строк x {data.shape[1]} колонок")

df = load_and_merge()

print(f"\n  После JOIN             : {df.shape[0]} строк x {df.shape[1]} колонок")
print(f"  Период                 : {df['timestamp'].min().date()} -- {df['timestamp'].max().date()}")
print(f"  Станций                : {df['station_id'].nunique()}")
print(f"  Часов на станцию       : {data.groupby('station_id').size().unique()[0]}")

# ============================================================
# 2. Пропуски
# ============================================================
print("\n" + "=" * 60)
print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
print("=" * 60)

nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls) == 0:
    print("  Пропущенных значений нет.")
else:
    for col, n in nulls.items():
        print(f"  {col:35s}: {n} пропусков")
    fill_missing(df)
    print(f"\n  Заполнено: holiday_name -> 'нет_праздника', ad_channel -> 'нет_рекламы'")
    print(f"  Пропусков после заполнения: {df.isnull().sum().sum()}")

# ============================================================
# 3. Основная статистика по целевым переменным
# ============================================================
fuel_cols = [c for c in TARGET_COLS if c.startswith("sales_")]
shop_cols = [c for c in TARGET_COLS if c.startswith("shop_")]

print("\n" + "=" * 60)
print("ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ — ТОПЛИВО (литры/час)")
print("=" * 60)
print(df[fuel_cols].describe().round(2).to_string())

print("\n" + "=" * 60)
print("ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ — МАГАЗИН (руб/час)")
print("=" * 60)
print(df[shop_cols].describe().round(2).to_string())

print("\n" + "=" * 60)
print("ПРОДАЖИ ТОПЛИВА ПО СТАНЦИЯМ (сумма за год, тыс. литров)")
print("=" * 60)
station_fuel = (
    df.groupby(["station_id", "station_name"])[fuel_cols].sum() / 1000
).round(1)
print(station_fuel.to_string())

print("\n" + "=" * 60)
print("ВЫРУЧКА МАГАЗИНА ПО СТАНЦИЯМ (сумма за год, тыс. руб)")
print("=" * 60)
station_shop = (
    df.groupby(["station_id", "station_name"])[shop_cols].sum() / 1000
).round(1)
print(station_shop.to_string())

# ============================================================
# 4. Обзор групп переменных
# ============================================================
print("\n" + "=" * 60)
print("СТРУКТУРА ПЕРЕМЕННЫХ")
print("=" * 60)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
bin_cols = [c for c in num_cols if df[c].dropna().isin([0, 1]).all()]
int_cols = [c for c in num_cols if c not in bin_cols]

print(f"  Всего колонок      : {df.shape[1]}")
print(f"  Категориальные     : {len(cat_cols)}  ({cat_cols})")
print(f"  Интервальные       : {len(int_cols)}")
print(f"  Бинарные (0/1)     : {len(bin_cols)}")

# ============================================================
# 5. Сохранение
# ============================================================
df.to_csv("data/merged_data.csv", index=False)
print(f"\n  Сохранено: data/merged_data.csv ({df.shape[0]} x {df.shape[1]})")
print("\nСледующий шаг: python eda/eda_column_analysis.py")
