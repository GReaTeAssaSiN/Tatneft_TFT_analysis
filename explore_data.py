"""
Загрузка и объединение исходных CSV-файлов для 5 АЗС.
Запускать из корня проекта: python explore_data.py
Результат: data/merged_data.csv
"""

import os
import sys

import pandas as pd

from utils.data_utils import fill_missing, load_and_merge

os.makedirs("data", exist_ok=True)

# ── 1. Проверка исходных файлов ───────────────────────────────
SRC_META = "SourceDataForWork/5stations_metadata.csv"
SRC_DATA = "SourceDataForWork/5stations_data.csv"

print("=" * 60)
print("ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ИСХОДНЫХ ДАННЫХ")
print("=" * 60)

for path in (SRC_META, SRC_DATA):
    if not os.path.exists(path):
        print(f"\n  [ОШИБКА] Файл не найден: {path}")
        print("  Убедитесь, что папка SourceDataForWork/ присутствует в корне проекта.")
        sys.exit(1)

meta = pd.read_csv(SRC_META)
data = pd.read_csv(SRC_DATA, parse_dates=["timestamp"])

print(f"  5stations_metadata.csv : {meta.shape[0]} строк x {meta.shape[1]} колонок")
print(f"  5stations_data.csv     : {data.shape[0]} строк x {data.shape[1]} колонок")

# ── 2. JOIN ───────────────────────────────────────────────────
df = load_and_merge()

print(f"\n  Тип JOIN               : metadata LEFT JOIN data ON station_id")
print(f"  После JOIN             : {df.shape[0]} строк x {df.shape[1]} колонок")
print(f"  Период                 : {df['timestamp'].min().date()} -- {df['timestamp'].max().date()}")
print(f"  Станций                : {df['station_id'].nunique()}")
print(f"  Часов на станцию       : {data.groupby('station_id').size().iloc[0]}")

# ── 3. Пропуски ───────────────────────────────────────────────
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

# ── 4. Сохранение ─────────────────────────────────────────────
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
