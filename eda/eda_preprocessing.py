"""
EDA-предобработка данных для TFT-модели.
Выполняет все этапы преобразования и выводит ключевые шаги в консоль.

Запускать из корня проекта: python eda/eda_preprocessing.py
Входные данные : data/merged_data.csv
Выходные данные:
  data/prepared_data.csv  — полный обработанный датафрейм
  data/train.csv          — Jan–Oct 2023 (~83.6%)
  data/val.csv            — Nov 2023 (~8.2%)
  data/test.csv           — Dec 2023 (~8.2%)
  tft/scalers.pkl         — {station_id: {col: (mean, std)}} + log1p_cols
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_utils import (
    CYCLICAL_FEATURES,
    FILL_MAP,
    LOG_COLS,
    STATIC_REALS,
    TARGET_COLS,
    add_cyclical_encoding,
    fill_missing,
)

os.makedirs("data", exist_ok=True)
os.makedirs("tft", exist_ok=True)

TRAIN_END = pd.Timestamp("2023-10-31 23:00:00")
VAL_END = pd.Timestamp("2023-11-30 23:00:00")

print("=" * 60)
print("EDA-ПРЕДОБРАБОТКА ДЛЯ TFT")
print("=" * 60)

df = pd.read_csv("data/merged_data.csv", parse_dates=["timestamp"])
print(f"Загружено: {df.shape[0]} строк x {df.shape[1]} колонок\n")

# ============================================================
# Шаг 1. Заполнение пропусков
# ============================================================
print("[1/8] Заполнение пропусков...")
nulls_before = df.isnull().sum()
nulls_before = nulls_before[nulls_before > 0]

fill_missing(df)

for col, n in nulls_before.items():
    val = FILL_MAP.get(col, "?")
    print(f"  {col}: {n} пропусков -> '{val}'")
print(f"  Пропусков после: {df.isnull().sum().sum()}")

# ============================================================
# Шаг 2. Winsorization выбросов (IQR)
# ============================================================
print("\n[2/8] Winsorization выбросов (метод IQR)...")

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
binary_cols = [c for c in num_cols if df[c].dropna().isin([0, 1]).all()]
static_skip = set(c for c in STATIC_REALS if c in df.columns)
interval_cols = [c for c in num_cols if c not in binary_cols and c not in static_skip]

n_clipped = 0
for col in interval_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out > 0:
        df[col] = df[col].clip(lower=lo, upper=hi)
        n_clipped += n_out

print(f"  Интервальных колонок: {len(interval_cols)}")
print(f"  Пропущено статических (metadata): {len(static_skip)} колонок — паспортные данные не меняются")
print(f"  Значений скорректировано (winsorized): {n_clipped}")
print(f"  Строки не удаляются — временной ряд должен быть непрерывным")

# ============================================================
# Шаг 3. Label Encoding категориальных переменных
# ============================================================
print("\n[3/8] Label Encoding категориальных переменных...")

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
# station_name и timestamp оставляем как есть
encode_cols = [c for c in cat_cols if c not in ["station_name", "timestamp"]]

label_encoders = {}
for col in encode_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"  Закодировано: {len(encode_cols)} колонок -> добавлены _enc суффиксы")
for col in encode_cols:
    n_cls = len(label_encoders[col].classes_)
    print(f"    {col:20s}: {n_cls} классов")

# ============================================================
# Шаг 4. Циклическое кодирование временных признаков
# ============================================================
print("\n[4/8] Циклическое кодирование (sin/cos) временных признаков...")
print("  Решает проблему разрыва: hour=23 и hour=0 будут близки на окружности")

add_cyclical_encoding(df)

for col, period in CYCLICAL_FEATURES.items():
    print(f"  {col:15s} (period={period:2d}) -> {col}_sin, {col}_cos")

# ============================================================
# Шаг 5. Монотонный индекс времени (time_idx)
# ============================================================
print("\n[5/8] Построение time_idx (порядковый номер часа per-station)...")

df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
df["time_idx"] = df.groupby("station_id").cumcount()

print(f"  time_idx: 0 -- {df['time_idx'].max()} (на каждой станции)")
print(f"  Количество станций: {df['station_id'].nunique()}")

# ============================================================
# Шаг 6. Log1p преобразование целевых переменных
# ============================================================
print("\n[6/8] Log1p преобразование целевых переменных...")
print("  log1p(x) = log(1+x) — убирает правостороннюю асимметрию продаж")
print("")
print(f"  {'Колонка':30s} {'skew до':>10} {'skew после':>12}")
print(f"  {'-'*30} {'-'*10} {'-'*12}")

actual_log_cols = [c for c in LOG_COLS if c in df.columns]
for col in actual_log_cols:
    skew_before = df[col].skew()
    df[col + "_orig"] = df[col].copy()
    df[col] = np.log1p(df[col])
    skew_after = df[col].skew()
    print(f"  {col:30s} {skew_before:>10.3f} {skew_after:>12.3f}")

print(f"\n  Оригиналы сохранены с суффиксом _orig ({len(actual_log_cols)} колонок)")

# ============================================================
# Шаг 7. Z-score нормализация per-station
# ============================================================
print("\n[7/8] Z-score нормализация (per-station)...")
print("  Формула: (x - mean_station) / std_station")
print("  Исключены: статические (metadata), бинарные, _enc, _orig, log1p-колонки (цели)")

skip_cols = set(
    ["timestamp", "station_id", "time_idx", "station_name"]
    + list(static_skip)
    + binary_cols
    + [c for c in df.columns if c.endswith("_enc")]
    + [c for c in df.columns if c.endswith("_orig")]
    + actual_log_cols
)
norm_cols = [
    c for c in df.select_dtypes(include=["number"]).columns if c not in skip_cols
]
df[norm_cols] = df[norm_cols].astype(float)

scalers = {}
for sid in df["station_id"].unique():
    scalers[sid] = {}
    mask = df["station_id"] == sid
    for col in norm_cols:
        mean = df.loc[mask, col].mean()
        std = df.loc[mask, col].std()
        if std > 0:
            df.loc[mask, col] = (df.loc[mask, col] - mean) / std
        scalers[sid][col] = (mean, std)

print(f"  Нормализовано: {len(norm_cols)} колонок x {df['station_id'].nunique()} станций")

# ============================================================
# Шаг 8. Темпоральный сплит и сохранение
# ============================================================
print("\n[8/8] Темпоральный сплит и сохранение файлов...")

train = df[df["timestamp"] <= TRAIN_END].copy()
val = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)].copy()
test = df[df["timestamp"] > VAL_END].copy()

print(f"  Train: {train['timestamp'].min().date()} -- {TRAIN_END.date()}  {len(train):6d} строк ({len(train)/len(df)*100:.1f}%)")
print(f"  Val  : {(TRAIN_END + pd.Timedelta(hours=1)).date()} -- {VAL_END.date()}  {len(val):6d} строк ({len(val)/len(df)*100:.1f}%)")
print(f"  Test : {(VAL_END + pd.Timedelta(hours=1)).date()} -- {df['timestamp'].max().date()}  {len(test):6d} строк ({len(test)/len(df)*100:.1f}%)")

# Сохранение файлов
df.to_csv("data/prepared_data.csv", index=False)
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

with open("tft/scalers.pkl", "wb") as f:
    pickle.dump({"scalers": scalers, "log1p_cols": actual_log_cols}, f)

print(f"\n  data/prepared_data.csv  : {df.shape[0]} строк x {df.shape[1]} колонок")
print(f"  data/train.csv          : {len(train)} строк")
print(f"  data/val.csv            : {len(val)} строк")
print(f"  data/test.csv           : {len(test)} строк")
print(f"  tft/scalers.pkl         : {df['station_id'].nunique()} станций, {len(norm_cols)} колонок")

print("\n" + "=" * 60)
print("Предобработка завершена.")
print("Следующий шаг: python eda/tft_report.py")
print("=" * 60)
