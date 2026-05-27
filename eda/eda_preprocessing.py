"""
Предобработка данных для прогностической модели.
Выполняет все этапы преобразования согласно reports/column_analysis.md.

Запускать из корня проекта: python eda/eda_preprocessing.py
Входные данные : data/merged_data.csv
Выходные данные:
  data/prepared_data.csv  — полный обработанный датафрейм
  data/train.csv          — Jan–Oct 2025 (~83.6%)
  data/val.csv            — Nov 2025 с 30-дневным контекстом (~11.5%)
  data/test.csv           — Dec 2025 (~8.2%)
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
    ENCODER_LENGTH,
    EXCLUDED_COLS,
    FILL_MAP,
    LOG_COLS,
    STATIC_REALS,
    TARGET_COLS,
    TRAIN_END,
    VAL_END,
    add_cyclical_encoding,
    fill_missing,
)

os.makedirs("data", exist_ok=True)
os.makedirs("tft", exist_ok=True)

# Колонки, требующие Label Encoding.
# Строковые категории — определяются автоматически по dtype object.
# Числовые коды категорий — перечисляются явно: значения являются кодами
# (например, settlement_size=3 означает «город 250–500 тыс.», не число 3).
NUM_CAT_ENCODE = ["road_level", "settlement_size", "distance_to_city_km"]

print("=" * 60)
print("ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ МОДЕЛИ")
print("=" * 60)

# ============================================================
# Загрузка
# ============================================================
df = pd.read_csv("data/merged_data.csv", parse_dates=["date"])
print(f"Загружено: {df.shape[0]} строк x {df.shape[1]} колонок\n")

# ============================================================
# Шаг 1. Заполнение пропусков
# ============================================================
print("[1/7] Заполнение пропусков...")

nulls_before = df.isnull().sum()
nulls_before = nulls_before[nulls_before > 0]
fill_missing(df)

if nulls_before.empty:
    print("  Пропусков не обнаружено.")
else:
    for col, n in nulls_before.items():
        val = FILL_MAP.get(col, "?")
        print(f"  {col}: {n} пропусков -> '{val}'")
print(f"  Пропусков после заполнения: {df.isnull().sum().sum()}")

# ============================================================
# Шаг 2. Исключение избыточных колонок
# ============================================================
print("\n[2/7] Исключение избыточных колонок...")

excluded_present = [c for c in EXCLUDED_COLS if c in df.columns]
df = df.drop(columns=excluded_present)

print(f"  Исключено {len(excluded_present)} колонок:")
for col in excluded_present:
    print(f"    - {col}")
print(f"  Осталось в датафрейме: {df.shape[1]} колонок")

# Вспомогательные множества для шагов 3 и 6
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
# Бинарные: только 0 и 1 — не нормализуются
binary_cols = [c for c in num_cols if df[c].dropna().isin([0, 1]).all()]
# Числовые коды категорий — тоже не нормализуются (войдут в _enc)
num_cat_present = [c for c in NUM_CAT_ENCODE if c in df.columns]
# Паспортные данные АЗС — не нормализуются
static_real_present = [c for c in STATIC_REALS if c in df.columns]

# ============================================================
# Шаг 3. Label Encoding категориальных переменных
# ============================================================
print("\n[3/7] Label Encoding категориальных переменных...")

# Строковые категории (dtype str / object)
str_cat_cols = df.select_dtypes(include=["str", "object"]).columns.tolist()
str_cat_cols = [c for c in str_cat_cols if c not in ("station_id",)]

# Объединяем: строки + числовые коды категорий
encode_cols = str_cat_cols + num_cat_present

# NaNLabelEncoder-стиль: fit на полном df, чтобы val/test не давали KeyError
label_encoders: dict = {}
enc_dict: dict = {}
for col in encode_cols:
    le = LabelEncoder()
    enc_dict[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df = pd.concat([df, pd.DataFrame(enc_dict, index=df.index)], axis=1)
df = df.copy()  # дефрагментация после concat

print(f"  Закодировано {len(encode_cols)} колонок:")
for col in encode_cols:
    n_cls = len(label_encoders[col].classes_)
    dtype_label = "object" if col in str_cat_cols else "int (код)"
    print(f"    {col:35s}: {n_cls} классов  [{dtype_label}]")

# ============================================================
# Шаг 4. Циклическое кодирование временных признаков
# ============================================================
print("\n[4/7] Циклическое кодирование (sin/cos)...")
print("  Устраняет разрыв на числовой оси: day=6 (Вс) и day=0 (Пн) близки на окружности")

add_cyclical_encoding(df)

for col, period in CYCLICAL_FEATURES.items():
    print(f"  {col:15s} (period={period:2d}) -> {col}_sin, {col}_cos")

# ============================================================
# Шаг 5. Монотонный индекс времени (time_idx)
# ============================================================
print("\n[5/7] Построение time_idx (порядковый номер дня per-station)...")

df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
df["time_idx"] = df.groupby("station_id").cumcount()

print(f"  time_idx: 0 — {df['time_idx'].max()}  (на каждой станции)")
print(f"  Станций: {df['station_id'].nunique()}, дней/станцию: {df['time_idx'].max() + 1}")

# ============================================================
# Шаг 6. Log1p преобразование целевых переменных
# ============================================================
print("\n[6/7] Log1p преобразование целевых переменных...")
print("  log1p(x) = log(1+x): устраняет правосторонний скос, стабилизирует нули")
print()
print(f"  {'Колонка':38s} {'skew до':>8} {'skew после':>10}")
print(f"  {'-'*38} {'-'*8} {'-'*10}")

actual_log_cols = [c for c in LOG_COLS if c in df.columns]

skews_before = {col: df[col].skew() for col in actual_log_cols}

orig_dict = {col + "_orig": df[col].copy() for col in actual_log_cols}
df = pd.concat([df, pd.DataFrame(orig_dict, index=df.index)], axis=1)
df[actual_log_cols] = np.log1p(df[actual_log_cols])

for col in actual_log_cols:
    print(f"  {col:38s} {skews_before[col]:>8.3f} {df[col].skew():>10.3f}")

print(f"\n  Оригиналы сохранены в _orig ({len(actual_log_cols)} колонок)")

df = df.copy()  # дефрагментация после пошаговых concat

# ============================================================
# Шаг 7. Z-score нормализация per-station
# ============================================================
print("\n[7/7] Z-score нормализация (per-station, статистика только по train)...")
print("  Формула: z = (x - mean_train) / std_train")
print("  Статистика вычисляется только по Jan–Oct 2025 -> нет утечки из val/test")

# Колонки, исключаемые из Z-score
skip_cols = set(
    ["date", "station_id", "time_idx"]
    + static_real_present          # паспортные данные — не нормализуются
    + binary_cols                  # бинарные [0/1] — масштаб уже нормализован
    + num_cat_present              # числовые коды -> заменяются _enc
    + [c for c in df.columns if c.endswith("_enc")]   # уже закодированы
    + [c for c in df.columns if c.endswith("_orig")]  # служебные оригиналы
    + actual_log_cols              # цели: нормализуются отдельно на уровне модели
)

norm_cols = [
    c for c in df.select_dtypes(include=["number"]).columns
    if c not in skip_cols
]
df[norm_cols] = df[norm_cols].astype(float)

train_mask = df["date"] <= TRAIN_END
scalers: dict = {}

for sid in df["station_id"].unique():
    scalers[sid] = {}
    mask_sid       = df["station_id"] == sid
    train_mask_sid = mask_sid & train_mask

    for col in norm_cols:
        mean = float(df.loc[train_mask_sid, col].mean())
        std  = float(df.loc[train_mask_sid, col].std())
        if std > 0:
            df.loc[mask_sid, col] = (df.loc[mask_sid, col] - mean) / std
        scalers[sid][col] = (mean, std)

zero_std_cols = [
    col for col in norm_cols
    if all(
        df.loc[df["station_id"] == sid, col].std(ddof=0) == 0
        for sid in df["station_id"].unique()
    )
]
print(f"  Нормализовано: {len(norm_cols)} колонок x {df['station_id'].nunique()} станций")
if zero_std_cols:
    print(f"  Пропущено (std=0 во всех станциях): {zero_std_cols}")

# ============================================================
# Темпоральный сплит и сохранение
# ============================================================
print("\nТемпоральный сплит и сохранение файлов...")

# Val включает ENCODER_LENGTH дней контекста из train,
# чтобы первые окна ноября имели полную историю
val_ctx_start = VAL_END.replace(day=1) - pd.Timedelta(days=ENCODER_LENGTH)

train = df[df["date"] <= TRAIN_END].copy()
val   = df[(df["date"] >= val_ctx_start) & (df["date"] <= VAL_END)].copy()
test  = df[df["date"] > VAL_END].copy()

total = len(df)
print(f"  Train: {train['date'].min().date()} — {train['date'].max().date()}  {len(train):6d} строк ({len(train)/total*100:.1f}%)")
print(f"  Val  : {val['date'].min().date()} — {val['date'].max().date()}  {len(val):6d} строк ({len(val)/total*100:.1f}%)  [вкл. {ENCODER_LENGTH} дн. контекста]")
print(f"  Test : {test['date'].min().date()} — {test['date'].max().date()}  {len(test):6d} строк ({len(test)/total*100:.1f}%)")

df.to_csv("data/prepared_data.csv", index=False)
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

with open("tft/scalers.pkl", "wb") as f:
    pickle.dump({"scalers": scalers, "log1p_cols": actual_log_cols}, f)

print(f"\n  data/prepared_data.csv : {df.shape[0]} строк x {df.shape[1]} колонок")
print(f"  data/train.csv         : {len(train)} строк")
print(f"  data/val.csv           : {len(val)} строк")
print(f"  data/test.csv          : {len(test)} строк")
print(f"  tft/scalers.pkl        : {df['station_id'].nunique()} станций, {len(norm_cols)} колонок")

print("\n" + "=" * 60)
print("Готово.")
print("  Следующий шаг: python tft/prepare_dataset.py")
print("=" * 60)
