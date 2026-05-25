"""
Подготовка TimeSeriesDataSet для TFT модели.
Запускать из корня проекта: python tft/prepare_dataset.py
"""

import os
import pickle
import sys
import warnings

# pytorch_forecasting передаёт read-only ndarray в torch.from_numpy() внутри
# NaNLabelEncoder — это баг библиотеки, наш код исправить его не может.
# Предупреждение не влияет на корректность: pytorch_forecasting тут же
# делает копию массива, так что UB не возникает. Подавляем точечно.
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (MultiNormalizer,
                                               NaNLabelEncoder,
                                               TorchNormalizer)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_utils import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    STATIC_CATS,
    STATIC_REALS,
    TARGET_COLS,
    TIME_VARYING_KNOWN_CATS,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
    TRAIN_END,
    VAL_END,
)

# ============================================================
# Параметры окна
# ============================================================
BATCH_SIZE = 64

# ============================================================
# Группы переменных TFT — импортированы из utils/data_utils.py
# ============================================================
# STATIC_CATS, STATIC_REALS, TARGET_COLS,
# TIME_VARYING_KNOWN_CATS, TIME_VARYING_KNOWN_REALS,
# TIME_VARYING_UNKNOWN_REALS

# ============================================================
# Загрузка данных
# ============================================================
print("=" * 60)
print("Загрузка prepared_data.csv")
print("=" * 60)

df = pd.read_csv("data/prepared_data.csv", parse_dates=["timestamp"])
df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

print(f"  Датафрейм  : {df.shape[0]} строк x {df.shape[1]} колонок")
print(f"  Станций    : {df['station_id'].nunique()}")
print(f"  Период     : {df['timestamp'].min().date()} — {df['timestamp'].max().date()}")

# ============================================================
# Проверка наличия всех колонок
# ============================================================
all_needed = (
    STATIC_CATS
    + STATIC_REALS
    + TIME_VARYING_KNOWN_CATS
    + TIME_VARYING_KNOWN_REALS
    + TIME_VARYING_UNKNOWN_REALS
    + ["station_id", "time_idx"]
)
missing = [c for c in set(all_needed) if c not in df.columns]

if missing:
    print(f"\n[ВНИМАНИЕ] Отсутствуют колонки: {missing}")
    print("Фильтруем списки переменных...")
    STATIC_CATS = [c for c in STATIC_CATS if c in df.columns]
    STATIC_REALS = [c for c in STATIC_REALS if c in df.columns]
    TIME_VARYING_KNOWN_CATS = [c for c in TIME_VARYING_KNOWN_CATS if c in df.columns]
    TIME_VARYING_KNOWN_REALS = [c for c in TIME_VARYING_KNOWN_REALS if c in df.columns]
    TIME_VARYING_UNKNOWN_REALS = [
        c for c in TIME_VARYING_UNKNOWN_REALS if c in df.columns
    ]
else:
    print("  Все необходимые колонки присутствуют.")

# ============================================================
# Типы данных
# ============================================================
# station_id — строка (group identifier)
df["station_id"] = df["station_id"].astype(str)

# Категориальные колонки — строки (pytorch-forecasting строит словарь сам)
# weather_condition_enc теперь в TIME_VARYING_KNOWN_CATS, а не в unknown_reals
for col in STATIC_CATS + TIME_VARYING_KNOWN_CATS:
    if col in df.columns:
        df[col] = df[col].astype(str)

# ============================================================
# Предобучение категориальных энкодеров на полном датафрейме
# ============================================================
# Энкодеры обучаются на всех 43800 строках (весь 2023 год), чтобы словарь
# содержал все возможные категории — включая праздники только в ноябре/декабре.
# Это исключает KeyError при создании val/test датасетов.
print("\nПредобучение категориальных энкодеров на полном df...")
cat_encoders = {}
for col in STATIC_CATS + TIME_VARYING_KNOWN_CATS + ["station_id"]:
    if col in df.columns:
        enc = NaNLabelEncoder(add_nan=False)
        enc.fit(df[col])
        cat_encoders[col] = enc
        print(f"  {col:30s}: {len(enc.classes_)} уникальных значений")
print("  Готово — неизвестных категорий при val/test не будет.")

# ============================================================
# Сплиты
# ============================================================
# Примечание: val_df включает данные train для encoder context
train_df = df[df["timestamp"] <= TRAIN_END].copy()
val_df = df[df["timestamp"] <= VAL_END].copy()  # jan–nov (encoder видит oct)
test_df = df.copy()  # jan–dec (encoder видит nov)

print(f"\nСплит:")
print(
    f"  train : {train_df['timestamp'].min().date()} — {train_df['timestamp'].max().date()}  ({len(train_df)} строк)"
)
print(
    f"  val   : {val_df['timestamp'].min().date()}  — {val_df['timestamp'].max().date()}  ({len(val_df)} строк, включает train для контекста)"
)
print(
    f"  test  : {test_df['timestamp'].min().date()} — {test_df['timestamp'].max().date()}  ({len(test_df)} строк)"
)

# ============================================================
# TimeSeriesDataSet — train
# ============================================================
print("\n" + "=" * 60)
print("Создание TimeSeriesDataSet (train)")
print("=" * 60)

training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=TARGET_COLS,
    group_ids=["station_id"],
    min_encoder_length=ENCODER_LENGTH // 2,  # мин. 3.5 суток (84 ч)
    max_encoder_length=ENCODER_LENGTH,        # макс. 7 суток (168 ч)
    min_prediction_length=1,
    max_prediction_length=PREDICTION_LENGTH,  # до 1 суток (24 ч)
    static_categoricals=STATIC_CATS,
    static_reals=STATIC_REALS,
    time_varying_known_categoricals=TIME_VARYING_KNOWN_CATS,
    time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
    categorical_encoders=cat_encoders,
    # Данные уже log1p + z-score нормализованы.
    # TorchNormalizer (метод robust) — реализован на PyTorch, без sklearn внутри,
    # поэтому не генерирует предупреждений о feature names.
    target_normalizer=MultiNormalizer(
        [TorchNormalizer(method="robust", center=True) for _ in TARGET_COLS]
    ),
    add_relative_time_idx=True,  # относительный индекс как known future
    add_target_scales=True,  # масштаб цели как static real
    add_encoder_length=True,  # длина энкодера как static real
    allow_missing_timesteps=False,
)

print(f"  Обучающих сэмплов: {len(training)}")
print(
    f"  Параметры окна   : encoder={ENCODER_LENGTH}ч ({ENCODER_LENGTH//24} сут.), "
    f"prediction={PREDICTION_LENGTH}ч ({PREDICTION_LENGTH//24} сут.)"
)

# ============================================================
# TimeSeriesDataSet — val и test
# ============================================================
print("\nСоздание TimeSeriesDataSet (val, test)...")

validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)

print(f"  Валидационных сэмплов: {len(validation)}")
print(f"  Тестовых сэмплов     : {len(testing)}")

# ============================================================
# DataLoaders
# ============================================================
print("\nСоздание DataLoaders...")

train_loader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)
val_loader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0
)
test_loader = testing.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0
)

# Быстрая проверка батча
x, y = next(iter(train_loader))
targets, weights = y
print(f"\nПроверка батча:")
print(f"  encoder_cont shape : {x['encoder_cont'].shape}")
print(f"  decoder_cont shape : {x['decoder_cont'].shape}")
print(f"  targets (multi)    : {len(targets)} переменных, shape={targets[0].shape}")

# ============================================================
# Сохранение
# ============================================================
print("\n" + "=" * 60)
print("Сохранение")
print("=" * 60)

os.makedirs("tft", exist_ok=True)

with open("tft/training_dataset.pkl", "wb") as f:
    pickle.dump(training, f)

dataset_config = {
    "encoder_length": ENCODER_LENGTH,
    "prediction_length": PREDICTION_LENGTH,
    "batch_size": BATCH_SIZE,
    "target_cols": TARGET_COLS,
    "static_cats": STATIC_CATS,
    "static_reals": STATIC_REALS,
    "known_cats": TIME_VARYING_KNOWN_CATS,
    "known_reals": TIME_VARYING_KNOWN_REALS,
    "unknown_reals": TIME_VARYING_UNKNOWN_REALS,
    "n_stations": df["station_id"].nunique(),
    "train_end": str(TRAIN_END.date()),
    "val_end": str(VAL_END.date()),
}

with open("tft/dataset_config.pkl", "wb") as f:
    pickle.dump(dataset_config, f)

print("  tft/training_dataset.pkl — объект TimeSeriesDataSet (для train.py)")
print("  tft/dataset_config.pkl   — конфигурация датасета (параметры, списки колонок)")

print("\n" + "=" * 60)
print("Готово.")
print("  Сохранено : tft/training_dataset.pkl, tft/dataset_config.pkl")
print("  Следующий шаг: python tft/train.py")
print("=" * 60)
