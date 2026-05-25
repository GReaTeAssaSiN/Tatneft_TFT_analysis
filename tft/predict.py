"""
Инференс TFT модели: предсказания на декабрь 2023.
Запускать из корня проекта: python tft/predict.py
Результат: data/predictions.csv, data/metrics.csv
"""

import glob
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.torch_compat  # noqa: F401 — патч torch.load для PyTorch 2.6
from utils.data_utils import (
    Q_HI,
    Q_LO,
    Q_MED,
    TARGET_COLS,
    TEST_END,
    TEST_START,
)

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

os.makedirs("data", exist_ok=True)

# ============================================================
# Загрузка модели
# ============================================================
print("=" * 60)
print("ИНФЕРЕНС TFT — ДЕКАБРЬ 2023")
print("=" * 60)

if os.path.exists("tft/model.ckpt"):
    ckpt_path = "tft/model.ckpt"
else:
    ckpt_files = sorted(glob.glob("tft/checkpoints/*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError("Чекпоинт не найден. Запустите python tft/train.py")
    ckpt_path = ckpt_files[-1]

print(f"\nЧекпоинт  : {ckpt_path}")
model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
model.eval()
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# Конфиг и данные
# ============================================================
with open("tft/training_dataset.pkl", "rb") as f:
    training = pickle.load(f)

with open("tft/dataset_config.pkl", "rb") as f:
    config = pickle.load(f)

BATCH_SIZE = config["batch_size"]
PRED_LEN = config["prediction_length"]   # 24
ENC_LEN = config["encoder_length"]       # 168

df = pd.read_csv("data/prepared_data.csv", parse_dates=["timestamp"])
df["station_id"] = df["station_id"].astype(str)
for col in config["static_cats"] + config["known_cats"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Контекст: ENC_LEN часов до декабря (для первых декабрьских окон) + весь декабрь
# TEST_START, TEST_END импортированы из utils.data_utils
CONTEXT_START = TEST_START - pd.Timedelta(hours=ENC_LEN)

test_df = df[df["timestamp"] >= CONTEXT_START].copy()
print(f"\nСкользящие окна: encoder={ENC_LEN}ч → decoder=24ч")
print(f"  Первое окно  : {CONTEXT_START.date()} – {(TEST_START - pd.Timedelta(hours=1)).date()} → прогноз {TEST_START.date()}")
print(f"  Последнее    : {(TEST_END - pd.Timedelta(hours=ENC_LEN)).date()} – {(TEST_END - pd.Timedelta(hours=1)).date()} → прогноз {TEST_END.date()}")
print(f"  Декабрь      : {TEST_START.date()} — {TEST_END.date()}")

testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
test_loader = testing.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0
)
print(f"Сэмплов   : {len(testing)}")

# ============================================================
# Предсказания
# ============================================================
print("\nВычисление предсказаний...")

# mode="quantiles" → TorchNormalizer^-1 применяется автоматически
# Результат в log1p-пространстве (т.к. TorchNormalizer обучался на log1p значениях)
# Далее применяем expm1 чтобы вернуться в исходные единицы (л/ч, руб/ч)
result = model.predict(
    test_loader,
    mode="quantiles",
    return_index=True,
    trainer_kwargs={"logger": False, "enable_progress_bar": True},
)

# Распаковка: pytorch-forecasting возвращает named tuple или обычный tuple
if hasattr(result, "output"):
    preds = result.output   # list[Tensor(n, pred_len, n_q)] per target
    idx_df = result.index   # DataFrame: time_idx (первый шаг декодера), station_id (закодированный)
else:
    preds, idx_df = result

# pytorch-forecasting возвращает station_id как закодированный integer (0,1,2,...),
# а не оригинальный строковый ID. Декодируем обратно через NaNLabelEncoder.
sid_encoder = training.categorical_encoders.get("station_id")
if sid_encoder is not None:
    idx_df = idx_df.copy()
    idx_df["station_id"] = sid_encoder.inverse_transform(
        pd.Series(idx_df["station_id"].astype(int))
    )

n_samples = len(idx_df)
n_q = preds[0].shape[-1]    # 7 квантилей: QUANTILE_LEVELS из data_utils
# Q_MED, Q_LO, Q_HI импортированы из utils.data_utils

print(f"Сэмплов: {n_samples}, горизонт: {PRED_LEN} ч, квантилей: {n_q}")

# Переводим тензоры в numpy (на GPU — явно перемещаем на CPU)
pred_np = [p.detach().cpu().numpy() for p in preds]  # list[(n, pred_len, n_q)]

# ============================================================
# Маппинг time_idx → timestamp
# ============================================================
# time_idx в idx_df — это первый шаг декодера (первый предсказываемый час)
ts_lookup = (
    df[["station_id", "time_idx", "timestamp"]]
    .drop_duplicates()
    .set_index(["station_id", "time_idx"])["timestamp"]
)

idx_df = idx_df.copy()
idx_df["time_idx"] = idx_df["time_idx"].astype(int)

# ============================================================
# Сборка predictions.csv (векторизованно)
# ============================================================
print("Сборка таблицы предсказаний...")

# Расширяем каждый сэмпл до PRED_LEN строк (по одной на каждый горизонт)
idx_exp = idx_df.loc[idx_df.index.repeat(PRED_LEN)].reset_index(drop=True)
horizon_arr = np.tile(np.arange(PRED_LEN), n_samples)   # 0, 1, ..., 23, 0, 1, ...
sample_arr = np.repeat(np.arange(n_samples), PRED_LEN)  # 0,0,...,0, 1,1,...

# Определяем timestamp первого шага для каждого сэмпла
start_ts = idx_df.apply(
    lambda r: ts_lookup.get((r["station_id"], int(r["time_idx"]))), axis=1
).values
start_ts_exp = np.repeat(start_ts, PRED_LEN)

# time_idx из idx_df — это первый шаг декодера (проверено).
# horizon_arr: 0=первый предсказываемый час, 23=последний.
idx_exp["timestamp"] = pd.to_datetime(start_ts_exp) + pd.to_timedelta(horizon_arr, unit="h")
idx_exp["horizon_h"] = horizon_arr + 1  # 1-индексированный горизонт

# Фильтрация: только декабрь 2023 (окна с конца декабря вылезают в январь)
mask = (idx_exp["timestamp"] >= TEST_START) & (idx_exp["timestamp"] <= TEST_END)
idx_exp = idx_exp[mask].reset_index(drop=True)
s_idx = sample_arr[mask.values]
h_idx = horizon_arr[mask.values]

# Добавляем предсказания и доверительные интервалы
for t_i, col in enumerate(TARGET_COLS):
    arr = pred_np[t_i]                                 # (n_samples, pred_len, n_q)
    log_med = arr[s_idx, h_idx, Q_MED]
    idx_exp[f"{col}_pred"] = np.expm1(np.maximum(log_med, 0.0))

    if n_q >= 5:
        idx_exp[f"{col}_q10"] = np.expm1(np.maximum(arr[s_idx, h_idx, Q_LO], 0.0))
        idx_exp[f"{col}_q90"] = np.expm1(np.maximum(arr[s_idx, h_idx, Q_HI], 0.0))

# Для каждого (station, timestamp) берём предсказание с наименьшим горизонтом
# (1-шаговый роллинговый прогноз — наиболее точный)
pred_cols = (
    [f"{col}_pred" for col in TARGET_COLS]
    + ([f"{col}_q10" for col in TARGET_COLS] if n_q >= 5 else [])
    + ([f"{col}_q90" for col in TARGET_COLS] if n_q >= 5 else [])
)

pred_df = (
    idx_exp[["station_id", "timestamp", "horizon_h"] + pred_cols]
    .sort_values(["station_id", "timestamp", "horizon_h"])
    .groupby(["station_id", "timestamp"], as_index=False)
    .first()
)

print(f"Уникальных предсказаний: {len(pred_df)}")

# ============================================================
# Фактические значения за декабрь
# ============================================================
# В prepared_data.csv целевые переменные уже в log1p (z-score к ним не применялся)
actual_df = df[(df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)][
    ["station_id", "timestamp"] + TARGET_COLS
].copy()

actual_df[TARGET_COLS] = np.expm1(actual_df[TARGET_COLS].clip(lower=0))
actual_df = actual_df.rename(columns={col: f"{col}_actual" for col in TARGET_COLS})

pred_df = pred_df.merge(actual_df, on=["station_id", "timestamp"], how="left")

pred_df.to_csv("data/predictions.csv", index=False)
print(f"Сохранено : data/predictions.csv ({pred_df.shape[0]} × {pred_df.shape[1]})")

# ============================================================
# Метрики (MAE, RMSE, MAPE)
# ============================================================
print("\n" + "=" * 60)
print("МЕТРИКИ ПО СТАНЦИЯМ")
print("  Топливо: л/ч  |  Магазин: руб/ч")
print("=" * 60)

rows_m = []
for sid in sorted(pred_df["station_id"].unique()):
    s = pred_df[pred_df["station_id"] == sid]
    for col in TARGET_COLS:
        yp = s[f"{col}_pred"].to_numpy(dtype=float)
        ya = s[f"{col}_actual"].to_numpy(dtype=float)
        ok = ~np.isnan(ya) & ~np.isnan(yp)
        if ok.sum() == 0:
            continue
        yp, ya = yp[ok], ya[ok]
        mae = float(np.mean(np.abs(yp - ya)))
        rmse = float(np.sqrt(np.mean((yp - ya) ** 2)))
        nz = ya > 0.01
        mape = (
            float(np.mean(np.abs((yp[nz] - ya[nz]) / ya[nz])) * 100)
            if nz.sum() > 0
            else None
        )
        rows_m.append(
            {
                "station_id": sid,
                "target": col,
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "MAPE_%": round(mape, 2) if mape is not None else None,
                "n": int(ok.sum()),
            }
        )

metrics_df = pd.DataFrame(rows_m)
print(metrics_df.to_string(index=False))

print("\n" + "=" * 60)
print("СВОДНЫЕ МЕТРИКИ (среднее по 5 станциям)")
print("=" * 60)
summary = metrics_df.groupby("target")[["MAE", "RMSE", "MAPE_%"]].mean().round(2)
print(summary.to_string())

metrics_df.to_csv("data/metrics.csv", index=False)
print(f"\nСохранено : data/metrics.csv")

print("\n" + "=" * 60)
print("Готово.")
print("  Сохранено : data/predictions.csv, data/metrics.csv")
print("  Следующий шаг: streamlit run dashboard/forecast_dashboard.py")
print("              или: streamlit run dashboard/tft_interpretation.py")
print("=" * 60)
