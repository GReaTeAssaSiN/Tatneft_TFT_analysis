"""
Обучение TFT модели.
Запускать из корня проекта: python tft/train.py
"""

import glob
import os
import pickle
import shutil
import sys

os.environ["LIGHTNING_DISABLE_REMOTE_TIPS"] = "1"

import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.torch_compat  # noqa: F401 — патч torch.load для PyTorch 2.6

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# ============================================================
# Определение устройства
# ============================================================
if torch.cuda.is_available():
    ACCELERATOR = "gpu"
    DEVICE_NAME = torch.cuda.get_device_name(0)
    BATCH_SIZE = 64
    EPOCHS = 80
    HIDDEN_SIZE = 128
    ATTN_HEADS = 4
else:
    ACCELERATOR = "cpu"
    DEVICE_NAME = "CPU"
    BATCH_SIZE = 32
    EPOCHS = 50
    HIDDEN_SIZE = 64
    ATTN_HEADS = 2

LEARNING_RATE = 3e-4
DROPOUT = 0.15
HIDDEN_CONTINUOUS = 64
GRADIENT_CLIP = 1.0

# ============================================================
# Параметры обучения
# ============================================================
print("=" * 60)
print("ПАРАМЕТРЫ ОБУЧЕНИЯ TFT")
print("=" * 60)
print(f"  Устройство          : {ACCELERATOR.upper()} ({DEVICE_NAME})")
print(f"  Эпох                : {EPOCHS}")
print(f"  Batch size          : {BATCH_SIZE}")
print(f"  Learning rate       : {LEARNING_RATE}")
print(f"  Hidden size         : {HIDDEN_SIZE}")
print(f"  Attention heads     : {ATTN_HEADS}")
print(f"  Hidden continuous   : {HIDDEN_CONTINUOUS}")
print(f"  Dropout             : {DROPOUT}")
print(f"  Gradient clip       : {GRADIENT_CLIP}")
print("=" * 60)

# ============================================================
# Загрузка датасета
# ============================================================
print("\nЗагрузка датасета...")

with open("tft/training_dataset.pkl", "rb") as f:
    training = pickle.load(f)

with open("tft/dataset_config.pkl", "rb") as f:
    config = pickle.load(f)

print(f"  Целевые переменные  : {config['target_cols']}")
print(f"  Encoder length      : {config['encoder_length']} ч")
print(f"  Prediction length   : {config['prediction_length']} ч")
print(f"  Станций             : {config['n_stations']}")

# ============================================================
# DataLoaders
# ============================================================
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

df = pd.read_csv("data/prepared_data.csv", parse_dates=["timestamp"])
df["station_id"] = df["station_id"].astype(str)

for col in config["static_cats"] + config["known_cats"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

TRAIN_END = pd.Timestamp(config["train_end"] + " 23:00:00")
VAL_END = pd.Timestamp(config["val_end"] + " 23:00:00")

train_df = df[df["timestamp"] <= TRAIN_END].copy()
val_df = df[df["timestamp"] <= VAL_END].copy()

validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

train_loader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)
val_loader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0
)

print(f"\n  Train батчей        : {len(train_loader)}")
print(f"  Val батчей          : {len(val_loader)}")

# ============================================================
# Модель
# ============================================================
print("\nСоздание модели TFT...")

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTN_HEADS,
    dropout=DROPOUT,
    hidden_continuous_size=HIDDEN_CONTINUOUS,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=5,
)

total_params = sum(p.numel() for p in tft.parameters())
print(f"  Параметров модели   : {total_params:,}")

# ============================================================
# Колбэк: синхронизация model.ckpt при каждом новом лучшем чекпоинте
# ============================================================

class BestModelSync(pl.Callback):
    """Перезаписывает tft/model.ckpt каждый раз, когда ModelCheckpoint
    сохраняет новый лучший чекпоинт — без ожидания конца обучения."""

    def __init__(self, checkpoint_cb: ModelCheckpoint, dest: str = "tft/model.ckpt"):
        self.checkpoint_cb = checkpoint_cb
        self.dest = dest
        self._last_synced: str = ""

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        best = self.checkpoint_cb.best_model_path
        if best and os.path.exists(best) and best != self._last_synced:
            shutil.copy(best, self.dest)
            self._last_synced = best
            score = self.checkpoint_cb.best_model_score
            print(f"  → model.ckpt обновлён  "
                  f"[epoch {trainer.current_epoch}  val_loss={score:.4f}]")


# ============================================================
# Callbacks
# ============================================================
os.makedirs("tft/checkpoints", exist_ok=True)
os.makedirs("tft/logs", exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    dirpath="tft/checkpoints",
    filename="tft-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
)

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=12,
    mode="min",
    verbose=True,
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
sync_cb = BestModelSync(checkpoint_cb)

tb_logger = TensorBoardLogger("tft/logs", name="tft_model")

# Автоопределение последнего чекпоинта для продолжения обучения
ckpt_files = sorted(glob.glob("tft/checkpoints/*.ckpt"))
resume_ckpt = ckpt_files[-1] if ckpt_files else None
if resume_ckpt:
    print(f"\nПродолжение обучения с чекпоинта: {resume_ckpt}")
else:
    print("\nЧекпоинт не найден — обучение с нуля.")

# ============================================================
# Обучение
# ============================================================
print(f"\nЗапуск обучения ({EPOCHS} эпох, {ACCELERATOR.upper()})...")
print("-" * 60)
print("Что сохраняется в процессе:")
print("  tft/checkpoints/  — чекпоинты модели после каждой эпохи.")
print("                      Хранится только лучший (monitor=val_loss).")
print("                      Файл: tft-NN-0.XXXX.ckpt")
print("  tft/logs/         — логи Lightning (loss по шагам и эпохам).")
print("                      Можно открыть через Board:")
print("                      tensorboard --logdir tft/logs")
print("-" * 60)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator=ACCELERATOR,
    devices=1,
    gradient_clip_val=GRADIENT_CLIP,
    callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, sync_cb],
    enable_model_summary=True,
    log_every_n_steps=10,
    logger=tb_logger,
)

trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=resume_ckpt,
)

# ============================================================
# Сохранение лучшей модели
# ============================================================
best_path = checkpoint_cb.best_model_path
print(f"\nЛучший чекпоинт     : {best_path}")
print(f"Лучший val_loss     : {checkpoint_cb.best_model_score:.4f}")

if best_path and not os.path.exists("tft/model.ckpt"):
    shutil.copy(best_path, "tft/model.ckpt")
    print(f"Сохранено           : tft/model.ckpt (финальный fallback)")

print("\n" + "=" * 60)
print("Готово.")
print("  Сохранено : tft/model.ckpt")
print("  Следующий шаг: python tft/predict.py")
print("=" * 60)