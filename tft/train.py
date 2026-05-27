"""
Обучение TFT модели.
Запускать из корня проекта: python tft/train.py
Входные данные : tft/training_dataset.pkl, tft/dataset_config.pkl (из prepare_dataset.py)
Выходные данные: tft/model.ckpt, tft/checkpoints/, tft/logs/
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
HIDDEN_CONTINUOUS = HIDDEN_SIZE // 2  # рекомендация TFT: hidden_continuous <= hidden_size / 2
GRADIENT_CLIP = 1.0

# ============================================================
# Загрузка конфигурации гиперпараметров (tft/train_config.json)
# Если файл есть — переопределяем дефолты из секции выше.
# ============================================================
_PATIENCE = 12  # EarlyStopping patience (default)
_cfg_path = os.path.join(os.path.dirname(__file__), "train_config.json")
if os.path.exists(_cfg_path):
    import json as _json_cfg
    with open(_cfg_path, encoding="utf-8") as _f:
        _cfg = _json_cfg.load(_f)
    BATCH_SIZE       = int(_cfg.get("batch_size",            BATCH_SIZE))
    EPOCHS           = int(_cfg.get("max_epochs",            EPOCHS))
    HIDDEN_SIZE      = int(_cfg.get("hidden_size",           HIDDEN_SIZE))
    ATTN_HEADS       = int(_cfg.get("attention_head_size",   ATTN_HEADS))
    LEARNING_RATE    = float(_cfg.get("learning_rate",       LEARNING_RATE))
    DROPOUT          = float(_cfg.get("dropout",             DROPOUT))
    GRADIENT_CLIP    = float(_cfg.get("gradient_clip",       GRADIENT_CLIP))
    HIDDEN_CONTINUOUS = int(_cfg.get("hidden_continuous_size", HIDDEN_SIZE // 2))
    _PATIENCE        = int(_cfg.get("patience",              _PATIENCE))
    print(f"  Конфиг загружен     : {_cfg_path}")

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
enc_d = config['encoder_length']
pred_d = config['prediction_length']
print(f"  Encoder length      : {enc_d} дн. (ретроспектива 1 месяц)")
print(f"  Prediction length   : {pred_d} дн. (горизонт прогноза 1 неделя)")
print(f"  Станций             : {config['n_stations']}")

# ============================================================
# DataLoaders
# ============================================================
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

df = pd.read_csv("data/prepared_data.csv", parse_dates=["date"])
df["station_id"] = df["station_id"].astype(str)

for col in config["static_cats"] + config["known_cats"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

TRAIN_END = pd.Timestamp(config["train_end"])
VAL_END   = pd.Timestamp(config["val_end"])

# val_df начинается с TRAIN_END - ENCODER_LENGTH + 1 = Oct 2.
# Это гарантирует, что все декодерные окна попадают в ноябрь (out-of-sample),
# а не в Jan–Oct (train-период), что давало бы смещённый val_loss.
_enc = config["encoder_length"]
VAL_START = TRAIN_END - pd.Timedelta(days=_enc - 1)

train_df = df[df["date"] <= TRAIN_END].copy()
val_df   = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()

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
    # log_interval: каждые N шагов пишет prediction-графики в TensorBoard.
    # При encoder=30 дн. / decoder=7 дн., 5 станций, batch=32 → ~48–60 батчей/эпоха.
    # log_interval=10 → ~5 точек на эпоху (достаточная детализация).
    log_interval=10,
    reduce_on_plateau_patience=5,
)

total_params = sum(p.numel() for p in tft.parameters())
print(f"  Параметров модели   : {total_params:,}")

# ============================================================
# Callbacks
# ============================================================
os.makedirs("tft/checkpoints", exist_ok=True)
os.makedirs("tft/logs", exist_ok=True)

# Версионный чекпоинт: имя содержит epoch + val_loss, удобен для восстановления.
checkpoint_cb = ModelCheckpoint(
    dirpath="tft/checkpoints",
    filename="tft-epoch={epoch:02d}-val_loss={val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

# BestModelSync: копирует лучший чекпоинт в tft/model.ckpt после каждого улучшения.
# Заменяет второй ModelCheckpoint — Lightning 2.x не допускает два экземпляра
# ModelCheckpoint с одинаковым state_key (RuntimeError).
class BestModelSync(pl.Callback):
    """Создаёт/обновляет tft/model.ckpt двумя путями:
    1. on_train_epoch_end  — страховка: сохраняет текущее состояние сразу после
       обучающей части эпохи, не дожидаясь валидации. Гарантирует, что файл
       существует уже после первой эпохи, даже если обучение будет прервано.
    2. on_validation_epoch_end — основной триггер: перезаписывает файл лучшим
       чекпоинтом по val_loss (через ModelCheckpoint), когда val_loss улучшился.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        # Страховочное сохранение после каждой train-эпохи.
        # trainer.save_checkpoint сохраняет полный Lightning-чекпоинт
        # (веса + оптимайзер + lr_scheduler), совместимый с load_from_checkpoint.
        # on_validation_epoch_end позже перезапишет его лучшим по val_loss.
        trainer.save_checkpoint("tft/model.ckpt")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Основной триггер: заменяем страховочный файл лучшим чекпоинтом.
        src = checkpoint_cb.best_model_path
        if src and os.path.exists(src):
            shutil.copy(src, "tft/model.ckpt")

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=_PATIENCE,
    mode="min",
    verbose=True,
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

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
    # BestModelSync стоит после checkpoint_cb — чтобы best_model_path уже был обновлён.
    callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, BestModelSync()],
    enable_model_summary=True,
    # При encoder=30 дн. / decoder=7 дн., 5 станций, batch=32 → ~48–60 батчей/эпоха.
    # log_every_n_steps=10 → ~5–6 точек на эпоху в TensorBoard (достаточно).
    # num_sanity_val_steps=1 → 1 батч для проверки val pipeline вместо 2.
    # precision="32-true" → явный float32, убирает предупреждение Lightning.
    log_every_n_steps=10,
    num_sanity_val_steps=1,
    precision="32-true",
    logger=tb_logger,
)

# Lightning при создании Trainer-а вызывает warnings.simplefilter("default"),
# сбрасывая фильтры, установленные выше. Поэтому переустанавливаем ПОСЛЕ Trainer.
# Также глушим канал logging.captureWarnings — Lightning роутит предупреждения туда.
import logging as _logging
warnings.filterwarnings("ignore")
_logging.getLogger("py.warnings").setLevel(_logging.CRITICAL)
_logging.getLogger("py.warnings").propagate = False

try:
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt,
    )
except (RuntimeError, Exception) as _e:
    # Если чекпоинт несовместим с текущей архитектурой (size mismatch после
    # изменения переменных/гиперпараметров) — удаляем его и стартуем с нуля.
    _emsg = str(_e)
    if resume_ckpt and ("size mismatch" in _emsg or "unexpected key" in _emsg or "missing key" in _emsg):
        print(f"\n[WARNING] Чекпоинт несовместим с архитектурой: {_emsg[:200]}")
        print("[WARNING] Удаляем несовместимые чекпоинты и обучаем с нуля...")
        for _f in glob.glob("tft/checkpoints/*.ckpt"):
            os.remove(_f)
        if os.path.exists("tft/model.ckpt"):
            os.remove("tft/model.ckpt")
        trainer.fit(
            tft,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=None,  # с нуля
        )
    else:
        raise

# ============================================================
# Итог
# ============================================================
best_path = checkpoint_cb.best_model_path
print(f"\nЛучший чекпоинт     : {best_path}")
print(f"Лучший val_loss     : {checkpoint_cb.best_model_score:.4f}")

# BestModelSync копирует model.ckpt на лету при каждом улучшении val_loss.
# Финальный fallback: если по какой-то причине файл не появился — копируем явно.
if not os.path.exists("tft/model.ckpt") and best_path:
    shutil.copy(best_path, "tft/model.ckpt")
    print("Сохранено           : tft/model.ckpt (fallback)")
else:
    print("Сохранено           : tft/model.ckpt ✓")

print("\n" + "=" * 60)
print("Готово.")
print("  Сохранено : tft/model.ckpt")
print("  Следующий шаг: python tft/predict.py")
print("=" * 60)