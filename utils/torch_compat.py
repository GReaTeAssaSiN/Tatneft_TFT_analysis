"""
Совместимость с PyTorch 2.6: патч torch.load для weights_only=False.

PyTorch 2.6 изменил дефолт weights_only=True, что блокирует классы
pytorch_forecasting и numpy при загрузке чекпоинтов через Lightning.
Импортировать этот модуль ДО вызова load_from_checkpoint / trainer.fit с ckpt_path.
Безопасно: чекпоинты создаются самим проектом.
"""

import logging as _logging
import warnings
import torch as _torch

_orig = _torch.load


def _patched_load(f, *args, **kwargs):
    # Принудительно устанавливаем weights_only=False независимо от того,
    # что передал вызывающий код (в том числе убираем weights_only=True по умолчанию).
    kwargs["weights_only"] = False
    return _orig(f, *args, **kwargs)


_torch.load = _patched_load

# Подавляем предупреждение PyTorch 2.6 о weights_only.
# 1. Python warnings — для кода, вызывающего torch.load через torch.load(...)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")   # любая категория

# 2. logging.captureWarnings — Lightning роутит warnings через logging.
#    Глушим канал py.warnings, чтобы они не всплывали через прогресс-бар.
_logging.getLogger("py.warnings").setLevel(_logging.CRITICAL)
_logging.getLogger("py.warnings").propagate = False
