"""
Совместимость с PyTorch 2.6: патч torch.load для weights_only=False.

PyTorch 2.6 изменил дефолт weights_only=True, что блокирует классы
pytorch_forecasting и numpy при загрузке чекпоинтов через Lightning.
Импортировать этот модуль ДО вызова load_from_checkpoint / trainer.fit с ckpt_path.
Безопасно: чекпоинты создаются самим проектом.
"""

import torch as _torch

_orig = _torch.load


def _patched_load(f, map_location=None, pickle_module=None, weights_only=True, **kw):
    return _orig(f, map_location=map_location, weights_only=False, **kw)


_torch.load = _patched_load
