"""
Дашборд управления обучением TFT.
Запускать из корня проекта:
    streamlit run dashboard/train_dashboard.py
"""

import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Пути
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT, "tft", "train_config.json")
LOG_DIR     = os.path.join(ROOT, "tft", "logs")
CKPT_DIR    = os.path.join(ROOT, "tft", "checkpoints")
MODEL_PATH  = os.path.join(ROOT, "tft", "model.ckpt")
TRAIN_PY    = os.path.join(ROOT, "tft", "train.py")

# ============================================================
# Цвета
# ============================================================
GOLD       = "#c8a84b"
GREEN      = "#4ECB71"
RED        = "#E24B4A"
BLUE       = "#2E75B6"
TEAL       = "#1ABC9C"
GRAY       = "#8B949E"
CARD_BG    = "#13161f"
GRID_COLOR = "#1e2235"

# ============================================================
# Конфигурация страницы
# ============================================================
st.set_page_config(
    page_title="TFT Обучение",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background-color: #0d1117; color: #c9d1d9; }}
[data-testid="stHeader"] {{ background-color: #0d1117; }}
[data-testid="stSidebar"] {{ background-color: #0d1117; }}
section[data-testid="stSidebar"] {{ background-color: #0d1117; }}
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD_BG};
    border-radius: 8px;
    padding: 2px 4px;
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {GRAY};
    background-color: transparent;
    border-radius: 6px;
    padding: 6px 16px;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD} !important;
    background-color: #1e2235 !important;
}}
div[data-testid="metric-container"] {{
    background-color: {CARD_BG};
    border: 1px solid {GRID_COLOR};
    border-radius: 8px;
    padding: 12px 16px;
}}
.stCodeBlock {{ background-color: #0a0e17 !important; }}
pre code {{ font-size: 11px !important; line-height: 1.4 !important; }}
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label {{
    color: {GRAY} !important;
    font-size: 12px !important;
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session state
# ============================================================
def _init_state():
    defaults = {
        "train_proc":       None,
        "train_output":     [],
        "train_start_time": None,
        "train_end_time":   None,
        "output_queue":     queue.Queue(),
        "reader_thread":    None,
        # Значения пресета для заполнения виджетов (без автосохранения на диск)
        "_preset_values":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ============================================================
# Конфиг
# ============================================================
_CPU_DEFAULTS = {
    "batch_size": 32, "max_epochs": 50,
    "hidden_size": 64, "attention_head_size": 2,
    "hidden_continuous_size": 32,
    "learning_rate": 3e-4, "dropout": 0.15,
    "gradient_clip": 1.0, "patience": 12,
}
_GPU_DEFAULTS = {
    "batch_size": 64, "max_epochs": 80,
    "hidden_size": 128, "attention_head_size": 4,
    "hidden_continuous_size": 64,
    "learning_rate": 3e-4, "dropout": 0.15,
    "gradient_clip": 1.0, "patience": 12,
}


def load_config() -> dict:
    """Загружает train_config.json или возвращает CPU-дефолты."""
    cfg = dict(_CPU_DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update(saved)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def delete_config():
    if os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)


# ============================================================
# TensorBoard: чтение кривых потерь
# ============================================================
def read_tb_losses() -> list[tuple[str, object, object]]:
    """
    Читает ВСЕ версии TensorBoard-логов.
    Возвращает list[(label, train_df, val_df)] — от старых к новым.
    label = "v0", "v1", ... "vN (текущий)"
    """
    import pandas as _pd

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return []

    versions = sorted(glob.glob(os.path.join(LOG_DIR, "tft_model", "version_*")))
    if not versions:
        return []

    result = []
    for v_path in versions:
        v_num = os.path.basename(v_path).replace("version_", "v")
        ea = EventAccumulator(v_path, size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            # Версия есть, но прочитать не удалось — добавляем с пустыми данными
            result.append((v_num, None, None))
            continue

        tags = ea.Tags().get("scalars", [])

        def _to_df(tag):
            if tag not in tags:
                return None
            evs = ea.Scalars(tag)
            if not evs:
                return None
            return _pd.DataFrame({"step":      [e.step      for e in evs],
                                   "value":     [e.value     for e in evs],
                                   "wall_time": [e.wall_time for e in evs]})

        train_df = _to_df("train_loss_step")
        if train_df is None:
            train_df = _to_df("train_loss_epoch")
        if train_df is None:
            train_df = _to_df("train_loss")
        val_df = _to_df("val_loss")

        # Добавляем всегда — даже если оба None (пустая/прерванная версия)
        result.append((v_num, train_df, val_df))

    # Помечаем последнюю версию как текущую
    if result:
        label, td, vd = result[-1]
        result[-1] = (f"{label} (текущий)", td, vd)

    return result


def get_tb_versions() -> list[str]:
    return sorted(glob.glob(os.path.join(LOG_DIR, "tft_model", "version_*")))


# ============================================================
# Управление процессом обучения
# ============================================================
def is_training() -> bool:
    proc = st.session_state.train_proc
    return proc is not None and proc.poll() is None


def start_training():
    if is_training():
        return
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, TRAIN_PY],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=env,
    )
    st.session_state.train_proc       = proc
    st.session_state.train_output     = []
    st.session_state.train_start_time = time.time()
    st.session_state.train_end_time   = None
    st.session_state.output_queue     = queue.Queue()

    def _reader(p, q):
        for line in p.stdout:
            q.put(line.rstrip("\r\n"))
        q.put(None)  # sentinel

    t = threading.Thread(
        target=_reader,
        args=(proc, st.session_state.output_queue),
        daemon=True,
    )
    t.start()
    st.session_state.reader_thread = t


def stop_training():
    proc = st.session_state.train_proc
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    if st.session_state.train_end_time is None:
        st.session_state.train_end_time = time.time()


def drain_queue() -> bool:
    """Сливает очередь вывода в session_state.train_output. True = были новые строки."""
    q = st.session_state.output_queue
    new = False
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            # Процесс завершился — фиксируем время
            if st.session_state.train_end_time is None:
                st.session_state.train_end_time = time.time()
            break
        st.session_state.train_output.append(line)
        new = True
    return new


def elapsed_str(start, end=None) -> str:
    if start is None:
        return "—"
    secs = int((end or time.time()) - start)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# Рендер вывода процесса обучения
# ============================================================
def render_output_html(lines: list[str]) -> str:
    """
    Преобразует сырые строки вывода train.py в цветной HTML.
    - Дедуплицирует прогресс-бары (оставляет последнее состояние каждого)
    - Фильтрует мусор от Lightning (советы, GPU-строки)
    - Убирает таблицу параметров модели (заменяет сводкой)
    - Раскрашивает: разделители, эпохи, val_loss, ошибки
    """
    import html as _html
    import re as _re

    # ── Шаг 1: очистка \r и дедупликация прогресс-баров ──────────
    _pb_re = _re.compile(r'^(.+?):\s+\d+%\|')
    deduped: list[str] = []
    last_pb_key: str | None = None

    for raw in lines:
        line = raw.replace('\r', '')
        m = _pb_re.match(line)
        if m:
            key = m.group(1).strip()
            if key == last_pb_key and deduped:
                deduped[-1] = line      # перезаписываем предыдущий бар
            else:
                deduped.append(line)
                last_pb_key = key
        else:
            last_pb_key = None
            deduped.append(line)

    # ── Шаг 2: раскраска ─────────────────────────────────────────
    # Паттерн строки параметров модели
    _tbl_row   = _re.compile(r'^\s*\d+\s+\|\s+\w')
    _tbl_hdr   = _re.compile(r'\|\s*Name\s*\|')
    _param_sum = _re.compile(r'(Trainable|Non-trainable|Total)\s+params')
    _separator = _re.compile(r'^[=\-]{8,}\s*$')
    _epoch_pb  = _re.compile(r'^Epoch\s+\d+:')
    _val_loss  = _re.compile(r'val_loss', _re.IGNORECASE)
    _kv_line   = _re.compile(r'^\s{2,}\S[^:]{2,}:\s+\S')
    _all_caps  = _re.compile(r'^[A-ZА-ЯЁ0-9\s\-–—]{8,}$')

    in_model_table = False
    html_parts: list[str] = []

    def _span(text: str, color: str, bold: bool = False, dim: bool = False) -> str:
        opacity = "opacity:0.35;" if dim else ""
        weight  = "font-weight:700;" if bold else ""
        return f'<span style="color:{color};{weight}{opacity}">{text}</span>'

    for line in deduped:
        stripped = line.strip()
        esc      = _html.escape(line)

        # ─ Пустая строка ─
        if not stripped:
            html_parts.append('')
            continue

        # ─ Фильтр: советы Lightning ─
        if stripped.startswith('💡') or ('Tip:' in stripped and 'litlogger' in stripped.lower()):
            continue

        # ─ GPU/TPU/HPU available — приглушённо, но читаемо ─
        if _re.match(r'^(GPU|TPU|HPU)\s+available:', stripped):
            html_parts.append(_span(esc, '#5a6270'))
            continue

        # ─ Таблица слоёв модели (заголовок) ─
        if _tbl_hdr.search(stripped):
            in_model_table = True
            html_parts.append(_span('  ┌─ Архитектура модели ────────────────', GRAY))
            continue

        # ─ Таблица слоёв модели (строки) ─
        if in_model_table and (_tbl_row.match(stripped) or stripped.startswith('|')):
            html_parts.append(_span(esc, '#4a5260'))
            continue

        # ─ Сводка параметров (Trainable / Total) ─
        if _param_sum.search(stripped):
            in_model_table = False
            html_parts.append(_span(esc, BLUE, bold=True))
            continue

        # ─ Прочие строки после таблицы сбрасывают флаг ─
        if in_model_table and not stripped.startswith('|'):
            in_model_table = False

        # ─ Разделители === / --- ─
        if _separator.match(stripped):
            html_parts.append(_span(esc, '#2a2f45'))
            continue

        # ─ Ошибки ─
        if _re.search(r'(Error|Traceback|Exception)', stripped):
            html_parts.append(_span(esc, RED, bold=True))
            continue

        # ─ val_loss (EarlyStopping, ModelCheckpoint, метрики) ─
        if _val_loss.search(stripped):
            html_parts.append(_span(esc, GREEN))
            continue

        # ─ Сохранено / лучший чекпоинт ─
        if _re.search(r'(Сохранено|model\.ckpt|Лучший|best)', stripped, _re.IGNORECASE):
            html_parts.append(_span(esc, GREEN, bold=True))
            continue

        # ─ Прогресс-бар Epoch ─
        if _epoch_pb.match(stripped):
            html_parts.append(_span(esc, GOLD))
            continue

        # ─ Прогресс-бары Sanity/Validation ─
        if _re.search(r'\d+%\|', stripped) or 'it/s' in stripped:
            html_parts.append(_span(esc, TEAL))
            continue

        # ─ Restoring / Restored ─
        if _re.match(r'Rest(oring|ored)', stripped):
            html_parts.append(_span(esc, GRAY))
            continue

        # ─ Продолжение обучения с чекпоинта ─
        if 'чекпоинта' in stripped or 'обучение с нуля' in stripped:
            html_parts.append(_span(esc, TEAL))
            continue

        # ─ Заголовки секций (CAPS или KEY:VALUE с отступом) ─
        if _all_caps.match(stripped) and len(stripped) > 6:
            html_parts.append(_span(esc, GOLD, bold=True))
            continue

        # ─ Ключ : Значение (отступ, >=2 пробела) ─
        if _kv_line.match(line):
            # Подсвечиваем ключ серым, значение — белым
            colon_idx = esc.find(':')
            if colon_idx > 0:
                key_part = esc[:colon_idx + 1]
                val_part = esc[colon_idx + 1:]
                html_parts.append(
                    f'<span style="color:{GRAY};">{key_part}</span>'
                    f'<span style="color:#c9d1d9;">{val_part}</span>'
                )
                continue

        # ─ По умолчанию ─
        html_parts.append(f'<span style="color:#c9d1d9;">{esc}</span>')

    return '\n'.join(html_parts)


# ============================================================
# Утилиты UI
# ============================================================
def section_header(icon: str, title: str, subtitle: str = "", color: str = GOLD):
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:16px 0 12px 0;">
  <span style="color:{color};font-size:11px;font-weight:700;letter-spacing:.1em;
               text-transform:uppercase;white-space:nowrap;">{icon} {title}</span>
  <span style="flex:1;height:1px;background:#2a2f45;"></span>
  {"" if not subtitle else
   f'<span style="color:#4b5563;font-size:11px;white-space:nowrap;">{subtitle}</span>'}
</div>
""", unsafe_allow_html=True)


def kpi(label, value, delta=None, color=GOLD):
    delta_html = ""
    if delta:
        delta_html = f'<div style="font-size:11px;color:{GRAY};margin-top:2px;">{delta}</div>'
    st.markdown(f"""
<div style="background:{CARD_BG};border:1px solid {GRID_COLOR};border-radius:8px;
            padding:12px 16px;margin-bottom:8px;">
  <div style="font-size:11px;color:{GRAY};letter-spacing:.07em;text-transform:uppercase;
              margin-bottom:4px;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:{color};">{value}</div>
  {delta_html}
</div>
""", unsafe_allow_html=True)


# ============================================================
# Заголовок страницы
# ============================================================
st.markdown(
    f"<h1 style='color:{GOLD};margin:0 0 2px 0;font-size:26px;'>🧠 TFT — Управление обучением</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:{GRAY};margin:0 0 18px 0;font-size:13px;'>"
    "Настройка гиперпараметров · Запуск обучения · Мониторинг результатов</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["⚙️  Гиперпараметры", "🚀  Обучение", "📈  Результаты"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Гиперпараметры
# ════════════════════════════════════════════════════════════════
with tab1:
    # Загружаем конфиг с диска (или дефолты)
    cfg = load_config()
    # Если выбран пресет — применяем значения в UI без сохранения на диск
    if st.session_state._preset_values is not None:
        cfg.update(st.session_state._preset_values)
        st.session_state._preset_values = None

    config_exists = os.path.exists(CONFIG_PATH)

    section_header("⚙️", "АРХИТЕКТУРА МОДЕЛИ")
    arch1, arch2, arch3 = st.columns(3)

    with arch1:
        st.markdown(f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 8px 0;'>"
                    "Размеры скрытых слоёв</p>", unsafe_allow_html=True)
        hidden_size = st.select_slider(
            "Hidden size",
            options=[32, 64, 128, 256],
            value=cfg["hidden_size"],
            help="Размер скрытого состояния LSTM и VSN. Увеличение → больше параметров и памяти.",
        )
        attn_heads = st.select_slider(
            "Attention heads",
            options=[1, 2, 4, 8],
            value=cfg["attention_head_size"],
            help="Число голов многоголового внимания. hidden_size должен делиться на heads.",
        )
        hidden_cont = st.number_input(
            "Hidden continuous size",
            min_value=8, max_value=256, step=8,
            value=cfg["hidden_continuous_size"],
            help="Рекомендация TFT: ≤ hidden_size / 2. Размер проекции непрерывных входов.",
        )

    with arch2:
        st.markdown(f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 8px 0;'>"
                    "Регуляризация и оптимизация</p>", unsafe_allow_html=True)
        lr_val = float(cfg["learning_rate"])
        lr = st.number_input(
            "Learning rate",
            min_value=1e-5, max_value=1e-2, step=1e-5,
            value=lr_val, format="%.5f",
            help="Скорость обучения. Рекомендация: 1e-4 … 5e-4.",
        )
        dropout = st.slider(
            "Dropout",
            min_value=0.0, max_value=0.5, step=0.05,
            value=float(cfg["dropout"]),
            help="Вероятность Dropout. Снижает переобучение.",
        )
        grad_clip = st.slider(
            "Gradient clip",
            min_value=0.1, max_value=5.0, step=0.1,
            value=float(cfg["gradient_clip"]),
            help="Обрезка градиентов по норме. Стабилизирует обучение.",
        )

    with arch3:
        st.markdown(f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 8px 0;'>"
                    "Расписание обучения</p>", unsafe_allow_html=True)
        batch_size = st.select_slider(
            "Batch size",
            options=[16, 32, 64, 128],
            value=cfg["batch_size"],
            help="Размер мини-батча. CPU: 32, GPU: 64.",
        )
        max_epochs = st.number_input(
            "Max epochs",
            min_value=1, max_value=300, step=5,
            value=int(cfg["max_epochs"]),
            help="Максимальное число эпох. EarlyStopping может остановить раньше.",
        )
        patience = st.number_input(
            "Early stopping patience",
            min_value=1, max_value=50, step=1,
            value=int(cfg["patience"]),
            help="Сколько эпох без улучшения val_loss допускается до остановки.",
        )

    # Предупреждение если hidden_cont > hidden_size // 2
    if hidden_cont > hidden_size // 2:
        st.warning(
            f"⚠️ Hidden continuous ({hidden_cont}) > hidden_size/2 ({hidden_size // 2}). "
            "Рекомендация TFT: hidden_continuous ≤ hidden_size / 2."
        )
    # Предупреждение если hidden_size % attn_heads != 0
    if hidden_size % attn_heads != 0:
        st.error(
            f"❌ hidden_size ({hidden_size}) не делится на attention_heads ({attn_heads}). "
            "Измените параметры — иначе модель не создастся."
        )

    section_header("💾", "ДЕЙСТВИЯ", color=GRAY)
    btn1, btn2, btn3, btn4, btn5 = st.columns(5)

    new_cfg = {
        "batch_size":            batch_size,
        "max_epochs":            max_epochs,
        "hidden_size":           hidden_size,
        "attention_head_size":   attn_heads,
        "hidden_continuous_size": hidden_cont,
        "learning_rate":         lr,
        "dropout":               dropout,
        "gradient_clip":         grad_clip,
        "patience":              patience,
    }

    with btn1:
        if st.button("💾  Сохранить конфиг", type="primary", width="stretch"):
            save_config(new_cfg)
            st.toast("Конфиг сохранён!", icon="✅")
            st.rerun()

    with btn2:
        if config_exists:
            if st.button(
                "📂  Загрузить из файла",
                width="stretch",
                help="Сбросить ползунки к значениям из сохранённого tft/train_config.json",
            ):
                with open(CONFIG_PATH, encoding="utf-8") as _f:
                    _saved = json.load(_f)
                st.session_state._preset_values = _saved
                st.toast("Конфиг загружен из файла!", icon="📂")
                st.rerun()
        else:
            st.button(
                "📂  Загрузить из файла",
                disabled=True,
                width="stretch",
                help="Нет сохранённого конфига — нечего загружать",
            )

    with btn3:
        if st.button("🔄  По умолчанию (CPU)", width="stretch"):
            st.session_state._preset_values = dict(_CPU_DEFAULTS)
            st.rerun()

    with btn4:
        if st.button("🎮  Пресет GPU", width="stretch"):
            st.session_state._preset_values = dict(_GPU_DEFAULTS)
            st.rerun()

    with btn5:
        if config_exists:
            if st.button("🗑️  Удалить конфиг", width="stretch"):
                delete_config()
                st.rerun()
        else:
            st.markdown(
                f"<div style='color:#e0a020;font-size:12px;font-weight:600;"
                f"padding:8px 0;letter-spacing:.02em;'>"
                "⚠️ Конфиг не сохранён</div>"
                f"<div style='color:{GRAY};font-size:11px;margin-top:-4px;'>"
                "используются авто-дефолты</div>",
                unsafe_allow_html=True,
            )

    # Превью JSON
    with st.expander("📄  Превью конфига (JSON)", expanded=False):
        st.json(new_cfg)

    if config_exists:
        st.markdown(
            f"<p style='color:{GRAY};font-size:11px;margin-top:8px;'>"
            f"📂 Активный конфиг: <code style='color:{TEAL};'>{CONFIG_PATH}</code></p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='color:{GRAY};font-size:11px;margin-top:8px;'>"
            "⚠️ Конфиг не сохранён. Нажмите «Сохранить конфиг» перед запуском.</p>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════
# TAB 2 — Обучение
# ════════════════════════════════════════════════════════════════
with tab2:
    # Слить новые строки из фонового процесса
    drain_queue()

    training_active   = is_training()
    has_output        = bool(st.session_state.train_output)
    proc              = st.session_state.train_proc
    exit_code         = proc.returncode if (proc and not training_active) else None
    start_t           = st.session_state.train_start_time
    end_t             = st.session_state.train_end_time

    elapsed = elapsed_str(start_t, end_t if not training_active else None)

    # ── Кнопки управления ────────────────────────────────────────
    section_header("🚀", "УПРАВЛЕНИЕ ОБУЧЕНИЕМ")
    ctrl1, ctrl2, _ = st.columns([2, 2, 6])

    with ctrl1:
        if training_active:
            if st.button("⏹  Остановить", type="secondary", width="stretch"):
                stop_training()
                st.rerun()
        else:
            prereqs_ok = (
                os.path.exists(os.path.join(ROOT, "tft", "training_dataset.pkl"))
                and os.path.exists(os.path.join(ROOT, "data", "prepared_data.csv"))
            )
            if not prereqs_ok:
                st.button("▶️  Запустить", disabled=True, width="stretch")
            else:
                if st.button("▶️  Запустить обучение", type="primary", width="stretch"):
                    start_training()
                    st.rerun()
            # Кнопка сброса — под кнопкой запуска, только если есть предыдущий запуск
            if proc is not None:
                if st.button(
                    "🔄  Сбросить состояние",
                    width="stretch",
                    help="Очищает данные о предыдущем запуске. Нажмите, если кнопка «Запустить» недоступна после ошибки.",
                ):
                    st.session_state.train_proc       = None
                    st.session_state.train_output     = []
                    st.session_state.train_start_time = None
                    st.session_state.train_end_time   = None
                    st.rerun()

    with ctrl2:
        if training_active:
            st.metric("Статус", "⚡ Обучение", f"прошло {elapsed}")
        elif has_output:
            if exit_code == 0:
                st.metric("Статус", "✅ Завершено", f"за {elapsed}")
            else:
                code_str = f"код {exit_code}" if exit_code is not None else "остановлено"
                st.metric("Статус", "🛑 Прервано", code_str)
        else:
            st.metric("Статус", "💤 Ожидание", "")

    # ── Предупреждение если нет пресетов ─────────────────────────
    if not os.path.exists(os.path.join(ROOT, "tft", "training_dataset.pkl")):
        st.warning(
            "⚠️ Файл `tft/training_dataset.pkl` не найден. "
            "Сначала запустите `python tft/prepare_dataset.py`."
        )
    elif not os.path.exists(os.path.join(ROOT, "data", "prepared_data.csv")):
        st.warning(
            "⚠️ Файл `data/prepared_data.csv` не найден. "
            "Сначала запустите `python eda/eda_preprocessing.py`."
        )

    # ── Вывод процесса ───────────────────────────────────────────
    section_header("📋", "ВЫВОД ПРОЦЕССА",
                   subtitle=f"{len(st.session_state.train_output)} строк" if has_output else "")

    if has_output:
        lines = st.session_state.train_output
        display_lines = lines[-300:]
        colored_html = render_output_html(display_lines)

        st.markdown(f"""
<div style="background:#0a0e17;border:1px solid {GRID_COLOR};border-radius:6px;
            padding:12px 16px;height:440px;overflow-y:auto;font-family:'Consolas','Courier New',monospace;
            font-size:11px;line-height:1.6;white-space:pre-wrap;"
     id="train-output">
{colored_html}
</div>
<script>
const el = document.getElementById('train-output');
if(el) el.scrollTop = el.scrollHeight;
</script>
""", unsafe_allow_html=True)
        if len(lines) > 300:
            st.caption(f"Показаны последние 300 из {len(lines)} строк.")
    else:
        st.markdown(
            f"<div style='color:{GRAY};font-size:12px;padding:40px 0;text-align:center;'>"
            "Нажмите «Запустить обучение» для начала. Вывод появится здесь.</div>",
            unsafe_allow_html=True,
        )

    # ── Авто-обновление пока идёт обучение ──────────────────────
    if training_active:
        time.sleep(2)
        st.rerun()


# ════════════════════════════════════════════════════════════════
# TAB 3 — Результаты
# ════════════════════════════════════════════════════════════════
with tab3:
    section_header("🏆", "СОСТОЯНИЕ МОДЕЛИ")

    # KPI-строка
    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)

    with kpi_c1:
        if os.path.exists(MODEL_PATH):
            mtime = os.path.getmtime(MODEL_PATH)
            import datetime as _dt
            ts = _dt.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M")
            size_mb = os.path.getsize(MODEL_PATH) / 1e6
            kpi("Лучшая модель", f"{size_mb:.1f} МБ", f"обновлено {ts}", GREEN)
        else:
            kpi("Лучшая модель", "нет", "запустите обучение", GRAY)

    with kpi_c2:
        ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")))
        if ckpt_files:
            last_name = os.path.basename(ckpt_files[-1])
            kpi("Последний чекпоинт", last_name[:30], f"{len(ckpt_files)} файл(ов)", BLUE)
        else:
            kpi("Последний чекпоинт", "нет", "", GRAY)

    with kpi_c3:
        versions = get_tb_versions()
        kpi("TensorBoard версии", str(len(versions)), "run tensorboard --logdir tft/logs", TEAL)

    # ── Кривые потерь ────────────────────────────────────────────
    section_header("📈", "КРИВЫЕ ПОТЕРЬ", subtitle="из TensorBoard логов")

    try:
        all_versions = read_tb_losses()
    except Exception as e:
        all_versions = []
        st.caption(f"Ошибка чтения логов: {e}")

    if all_versions:
        import pandas as _pd_stats

        fig = go.Figure()
        n = len(all_versions)
        _old_train_shown = False
        _old_val_shown   = False

        for i, (label, train_df, val_df) in enumerate(all_versions):
            is_latest = (i == n - 1)

            if is_latest:
                # ── Текущий запуск — яркие цвета ──────────────────────
                if train_df is not None and len(train_df):
                    fig.add_trace(go.Scatter(
                        x=train_df["step"], y=train_df["value"],
                        name=f"Train loss ({label})",
                        mode="lines",
                        line=dict(color=GOLD, width=2.5),
                        legendgroup="cur",
                    ))
                if val_df is not None and len(val_df):
                    _single = len(val_df) == 1
                    fig.add_trace(go.Scatter(
                        x=val_df["step"], y=val_df["value"],
                        name=f"Val loss ({label})",
                        mode="markers" if _single else "lines+markers",
                        line=dict(color=GREEN, width=2),
                        marker=dict(size=10 if _single else 6, color=GREEN,
                                    line=dict(color="#fff", width=1)),
                        legendgroup="cur_val",
                    ))
                    if not _single:
                        _bi = val_df["value"].idxmin()
                        fig.add_trace(go.Scatter(
                            x=[val_df.loc[_bi, "step"]], y=[val_df.loc[_bi, "value"]],
                            mode="markers+text",
                            name=f"Best val: {val_df['value'].min():.4f}",
                            marker=dict(symbol="star", size=16, color=GREEN,
                                        line=dict(color="#fff", width=1)),
                            text=[f"  {val_df['value'].min():.4f}"],
                            textfont=dict(color=GREEN, size=11),
                            textposition="middle right",
                            legendgroup="best",
                        ))
            else:
                # ── Прошлые запуски — зелёные ─────────────────────────
                if train_df is not None and len(train_df):
                    fig.add_trace(go.Scatter(
                        x=train_df["step"], y=train_df["value"],
                        name="Train (прошлые)",
                        mode="lines",
                        line=dict(color=GREEN, width=1.2),
                        opacity=0.55,
                        legendgroup="old_train",
                        showlegend=not _old_train_shown,
                    ))
                    _old_train_shown = True
                if val_df is not None and len(val_df):
                    fig.add_trace(go.Scatter(
                        x=val_df["step"], y=val_df["value"],
                        name="Val (прошлые)",
                        mode="markers",
                        marker=dict(size=8, color=GREEN, opacity=0.6,
                                    line=dict(color="#fff", width=0.8)),
                        legendgroup="old_val",
                        showlegend=not _old_val_shown,
                    ))
                    _old_val_shown = True

        fig.update_layout(
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            font=dict(color="#c9d1d9", size=11),
            xaxis=dict(title="Шаг обучения", gridcolor=GRID_COLOR,
                       zerolinecolor=GRID_COLOR, tickformat=",d"),
            yaxis=dict(title="QuantileLoss", gridcolor=GRID_COLOR,
                       zerolinecolor=GRID_COLOR),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
            ),
            margin=dict(l=55, r=20, t=55, b=50),
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        # ── Таблица метрик по всем запускам ───────────────────────────
        section_header("📊", "МЕТРИКИ ПО ЗАПУСКАМ", color=GRAY)
        def _fmt_dur(seconds):
            if seconds is None or seconds <= 0:
                return "—"
            h, r = divmod(int(seconds), 3600)
            m, s = divmod(r, 60)
            if h > 0:
                return f"{h}ч {m:02d}мин"
            elif m > 0:
                return f"{m}мин {s:02d}с"
            return f"{s}с"

        rows = []
        for label, train_df, val_df in all_versions:
            n_train_pts = len(train_df) if train_df is not None else 0
            n_val_pts   = len(val_df)   if val_df   is not None else 0

            # Длительность — из wall_time первого и последнего события
            duration = None
            if train_df is not None and n_train_pts >= 2 and "wall_time" in train_df.columns:
                duration = train_df["wall_time"].iloc[-1] - train_df["wall_time"].iloc[0]
            elif val_df is not None and n_val_pts >= 2 and "wall_time" in val_df.columns:
                duration = val_df["wall_time"].iloc[-1] - val_df["wall_time"].iloc[0]

            rows.append({
                "Запуск":          label,
                "Статус":          "✅ завершён" if n_val_pts > 0 else ("⚡ прерван" if n_train_pts > 0 else "❌ нет данных"),
                "Время":           _fmt_dur(duration),
                "Val эпох":        str(n_val_pts) if n_val_pts > 0 else "—",
                "Last train loss": f"{train_df['value'].iloc[-1]:.4f}"
                                   if n_train_pts else "—",
                "Best val loss":   f"{val_df['value'].min():.4f}"
                                   if n_val_pts else "—",
            })
        _stats_df = _pd_stats.DataFrame(rows)
        # Подсветка текущего (последнего) запуска
        def _style_row(row):
            if "(текущий)" in str(row["Запуск"]):
                return [f"color:{GOLD};font-weight:700"] * len(row)
            return ["color:#c9d1d9"] * len(row)
        st.dataframe(
            _stats_df.style.apply(_style_row, axis=1),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info(
            "📭 Данные обучения не найдены. Запустите обучение — кривые потерь "
            "появятся здесь автоматически (обновите страницу после запуска)."
        )

    # ── Управление логами (удаление версий) ─────────────────────
    section_header("🗑️", "УПРАВЛЕНИЕ ЛОГАМИ", color=GRAY)
    _log_versions = get_tb_versions()
    if _log_versions:
        with st.expander(f"Удалить версии логов TensorBoard ({len(_log_versions)} доступно)"):
            st.markdown(
                f"<p style='color:{GRAY};font-size:12px;margin:0 0 10px 0;'>"
                "Выберите версии для удаления. Операция <b style='color:{RED};'>необратима</b>.</p>",
                unsafe_allow_html=True,
            )
            _to_delete = []
            _cols_per_row = 3
            _v_rows = [_log_versions[i:i+_cols_per_row] for i in range(0, len(_log_versions), _cols_per_row)]
            for _row in _v_rows:
                _vcols = st.columns(_cols_per_row)
                for _ci, _v in enumerate(_row):
                    with _vcols[_ci]:
                        _vn = os.path.basename(_v)
                        if st.checkbox(_vn, key=f"del_log_{_vn}"):
                            _to_delete.append(_v)

            if _to_delete:
                st.warning(
                    f"⚠️ Будет удалено {len(_to_delete)} версий: "
                    + ", ".join(os.path.basename(_v) for _v in _to_delete)
                )
                _confirm = st.checkbox(
                    "✅ Подтверждаю удаление выбранных версий",
                    key="confirm_del_logs",
                )
                if _confirm:
                    if st.button("🗑️  Удалить выбранные", type="primary", key="btn_del_sel"):
                        for _v in _to_delete:
                            shutil.rmtree(_v, ignore_errors=True)
                        st.toast(f"Удалено {len(_to_delete)} версий логов", icon="🗑️")
                        st.rerun()

            st.markdown(
                f"<hr style='border-color:{GRID_COLOR};margin:14px 0;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:{RED};font-size:11px;font-weight:600;margin:0 0 6px 0;'>"
                "⚠️ Опасная зона — удалить все версии:</p>",
                unsafe_allow_html=True,
            )
            _confirm_all = st.checkbox(
                "✅ Подтверждаю удаление ВСЕХ версий логов",
                key="confirm_del_all_logs",
            )
            if _confirm_all:
                if st.button("🗑️  Удалить ВСЕ версии", type="primary", key="btn_del_all"):
                    for _v in _log_versions:
                        shutil.rmtree(_v, ignore_errors=True)
                    st.toast("Все версии логов удалены", icon="🗑️")
                    st.rerun()
    else:
        st.markdown(
            f"<p style='color:{GRAY};font-size:12px;padding:4px 0;'>"
            "Нет версий логов для удаления.</p>",
            unsafe_allow_html=True,
        )

    # ── Активный конфиг ─────────────────────────────────────────
    section_header("⚙️", "АКТИВНЫЙ КОНФИГ ГИПЕРПАРАМЕТРОВ", color=GRAY)
    current_cfg = load_config()
    cfg_c1, cfg_c2 = st.columns(2)

    with cfg_c1:
        st.markdown(f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 6px 0;'>"
                    "Архитектура</p>", unsafe_allow_html=True)
        arch_rows = [
            ("Hidden size",           current_cfg["hidden_size"]),
            ("Attention heads",       current_cfg["attention_head_size"]),
            ("Hidden continuous",     current_cfg["hidden_continuous_size"]),
        ]
        for label, val in arch_rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
                f"border-bottom:1px solid {GRID_COLOR};font-size:12px;'>"
                f"<span style='color:{GRAY};'>{label}</span>"
                f"<span style='color:#c9d1d9;font-weight:600;'>{val}</span></div>",
                unsafe_allow_html=True,
            )

    with cfg_c2:
        st.markdown(f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 6px 0;'>"
                    "Обучение</p>", unsafe_allow_html=True)
        train_rows = [
            ("Learning rate",   f"{current_cfg['learning_rate']:.5f}"),
            ("Dropout",         f"{current_cfg['dropout']:.2f}"),
            ("Gradient clip",   f"{current_cfg['gradient_clip']:.1f}"),
            ("Batch size",      current_cfg["batch_size"]),
            ("Max epochs",      current_cfg["max_epochs"]),
            ("Patience",        current_cfg["patience"]),
        ]
        for label, val in train_rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
                f"border-bottom:1px solid {GRID_COLOR};font-size:12px;'>"
                f"<span style='color:{GRAY};'>{label}</span>"
                f"<span style='color:#c9d1d9;font-weight:600;'>{val}</span></div>",
                unsafe_allow_html=True,
            )

    src = f"`{CONFIG_PATH}`" if os.path.exists(CONFIG_PATH) else "дефолты (конфиг не сохранён)"
    st.markdown(
        f"<p style='color:{GRAY};font-size:11px;margin-top:8px;'>Источник: {src}</p>",
        unsafe_allow_html=True,
    )

    # ── Авто-обновление / кнопка ────────────────────────────────
    if is_training():
        st.markdown(
            f"<p style='color:{TEAL};font-size:11px;margin-top:4px;'>"
            "⚡ Обучение идёт — страница обновляется каждые 10 сек</p>",
            unsafe_allow_html=True,
        )
        time.sleep(10)
        st.rerun()
    else:
        if st.button("🔄  Обновить результаты", width="content"):
            st.rerun()
