"""
app_dashboard.py — Главный дашборд АЗС Татнефть.
Объединяет EDA-анализ и прогнозы TFT в единый интерфейс.
Запуск (из корня проекта): streamlit run dashboard/app_dashboard.py
"""

import datetime
import glob
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import utils.torch_compat  # noqa: F401
except Exception:
    pass

from utils.data_utils import (
    ENCODER_LENGTH,
    LOG_COLS,
    PREDICTION_LENGTH,
    Q_HI,
    Q_LO,
    Q_MED,
    STATIC_REALS,
    TARGET_COLS,
    TEST_END,
    TEST_START,
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Татнефть АЗС — Аналитика",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp,[data-testid="stAppViewContainer"]{background:#0f1117;color:#e8eaf0}
    [data-testid="stHeader"]{background:#13161f;border-bottom:1px solid #1e2235}
    .block-container{padding-top:3.5rem;padding-bottom:1rem}
    .dash-header{background:#13161f;border:1px solid #1e2235;border-radius:10px;
        padding:10px 16px;margin-bottom:14px;display:flex;align-items:center;
        gap:10px;flex-wrap:nowrap}
    .dash-logo{background:linear-gradient(135deg,#c8a84b,#e8c96b);color:#0a0c13;
        font-weight:700;font-size:11px;padding:4px 8px;border-radius:5px;white-space:nowrap}
    .dash-title{font-size:13px;font-weight:600;color:#e8eaf0;margin:0}
    .dash-sep{color:#2a2f45;font-size:16px}
    .dash-sub{font-size:11px;color:#8891a8;margin:0}
    .kpi-card{background:#13161f;border:1px solid #1e2235;border-radius:10px;
        padding:14px 16px;text-align:center}
    .kpi-label{font-size:11px;color:#6b7280;text-transform:uppercase;
        letter-spacing:.6px;margin-bottom:6px}
    .kpi-val{font-size:22px;font-weight:700;color:#c8a84b}
    .kpi-sub{font-size:11px;color:#8891a8;margin-top:3px}
    .info-box{background:#13161f;border-left:3px solid #2E75B6;
        border-radius:0 8px 8px 0;padding:.8rem 1rem;
        margin:.5rem 0;font-size:.88rem;color:#d4dbe8}
    [data-testid="stSelectbox"]>div,[data-testid="stMultiSelect"]>div{
        background:#1e2235!important;border:1px solid #2a2f45!important;border-radius:6px!important}
    [data-testid="stSelectbox"] label,[data-testid="stMultiSelect"] label{
        color:#8891a8!important;font-size:11px!important;
        text-transform:uppercase;letter-spacing:.5px}
    [data-baseweb="tag"]{background:#1e2235!important;border:1px solid #2a2f45!important;
        border-radius:4px!important;max-width:100%!important}
    [data-baseweb="tag"] span{color:#c8a84b!important;font-size:12px!important;
        white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;
        max-width:220px!important;display:inline-block!important}
    [data-testid="stTabs"] [role="tablist"]{background:#0a0c13;border-radius:8px;padding:3px;gap:2px}
    [data-testid="stTabs"] button[role="tab"]{background:transparent!important;
        color:#8891a8!important;border-radius:6px!important;font-size:13px!important;
        padding:6px 16px!important}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
        background:#1e2235!important;color:#e8eaf0!important}
    hr{border-color:#1e2235}
    [data-testid="collapsedControl"],section[data-testid="stSidebar"]{display:none}
    /* ── dropdown list background ── */
    [data-baseweb="popover"],[data-baseweb="menu"]{
        background:#1e2235!important;border:1px solid #2a2f45!important;
        border-radius:8px!important}
    [role="listbox"],[role="option"]{
        background:#1e2235!important;color:#e8eaf0!important;
        outline:none!important;box-shadow:none!important}
    [data-baseweb="popover"] [role="option"] span,
    [data-baseweb="popover"] [role="option"] div,
    [data-baseweb="popover"] [data-baseweb="menu-item"],
    [role="listbox"] [role="option"]:first-child,
    [role="listbox"] [role="option"]:first-child span,
    [role="listbox"] [role="option"]:first-child div{
        color:#e8eaf0!important;background:transparent!important}
    [role="option"]{transition:none!important}
    [role="option"]:hover,[role="option"]:focus,[role="option"]:focus-visible,
    [role="option"][aria-selected="true"],[role="option"][data-highlighted]{
        background:#2a2f45!important;color:#c8a84b!important;
        outline:none!important;box-shadow:none!important}
    [role="option"]:hover span,[role="option"]:hover div,
    [role="option"]:focus span,[role="option"]:focus div,
    [role="option"][aria-selected="true"] span,[role="option"][aria-selected="true"] div,
    [role="option"][data-highlighted] span,[role="option"][data-highlighted] div,
    [data-baseweb="popover"] [role="option"]:hover [data-baseweb="menu-item"],
    [data-baseweb="popover"] [role="option"][aria-selected="true"] [data-baseweb="menu-item"],
    [data-baseweb="popover"] [role="option"][data-highlighted] [data-baseweb="menu-item"]{
        color:#c8a84b!important}
    [data-baseweb="select"] [data-baseweb="icon"]{color:#8891a8!important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════
FUEL_COLS = [c for c in TARGET_COLS if c.startswith("sales_")]
SHOP_COLS = [c for c in TARGET_COLS if c.startswith("shop_")]

FUEL_LABELS = {
    "sales_AI92": "АИ-92", "sales_AI95": "АИ-95", "sales_AI98": "АИ-98",
    "sales_DT_EURO": "ДТ Евро+", "sales_DT_TANEKO": "ДТ ТАНЕКО",
    "sales_DT_SUMMER": "ДТ Летнее", "sales_DT_WINTER": "ДТ Зимнее",
}
SHOP_LABELS = {
    "shop_напитки": "Напитки", "shop_закуски": "Закуски",
    "shop_автотовары": "Автотовары", "shop_кофе": "Кофе", "shop_табак": "Табак",
}
TARGET_LABELS = {**FUEL_LABELS, **SHOP_LABELS}

FUEL_HEX = {
    "sales_AI92": "#c8a84b", "sales_AI95": "#2E75B6", "sales_AI98": "#9B59B6",
    "sales_DT_EURO": "#4ECB71", "sales_DT_TANEKO": "#1ABC9C",
    "sales_DT_SUMMER": "#E67E22", "sales_DT_WINTER": "#E24B4A",
}
SHOP_HEX = {
    "shop_напитки": "#2E75B6", "shop_закуски": "#c8a84b",
    "shop_автотовары": "#4ECB71", "shop_кофе": "#E67E22", "shop_табак": "#9B59B6",
}
STATION_PALETTE = ["#c8a84b", "#2E75B6", "#4ECB71", "#E24B4A", "#9B59B6", "#1ABC9C", "#E67E22"]
TRAFFIC_MAP = {
    "traffic_Passengers_cars": "Легковые", "traffic_Truck_short": "Груз. малые",
    "traffic_Truck": "Грузовые", "traffic_Truck_long": "Фуры",
    "traffic_Transporter": "Микроавт.", "traffic_Undefined": "Прочие",
}
MONTHS_RU    = {1:"Январь",2:"Февраль",3:"Март",4:"Апрель",5:"Май",6:"Июнь",
                7:"Июль",8:"Август",9:"Сентябрь",10:"Октябрь",11:"Ноябрь",12:"Декабрь"}
MONTHS_SHORT = {k: v[:3] for k, v in MONTHS_RU.items()}
DAY_NAMES    = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
SEASONS_RU   = {"winter":"Зима","spring":"Весна","summer":"Лето","autumn":"Осень"}
SEASONS_ENG  = {v: k for k, v in SEASONS_RU.items()}

GOLD        = "#c8a84b"
GREEN       = "#4ECB71"
RED         = "#E24B4A"
BLUE        = "#2E75B6"
TEAL        = "#1ABC9C"
GRAY        = "#8B949E"
GRID_COLOR  = "#1e2235"
CARD_BG     = "#13161f"
PLOTLY_THEME = dict(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                    font_color="#e8eaf0", font_size=12)

# ══════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════

def sfig(fig, height=None):
    kw = {**PLOTLY_THEME,
          "margin": dict(l=8, r=8, t=36, b=8),
          "legend": dict(bgcolor=CARD_BG, bordercolor=GRID_COLOR, borderwidth=1),
          "xaxis":  dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
          "yaxis":  dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR)}
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    return fig


def add_regline(fig, x_arr, y_arr, color=RED):
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    xv, yv = x_arr[valid], y_arr[valid]
    if len(xv) < 5:
        return None
    m, b = np.polyfit(xv, yv, 1)
    r    = float(np.corrcoef(xv, yv)[0, 1])
    xr   = np.linspace(xv.min(), xv.max(), 50)
    fig.add_trace(go.Scatter(x=xr, y=m*xr+b, mode="lines",
        line={"color": color, "width": 2}, showlegend=False,
        hovertemplate=f"Тренд (r={r:.2f})<extra></extra>"))
    return r


def kpi_card(col, label, value, sub="", accent=GOLD):
    with col:
        st.markdown(
            f'<div class="kpi-card" style="border-top:3px solid {accent}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-val" style="color:{accent}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True)


def rec_card(tag, body, color=GREEN):
    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:10px;padding:13px 17px;'
        f'margin-bottom:9px;border-left:4px solid {color};">'
        f'<div style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.5px;color:{color};margin-bottom:3px;">{tag}</div>'
        f'<div style="font-size:.9rem;color:#e8eaf0;">{body}</div></div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_data.csv", parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    fc = [c for c in FUEL_COLS if c in df.columns]
    sc = [c for c in SHOP_COLS if c in df.columns]
    tc = [c for c in TRAFFIC_MAP if c in df.columns]
    if "total_fuel_sales"   not in df.columns: df["total_fuel_sales"]   = df[fc].sum(axis=1)
    if "shop_total_revenue" not in df.columns: df["shop_total_revenue"] = df[sc].sum(axis=1)
    if "total_traffic"      not in df.columns: df["total_traffic"]      = df[tc].sum(axis=1)
    for pair, price in [("competitor_price_AI92","price_AI92"),
                        ("competitor_price_AI95","price_AI95")]:
        if pair in df.columns and price in df.columns:
            key = "comp_ratio_" + pair.split("_")[-1]
            df[key] = df[pair] / df[price].replace(0, np.nan)
    _dt = next((c for c in ["price_DT_EURO","price_DT_TANEKO"] if c in df.columns), None)
    if "competitor_price_DT" in df.columns and _dt:
        df["comp_ratio_DT"] = df["competitor_price_DT"] / df[_dt].replace(0, np.nan)
    if "ad_channel"   in df.columns: df["ad_channel"]   = df["ad_channel"].fillna("нет_рекламы")
    if "holiday_name" in df.columns: df["holiday_name"] = df["holiday_name"].fillna("нет_праздника")
    return df


@st.cache_data
def load_predictions():
    p = "data/predictions.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    return df


@st.cache_data
def load_metrics_tft():
    p = "data/metrics.csv"
    if not os.path.exists(p): return None
    return pd.read_csv(p)


@st.cache_data
def load_prepared():
    p = "data/prepared_data.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    dec = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    for col in TARGET_COLS:
        if col in df.columns:
            df.loc[dec, col] = np.expm1(df.loc[dec, col].clip(lower=0))
    return df


@st.cache_data
def load_ad_map():
    p = "data/prepared_data.csv"
    if not os.path.exists(p): return {}
    df = pd.read_csv(p, usecols=["ad_channel", "ad_channel_enc"])
    return (df.drop_duplicates().sort_values("ad_channel_enc")
            .set_index("ad_channel_enc")["ad_channel"].to_dict())


@st.cache_data
def load_scalers():
    p = "tft/scalers.pkl"
    if not os.path.exists(p): return {}
    with open(p, "rb") as f: return pickle.load(f)


@st.cache_resource
def load_tft():
    try:
        from pytorch_forecasting import TemporalFusionTransformer
        ckpt = "tft/model.ckpt"
        if not os.path.exists(ckpt):
            ckpts = sorted(glob.glob("tft/checkpoints/*.ckpt"))
            if not ckpts: return None, None, None, "Чекпоинт не найден"
            ckpt = ckpts[-1]
        with open("tft/training_dataset.pkl", "rb") as f: training = pickle.load(f)
        with open("tft/dataset_config.pkl",   "rb") as f: cfg      = pickle.load(f)
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
        model.eval()
        # Patch ALL nn.Embedding modules to clamp indices → prevents IndexError on OOV
        import torch
        for _name, _emb in model.named_modules():
            if isinstance(_emb, torch.nn.Embedding):
                _n = _emb.num_embeddings
                _orig = _emb.forward
                _emb.forward = (lambda x, o=_orig, n=_n: o(x.clamp(0, n - 1)))

        # Detect model ↔ training_dataset categorical mismatch and patch training in-place.
        # Root cause: model.ckpt may have been saved with more categoricals than
        # the current training_dataset.pkl (e.g., season_enc was added/removed).
        # from_dataset() uses training.categoricals to build the tensor; if the model
        # expects more columns than the tensor has, it crashes with "index N out of bounds".
        try:
            from pytorch_forecasting.data.encoders import NaNLabelEncoder as _NLE
            _model_cats = list(getattr(
                getattr(model, "input_embeddings", None), "x_categoricals",
                getattr(model.hparams, "x_categoricals", [])
            ))
            _train_cats = list(getattr(training, "categoricals", []))
            _missing = [c for c in _model_cats if c not in _train_cats]
            if _missing:
                # Try to fit real encoders from prepared_data.csv
                _prep_snap = None
                if os.path.exists("data/prepared_data.csv"):
                    try:
                        _prep_snap = pd.read_csv("data/prepared_data.csv", nrows=200)
                    except Exception:
                        pass
                for _col in _missing:
                    if _col in training.categorical_encoders:
                        continue
                    _enc = _NLE(add_nan=True)
                    if _prep_snap is not None and _col in _prep_snap.columns:
                        _enc.fit(_prep_snap[_col].astype(str).dropna())
                    else:
                        _enc.fit(pd.Series(["0"]))
                    training.categorical_encoders[_col] = _enc
                    # Season / temporal → time_varying_known; others → static
                    _tgt_list = (training.time_varying_known_categoricals
                                 if any(k in _col for k in ("season", "hour", "day", "month", "week"))
                                 else training.static_categoricals)
                    if _col not in _tgt_list:
                        _tgt_list.append(_col)
        except Exception:
            pass

        return model, training, cfg, None
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data(show_spinner=False)
def compute_interpretation(_model, _training, tft_cfg_key,
                           period_start, period_end, context_start_str, station):
    """Прогоняет TFT в режиме raw и возвращает numpy-словарь весов VSN + attention.
    При любой ошибке возвращает {"_error": "<текст>"} вместо исключения."""
    import traceback as _tb
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        df = load_prepared()
        if df is None:
            return {"_error": "Файл data/prepared_data.csv не найден. Выполните: python eda/eda_preprocessing.py"}

        interp_df = df[
            (df["timestamp"] >= pd.Timestamp(context_start_str)) &
            (df["timestamp"] <= pd.Timestamp(period_end + " 23:00:00"))
        ].copy().reset_index(drop=True)

        if station != "Все":
            interp_df = interp_df[interp_df["station_id"] == station].reset_index(drop=True)

        if interp_df.empty:
            return {"_error": f"Нет данных для выбранного периода/станции ({period_start}–{period_end}, {station})"}

        # Читаем каждый список отдельно — в pytorch-forecasting 1.x categoricals может быть
        # frozen-атрибутом, а не @property, поэтому после патча load_tft() только
        # time_varying_known_categoricals обновляется; categoricals остаётся старым.
        cat_cols = list(dict.fromkeys(
            list(getattr(_training, "static_categoricals", None) or [])
            + list(getattr(_training, "time_varying_known_categoricals", None) or [])
            + list(getattr(_training, "time_varying_unknown_categoricals", None) or [])
        ))

        # Добавляем отсутствующие категориальные колонки.
        # Временные категориальные (season_enc и т.п.) выводим через month→значение.
        for col in cat_cols:
            if col not in interp_df.columns:
                _known, _first = _encoder_classes(_training.categorical_encoders.get(col))
                fallback = _first if _first is not None else "0"
                if col in df.columns and "month" in df.columns and "month" in interp_df.columns:
                    month_map = df.groupby("month")[col].first().to_dict()
                    if month_map:
                        interp_df[col] = interp_df["month"].map(month_map).fillna(fallback)
                        continue
                interp_df[col] = fallback

        # Конвертируем float→int→str ("2.0"→"2"), заполняем NaN, зажимаем до известных классов
        for col in cat_cols:
            if col not in interp_df.columns:
                continue
            _known, _first = _encoder_classes(_training.categorical_encoders.get(col))
            fallback = _first if _first is not None else "0"
            try:
                s = interp_df[col].ffill().bfill().fillna(fallback)
                try:
                    interp_df[col] = s.astype(float).astype(int).astype(str)
                except (ValueError, TypeError):
                    interp_df[col] = s.astype(str)
            except Exception:
                interp_df[col] = fallback
            # Clamp to encoder's known classes → prevents index out of range
            if _known and fallback is not None:
                interp_df[col] = interp_df[col].map(
                    lambda v, k=_known, f=fallback: v if v in k else f
                )

        # Финальная страховка: все колонки, требуемые training, должны быть в interp_df
        for _c in cat_cols:
            if _c not in interp_df.columns:
                interp_df[_c] = "0"

        ds  = TimeSeriesDataSet.from_dataset(_training, interp_df, stop_randomization=True)
        ldr = ds.to_dataloader(train=False, batch_size=32, num_workers=0)

        raw = _model.predict(ldr, mode="raw",
                             trainer_kwargs={"logger": False, "enable_progress_bar": False})

        # predict(mode="raw") в pytorch-forecasting 1.7 возвращает plain dict
        # с ключами prediction/decoder_attention/encoder_attention/...
        # (результат _concatenate_output). interpret_output нужен ВЕСЬ dict,
        # а не только raw["prediction"].
        if isinstance(raw, dict) and "decoder_attention" in raw:
            out = raw  # полный dict — то, что ждёт interpret_output
        elif hasattr(raw, "output") and raw.output is not None:
            out = raw.output  # Prediction NamedTuple → берём .output
        elif isinstance(raw, dict):
            out = raw  # передаём как есть, пусть interpret_output разберётся
        elif isinstance(raw, (list, tuple)) and hasattr(raw, "_fields"):
            out = raw  # NamedTuple с OutputMixIn — поддерживает str-индекс
        elif isinstance(raw, (list, tuple)):
            out = raw[0]
        else:
            out = raw
        # Если после всей цепочки out всё ещё не поддерживает str-индексацию
        # (например, чистый list) — конвертируем через _asdict если возможно
        if not isinstance(out, dict) and hasattr(out, "_asdict"):
            out = out._asdict()

        interp = _model.interpret_output(out, reduction="mean")

        if isinstance(interp, dict):
            result = {k: v.detach().cpu().numpy() for k, v in interp.items()
                      if hasattr(v, "detach")}
        elif hasattr(interp, "_fields"):
            result = {k: getattr(interp, k).detach().cpu().numpy()
                      for k in interp._fields
                      if hasattr(getattr(interp, k), "detach")}
        else:
            return {"_error": f"Неожиданный тип interpret_output: {type(interp)}"}

        if not result:
            return {"_error": "interpret_output вернул пустой словарь. "
                              "Чекпоинт может быть несовместим."}
        return result

    except Exception as e:
        return {"_error": f"{e}\n\n{_tb.format_exc()}"}


def build_future_ctx(sc_st, ps, prepared_df_all):
    st_df = prepared_df_all[prepared_df_all["station_id"] == sc_st].sort_values("timestamp")
    encoder = st_df[st_df["timestamp"] < ps].tail(ENCODER_LENGTH).copy()
    if encoder.empty: return None, "Нет данных по выбранной станции"
    template  = encoder.iloc[-1].copy()
    last_tidx = int(template["time_idx"])
    season_enc_map = (prepared_df_all.groupby(prepared_df_all["timestamp"].dt.month)["season_enc"]
                      .first().to_dict())
    hol_enc_map = prepared_df_all.groupby([
        prepared_df_all["timestamp"].dt.month,
        prepared_df_all["timestamp"].dt.day,
    ])[["is_holiday","holiday_name_enc"]].first()
    no_hol_enc = prepared_df_all[prepared_df_all["is_holiday"]==0]["holiday_name_enc"].iloc[0]
    dec_rows = []
    for h in range(PREDICTION_LENGTH):
        ts  = ps + pd.Timedelta(hours=h)
        row = template.copy()
        row["timestamp"] = ts
        row["time_idx"]  = last_tidx + h + 1
        hr, dow, m = ts.hour, ts.dayofweek, ts.month
        woy = int(ts.isocalendar().week)
        row.update({"hour": hr,
                    "hour_sin":  np.sin(2*np.pi*hr/24), "hour_cos":  np.cos(2*np.pi*hr/24),
                    "day_of_week": dow,
                    "day_of_week_sin": np.sin(2*np.pi*dow/7), "day_of_week_cos": np.cos(2*np.pi*dow/7),
                    "week_of_year": woy,
                    "week_of_year_sin": np.sin(2*np.pi*woy/52), "week_of_year_cos": np.cos(2*np.pi*woy/52),
                    "month": m,
                    "month_sin": np.sin(2*np.pi*m/12), "month_cos": np.cos(2*np.pi*m/12),
                    "season_enc": season_enc_map.get(m, template["season_enc"]),
                    "is_weekend": int(dow >= 5),
                    "is_rush_hour": int(hr in [7,8,9,17,18,19]),
                    "is_night": int(hr < 6 or hr >= 22),
                    "is_shop_open": int(5 <= hr <= 21)})
        hol_key = (m, ts.day)
        if hol_key in hol_enc_map.index:
            hol_row = hol_enc_map.loc[hol_key]
            row["is_holiday"]       = int(hol_row["is_holiday"])
            row["holiday_name_enc"] = hol_row["holiday_name_enc"]
        else:
            row["is_holiday"] = 0; row["holiday_name_enc"] = no_hol_enc
        for col in TARGET_COLS: row[col] = 0.0
        dec_rows.append(row)
    return pd.concat([encoder, pd.DataFrame(dec_rows)], ignore_index=True), None


# ══════════════════════════════════════════════════════════════
# ЗАГРУЗКА НА УРОВНЕ МОДУЛЯ
# ══════════════════════════════════════════════════════════════
try:
    df = load_data()
except Exception:
    st.error("Файл `data/merged_data.csv` не найден. Выполните: `python explore_data.py`")
    st.stop()

pred_df     = load_predictions()
metrics_df  = load_metrics_tft()
prepared_df = load_prepared()
ad_map      = load_ad_map()
scalers     = load_scalers()

_all_stations = sorted(df["station_name"].unique())
_all_months   = sorted(df["month"].unique().tolist())
_sid_by_name  = df.groupby("station_name")["station_id"].first().to_dict()
_name_by_sid  = {v: k for k, v in _sid_by_name.items()}
_road_by_name = (df.groupby("station_name")["road_type"].first().to_dict()
                 if "road_type" in df.columns else {})
_station_display = {
    name: f"{name}  ·  {_road_by_name[name]}" if _road_by_name.get(name) else name
    for name in _all_stations
}

dist_col = "distance_to_city_km"
dist_min = float(df[dist_col].min()) if dist_col in df.columns else 0.0
dist_max = float(df[dist_col].max()) if dist_col in df.columns else 100.0

# ══════════════════════════════════════════════════════════════
# ШАПКА
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <span class="dash-logo">ТАТНЕФТЬ</span>
  <span class="dash-title">Аналитика продаж АЗС</span>
  <span class="dash-sep">|</span>
  <span class="dash-sub">5 АЗС · Почасовые данные · 2023 · TFT-прогноз</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЕ ФИЛЬТРЫ — применяются ко всем вкладкам
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:4px 0 10px 0;">
  <span style="color:{GOLD};font-size:11px;font-weight:700;letter-spacing:.1em;
               text-transform:uppercase;white-space:nowrap;">🌐 Глобальные фильтры</span>
  <span style="flex:1;height:1px;background:#2a2f45;"></span>
  <span style="color:#4b5563;font-size:11px;white-space:nowrap;">
    применяются ко всем вкладкам дашборда
  </span>
</div>
""", unsafe_allow_html=True)
gf1, gf2 = st.columns([3, 7])
with gf1:
    st.markdown(
        f'<p style="font-size:10px;color:{GRAY};letter-spacing:.09em;font-weight:700;'
        f'margin:0 0 3px 0;text-transform:uppercase;">🏪 &nbsp;Станции</p>',
        unsafe_allow_html=True)
    sel_stations_multi = st.multiselect(
        "Станции", _all_stations, default=_all_stations,
        format_func=lambda x: _station_display.get(x, x),
        label_visibility="collapsed", key="g_stations_multi",
    )
with gf2:
    st.markdown(
        f'<p style="font-size:10px;color:{GRAY};letter-spacing:.09em;font-weight:700;'
        f'margin:0 0 3px 0;text-transform:uppercase;">📅 &nbsp;Период</p>',
        unsafe_allow_html=True)
    sel_months = st.multiselect(
        "Месяцы", _all_months, default=_all_months,
        format_func=lambda x: MONTHS_RU[x],
        label_visibility="collapsed", key="g_months",
    )

sel_months_eff   = sel_months or _all_months
sel_stations_eff = sel_stations_multi or _all_stations

# ── Применение глобальных фильтров ────────────────────────────
base_mask = df["station_name"].isin(sel_stations_eff) & df["month"].isin(sel_months_eff)
df_ov = df[base_mask].copy()
df_ov["total_sales"]      = df_ov[[c for c in FUEL_COLS if c in df_ov.columns]].sum(axis=1)
df_ov["total_shop_sales"] = df_ov[[c for c in SHOP_COLS if c in df_ov.columns]].sum(axis=1)
stn_clr = {s: STATION_PALETTE[i % len(STATION_PALETTE)] for i, s in enumerate(_all_stations)}

# ── KPI-строка ────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
_peak_m = int(df_ov.groupby("month")["total_sales"].sum().idxmax()) if len(df_ov) > 0 else 1
kpi_card(k1, "Продажи топлива", f"{df_ov['total_sales'].sum():,.0f} л", "за период")
kpi_card(k2, "Выручка магазина", f"{df_ov['total_shop_sales'].sum():,.0f} руб.", "за период")
kpi_card(k3, "Ср. трафик/час",
         f"{df_ov['total_traffic'].mean():.0f}" if "total_traffic" in df_ov.columns else "—", "авт/час")
kpi_card(k4, "Станций", str(df_ov["station_name"].nunique()), "в выборке")
kpi_card(k5, "Пиковый месяц", MONTHS_SHORT.get(_peak_m,"—"), "максимум продаж")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# OUTER TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Обзор",
    "🔬  Анализ данных",
    "🤖  Прогноз TFT",
    "💡  Рекомендации",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — ОБЗОР
# ══════════════════════════════════════════════════════════════
with tab1:
    sub1a, sub1b, sub1c = st.tabs([
        "📈  Сводка сети",
        "🕐  Паттерны продаж",
        "🏪  Сравнение АЗС",
    ])

    # ── Сводка сети ───────────────────────────────────────────
    with sub1a:
        col_l, col_r = st.columns(2)
        with col_l:
            monthly = df_ov.groupby("month")["total_sales"].sum().reset_index()
            monthly["m_name"] = monthly["month"].map(MONTHS_SHORT)
            _peak = int(monthly.loc[monthly["total_sales"].idxmax(),"month"]) if len(monthly) else 1
            fig_m = go.Figure(go.Bar(
                x=monthly["m_name"], y=monthly["total_sales"],
                marker_color=[GOLD if m==_peak else "rgba(200,168,75,0.35)" for m in monthly["month"]],
                hovertemplate="%{x}: %{y:,.0f} л<extra></extra>",
            ))
            fig_m.update_layout(title="Продажи по месяцам", yaxis_title="Литры", showlegend=False)
            st.plotly_chart(sfig(fig_m), key="ap001", width="stretch")

        with col_r:
            ft = {FUEL_LABELS[c]: df_ov[c].sum() for c in FUEL_COLS if c in df_ov.columns}
            if ft:
                fig_pie = go.Figure(go.Pie(
                    labels=list(ft.keys()), values=list(ft.values()),
                    marker_colors=[FUEL_HEX.get(c,GOLD) for c in FUEL_COLS if c in df_ov.columns],
                    hole=0.4, hovertemplate="%{label}: %{value:,.0f} л (%{percent})<extra></extra>",
                ))
                fig_pie.update_layout(title="Структура продаж топлива")
                st.plotly_chart(sfig(fig_pie), key="ap002", width="stretch")

        col_l, col_r = st.columns(2)
        with col_l:
            sh = df_ov.groupby("station_name")[[c for c in SHOP_COLS if c in df_ov.columns]].sum().reset_index()
            shm = sh.melt(id_vars="station_name", var_name="cat", value_name="rev")
            shm["cat_label"] = shm["cat"].map(SHOP_LABELS)
            fig_sh = px.bar(shm, x="station_name", y="rev", color="cat_label", barmode="stack",
                            color_discrete_map={SHOP_LABELS[c]: SHOP_HEX.get(c,"#60a5fa") for c in SHOP_COLS},
                            labels={"station_name":"Станция","rev":"Выручка, руб.","cat_label":"Категория"})
            fig_sh.update_layout(title="Выручка магазина по станциям", legend=dict(orientation="h",y=-0.28))
            st.plotly_chart(sfig(fig_sh), key="ap003", width="stretch")

        with col_r:
            by_st = (df_ov.groupby("station_name")["total_sales"]
                     .sum().reset_index().sort_values("total_sales",ascending=True))
            fig_st = go.Figure(go.Bar(
                x=by_st["total_sales"], y=by_st["station_name"], orientation="h",
                marker_color=[stn_clr.get(s,GOLD) for s in by_st["station_name"]],
                hovertemplate="%{y}: %{x:,.0f} л<extra></extra>",
            ))
            fig_st.update_layout(title="Суммарные продажи топлива по станциям",
                                  xaxis_title="Литры", showlegend=False)
            st.plotly_chart(sfig(fig_st), key="ap004", width="stretch")

        tc_avail = [c for c in TRAFFIC_MAP if c in df_ov.columns]
        if tc_avail:
            tr_m = df_ov.groupby("month")[tc_avail].mean().reset_index()
            fig_tr = go.Figure()
            for c in tc_avail:
                fig_tr.add_trace(go.Bar(x=tr_m["month"].map(MONTHS_SHORT), y=tr_m[c],
                                        name=TRAFFIC_MAP[c]))
            fig_tr.update_layout(barmode="stack", title="Трафик по типу ТС (ср. авт./час)",
                                  legend=dict(orientation="h",y=-0.28))
            st.plotly_chart(sfig(fig_tr), key="ap005", width="stretch")

        st.markdown("### Магазин: структура выручки")
        shop_cats = {c: SHOP_LABELS[c] for c in SHOP_COLS if c in df_ov.columns}
        if shop_cats:
            col_l, col_r = st.columns(2)
            with col_l:
                shop_means = {v: df_ov[k].mean() for k,v in shop_cats.items()}
                fig_sp = px.pie(values=list(shop_means.values()), names=list(shop_means.keys()),
                                color_discrete_sequence=list(SHOP_HEX.values()))
                fig_sp.update_layout(title="Доля категорий в выручке магазина")
                st.plotly_chart(sfig(fig_sp), key="ap040", width="stretch")
            with col_r:
                sh_h = df_ov.groupby(df_ov["timestamp"].dt.hour)[list(shop_cats.keys())].mean().reset_index()
                sh_h.columns = ["hour"] + list(shop_cats.keys())
                sh_hm = sh_h.melt(id_vars="hour", var_name="cat", value_name="rev")
                sh_hm["cat_label"] = sh_hm["cat"].map(SHOP_LABELS)
                fig_sh4 = px.line(sh_hm, x="hour", y="rev", color="cat_label",
                                  color_discrete_map={SHOP_LABELS[c]: SHOP_HEX.get(c,"#60a5fa") for c in shop_cats},
                                  labels={"hour":"Час суток","rev":"Ср. руб./час","cat_label":"Категория"})
                fig_sh4.update_layout(title="Выручка магазина по часу суток")
                fig_sh4.update_xaxes(dtick=2)
                st.plotly_chart(sfig(fig_sh4), key="ap041", width="stretch")

    # ── Паттерны продаж ───────────────────────────────────────
    with sub1b:
        col_l, col_r = st.columns(2)
        with col_l:
            hourly = df_ov.groupby(df_ov["timestamp"].dt.hour)["total_sales"].mean().reset_index()
            hourly.columns = ["hour","sales"]
            ph = int(hourly.set_index("hour")["sales"].idxmax())
            fig_h = go.Figure(go.Scatter(x=hourly["hour"], y=hourly["sales"],
                mode="lines+markers", line=dict(color=GOLD,width=2), marker=dict(size=5),
                fill="tozeroy", fillcolor="rgba(200,168,75,0.1)",
                hovertemplate="Час %{x}: %{y:.1f} л/час<extra></extra>"))
            fig_h.add_vline(x=ph, line_dash="dash", line_color=RED,
                            annotation_text=f"Пик {ph}:00", annotation_position="top right")
            fig_h.update_layout(title="Суточный профиль (ср. л/час)",
                                  xaxis_title="Час суток", yaxis_title="Литры/час", showlegend=False)
            fig_h.update_xaxes(dtick=2)
            st.plotly_chart(sfig(fig_h), key="ap006", width="stretch")

        with col_r:
            dg = df_ov.groupby(df_ov["timestamp"].dt.dayofweek)["total_sales"].mean().reset_index()
            dg.columns = ["dow","sales"]
            dg["label"] = dg["dow"].map(lambda d: DAY_NAMES[d])
            fig_d = px.bar(dg, x="label", y="sales",
                           labels={"label":"","sales":"Ср. продажи, л/час"},
                           color_discrete_sequence=[GOLD])
            fig_d.update_layout(title="Продажи по дням недели")
            st.plotly_chart(sfig(fig_d), key="ap007", width="stretch")

        if "season" in df_ov.columns:
            s_grp = (df_ov.groupby("season")["total_sales"].mean()
                     .reindex(["winter","spring","summer","autumn"]).reset_index())
            s_grp["label"] = s_grp["season"].map(SEASONS_RU)
            fig_s = go.Figure(go.Bar(x=s_grp["label"], y=s_grp["total_sales"],
                marker_color=[BLUE,GREEN,GOLD,"#E67E22"],
                hovertemplate="%{x}: %{y:.1f} л/час<extra></extra>"))
            fig_s.update_layout(title="Продажи по сезонам (ср. л/час)", showlegend=False)
            st.plotly_chart(sfig(fig_s), key="ap008", width="stretch")

        fuel_h_keys = [c for c in FUEL_COLS if c in df_ov.columns]
        if fuel_h_keys:
            # st.radio(horizontal=True) вместо st.selectbox: radio рендерит элементы
            # как обычные DOM-spans, а не BaseWeb-popup с виртуальным списком,
            # поэтому баг с исчезновением первого кириллического элемента не возникает.
            heat_col = st.radio(
                "Топливо для тепловой карты",
                options=fuel_h_keys,
                format_func=lambda c: FUEL_LABELS.get(c, c),
                horizontal=True,
                key="heat_fuel_radio")
            heat_choice = FUEL_LABELS.get(heat_col, heat_col)
            pivot = df_ov.pivot_table(index=df_ov["timestamp"].dt.hour,
                                       columns=df_ov["timestamp"].dt.dayofweek,
                                       values=heat_col, aggfunc="mean")
            pivot.index.name = "Час"
            pivot.columns = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"][:len(pivot.columns)]
            fig_hm = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                               labels={"x":"","y":"Час","color":"л/час"},
                               title=f"Тепловая карта: час × день — {heat_choice}")
            st.plotly_chart(sfig(fig_hm), key="ap009", width="stretch")

        st.divider()
        st.markdown("#### Влияние акций на продажи топлива")
        _ap1, _ap2, _ap3 = st.columns(3)
        for _pi, (_pcol, _plbl, _pccc, _pclr) in enumerate([
            ("promotion_fuel_active", "Акция на топливо", _ap1, GOLD),
            ("promotion_shop_active", "Акция на магазин", _ap2, GREEN),
            ("promotion_cafe_active", "Акция на кафе",    _ap3, BLUE),
        ]):
            if _pcol in df_ov.columns:
                _pg = df_ov.groupby(_pcol)["total_sales"].mean()
                with _pccc:
                    _fig_pp = go.Figure(go.Bar(
                        x=["Без акции", "С акцией"],
                        y=[_pg.get(0, 0), _pg.get(1, 0)],
                        marker_color=["#4a5568", _pclr],
                        hovertemplate="%{x}: %{y:.1f} л/час<extra></extra>",
                    ))
                    _fig_pp.update_layout(title=_plbl, showlegend=False, yaxis_title="Ср. л/час")
                    st.plotly_chart(sfig(_fig_pp, height=250), key=f"ap_promo_{_pi}", width="stretch")

    # ── Сравнение АЗС ─────────────────────────────────────────
    with sub1c:
        melt_st = df_ov.groupby("station_name")[[c for c in FUEL_COLS if c in df_ov.columns]].sum().reset_index()
        melt_st = melt_st.melt(id_vars="station_name", var_name="fuel", value_name="sales")
        melt_st["fuel_label"] = melt_st["fuel"].map(FUEL_LABELS)
        fig_st2 = px.bar(melt_st, x="station_name", y="sales", color="fuel_label", barmode="stack",
                         color_discrete_map={FUEL_LABELS[c]: FUEL_HEX.get(c,GOLD) for c in FUEL_COLS},
                         labels={"station_name":"Станция","sales":"Продажи, л","fuel_label":"Топливо"})
        fig_st2.update_layout(title="Структура топлива по АЗС",legend=dict(orientation="h",y=-0.28))
        st.plotly_chart(sfig(fig_st2), key="ap010", width="stretch")

        df_ov["week"] = df_ov["timestamp"].dt.isocalendar().week.astype(int)
        weekly = df_ov.groupby(["week","station_name"])["total_sales"].sum().reset_index()
        fig_dyn = px.line(weekly, x="week", y="total_sales", color="station_name",
                          color_discrete_map=stn_clr,
                          labels={"week":"Неделя","total_sales":"Продажи, л","station_name":"Станция"})
        fig_dyn.update_layout(title="Еженедельная динамика по станциям",legend=dict(orientation="h",y=-0.22))
        st.plotly_chart(sfig(fig_dyn), key="ap011", width="stretch")

        st.divider()
        st.markdown("### Характеристики АЗС")
        col_l, col_r = st.columns(2)
        with col_l:
            if "road_type" in df_ov.columns:
                rt = (df_ov.groupby("road_type")["total_sales"].mean()
                      .reset_index().sort_values("total_sales",ascending=True))
                fig_rt = px.bar(rt, x="total_sales", y="road_type", orientation="h",
                                labels={"total_sales":"Ср. л/час","road_type":""},
                                color_discrete_sequence=[GOLD])
                fig_rt.update_layout(title="Продажи по типу дороги")
                st.plotly_chart(sfig(fig_rt), key="ap012", width="stretch")
        with col_r:
            if "settlement_size" in df_ov.columns:
                ss = (df_ov.groupby("settlement_size")["total_sales"].mean()
                      .reset_index().sort_values("total_sales",ascending=True))
                fig_ss = px.bar(ss, x="total_sales", y="settlement_size", orientation="h",
                                labels={"total_sales":"Ср. л/час","settlement_size":""},
                                color_discrete_sequence=[BLUE])
                fig_ss.update_layout(title="Продажи по размеру нас. пункта")
                st.plotly_chart(sfig(fig_ss), key="ap013", width="stretch")

        meta_df = df_ov.groupby("station_name").agg(
            avg_sales=("total_sales","mean"),
        ).reset_index()
        for mc in ["road_type","direction","distance_to_city_km","total_pumps",
                   "competitors_within_5km","customer_loyalty_score","staff_quality_score",
                   "staff_engagement_score","corporate_customer_ratio"]:
            if mc in df_ov.columns:
                meta_df[mc] = df_ov.groupby("station_name")[mc].first().values

        col_l, col_r = st.columns(2)
        with col_l:
            if "competitors_within_5km" in meta_df.columns:
                fig_comp = px.bar(meta_df.sort_values("competitors_within_5km"),
                    x="station_name", y="competitors_within_5km",
                    color="station_name", color_discrete_map=stn_clr,
                    labels={"station_name":"","competitors_within_5km":"Конкурентов"})
                fig_comp.update_layout(title="Конкуренты в радиусе 5 км", showlegend=False)
                st.plotly_chart(sfig(fig_comp), key="ap014", width="stretch")
        with col_r:
            score_cols = ["customer_loyalty_score","staff_quality_score",
                          "staff_engagement_score","corporate_customer_ratio"]
            score_names = {"customer_loyalty_score":"Лояльность","staff_quality_score":"Качество",
                           "staff_engagement_score":"Вовлечённость","corporate_customer_ratio":"Корп. доля"}
            avail_sc = [c for c in score_cols if c in meta_df.columns]
            if avail_sc:
                sc_melt = meta_df[["station_name"]+avail_sc].melt(id_vars="station_name",
                    var_name="metric", value_name="value")
                sc_melt["metric"] = sc_melt["metric"].map(score_names)
                fig_sc = px.bar(sc_melt, x="station_name", y="value", color="metric", barmode="group",
                                labels={"station_name":"","value":"Значение","metric":"Метрика"})
                fig_sc.update_layout(title="Качественные метрики АЗС",legend=dict(orientation="h",y=-0.28))
                st.plotly_chart(sfig(fig_sc), key="ap015", width="stretch")

        svc_map = {"has_car_wash":"Автомойка","has_tire_service":"Шиномонтаж",
                   "has_cafe":"Кафе","has_hotel":"Отель","has_shop":"Магазин"}
        svc_avail = [c for c in svc_map if c in df_ov.columns]
        if svc_avail:
            svc_df = df_ov.groupby("station_name")[svc_avail].first().rename(columns=svc_map)
            fig_svc = px.imshow(svc_df, text_auto=True, aspect="auto",
                                color_continuous_scale=[GRID_COLOR, GOLD],
                                labels={"color":"Наличие"})
            fig_svc.update_layout(title="Услуги АЗС (1 — есть, 0 — нет)", height=220)
            st.plotly_chart(sfig(fig_svc), key="ap016", width="stretch")


# ══════════════════════════════════════════════════════════════
# TAB 2 — АНАЛИЗ ДАННЫХ
# ══════════════════════════════════════════════════════════════
with tab2:
    # ── Фильтры вкладки «Анализ данных» ─────────────────────────
    _FSEC = '<p style="font-size:11px;color:#6b7280;margin:8px 0 4px 0;letter-spacing:.04em;font-weight:600;">'
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:4px 0 10px 0;">
  <span style="color:{TEAL};font-size:11px;font-weight:700;letter-spacing:.1em;
               text-transform:uppercase;white-space:nowrap;">📊 Фильтры анализа</span>
  <span style="flex:1;height:1px;background:#2a2f45;"></span>
  <span style="color:#4b5563;font-size:11px;white-space:nowrap;">
    влияют только на эту вкладку
  </span>
</div>
""", unsafe_allow_html=True)
    tf1, tf2 = st.columns(2)
    with tf1:
        st.markdown(
            f'<p style="font-size:10px;color:{GRAY};letter-spacing:.09em;font-weight:700;'
            f'margin:0 0 3px 0;text-transform:uppercase;">⛽ &nbsp;Виды топлива</p>',
            unsafe_allow_html=True)
        adv_fuels = st.multiselect(
            "Виды топлива", list(FUEL_LABELS.values()),
            default=list(FUEL_LABELS.values()), key="adv_fuels",
            label_visibility="collapsed")
    with tf2:
        st.markdown(
            f'<p style="font-size:10px;color:{GRAY};letter-spacing:.09em;font-weight:700;'
            f'margin:0 0 3px 0;text-transform:uppercase;">🛒 &nbsp;Товары магазина</p>',
            unsafe_allow_html=True)
        adv_shops = st.multiselect(
            "Товары магазина", list(SHOP_LABELS.values()),
            default=list(SHOP_LABELS.values()), key="adv_shops",
            label_visibility="collapsed")
    with st.expander("🔍  Дополнительные фильтры"):
        _road_opts2 = sorted(df["road_type"].dropna().unique()) if "road_type" in df.columns else []
        _dir_opts2  = sorted(df["direction"].dropna().unique()) if "direction" in df.columns else []
        _sett_opts2 = sorted(df["settlement_size"].dropna().unique()) if "settlement_size" in df.columns else []
        st.markdown(f'{_FSEC}📅 &nbsp;ПЕРИОД И ТИП ДНЯ</p>', unsafe_allow_html=True)
        afc1, afc2, afc3, afc4 = st.columns(4)
        with afc1:
            adv_season   = st.multiselect("Сезон", list(SEASONS_RU.values()), placeholder="Все", key="adv_season")
        with afc2:
            adv_day_type = st.multiselect("Тип дня", ["Будни","Выходные","Праздники"], placeholder="Все", key="adv_daytype")
        with afc3:
            adv_holiday  = st.selectbox("Праздники", ["Все","Только праздники","Без праздников"], key="adv_hol")
        with afc4:
            adv_promo    = st.multiselect("Акции", ["Топливо","Магазин","Кафе","Без акций"], placeholder="Любые", key="adv_promo")
        st.markdown(f'{_FSEC}🌤 &nbsp;ВНЕШНИЕ УСЛОВИЯ</p>', unsafe_allow_html=True)
        wfc1, wfc2, wfc3, wfc4 = st.columns(4)
        with wfc1:
            adv_weather  = st.multiselect("Погода", ["Ясно","Дождь","Снег","Туман"], placeholder="Любая", key="adv_weather")
        st.markdown(f'{_FSEC}🏪 &nbsp;ХАРАКТЕРИСТИКИ АЗС</p>', unsafe_allow_html=True)
        bfc1, bfc2, bfc3, bfc4 = st.columns(4)
        with bfc1:
            adv_road  = st.multiselect("Тип дороги", _road_opts2, placeholder="Все", key="adv_road")
        with bfc2:
            adv_dir   = st.multiselect("Направление", _dir_opts2, placeholder="Все", key="adv_dir")
        with bfc3:
            adv_dist  = (st.slider("Расст. до города, км", dist_min, dist_max,
                                   (dist_min, dist_max), step=0.5, key="adv_dist")
                         if dist_col in df.columns else (dist_min, dist_max))
        with bfc4:
            adv_sett  = st.multiselect("Населённый пункт", _sett_opts2, placeholder="Все", key="adv_sett")

    # ── Вычисление fdf из выбранных фильтров ─────────────────────
    adv_fuel_cols = [k for k, v in FUEL_LABELS.items() if v in adv_fuels] or list(FUEL_LABELS.keys())
    adv_shop_cols = [k for k, v in SHOP_LABELS.items() if v in adv_shops] or list(SHOP_LABELS.keys())
    adv_mask = base_mask.copy()
    if adv_season and "season" in df.columns:
        adv_mask &= df["season"].isin([SEASONS_ENG[s] for s in adv_season])
    if adv_day_type:
        _dm = pd.Series(False, index=df.index)
        if "Будни"     in adv_day_type: _dm |= (df["is_weekend"] == 0)
        if "Выходные"  in adv_day_type: _dm |= (df["is_weekend"] == 1)
        if "Праздники" in adv_day_type: _dm |= (df["is_holiday"] == 1)
        adv_mask &= _dm
    if adv_weather:
        _wm = pd.Series(False, index=df.index)
        if "Ясно"  in adv_weather: _wm |= (df["is_rain"]==0)&(df["is_snow"]==0)&(df["is_fog"]==0)
        if "Дождь" in adv_weather: _wm |= (df["is_rain"] == 1)
        if "Снег"  in adv_weather: _wm |= (df["is_snow"] == 1)
        if "Туман" in adv_weather: _wm |= (df["is_fog"]  == 1)
        adv_mask &= _wm
    if adv_promo:
        _pm = pd.Series(False, index=df.index)
        if "Топливо"   in adv_promo: _pm |= (df["promotion_fuel_active"] == 1)
        if "Магазин"   in adv_promo: _pm |= (df["promotion_shop_active"] == 1)
        if "Кафе"      in adv_promo: _pm |= (df["promotion_cafe_active"] == 1)
        if "Без акций" in adv_promo:
            _pm |= ((df["promotion_fuel_active"]==0)&(df["promotion_shop_active"]==0)
                    &(df["promotion_cafe_active"]==0))
        adv_mask &= _pm
    if adv_holiday == "Только праздники":  adv_mask &= df["is_holiday"] == 1
    elif adv_holiday == "Без праздников":  adv_mask &= df["is_holiday"] == 0
    if dist_col in df.columns:
        adv_mask &= (df[dist_col] >= adv_dist[0]) & (df[dist_col] <= adv_dist[1])
    if adv_road and "road_type" in df.columns:       adv_mask &= df["road_type"].isin(adv_road)
    if adv_dir  and "direction"  in df.columns:      adv_mask &= df["direction"].isin(adv_dir)
    if adv_sett and "settlement_size" in df.columns: adv_mask &= df["settlement_size"].isin(adv_sett)
    fdf = df[adv_mask].copy()
    if fdf.empty:
        fdf = df_ov.copy()
    fdf["total_sales"]      = fdf[[c for c in adv_fuel_cols if c in fdf.columns]].sum(axis=1)
    fdf["total_shop_sales"] = fdf[[c for c in adv_shop_cols if c in fdf.columns]].sum(axis=1)

    if fdf.empty:
        st.warning("Нет данных для выбранных фильтров. Измените параметры фильтров.")
    else:
        sub2a, sub2b, sub2c, sub2d = st.tabs([
            "🌤️  Погода & Трафик",
            "🎯  Акции & Реклама",
            "🔗  Корреляции",
            "📊  Статистика",
        ])

        # ── Погода & Трафик ───────────────────────────────────
        with sub2a:
            fa_fuel_opts = ["Суммарно"] + [FUEL_LABELS[c] for c in adv_fuel_cols if c in fdf.columns]
            fa_fuel = st.selectbox("Вид топлива для анализа", fa_fuel_opts, key="t2a_fuel")
            fa_col  = "total_sales" if fa_fuel == "Суммарно" else next(k for k,v in FUEL_LABELS.items() if v==fa_fuel)
            sample  = fdf.sample(min(3000,len(fdf)), random_state=42)
            sx      = sample[fa_col].values.astype(float)

            col_l, col_r = st.columns(2)
            with col_l:
                if "temperature" in sample.columns:
                    x_temp = sample["temperature"].values.astype(float)
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(x=x_temp, y=sx, mode="markers",
                        marker=dict(color=GOLD,opacity=0.3,size=3), showlegend=False,
                        hovertemplate="T=%{x:.1f}°C  прод.=%{y:.1f}<extra></extra>"))
                    r_temp = add_regline(fig_temp, x_temp, sx)
                    fig_temp.update_layout(title=f"Температура vs {fa_fuel}"+(f"  (r={r_temp:.2f})" if r_temp else ""),
                                            xaxis_title="Температура, °C", yaxis_title="Продажи, л/час")
                    st.plotly_chart(sfig(fig_temp), key="ap020", width="stretch")
            with col_r:
                if "total_traffic" in sample.columns:
                    x_tr = sample["total_traffic"].values.astype(float)
                    fig_traf = go.Figure()
                    fig_traf.add_trace(go.Scatter(x=x_tr, y=sx, mode="markers",
                        marker=dict(color=BLUE,opacity=0.3,size=3), showlegend=False))
                    r_tr = add_regline(fig_traf, x_tr, sx)
                    fig_traf.update_layout(title=f"Трафик vs {fa_fuel}"+(f"  (r={r_tr:.2f})" if r_tr else ""),
                                            xaxis_title="Трафик, авт/час", yaxis_title="Продажи, л/час")
                    st.plotly_chart(sfig(fig_traf), key="ap021", width="stretch")

            col_l, col_r = st.columns(2)
            with col_l:
                if "traffic_Passengers_cars" in sample.columns:
                    xp = sample["traffic_Passengers_cars"].values.astype(float)
                    fig_pass = go.Figure()
                    fig_pass.add_trace(go.Scatter(x=xp,y=sx,mode="markers",
                        marker=dict(color=GOLD,opacity=0.3,size=3),showlegend=False))
                    r_pass = add_regline(fig_pass, xp, sx)
                    fig_pass.update_layout(title=f"Легковые → {fa_fuel}"+(f"  (r={r_pass:.2f})" if r_pass else ""),
                                            xaxis_title="Легковых авт./час", yaxis_title="Продажи, л/час")
                    st.plotly_chart(sfig(fig_pass), key="ap022", width="stretch")
            with col_r:
                trucks = [c for c in ["traffic_Truck_short","traffic_Truck","traffic_Truck_long"] if c in sample.columns]
                dt_col2 = next((c for c in ["sales_DT_EURO","sales_DT_TANEKO"] if c in sample.columns), None)
                if trucks and dt_col2:
                    x_truck = sample[trucks].sum(axis=1).values.astype(float)
                    y_dt    = sample[dt_col2].values.astype(float)
                    fig_tr2 = go.Figure()
                    fig_tr2.add_trace(go.Scatter(x=x_truck,y=y_dt,mode="markers",
                        marker=dict(color=GREEN,opacity=0.3,size=3),showlegend=False))
                    r_truck = add_regline(fig_tr2, x_truck, y_dt)
                    fig_tr2.update_layout(title=f"Грузовые → {FUEL_LABELS.get(dt_col2,'ДТ')}"+(f"  (r={r_truck:.2f})" if r_truck else ""),
                                           xaxis_title="Грузовых авт./час", yaxis_title="Продажи ДТ, л/час")
                    st.plotly_chart(sfig(fig_tr2), key="ap023", width="stretch")

            tr_keys   = [c for c in TRAFFIC_MAP if c in fdf.columns]
            fuel_keys = [c for c in adv_fuel_cols if c in fdf.columns]
            if tr_keys and fuel_keys:
                corr_mat = fdf[tr_keys+fuel_keys].corr().loc[tr_keys,fuel_keys]
                fig_chm  = go.Figure(go.Heatmap(
                    z=corr_mat.values,
                    x=[FUEL_LABELS[c] for c in corr_mat.columns],
                    y=[TRAFFIC_MAP[c]  for c in corr_mat.index],
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=corr_mat.values.round(2), texttemplate="%{text}",
                    hovertemplate="%{y} × %{x}: r=%{z:.3f}<extra></extra>"))
                fig_chm.update_layout(title="Корреляция: тип ТС × вид топлива", height=350)
                st.plotly_chart(sfig(fig_chm), key="ap024", width="stretch")

            col_l, col_r = st.columns(2)
            with col_l:
                if "weather_condition" in fdf.columns:
                    ws = (fdf.groupby("weather_condition")["total_sales"].mean()
                          .reset_index().sort_values("total_sales",ascending=True))
                    fig_w = px.bar(ws, x="total_sales", y="weather_condition", orientation="h",
                                   labels={"total_sales":"Ср. л/час","weather_condition":""},
                                   color_discrete_sequence=[GOLD])
                    fig_w.update_layout(title="Продажи по типу погоды")
                    st.plotly_chart(sfig(fig_w), key="ap025", width="stretch")
            with col_r:
                wf = {"is_rain":"Дождь","is_snow":"Снег","is_fog":"Туман"}
                flag_rows = []
                for cf,lf in wf.items():
                    if cf in fdf.columns:
                        for val,nm in [(0,f"Без: {lf}"),(1,lf)]:
                            flag_rows.append({"Условие":nm,"Продажи":fdf[fdf[cf]==val]["total_sales"].mean(),"Тип":lf})
                if flag_rows:
                    fig_fl = px.bar(pd.DataFrame(flag_rows), x="Условие", y="Продажи", color="Тип",
                                    labels={"Продажи":"Ср. л/час"},
                                    color_discrete_sequence=[GOLD,BLUE,GREEN])
                    fig_fl.update_layout(title="Влияние дождя / снега / тумана")
                    st.plotly_chart(sfig(fig_fl), key="ap026", width="stretch")

            wx_cols3 = st.columns(3)
            for _wi,(feat,x_lbl,clr) in enumerate([
                ("precipitation_mm","Осадки, мм",GOLD),
                ("visibility_km","Видимость, км",BLUE),
                ("wind_speed_ms","Ветер, м/с",GREEN),
            ]):
                if feat in sample.columns:
                    with wx_cols3[_wi]:
                        xf = sample[feat].values.astype(float)
                        fig_f = go.Figure()
                        fig_f.add_trace(go.Scatter(x=xf,y=sx,mode="markers",
                            marker=dict(color=clr,opacity=0.35,size=3),showlegend=False))
                        r_f = add_regline(fig_f, xf, sx)
                        fig_f.update_layout(title=f"{x_lbl} vs продажи"+(f"  (r={r_f:.2f})" if r_f else ""),
                                             xaxis_title=x_lbl, yaxis_title="л/час")
                        st.plotly_chart(sfig(fig_f), key=f"ap_wx_{_wi}", width="stretch")

        # ── Акции & Реклама ───────────────────────────────────
        with sub2b:
            st.markdown("### Реклама и продажи")
            col_l, col_r = st.columns(2)
            with col_l:
                if "ad_channel" in fdf.columns:
                    ag = (fdf.groupby("ad_channel")["total_sales"].mean()
                          .sort_values(ascending=False).reset_index())
                    no_ad = float(fdf[fdf["ad_channel"]=="нет_рекламы"]["total_sales"].mean()
                                  if "нет_рекламы" in fdf["ad_channel"].values else 0)
                    fig_ad = go.Figure(go.Bar(
                        x=ag["ad_channel"], y=ag["total_sales"],
                        marker_color=[GOLD if ch!="нет_рекламы" else "rgba(46,117,182,0.55)"
                                      for ch in ag["ad_channel"]],
                        hovertemplate="%{x}: %{y:.1f} л/час<extra></extra>"))
                    if no_ad>0:
                        fig_ad.add_hline(y=no_ad,line_dash="dot",line_color="#888",
                                          annotation_text="Без рекламы",annotation_position="top right")
                    fig_ad.update_layout(title="Продажи топлива по каналу рекламы",
                                          xaxis_title="Канал", yaxis_title="Ср. л/час", showlegend=False)
                    st.plotly_chart(sfig(fig_ad), key="ap030", width="stretch")
            with col_r:
                if "ad_channel" in fdf.columns and "shop_total_revenue" in fdf.columns:
                    sg2 = (fdf.groupby("ad_channel")["shop_total_revenue"].mean()
                           .sort_values(ascending=False).reset_index())
                    fig_sh2 = go.Figure(go.Bar(
                        x=sg2["ad_channel"], y=sg2["shop_total_revenue"],
                        marker_color=[TEAL if ch!="нет_рекламы" else "rgba(155,89,182,0.55)"
                                      for ch in sg2["ad_channel"]],
                        hovertemplate="%{x}: %{y:.1f} руб./час<extra></extra>"))
                    fig_sh2.update_layout(title="Выручка магазина по каналу рекламы",
                                           xaxis_title="Канал", yaxis_title="Ср. руб./час", showlegend=False)
                    st.plotly_chart(sfig(fig_sh2), key="ap031", width="stretch")

            st.markdown("### Эффективность акций по виду топлива")
            promo_radio = st.radio("Тип акции:", ["Акция на топливо","Акция на магазин","Акция на кафе"],
                                   horizontal=True, key="t2b_promo")
            promo_col_map = {"Акция на топливо":"promotion_fuel_active",
                             "Акция на магазин":"promotion_shop_active",
                             "Акция на кафе":"promotion_cafe_active"}
            sel_pcol = promo_col_map[promo_radio]
            col_l, col_r = st.columns(2)
            with col_l:
                if sel_pcol in fdf.columns and adv_fuel_cols:
                    pg = fdf.groupby(sel_pcol)[[c for c in adv_fuel_cols if c in fdf.columns]].mean()
                    fig_pr = go.Figure()
                    for val,clr2,lbl2 in [(0,BLUE,"Без акции"),(1,GOLD,"С акцией")]:
                        if val in pg.index:
                            fig_pr.add_trace(go.Bar(name=lbl2,
                                x=[FUEL_LABELS[c] for c in adv_fuel_cols if c in fdf.columns],
                                y=[pg.loc[val,c] for c in adv_fuel_cols if c in fdf.columns],
                                marker_color=clr2))
                    fig_pr.update_layout(barmode="group",title="Влияние акции на топливо",
                                          legend=dict(orientation="h",y=-0.3))
                    st.plotly_chart(sfig(fig_pr), key="ap032", width="stretch")
            with col_r:
                sh_avail = [c for c in adv_shop_cols if c in fdf.columns]
                if sel_pcol in fdf.columns and sh_avail:
                    sg3 = fdf.groupby(sel_pcol)[sh_avail].mean()
                    fig_sh3 = go.Figure()
                    for val,clr3,lbl3 in [(0,"#9B59B6","Без акции"),(1,TEAL,"С акцией")]:
                        if val in sg3.index:
                            fig_sh3.add_trace(go.Bar(name=lbl3,
                                x=[SHOP_LABELS[c] for c in sh_avail],
                                y=[sg3.loc[val,c] for c in sh_avail],
                                marker_color=clr3))
                    fig_sh3.update_layout(barmode="group",title="Влияние акции на магазин",
                                           legend=dict(orientation="h",y=-0.3))
                    st.plotly_chart(sfig(fig_sh3), key="ap033", width="stretch")

            st.markdown("### Временные паттерны и праздники")
            flag_cols3 = st.columns(3)
            for _fi,(ccol,flag,labels,colors,title) in enumerate([
                (flag_cols3[0],"is_holiday",{0:"Обычный день",1:"Праздник"},
                 {"Обычный день":GRID_COLOR,"Праздник":GREEN},"Праздники vs обычные"),
                (flag_cols3[1],"is_rush_hour",{0:"Обычный час",1:"Пиковый час"},
                 {"Обычный час":GRID_COLOR,"Пиковый час":GOLD},"Пиковые часы"),
                (flag_cols3[2],"is_weekend",{0:"Будни",1:"Выходные"},
                 {"Будни":GRID_COLOR,"Выходные":"#f59e0b"},"Будни vs Выходные"),
            ]):
                if flag in fdf.columns:
                    gf2 = fdf.groupby(flag)["total_sales"].mean().reset_index()
                    gf2["Метка"] = gf2[flag].map(labels)
                    with ccol:
                        fig_gf = px.bar(gf2, x="Метка", y="total_sales", color="Метка",
                                        color_discrete_map=colors,
                                        labels={"total_sales":"Ср. л/час"})
                        fig_gf.update_layout(title=title, showlegend=False)
                        st.plotly_chart(sfig(fig_gf, height=250), key=f"ap_flag_{_fi}", width="stretch")

            col_l, col_r = st.columns(2)
            with col_l:
                if "ad_active" in fdf.columns:
                    af = fdf.groupby("ad_active")["total_sales"].mean().reset_index()
                    af["Реклама"] = af["ad_active"].map({0:"Без рекламы",1:"С рекламой"})
                    fig_af = px.bar(af, x="Реклама", y="total_sales", color="Реклама",
                                    color_discrete_map={"Без рекламы":GRID_COLOR,"С рекламой":BLUE},
                                    labels={"total_sales":"Ср. л/час"})
                    fig_af.update_layout(title="Реклама активна vs нет", showlegend=False)
                    st.plotly_chart(sfig(fig_af, height=250), key="ap036", width="stretch")
            with col_r:
                if "holiday_name" in fdf.columns:
                    hn = (fdf[fdf["is_holiday"]==1].groupby("holiday_name")["total_sales"]
                          .mean().reset_index().sort_values("total_sales",ascending=True))
                    if not hn.empty:
                        fig_hn = px.bar(hn, x="total_sales", y="holiday_name", orientation="h",
                                        labels={"total_sales":"Ср. л/час","holiday_name":""},
                                        color_discrete_sequence=[GOLD])
                        fig_hn.update_layout(title="Продажи по праздникам")
                        st.plotly_chart(sfig(fig_hn), key="ap037", width="stretch")

        # ── Корреляции ────────────────────────────────────────
        with sub2c:
            corr_cols = adv_fuel_cols + [
                "temperature","precipitation_mm","wind_speed_ms","visibility_km",
                "total_traffic","shop_total_revenue",
                "promotion_fuel_active","promotion_shop_active","promotion_cafe_active",
                "ad_active","is_holiday","is_weekend","is_rush_hour","is_night",
                "competitor_price_AI92","competitor_price_AI95","competitor_price_DT",
            ]
            corr_cols = [c for c in corr_cols if c in fdf.columns]
            all_lbl = {**TARGET_LABELS, **{
                "temperature":"Температура","precipitation_mm":"Осадки",
                "wind_speed_ms":"Ветер","visibility_km":"Видимость",
                "total_traffic":"Трафик","shop_total_revenue":"Выручка магазина",
                "promotion_fuel_active":"Акция топливо","promotion_shop_active":"Акция магазин",
                "promotion_cafe_active":"Акция кафе","ad_active":"Реклама",
                "is_holiday":"Праздник","is_weekend":"Выходной",
                "is_rush_hour":"Пиковый час","is_night":"Ночь",
                "competitor_price_AI92":"Конк. АИ-92","competitor_price_AI95":"Конк. АИ-95",
                "competitor_price_DT":"Конк. ДТ",
            }}
            if corr_cols:
                corr = fdf[corr_cols].corr().round(2)
                corr.index   = [all_lbl.get(c,c) for c in corr.index]
                corr.columns = [all_lbl.get(c,c) for c in corr.columns]
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig_corr.update_layout(title="Матрица корреляций", height=580)
                st.plotly_chart(sfig(fig_corr), key="ap042", width="stretch")

                factor_cols_c = [c for c in corr_cols if c not in adv_fuel_cols]
                if factor_cols_c and "total_sales" in fdf.columns:
                    top = fdf[factor_cols_c+["total_sales"]].corr()["total_sales"].drop("total_sales")
                    top = top.abs().sort_values(ascending=True)
                    top.index = [all_lbl.get(c,c) for c in top.index]
                    fig_top = px.bar(top.reset_index(), x="total_sales", y="index", orientation="h",
                                     labels={"total_sales":"|Корреляция| с продажами","index":""},
                                     color_discrete_sequence=[GOLD])
                    fig_top.update_layout(title="Факторы по силе влияния на продажи")
                    st.plotly_chart(sfig(fig_top), key="ap043", width="stretch")

            st.divider()
            st.markdown("### Корреляция по группам факторов")
            _group_opts = {
                "Трафик":  [c for c in ["traffic_Passengers_cars","traffic_Truck_short","traffic_Truck",
                            "traffic_Truck_long","traffic_Transporter","traffic_Undefined","total_traffic"]
                            if c in fdf.columns],
                "Погода":  [c for c in ["temperature","precipitation_mm","wind_speed_ms","visibility_km",
                            "is_rain","is_snow","is_fog"] if c in fdf.columns],
                "Цены":    [c for c in ["price_AI92","price_AI95","price_AI98",
                            "price_DT_EURO","price_DT_TANEKO","price_DT_SUMMER","price_DT_WINTER",
                            "comp_ratio_AI92","comp_ratio_AI95","comp_ratio_DT"] if c in fdf.columns],
                "Акции":   [c for c in ["promotion_fuel_active","promotion_shop_active","promotion_cafe_active",
                            "ad_active","is_holiday","is_weekend","is_rush_hour","is_night"] if c in fdf.columns],
                "Магазин": [c for c in ["shop_напитки","shop_закуски","shop_автотовары",
                            "shop_кофе","shop_табак","shop_total_revenue"] if c in fdf.columns],
            }
            _chosen_grp = st.selectbox("Группа факторов", list(_group_opts.keys()), key="corr_group")
            _factor_set = _group_opts[_chosen_grp]
            _corr2_cols = [c for c in adv_fuel_cols + _factor_set if c in fdf.columns]
            if _corr2_cols:
                _extra_lbl = {
                    "traffic_Passengers_cars":"Легковые","traffic_Truck_short":"Груз. малые",
                    "traffic_Truck":"Грузовые","traffic_Truck_long":"Фуры",
                    "traffic_Transporter":"Микроавт.","traffic_Undefined":"Прочие",
                    "total_traffic":"Трафик",
                    "temperature":"Температура","precipitation_mm":"Осадки",
                    "wind_speed_ms":"Ветер","visibility_km":"Видимость",
                    "is_rain":"Дождь","is_snow":"Снег","is_fog":"Туман",
                    "price_AI92":"Цена АИ-92","price_AI95":"Цена АИ-95","price_AI98":"Цена АИ-98",
                    "price_DT_EURO":"Цена ДТ Евро+","price_DT_TANEKO":"Цена ДТ ТАНЕКО",
                    "price_DT_SUMMER":"Цена ДТ Лето","price_DT_WINTER":"Цена ДТ Зима",
                    "comp_ratio_AI92":"Соотн. АИ-92","comp_ratio_AI95":"Соотн. АИ-95",
                    "comp_ratio_DT":"Соотн. ДТ",
                    "promotion_fuel_active":"Акция топливо","promotion_shop_active":"Акция магазин",
                    "promotion_cafe_active":"Акция кафе","ad_active":"Реклама",
                    "is_holiday":"Праздник","is_weekend":"Выходной",
                    "is_rush_hour":"Пиковый час","is_night":"Ночь",
                    "shop_total_revenue":"Выручка магазина",
                    **SHOP_LABELS,
                }
                _all_lbl2 = {**TARGET_LABELS, **_extra_lbl}
                _corr2 = fdf[_corr2_cols].corr().round(2)
                _corr2.index   = [_all_lbl2.get(c, c) for c in _corr2.index]
                _corr2.columns = [_all_lbl2.get(c, c) for c in _corr2.columns]
                _fig_c2 = px.imshow(_corr2, text_auto=True, aspect="auto",
                                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                _fig_c2.update_layout(title=f"Корреляции: топливо × {_chosen_grp}", height=500)
                st.plotly_chart(sfig(_fig_c2), key="ap044", width="stretch")

                if adv_fuel_cols and _factor_set:
                    _top5_fuel_keys = [c for c in adv_fuel_cols if c in fdf.columns]
                    _top5_fc = st.radio(
                        "Топливо для топ-5", options=_top5_fuel_keys,
                        format_func=lambda c: FUEL_LABELS.get(c, c),
                        horizontal=True, key="top5_fuel")
                    _top5_sel = FUEL_LABELS.get(_top5_fc, _top5_fc)
                    if _top5_fc in fdf.columns:
                        _t5 = (fdf[[c for c in _factor_set if c in fdf.columns] + [_top5_fc]]
                               .corr()[_top5_fc].drop(_top5_fc)
                               .abs().sort_values(ascending=False).head(5).reset_index())
                        _t5.columns = ["Фактор", "|Корреляция|"]
                        _t5["Фактор"] = _t5["Фактор"].map(lambda x: _all_lbl2.get(x, x))
                        st.markdown(f"**Топ-5 факторов, влияющих на {_top5_sel}:**")
                        st.dataframe(_t5, hide_index=True)

            st.divider()
            st.markdown("### Цены конкурентов")
            st.markdown('<div class="info-box">Значение &gt;1 — конкурент дороже нас. &lt;1 — конкурент дешевле.</div>', unsafe_allow_html=True)
            price_pairs = []
            if "comp_ratio_AI92" in fdf.columns and "sales_AI92" in fdf.columns:
                price_pairs.append(("comp_ratio_AI92","sales_AI92","АИ-92",GOLD))
            if "comp_ratio_AI95" in fdf.columns and "sales_AI95" in fdf.columns:
                price_pairs.append(("comp_ratio_AI95","sales_AI95","АИ-95",BLUE))
            _dt_s = next((c for c in ["sales_DT_EURO","sales_DT_TANEKO"] if c in fdf.columns),None)
            if "comp_ratio_DT" in fdf.columns and _dt_s:
                price_pairs.append(("comp_ratio_DT",_dt_s,"ДТ",GREEN))
            if price_pairs:
                pr_cols = st.columns(len(price_pairs))
                for i,(rc,sc2,fl,clr2) in enumerate(price_pairs):
                    with pr_cols[i]:
                        sub2 = fdf[[rc,sc2]].dropna()
                        samp2 = sub2.sample(min(2000,len(sub2)),random_state=42)
                        xp2 = samp2[rc].values.astype(float)
                        yp2 = samp2[sc2].values.astype(float)
                        fig_pr2 = go.Figure()
                        fig_pr2.add_trace(go.Scatter(x=xp2,y=yp2,mode="markers",
                            marker=dict(color=clr2,opacity=0.3,size=3),showlegend=False))
                        r_pr2 = add_regline(fig_pr2, xp2, yp2)
                        fig_pr2.add_vline(x=1.0,line_dash="dot",line_color="#aaa",
                                           annotation_text="Равная цена",annotation_position="top")
                        fig_pr2.update_layout(title=f"{fl}: конкурент/наша"+(f"  (r={r_pr2:.2f})" if r_pr2 else ""),
                                               xaxis_title="Соотношение цен", yaxis_title=f"Продажи {fl}, л/час")
                        st.plotly_chart(sfig(fig_pr2), key=f"ap_price_{i}", width="stretch")
            else:
                st.info("Данные о ценах конкурентов не найдены.")

        # ── Статистика ────────────────────────────────────────
        with sub2d:
            st.markdown("### Распределения целевых переменных")
            all_tgt_opts = [c for c in adv_fuel_cols + adv_shop_cols if c in fdf.columns]
            if all_tgt_opts:
                tgt_choice = st.selectbox("Целевая переменная",
                    [TARGET_LABELS.get(c,c) for c in all_tgt_opts], key="dist_sel")
                sel_col = all_tgt_opts[[TARGET_LABELS.get(c,c) for c in all_tgt_opts].index(tgt_choice)]
                unit_lbl = "л/час" if sel_col.startswith("sales_") else "руб/час"
                skew_b = fdf[sel_col].skew()
                log_v  = np.log1p(fdf[sel_col])
                skew_a = log_v.skew()

                col_l, col_r = st.columns(2)
                with col_l:
                    fig_hb = px.histogram(fdf, x=sel_col, nbins=60,
                                          labels={sel_col:f"Значение, {unit_lbl}"},
                                          color_discrete_sequence=[GOLD])
                    fig_hb.update_layout(title=f"До log-transform | skew={skew_b:.3f}")
                    st.plotly_chart(sfig(fig_hb), key="ap046", width="stretch")
                with col_r:
                    fig_ha = px.histogram(x=log_v, nbins=60,
                                          labels={"x":f"log1p({tgt_choice})"},
                                          color_discrete_sequence=[BLUE])
                    fig_ha.update_layout(title=f"После log-transform | skew={skew_a:.3f}")
                    st.plotly_chart(sfig(fig_ha), key="ap047", width="stretch")

                col_l, col_r = st.columns(2)
                with col_l:
                    fig_viol = px.violin(fdf, x="station_name", y=sel_col,
                                         color="station_name", color_discrete_map=stn_clr,
                                         box=True, points=False,
                                         labels={"station_name":"Станция", sel_col:unit_lbl})
                    fig_viol.update_layout(title=f"Violin по станциям — {tgt_choice}", showlegend=False)
                    st.plotly_chart(sfig(fig_viol), key="ap048", width="stretch")
                with col_r:
                    fig_box = px.box(fdf, x="station_name", y=sel_col,
                                      color="station_name", color_discrete_map=stn_clr,
                                      labels={"station_name":"Станция", sel_col:unit_lbl})
                    fig_box.update_layout(title=f"Boxplot — {tgt_choice}", showlegend=False)
                    st.plotly_chart(sfig(fig_box), key="ap049", width="stretch")

            st.divider()
            st.markdown("### Блок 3 — Временные паттерны")
            _pat_choice = st.radio("Разрез", ["По часам", "По дням недели", "По месяцам"],
                                   horizontal=True, key="pat_choice")
            _day_map2 = {0:"Пн", 1:"Вт", 2:"Ср", 3:"Чт", 4:"Пт", 5:"Сб", 6:"Вс"}
            _pat_fuel_cols = [c for c in adv_fuel_cols if c in fdf.columns]
            if _pat_fuel_cols:
                if _pat_choice == "По часам":
                    _grp = fdf.groupby("hour")[_pat_fuel_cols].mean().reset_index()
                    _grp = _grp.melt(id_vars="hour", var_name="fuel", value_name="sales")
                    _grp["fuel"] = _grp["fuel"].map(FUEL_LABELS)
                    _fig_p = px.line(_grp, x="hour", y="sales", color="fuel",
                                     color_discrete_map={FUEL_LABELS[c]: FUEL_HEX.get(c, GOLD) for c in _pat_fuel_cols},
                                     markers=True,
                                     labels={"hour":"Час суток","sales":"Ср. продажи, л/час","fuel":"Топливо"})
                    _fig_p.update_layout(title="Средние продажи по часу суток")
                    _fig_p.update_xaxes(dtick=2)
                elif _pat_choice == "По дням недели":
                    fdf["_day_label2"] = fdf["day_of_week"].map(_day_map2)
                    _grp = (fdf.groupby(["day_of_week","_day_label2"])[_pat_fuel_cols]
                            .mean().reset_index().sort_values("day_of_week"))
                    _grp = _grp.melt(id_vars=["day_of_week","_day_label2"], var_name="fuel", value_name="sales")
                    _grp["fuel"] = _grp["fuel"].map(FUEL_LABELS)
                    _fig_p = px.bar(_grp, x="_day_label2", y="sales", color="fuel",
                                    color_discrete_map={FUEL_LABELS[c]: FUEL_HEX.get(c, GOLD) for c in _pat_fuel_cols},
                                    barmode="group",
                                    labels={"_day_label2":"День недели","sales":"Ср. продажи, л/час","fuel":"Топливо"})
                    _fig_p.update_layout(title="Средние продажи по дням недели")
                else:
                    _grp = fdf.groupby("month")[_pat_fuel_cols].mean().reset_index()
                    _grp = _grp.melt(id_vars="month", var_name="fuel", value_name="sales")
                    _grp["fuel"] = _grp["fuel"].map(FUEL_LABELS)
                    _fig_p = px.line(_grp, x="month", y="sales", color="fuel",
                                     color_discrete_map={FUEL_LABELS[c]: FUEL_HEX.get(c, GOLD) for c in _pat_fuel_cols},
                                     markers=True,
                                     labels={"month":"Месяц","sales":"Ср. продажи, л/час","fuel":"Топливо"})
                    _fig_p.update_layout(title="Сезонность продаж по месяцам")
                    _fig_p.update_xaxes(dtick=1)
                _fig_p.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(sfig(_fig_p), key="ap_pat", width="stretch")

            st.divider()
            st.markdown("### Таблица выбросов (метод IQR)")
            num_c    = df.select_dtypes(include=["number"]).columns
            bin_c    = [c for c in num_c if df[c].dropna().isin([0,1]).all()]
            static_c = set(STATIC_REALS)
            int_c    = [c for c in num_c if c not in bin_c and c not in static_c]
            out_rows = []
            for c in int_c:
                Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                n = int(((df[c] < lo) | (df[c] > hi)).sum())
                if n > 0:
                    out_rows.append({"Колонка":c,"Выбросов":n,
                                     "IQR нижняя":round(lo,2),"IQR верхняя":round(hi,2)})
            if out_rows:
                st.dataframe(pd.DataFrame(out_rows), hide_index=True)
            else:
                st.info("Выбросов по методу IQR не обнаружено.")


# ══════════════════════════════════════════════════════════════
# КОНСТАНТЫ СКОЛЬЗЯЩЕГО ПРОГНОЗА
# ══════════════════════════════════════════════════════════════
WEATHER_PRESETS = {
    "Норма":    {"temperature":-5,  "precipitation_mm":0.5, "visibility_km":8,  "wind_speed_ms":3, "is_snow":0,"is_rain":0,"is_fog":0},
    "Мороз":    {"temperature":-20, "precipitation_mm":0,   "visibility_km":15, "wind_speed_ms":5, "is_snow":0,"is_rain":0,"is_fog":0},
    "Снегопад": {"temperature":-3,  "precipitation_mm":8,   "visibility_km":2,  "wind_speed_ms":4, "is_snow":1,"is_rain":0,"is_fog":0},
    "Оттепель": {"temperature":2,   "precipitation_mm":3,   "visibility_km":6,  "wind_speed_ms":4, "is_snow":0,"is_rain":1,"is_fog":0},
}
WEATHER_NAMES = list(WEATHER_PRESETS.keys())
HORIZON_OPTIONS = {"1 день": 1, "7 дней (неделя)": 7, "30 дней (месяц)": 30}


def _encoder_classes(encoder):
    """Returns (known_set, first_val) from a NaNLabelEncoder (dict or array)."""
    if encoder is None or not hasattr(encoder, "classes_"):
        return None, None
    cls = encoder.classes_
    if isinstance(cls, dict):
        keys = list(cls.keys())
        return set(str(k) for k in keys), str(keys[0]) if keys else None
    try:
        lst = cls.tolist()
    except AttributeError:
        lst = list(cls)
    return (set(str(c) for c in lst), str(lst[0])) if lst else (None, None)


def _rolling_forecast(model, training, tft_cfg, prep_df, scalers_dict,
                      sc_st, start_ps, n_days, daily_conds, sel_target):
    """Скользящий прогноз: n_days × 24ч. Возвращает (DataFrame, error_str|None)."""
    try:
        return _rolling_forecast_impl(model, training, tft_cfg, prep_df, scalers_dict,
                                      sc_st, start_ps, n_days, daily_conds, sel_target)
    except Exception as _e:
        import traceback as _tb
        return None, f"{_e}\n\n{_tb.format_exc()}"


def _rolling_forecast_impl(model, training, tft_cfg, prep_df, scalers_dict,
                            sc_st, start_ps, n_days, daily_conds, sel_target):
    """Внутренняя реализация. Вызывается через _rolling_forecast с перехватом исключений."""
    from pytorch_forecasting import TimeSeriesDataSet

    enc_start = start_ps - pd.Timedelta(hours=ENCODER_LENGTH)
    syn_hist = prep_df[
        (prep_df["station_id"] == sc_st) &
        (prep_df["timestamp"] >= enc_start) &
        (prep_df["timestamp"] < start_ps)
    ].copy().reset_index(drop=True)
    if len(syn_hist) < ENCODER_LENGTH:
        syn_hist = (prep_df[prep_df["station_id"] == sc_st]
                    .sort_values("timestamp").tail(ENCODER_LENGTH)
                    .copy().reset_index(drop=True))
    if syn_hist.empty:
        return None, "Нет данных для выбранной станции"

    last_tidx = int(syn_hist["time_idx"].max())
    sid_key   = str(sc_st)
    traffic_cols = [c for c in syn_hist.columns if c.startswith("traffic_")]

    season_enc_map = (prep_df.groupby(prep_df["timestamp"].dt.month)["season_enc"]
                      .first().to_dict()) if "season_enc" in prep_df.columns else {}
    if "holiday_name_enc" in prep_df.columns:
        hol_enc_map = prep_df.groupby([
            prep_df["timestamp"].dt.month, prep_df["timestamp"].dt.day
        ])[["is_holiday","holiday_name_enc"]].first()
        no_hol_enc = int(prep_df[prep_df["is_holiday"]==0]["holiday_name_enc"].iloc[0])
    else:
        hol_enc_map, no_hol_enc = None, 0

    def _norm(col, raw_val):
        if (scalers_dict and sid_key in scalers_dict
                and isinstance(scalers_dict[sid_key], dict)
                and col in scalers_dict[sid_key]):
            m, s = scalers_dict[sid_key][col]
            return (raw_val - m) / s if s > 0 else 0.0
        return None

    all_rows = []
    for day in range(n_days):
        cond   = daily_conds[day] if day < len(daily_conds) else daily_conds[-1]
        day_ps = start_ps + pd.Timedelta(hours=day * 24)
        day_pe = day_ps + pd.Timedelta(hours=PREDICTION_LENGTH - 1)

        template  = syn_hist.iloc[-1].copy()
        wx_preset = WEATHER_PRESETS.get(cond.get("weather", "Норма"), WEATHER_PRESETS["Норма"])
        traf_mult = cond.get("traffic_pct", 100) / 100.0

        dec_rows = []
        for h in range(PREDICTION_LENGTH):
            ts  = day_ps + pd.Timedelta(hours=h)
            row = template.copy()
            row["timestamp"] = ts
            row["time_idx"]  = last_tidx + h + 1
            hr, dow, m = ts.hour, ts.dayofweek, ts.month
            woy = int(ts.isocalendar().week)
            row.update({
                "hour": hr,
                "hour_sin":  np.sin(2*np.pi*hr/24),  "hour_cos":  np.cos(2*np.pi*hr/24),
                "day_of_week": dow,
                "day_of_week_sin": np.sin(2*np.pi*dow/7), "day_of_week_cos": np.cos(2*np.pi*dow/7),
                "week_of_year": woy,
                "week_of_year_sin": np.sin(2*np.pi*woy/52), "week_of_year_cos": np.cos(2*np.pi*woy/52),
                "month": m,
                "month_sin": np.sin(2*np.pi*m/12), "month_cos": np.cos(2*np.pi*m/12),
                "is_weekend": int(dow >= 5),
                "is_rush_hour": int(hr in [7,8,9,17,18,19]),
                "is_night": int(hr < 6 or hr >= 22),
                "is_shop_open": int(5 <= hr <= 21),
                "promotion_fuel_active": int(cond.get("promo_fuel", False)),
                "promotion_shop_active": int(cond.get("promo_shop", False)),
                "ad_active": int(cond.get("ad_active", False)),
            })
            if "season_enc" in row.index:
                row["season_enc"] = season_enc_map.get(m, template.get("season_enc", 0))
            if hol_enc_map is not None:
                hk = (m, ts.day)
                if hk in hol_enc_map.index:
                    hd = hol_enc_map.loc[hk]
                    row["is_holiday"] = int(hd["is_holiday"])
                    row["holiday_name_enc"] = int(hd["holiday_name_enc"])
                else:
                    row["is_holiday"] = 0
                    if "holiday_name_enc" in row.index: row["holiday_name_enc"] = no_hol_enc
            for wk, wv in wx_preset.items():
                if wk in row.index:
                    normed = _norm(wk, wv)
                    row[wk] = normed if normed is not None else template.get(wk, wv)
            for tc in traffic_cols:
                if tc in row.index:
                    row[tc] = template.get(tc, 0) * traf_mult
            for col in TARGET_COLS:
                if col in row.index: row[col] = 0.0
            dec_rows.append(row)

        ctx = pd.concat([syn_hist.tail(ENCODER_LENGTH).copy().reset_index(drop=True),
                         pd.DataFrame(dec_rows)], ignore_index=True)
        # Читаем каждый список отдельно (см. compute_interpretation — аналогичная причина)
        _cat_cols = set(
            list(getattr(training, "static_categoricals", None) or [])
            + list(getattr(training, "time_varying_known_categoricals", None) or [])
            + list(getattr(training, "time_varying_unknown_categoricals", None) or [])
        )
        # Add any missing categorical columns — derive temporal ones from prep_df
        for col in _cat_cols:
            if col not in ctx.columns:
                _ek, _ef = _encoder_classes(training.categorical_encoders.get(col))
                _derived = False
                if "month" in ctx.columns and col in prep_df.columns and "month" in prep_df.columns:
                    _mmap = prep_df.groupby("month")[col].first().to_dict()
                    if _mmap:
                        ctx[col] = ctx["month"].map(_mmap)
                        _derived = True
                if not _derived:
                    ctx[col] = _ef if _ef is not None else "0"
        for col in _cat_cols:
            if col not in ctx.columns:
                continue
            try:
                ctx[col] = ctx[col].ffill().bfill().astype(float).astype(int).astype(str)
            except Exception:
                try: ctx[col] = ctx[col].ffill().bfill().astype(str)
                except Exception: pass
            _enc_known, _enc_first = _encoder_classes(training.categorical_encoders.get(col))
            if _enc_known and _enc_first is not None:
                ctx[col] = ctx[col].map(
                    lambda v, k=_enc_known, f=_enc_first: v if v in k else f
                )

        # Финальная страховка: все колонки, требуемые training, должны быть в ctx
        for _c in _cat_cols:
            if _c not in ctx.columns:
                ctx[_c] = "0"

        ds  = TimeSeriesDataSet.from_dataset(training, ctx, stop_randomization=True)
        ldr = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        raw = model.predict(ldr, mode="quantiles",
                            trainer_kwargs={"logger":False,"enable_progress_bar":False})

        # Normalise predict() return: extract list/tuple of per-target tensors.
        # pytorch-forecasting may return NamedTuple(.output), dict, list/tuple, or tensor.
        import torch as _torch
        if hasattr(raw, "output"):
            _raw_out = raw.output
        elif isinstance(raw, dict) and "prediction" in raw:
            _raw_out = raw["prediction"]
        elif isinstance(raw, (list, tuple)):
            _raw_out = raw
        else:
            _raw_out = raw

        # Helper: extract array[batch=0, h, q] for target index ti robustly
        def _get_target_arr(ti):
            if isinstance(_raw_out, (list, tuple)):
                if ti < len(_raw_out):
                    t = _raw_out[ti]
                    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.array(t)
                return None
            if isinstance(_raw_out, _torch.Tensor):
                t = _raw_out
                # Try [n_targets, batch, pred_len, n_quantiles]
                if t.ndim == 4 and t.shape[0] == len(TARGET_COLS) and ti < t.shape[0]:
                    return t[ti].detach().cpu().numpy()
                # Try [batch, n_targets, pred_len, n_quantiles]
                if t.ndim == 4 and t.shape[1] == len(TARGET_COLS) and ti < t.shape[1]:
                    return t[:, ti, :, :].detach().cpu().numpy()
                # Single tensor [batch, pred_len, n_quantiles]
                if t.ndim == 3:
                    return t.detach().cpu().numpy()
                return None
            return None

        ti  = TARGET_COLS.index(sel_target)
        arr = _get_target_arr(ti)
        if arr is None:
            return None, f"Не удалось извлечь прогноз для '{sel_target}' из выхода модели"

        day_log_p50 = {}
        for h in range(PREDICTION_LENGTH):
            ts2 = day_ps + pd.Timedelta(hours=h)
            if day_ps <= ts2 <= day_pe:
                all_rows.append({
                    "timestamp": ts2, "day": day,
                    "p10": float(np.expm1(max(arr[0, h, Q_LO], 0))),
                    "p50": float(np.expm1(max(arr[0, h, Q_MED], 0))),
                    "p90": float(np.expm1(max(arr[0, h, Q_HI], 0))),
                })
                day_log_p50[ts2] = float(arr[0, h, Q_MED])
        # Feedback: fill all-target P50 into dec_rows for next encoder window
        for ti2, tgt2 in enumerate(TARGET_COLS):
            arr2 = _get_target_arr(ti2)
            if arr2 is not None:
                for h in range(PREDICTION_LENGTH):
                    if h < len(dec_rows):
                        dec_rows[h][tgt2] = float(arr2[0, h, Q_MED])

        dec_df = pd.DataFrame(dec_rows).reset_index(drop=True)
        syn_hist = pd.concat([syn_hist, dec_df], ignore_index=True)
        last_tidx += PREDICTION_LENGTH

    return (pd.DataFrame(all_rows) if all_rows else None), None


# ══════════════════════════════════════════════════════════════
# TAB 3 — ПРОГНОЗ TFT
# ══════════════════════════════════════════════════════════════
with tab3:
    if pred_df is None and metrics_df is None:
        st.info("Файлы прогнозов не найдены. Выполните: `python tft/predict.py`")
    else:
        _t3c1, _t3c2, _t3c3 = st.columns([2, 2, 4])
        with _t3c1:
            sel_station_tft = st.selectbox(
                "Станция", ["Все станции"] + _all_stations,
                format_func=lambda x: _station_display.get(x, x) if x != "Все станции" else x,
                label_visibility="collapsed", key="tft_station",
            )
        with _t3c2:
            sel_target = st.selectbox(
                "Показатель", TARGET_COLS,
                format_func=lambda x: TARGET_LABELS.get(x, x),
                label_visibility="collapsed", key="tft_target",
            )
        target_label = TARGET_LABELS.get(sel_target, sel_target)
        unit         = "л/ч" if sel_target in FUEL_COLS else "руб/ч"
        sel_sid = (_sid_by_name.get(sel_station_tft) if sel_station_tft != "Все станции" else None)

        sub3a, sub3b, sub3c, sub3d = st.tabs([
            "📋  Метрики & Точность",
            "📈  Прогноз vs Факт",
            "🔮  Сценарий (What-if)",
            "🧠  Интерпретация VSN",
        ])

        _pf = pred_df.copy()  if pred_df  is not None else pd.DataFrame()
        _mf = metrics_df.copy() if metrics_df is not None else pd.DataFrame()
        if sel_sid and not _pf.empty:
            _pf = _pf[_pf["station_id"] == sel_sid]
        if sel_sid and not _mf.empty:
            _mf = _mf[_mf["station_id"].astype(str) == sel_sid]

        _tft_all_names = {**{c: FUEL_LABELS[c] for c in FUEL_COLS},
                          **{c: SHOP_LABELS[c] for c in SHOP_COLS}}
        _avail_tgt = [t for t in TARGET_COLS if f"{t}_pred" in (pred_df.columns if pred_df is not None else [])]

        # ── Метрики & Точность ────────────────────────────────
        with sub3a:
            if not _mf.empty:
                _mape_col = next((c for c in _mf.columns if "MAPE" in c.upper()), None)
                if _mape_col:
                    _med_mape = _mf[_mape_col].median()
                    _good_cnt = int((_mf[_mape_col] <= 15).sum())
                else:
                    _med_mape, _good_cnt = None, 0

                _r2_list = []
                for _t in _avail_tgt:
                    _sub = (pred_df if pred_df is not None else pd.DataFrame())
                    if sel_sid: _sub = _sub[_sub["station_id"]==sel_sid]
                    _sub2 = _sub[[f"{_t}_actual",f"{_t}_pred"]].dropna() if all(c in _sub.columns for c in [f"{_t}_actual",f"{_t}_pred"]) else pd.DataFrame()
                    if len(_sub2) > 1:
                        _ssr = ((_sub2[f"{_t}_actual"]-_sub2[f"{_t}_pred"])**2).sum()
                        _sst = ((_sub2[f"{_t}_actual"]-_sub2[f"{_t}_actual"].mean())**2).sum()
                        if _sst > 0: _r2_list.append(1-_ssr/_sst)
                _med_r2 = float(np.median(_r2_list)) if _r2_list else None

                k1,k2,k3,k4 = st.columns(4)
                kpi_card(k1,"Таргетов",str(len(_avail_tgt)),"в predictions.csv")
                kpi_card(k2,"Медиана MAPE",f"{_med_mape:.1f}%" if _med_mape else "—","ниже = лучше",
                         GREEN if _med_mape and _med_mape<=10 else (GOLD if _med_mape and _med_mape<=20 else RED))
                kpi_card(k3,"Медиана R²",f"{_med_r2:.3f}" if _med_r2 else "—","выше = лучше",
                         GREEN if _med_r2 and _med_r2>=0.8 else GOLD)
                kpi_card(k4,"MAPE ≤ 15%",str(_good_cnt),f"из {len(_avail_tgt)} целей",
                         GREEN if _good_cnt >= len(_avail_tgt)//2 else GOLD)
                st.markdown("<br>", unsafe_allow_html=True)

                if _mape_col:
                    col_acc, col_heat = st.columns([3,2])
                    with col_acc:
                        _acc_df = _mf.groupby("target",as_index=False)[_mape_col].mean()
                        _acc_df["name"] = _acc_df["target"].map(lambda x: _tft_all_names.get(x,x))
                        _acc_df = _acc_df.sort_values(_mape_col, na_position="first")
                        _bar_colors = [
                            GRAY if pd.isna(v) else (GREEN if v<=10 else (GOLD if v<=20 else RED))
                            for v in _acc_df[_mape_col]]
                        _bar_text = [
                            "нет продаж" if pd.isna(v) else f"{v:.1f}%"
                            for v in _acc_df[_mape_col]]
                        _bar_hover = [
                            "%{y}: нет данных о продажах<extra></extra>" if pd.isna(v)
                            else "%{y}: MAPE=%{x:.1f}%<extra></extra>"
                            for v in _acc_df[_mape_col]]
                        fig_acc = go.Figure(go.Bar(
                            x=_acc_df[_mape_col].fillna(0).round(1), y=_acc_df["name"],
                            orientation="h", marker_color=_bar_colors,
                            text=_bar_text, textposition="outside",
                            hovertemplate=_bar_hover))
                        fig_acc.add_vline(x=10,line_dash="dash",line_color=GREEN,opacity=0.6)
                        fig_acc.add_vline(x=20,line_dash="dash",line_color=GOLD,opacity=0.6)
                        fig_acc.update_layout(title="Точность по таргетам (MAPE, %)",
                                               xaxis_title="MAPE, %", yaxis_title="")
                        st.plotly_chart(sfig(fig_acc,height=380), key="ap_tft_acc", width="stretch")

                    with col_heat:
                        if metrics_df is not None and _mape_col in metrics_df.columns:
                            _pivot = metrics_df[metrics_df["target"].isin(_avail_tgt)].pivot_table(
                                index="station_id", columns="target", values=_mape_col, aggfunc="mean")
                            _pivot.columns = [_tft_all_names.get(c,c) for c in _pivot.columns]
                            _pivot.index = [_name_by_sid.get(str(i),f"АЗС {i}") for i in _pivot.index]
                            fig_heat2 = go.Figure(go.Heatmap(
                                z=_pivot.values, x=list(_pivot.columns), y=list(_pivot.index),
                                colorscale=[[0,GREEN],[0.5,GOLD],[1,RED]],
                                zmin=0, zmax=30,
                                text=np.round(_pivot.values,1).astype(str), texttemplate="%{text}%",
                                hovertemplate="АЗС: %{y}<br>Цель: %{x}<br>MAPE: %{z:.1f}%<extra></extra>"))
                            fig_heat2.update_layout(title="MAPE: станция × таргет")
                            st.plotly_chart(sfig(fig_heat2,height=380), key="ap_tft_heat", width="stretch")

                with st.expander("📋  Таблица метрик"):
                    _mt = _mf.copy()
                    _mt["Таргет"] = _mt["target"].map(lambda x: _tft_all_names.get(x,x))
                    _mt["Станция"] = _mt["station_id"].astype(str).map(lambda x: _name_by_sid.get(x,f"АЗС {x}"))
                    _disp_cols = ["Станция","Таргет"] + [c for c in _mt.columns
                                   if c not in {"target","station_id","Станция","Таргет"}]
                    st.dataframe(_mt[_disp_cols].sort_values(["Станция","Таргет"]).reset_index(drop=True),
                                 hide_index=True)
            else:
                st.info("Файл `data/metrics.csv` не найден.")

        # ── Прогноз vs Факт ───────────────────────────────────
        with sub3b:
            if _pf.empty:
                st.info("Нет данных прогноза для выбранной станции.")
            else:
                _tgt_sel = st.selectbox("Целевая переменная",
                    [t for t in _avail_tgt],
                    format_func=lambda x: _tft_all_names.get(x,x),
                    key="tft_tgt_b")
                _pc = f"{_tgt_sel}_pred"; _ac = f"{_tgt_sel}_actual"
                _q1c = f"{_tgt_sel}_q10"; _q9c = f"{_tgt_sel}_q90"

                if _pc in _pf.columns and _ac in _pf.columns:
                    _ts = _pf[["timestamp",_ac,_pc]].dropna().sort_values("timestamp")
                    fig_ts = go.Figure()
                    if _q1c in _pf.columns and _q9c in _pf.columns:
                        _bd = _pf[["timestamp",_q1c,_q9c]].dropna().sort_values("timestamp")
                        fig_ts.add_trace(go.Scatter(
                            x=pd.concat([_bd["timestamp"],_bd["timestamp"].iloc[::-1]]),
                            y=pd.concat([_bd[_q9c],_bd[_q1c].iloc[::-1]]),
                            fill="toself", fillcolor="rgba(46,117,182,0.15)",
                            line=dict(color="rgba(0,0,0,0)"), name="Q10–Q90", hoverinfo="skip"))
                    fig_ts.add_trace(go.Scatter(x=_ts["timestamp"],y=_ts[_ac],
                        mode="lines",name="Факт",line=dict(color=GOLD,width=1.5)))
                    fig_ts.add_trace(go.Scatter(x=_ts["timestamp"],y=_ts[_pc],
                        mode="lines",name="Прогноз P50",line=dict(color=BLUE,width=1.5,dash="dot")))
                    fig_ts.update_layout(
                        title=f"{_tft_all_names.get(_tgt_sel,_tgt_sel)} — прогноз vs факт (декабрь 2023)",
                        xaxis_title="Дата", yaxis_title=f"Значение, {unit}",
                        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
                    st.plotly_chart(sfig(fig_ts,height=400), key="ap_tft_ts", width="stretch")

                    _sc = _ts[[_ac,_pc]].dropna()
                    if len(_sc) >= 5:
                        col_sc, col_err = st.columns([3,2])
                        with col_sc:
                            fig_sc = go.Figure()
                            fig_sc.add_trace(go.Scatter(x=_sc[_ac],y=_sc[_pc],mode="markers",
                                marker=dict(color=GOLD,size=3,opacity=0.4),
                                hovertemplate="Факт: %{x:.2f}<br>Прогноз: %{y:.2f}<extra></extra>"))
                            _vmin = float(min(_sc[_ac].min(),_sc[_pc].min()))
                            _vmax = float(max(_sc[_ac].max(),_sc[_pc].max()))
                            fig_sc.add_trace(go.Scatter(x=[_vmin,_vmax],y=[_vmin,_vmax],
                                mode="lines",name="Идеал",line=dict(color=RED,dash="dash",width=1.5)))
                            _ssr = ((_sc[_ac]-_sc[_pc])**2).sum()
                            _sst = ((_sc[_ac]-_sc[_ac].mean())**2).sum()
                            _r2  = 1 - _ssr/_sst if _sst > 0 else float("nan")
                            fig_sc.update_layout(title=f"Прогноз vs Факт (R²={_r2:.3f})",
                                                  xaxis_title="Факт",yaxis_title="Прогноз")
                            st.plotly_chart(sfig(fig_sc,height=360), key="ap_tft_sc", width="stretch")
                        with col_err:
                            _err = _sc[_pc] - _sc[_ac]
                            fig_err = go.Figure(go.Histogram(x=_err, nbinsx=50,
                                marker_color=GOLD, opacity=0.8))
                            fig_err.add_vline(x=0,line_dash="dash",line_color=RED,opacity=0.8)
                            fig_err.update_layout(title="Распределение ошибок",
                                                   xaxis_title="Прогноз − Факт",yaxis_title="Кол-во")
                            st.plotly_chart(sfig(fig_err,height=360), key="ap_tft_err", width="stretch")
                else:
                    st.warning(f"Колонки `{_pc}` / `{_ac}` не найдены в predictions.csv.")

        # ── Сценарий (What-if) ────────────────────────────────
        with sub3c:
            if prepared_df is None:
                st.info("Файл `data/prepared_data.csv` не найден. Выполните: `python eda/eda_preprocessing.py`")
            elif sel_station_tft == "Все станции":
                st.info("Выберите конкретную станцию в фильтре «Станция» выше для запуска сценария.")
            else:
                st.markdown('<div class="info-box">Задайте дату начала, горизонт и условия каждого дня. '
                            'TFT строит прогноз скользящим окном 24ч. Первый запуск загружает модель (~30 с).</div>',
                            unsafe_allow_html=True)

                sc_st  = sel_sid
                # Align station_id format with prepared_df (may use "station_N" prefix)
                if prepared_df is not None and sc_st is not None:
                    _pids = set(prepared_df["station_id"].unique())
                    if str(sc_st) not in _pids:
                        _alt = f"station_{sc_st}"
                        if _alt in _pids:
                            sc_st = _alt
                sc_tgt = sel_target
                PRICE_MAP = {
                    "sales_AI92":"price_AI92","sales_AI95":"price_AI95","sales_AI98":"price_AI98",
                    "sales_DT_EURO":"price_DT_EURO","sales_DT_TANEKO":"price_DT_TANEKO",
                    "sales_DT_SUMMER":"price_DT_SUMMER","sales_DT_WINTER":"price_DT_WINTER",
                }

                # ── Параметры прогноза ────────────────────────
                ph1, ph2, ph3 = st.columns([2, 1, 2])
                with ph1:
                    sc_date = st.date_input("Дата начала",
                        value=datetime.date(2023,12,1),
                        min_value=datetime.date(2023,12,1),
                        max_value=datetime.date(2024,1,31), key="sc_date")
                with ph2:
                    sc_hour = st.selectbox("Час", list(range(24)),
                        format_func=lambda h: f"{h:02d}:00", key="sc_hour")
                with ph3:
                    sc_horizon_lbl = st.radio("Горизонт", list(HORIZON_OPTIONS.keys()),
                        horizontal=True, key="sc_horizon")
                sc_n_days = HORIZON_OPTIONS[sc_horizon_lbl]

                # ── Цена (только для топлива, 1 день) ────────
                sc_price_rub = None; base_p = None
                price_col_sc = PRICE_MAP.get(sc_tgt)
                if price_col_sc and price_col_sc in df.columns and sel_sid:
                    _st_mask = df["station_id"] == sel_sid
                    base_p   = float(df[_st_mask][price_col_sc].median()) if _st_mask.any() else 50.0
                    sc_price_rub = st.slider(
                        f"Цена {TARGET_LABELS.get(sc_tgt,sc_tgt)}, руб/л (применяется ко всем дням)",
                        min_value=round(base_p*0.88,1), max_value=round(base_p*1.12,1),
                        value=round(base_p,1), step=0.5, key="sc_price")
                    st.caption(f"Медиана 2023 г.: {base_p:.1f} руб/л · диапазон ±12%")

                # ── Настройка неизвестных факторов по дням ───
                st.markdown("##### Условия по дням (неизвестные факторы)")
                _dc1, _dc2, _dc3, _dc4, _dc5 = st.columns([2,2,2,2,1])
                _dc1.markdown("**Дата**"); _dc2.markdown("**Погода**")
                _dc3.markdown("**Трафик**"); _dc4.markdown("**Акции**"); _dc5.markdown("**Реклама**")

                daily_conds = []
                _week_labels = ["Неделя 1","Неделя 2","Неделя 3","Неделя 4","Неделя 5"]

                def _day_row(day_idx, start_date):
                    day_date = start_date + datetime.timedelta(days=day_idx)
                    c1, c2, c3, c4, c5 = st.columns([2,2,2,2,1])
                    with c1:
                        st.markdown(f"**{day_date.strftime('%d.%m')}** {['Пн','Вт','Ср','Чт','Пт','Сб','Вс'][day_date.weekday()]}")
                    with c2:
                        wx = st.selectbox("Погода", WEATHER_NAMES, key=f"sc_wx_{day_idx}", label_visibility="collapsed")
                    with c3:
                        tr = st.select_slider("Трафик", options=[50,75,100,125,150], value=100,
                                              format_func=lambda v: f"{v}%",
                                              key=f"sc_tr_{day_idx}", label_visibility="collapsed")
                    with c4:
                        pr = st.multiselect("Акции", ["Топливо","Магазин"], key=f"sc_pr_{day_idx}", label_visibility="collapsed")
                    with c5:
                        ad = st.checkbox("Реклама", key=f"sc_ad_{day_idx}", label_visibility="collapsed")
                    return {"weather": wx, "traffic_pct": tr,
                            "promo_fuel": "Топливо" in pr, "promo_shop": "Магазин" in pr,
                            "ad_active": ad}

                if sc_n_days <= 7:
                    for d in range(sc_n_days):
                        daily_conds.append(_day_row(d, sc_date))
                else:
                    for week in range((sc_n_days + 6) // 7):
                        ws, we = week*7, min(week*7+7, sc_n_days)
                        w_start = sc_date + datetime.timedelta(days=ws)
                        w_end   = sc_date + datetime.timedelta(days=we-1)
                        with st.expander(f"{_week_labels[week]} · {w_start.strftime('%d.%m')}–{w_end.strftime('%d.%m')}", expanded=(week==0)):
                            for d in range(ws, we):
                                daily_conds.append(_day_row(d, sc_date))

                run_btn = st.button("▶  Запустить прогноз", type="primary", key="sc_run")
                if run_btn:
                    prog = st.progress(0, text="Инициализация...")
                    prog.progress(10, text="Загрузка TFT-модели...")
                    model, training, tft_cfg, err = load_tft()
                    if err or model is None:
                        prog.empty(); st.error(f"Не удалось загрузить модель: {err}")
                    else:
                        try:
                            # Применить цену ко всем дням через scalers
                            if sc_price_rub is not None and price_col_sc and base_p and base_p > 0:
                                _sid_key = str(sc_st)
                                _sc_ok = (scalers and _sid_key in scalers
                                          and isinstance(scalers[_sid_key],dict)
                                          and price_col_sc in scalers[_sid_key])
                                for d in range(len(daily_conds)):
                                    if _sc_ok:
                                        _m,_s = scalers[_sid_key][price_col_sc]
                                        daily_conds[d]["price_col"] = price_col_sc
                                        daily_conds[d]["price_norm"] = (sc_price_rub-_m)/_s if _s>0 else 0.0
                                    else:
                                        daily_conds[d]["price_col"]  = price_col_sc
                                        daily_conds[d]["price_ratio"] = sc_price_rub/base_p

                            start_ps = pd.Timestamp(sc_date) + pd.Timedelta(hours=sc_hour)
                            _total = sc_n_days
                            prog.progress(20, text=f"Прогноз: 0/{_total} дней...")

                            sc_df, sc_err = _rolling_forecast(
                                model, training, tft_cfg, prepared_df, scalers,
                                sc_st, start_ps, sc_n_days, daily_conds, sc_tgt)

                            if sc_err:
                                prog.empty(); st.error(sc_err)
                            elif sc_df is None or sc_df.empty:
                                prog.empty(); st.warning("Прогноз не содержит данных.")
                            else:
                                prog.progress(100, text="Готово!")
                                prog.empty()

                                u_s   = "л/ч" if sc_tgt in FUEL_COLS else "руб/ч"
                                lbl_s = TARGET_LABELS.get(sc_tgt, sc_tgt)

                                val_mean = sc_df["p50"].mean()
                                val_peak = sc_df["p50"].max()
                                peak_ts  = sc_df.loc[sc_df["p50"].idxmax(), "timestamp"]

                                st.session_state["sc_result"] = {
                                    "val_mean":val_mean,"val_peak":val_peak,"peak_ts":peak_ts,
                                    "sc_tgt":sc_tgt,"sc_st":sc_st,"n_days":sc_n_days,
                                    "sc_pf":any(d.get("promo_fuel") for d in daily_conds),
                                    "sc_ps":any(d.get("promo_shop") for d in daily_conds),
                                    "sc_ad":any(d.get("ad_active") for d in daily_conds),
                                    "sc_ch":"—","base_mean":None,
                                    "price_delta":((sc_price_rub-base_p)/base_p*100
                                                   if sc_price_rub and base_p and base_p>0 else None),
                                    "unit":u_s,
                                }

                                km1, km2, km3 = st.columns(3)
                                kpi_card(km1, f"Старт {sc_date.strftime('%d.%m')} {sc_hour:02d}:00",
                                         f"{sc_n_days} д · {sc_n_days*24} ч", "горизонт прогноза")
                                kpi_card(km2, f"Ср. {lbl_s}", f"{val_mean:.1f} {u_s}", "P50 за горизонт")
                                kpi_card(km3, "Пик", f"{val_peak:.1f} {u_s}",
                                         peak_ts.strftime("%d.%m %H:%M"))

                                # График P10/P50/P90
                                fig_roll = go.Figure()
                                fig_roll.add_trace(go.Scatter(
                                    x=pd.concat([sc_df["timestamp"], sc_df["timestamp"][::-1]]),
                                    y=pd.concat([sc_df["p90"], sc_df["p10"][::-1]]),
                                    fill="toself", fillcolor="rgba(46,117,182,0.15)",
                                    line=dict(color="rgba(0,0,0,0)"), showlegend=True,
                                    name="P10–P90", hoverinfo="skip"))
                                fig_roll.add_trace(go.Scatter(
                                    x=sc_df["timestamp"], y=sc_df["p50"],
                                    name="P50 (медиана)", mode="lines",
                                    line=dict(color=GREEN, width=2)))
                                # Вертикальные разделители суток
                                for d in range(1, sc_n_days):
                                    _dl = start_ps + pd.Timedelta(hours=d*24)
                                    fig_roll.add_vline(x=_dl, line_dash="dot",
                                                       line_color=GRID_COLOR, line_width=1)
                                fig_roll.update_layout(
                                    title=f"{lbl_s} · {sel_station_tft} · {sc_horizon_lbl}",
                                    xaxis_title="Время", yaxis_title=u_s,
                                    legend=dict(orientation="h", yanchor="top",
                                                y=-0.2, x=0, xanchor="left"),
                                    margin=dict(b=70))
                                st.plotly_chart(sfig(fig_roll, height=400), key="ap_sc_chart", width="stretch")

                                # Сводная таблица по суткам
                                day_tbl = (sc_df.groupby("day")
                                           .agg(date=("timestamp","first"),
                                                p10=("p10","mean"), p50=("p50","mean"),
                                                p90=("p90","mean"), peak=("p50","max"))
                                           .reset_index())
                                day_tbl["date"] = day_tbl["date"].dt.strftime("%d.%m")
                                day_tbl.columns = ["День","Дата","P10","P50","P90","Пик"]
                                day_tbl["День"] += 1
                                st.dataframe(day_tbl.style.format(
                                    {c: "{:.1f}" for c in ["P10","P50","P90","Пик"]}),
                                    hide_index=True)

                        except Exception as e:
                            prog.empty(); st.error(f"Ошибка при расчёте: {e}")

        # ── Интерпретация VSN + Temporal Attention ────────────
        with sub3d:
            st.markdown("""
**TFT** специально спроектирован для интерпретируемости. Два механизма объясняют *что* важно
и *когда* модель обращается к прошлому:

| Механизм | Что показывает |
|---|---|
| **VSN** (Variable Selection Networks) | Важность каждой переменной в трёх группах |
| **Temporal Self-Attention** | На какие прошлые часы модель смотрит при прогнозировании |
""")
            _IPERIODS = {
                "Декабрь 2023 — тест":       ("2023-12-01", "2023-12-10"),
                "Ноябрь 2023 — валидация":   ("2023-11-01", "2023-11-30"),
                "Q4 2023 (окт–дек)":          ("2023-10-01", "2023-12-31"),
                "Весь 2023 год":              ("2023-01-08", "2023-12-31"),
            }
            _ic1, _ic2, _ic3 = st.columns([2, 2, 2])
            with _ic1:
                iperiod = st.selectbox("Период", list(_IPERIODS.keys()), key="interp_period")
            with _ic2:
                istation = st.selectbox(
                    "Станция",
                    ["Все"] + _all_stations,
                    format_func=lambda x: "Все станции" if x == "Все" else _station_display.get(x, x),
                    key="interp_station",
                )
            with _ic3:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                run_interp = st.button("🧠  Рассчитать", type="primary", key="run_interp")

            ips, ipe  = _IPERIODS[iperiod]
            _ictx     = pd.Timestamp(ips) - pd.Timedelta(hours=ENCODER_LENGTH)
            _ist_id = "Все" if istation == "Все" else str(_sid_by_name.get(istation, "Все"))
            # Normalize station_id format to match prepared_df ("station_N" prefix)
            if prepared_df is not None and _ist_id != "Все":
                _ipids = set(prepared_df["station_id"].unique())
                if _ist_id not in _ipids:
                    _ialt = f"station_{_ist_id}"
                    if _ialt in _ipids:
                        _ist_id = _ialt
            _ick = f"{ips}|{ipe}|{_ist_id}"

            if "_interp_results" not in st.session_state:
                st.session_state["_interp_results"] = {}
            if "_interp_active" not in st.session_state:
                st.session_state["_interp_active"] = None

            if run_interp:
                _iprog = st.progress(0, text="Загрузка TFT-модели...")
                _im, _itr, _ic_cfg, _ierr = load_tft()
                if _ierr or _im is None:
                    _iprog.empty()
                    st.error(f"Не удалось загрузить модель: {_ierr}")
                else:
                    try:
                        _iprog.progress(25, text="Подготовка данных...")
                        _iprog.progress(40, text="Прогон TFT (режим raw) — до 90 с...")
                        _ikey2 = str(sum(p.numel() for p in _im.parameters()))
                        _inp_r = compute_interpretation(
                            _im, _itr, _ikey2,
                            ips, ipe, str(_ictx), _ist_id,
                        )
                        _iprog.progress(100, text="Готово!")
                        _iprog.empty()
                        st.session_state["_interp_results"][_ick] = _inp_r
                        st.session_state["_interp_active"] = _ick
                    except Exception as _ie:
                        _iprog.empty()
                        import traceback as _tb
                        st.error(f"Ошибка вычисления: {_ie}")
                        with st.expander("Подробности"):
                            st.code(_tb.format_exc())

            _akey = st.session_state.get("_interp_active")
            _inp  = st.session_state.get("_interp_results", {}).get(_akey)

            if _inp is None:
                st.info(
                    "Нажмите **🧠 Рассчитать** для вычисления весов VSN.  \n"
                    "Первый расчёт занимает **30–90 секунд** "
                    "(прогон модели на скользящих окнах)."
                )
            elif isinstance(_inp, dict) and "_error" in _inp:
                st.error("Ошибка вычисления VSN")
                with st.expander("Подробности ошибки"):
                    st.code(_inp["_error"])
            else:
                _im2, _, _, _ = load_tft()
                if _im2 is None:
                    st.error("Модель не доступна.")
                else:
                    _is_lbl = ("Все станции" if istation == "Все"
                               else _station_display.get(istation, istation))
                    st.success(f"Интерпретация: **{iperiod}** | **{_is_lbl}**")

                    def _imp_chart(key, names_attr, cap, chart_key):
                        if key not in _inp:
                            st.caption("Данные недоступны для данного чекпоинта.")
                            return
                        vals  = _inp[key]
                        names = getattr(_im2, names_attr, [])
                        if vals.ndim > 1:
                            vals = vals.mean(axis=0)
                        n = min(len(names), len(vals))
                        if n == 0:
                            st.caption("Нет данных.")
                            return
                        imp = (
                            pd.DataFrame({"Переменная": list(names[:n]),
                                          "Важность":   vals[:n].tolist()})
                            .sort_values("Важность", ascending=False)
                        )
                        total = imp["Важность"].sum()
                        imp["Важность, %"] = (imp["Важность"] / total * 100).round(2)
                        n_top   = min(20, n)
                        imp_top = imp.head(n_top)
                        st.caption(cap)
                        _cc, _ct = st.columns([2, 1])
                        with _cc:
                            _fig = px.bar(
                                imp_top, x="Важность, %", y="Переменная", orientation="h",
                                color="Важность, %", color_continuous_scale="Blues",
                                height=max(350, n_top * 22),
                            )
                            _fig.update_layout(
                                coloraxis_showscale=False,
                                yaxis=dict(autorange="reversed"),
                                **PLOTLY_THEME, margin=dict(t=10, b=10, l=10),
                            )
                            st.plotly_chart(_fig, key=chart_key, width="stretch")
                        with _ct:
                            st.caption(
                                f"Топ-{n_top} из {n}. "
                                f"Показано {imp_top['Важность, %'].sum():.1f}%."
                            )
                            st.dataframe(
                                imp_top[["Переменная", "Важность, %"]].reset_index(drop=True),
                                hide_index=True, height=min(600, n_top * 38),
                            )

                    _ivs, _ive, _ivd = st.tabs([
                        "📌 Статические признаки",
                        "🔁 Прошлые наблюдения (Encoder)",
                        "🔮 Известные будущие (Decoder)",
                    ])
                    with _ivs:
                        _imp_chart(
                            "static_variables", "static_variables",
                            "Паспорт АЗС: тип дороги, площадь, число колонок. "
                            "Высокий вес → характеристика АЗС влияет на базовый уровень прогнозов.",
                            "vsn_static",
                        )
                    with _ive:
                        _imp_chart(
                            "encoder_variables", "encoder_variables",
                            "Наблюдаемые прошлые данные (энкодер): 12 целевых переменных как авторегрессивные входы. "
                            "Все ковариаты (погода, трафик, цены) — в известных будущих (KNOWN_REALS). "
                            "Высокий вес → модель опирается на «что продавали вчера/неделю назад».",
                            "vsn_encoder",
                        )
                    with _ivd:
                        _imp_chart(
                            "decoder_variables", "decoder_variables",
                            "Известные будущие данные: час суток, праздники, акции, цены топлива. "
                            "Высокий вес у hour_sin/cos → суточная сезонность критична для прогноза.",
                            "vsn_decoder",
                        )

                    st.divider()
                    st.markdown("#### Временно́е внимание (Temporal Self-Attention)")
                    st.caption(
                        "Насколько сильно модель обращается к каждому из 168 прошлых часов "
                        "при составлении прогноза (усреднено по всем 24 шагам горизонта)."
                    )

                    if "attention" in _inp:
                        attn = _inp["attention"]
                        if attn.ndim == 4:
                            attn = attn.mean(axis=(0, 1))
                        elif attn.ndim == 3:
                            attn = attn.mean(axis=0)

                        if attn.ndim == 2:
                            fig_a = px.imshow(
                                attn, color_continuous_scale="Viridis",
                                labels=dict(x="Часов назад", y="Шаг прогноза", color="Внимание"),
                                aspect="auto", height=400,
                            )
                            fig_a.update_layout(**PLOTLY_THEME, margin=dict(t=20, b=20))
                            st.plotly_chart(fig_a, key="ap_attn_hm", width="stretch")
                        else:
                            enc_steps = list(range(-len(attn), 0))
                            attn_pct  = attn / attn.sum() * 100
                            fig_a = go.Figure()
                            fig_a.add_trace(go.Bar(
                                x=enc_steps, y=attn_pct,
                                marker_color=attn_pct, marker_colorscale="Viridis",
                                showlegend=False,
                            ))
                            top3 = np.argsort(attn_pct)[-3:][::-1]
                            for _aidx in top3:
                                fig_a.add_annotation(
                                    x=enc_steps[_aidx], y=attn_pct[_aidx],
                                    text=f"{enc_steps[_aidx]} ч<br>{attn_pct[_aidx]:.1f}%",
                                    showarrow=True, arrowhead=2, arrowsize=1,
                                    font=dict(size=11, color="white"),
                                    bgcolor="rgba(80,80,80,0.7)",
                                    bordercolor="gray", borderwidth=1, ay=-40,
                                )
                            fig_a.update_layout(
                                xaxis_title="Часов назад (от момента прогноза)",
                                yaxis_title="Вес внимания, %", height=400,
                                xaxis=dict(
                                    tickmode="array",
                                    tickvals=list(range(-168, 0, 24)),
                                    ticktext=[f"−{abs(v)}ч (−{abs(v)//24}д)"
                                              for v in range(-168, 0, 24)],
                                ),
                                **PLOTLY_THEME, margin=dict(t=20, b=40),
                            )
                            st.plotly_chart(fig_a, key="ap_attn_bar", width="stretch")

                            top_h = enc_steps[int(np.argmax(attn_pct))]
                            st.markdown(
                                f"**Интерпретация:** наибольшее внимание — час **{top_h} ч** "
                                f"({abs(top_h)//24} сут. + {abs(top_h)%24} ч назад)."
                            )
                            if abs(top_h - (-24)) <= 3:
                                st.info("Доминирует **суточная** периодичность: "
                                        "модель в первую очередь смотрит на вчерашний час.")
                            elif abs(top_h - (-168)) <= 6:
                                st.info("Доминирует **недельная** периодичность: "
                                        "модель смотрит на тот же час неделю назад.")
                            else:
                                st.info(f"Основной паттерн не совпадает с суточным/недельным "
                                        f"— пик на {top_h} ч.")
                    else:
                        st.caption("Данные внимания недоступны для данного чекпоинта.")

                    with st.expander("📚 Архитектура TFT — краткая справка"):
                        st.markdown("""
**Temporal Fusion Transformer** (Lim et al., 2020) — архитектура с явной интерпретируемостью.

```
Входы:
  Static  ──► Static VSN  ──► Context vectors (c_s, c_e, c_h, c_c)
  Past    ──► Encoder VSN ──► LSTM (encoder) ──┐
  Future  ──► Decoder VSN ──► LSTM (decoder) ──┤
                                               ▼
                                  Temporal Self-Attention
                                               ▼
                                  QuantileLoss (P10 / P50 / P90)
```

- **VSN** — softmax-веса над переменными (сумма = 100%)
- **LSTM** — краткосрочная зависимость (локальный контекст)
- **Self-Attention** — долгосрочные паттерны (до 7 суток назад)
- **Static context** — инициализирует LSTM и VSN характеристиками АЗС

*Горизонт: 24 ч | Ретроспектива: 168 ч | Целей: 12 | Входов: 89*
                        """)


# ══════════════════════════════════════════════════════════════
# TAB 4 — РЕКОМЕНДАЦИИ
# ══════════════════════════════════════════════════════════════
with tab4:
    _t4c1, _t4c2, _t4c3 = st.columns([2, 2, 4])
    with _t4c1:
        sel_station_rec = st.selectbox(
            "Станция", ["Все станции"] + _all_stations,
            format_func=lambda x: _station_display.get(x, x) if x != "Все станции" else x,
            label_visibility="collapsed", key="rec_station",
        )
    with _t4c2:
        sel_target = st.selectbox(
            "Показатель", TARGET_COLS,
            format_func=lambda x: TARGET_LABELS.get(x, x),
            label_visibility="collapsed", key="rec_target",
        )
    sel_sid_rec      = (_sid_by_name.get(sel_station_rec) if sel_station_rec != "Все станции" else None)
    target_label_rec = TARGET_LABELS.get(sel_target, sel_target)
    unit             = "л/ч" if sel_target in FUEL_COLS else "руб/ч"

    st.caption(f"Паттерны — полный 2023 год · Станция: {sel_station_rec} · Показатель: {target_label_rec}")

    rd = df_ov.copy()

    # Динамический блок — результаты последнего сценария
    sr = st.session_state.get("sc_result")
    _sr_fresh = (sr is not None
                 and sr.get("sc_tgt") == sel_target
                 and str(sr.get("sc_st","")) == str(sel_sid_rec or ""))
    if _sr_fresh:
        lbl_sr = TARGET_LABELS.get(sr["sc_tgt"],sr["sc_tgt"])
        u_sr   = sr["unit"]
        st.markdown("#### Результат последнего сценария")
        if sr["base_mean"] is not None:
            dp_sr = (sr["val_mean"]-sr["base_mean"])/sr["base_mean"]*100 if sr["base_mean"]>0 else 0
            rec_card(f"Сценарий vs базовый · {lbl_sr}",
                     f"{'↑' if dp_sr>=0 else '↓'} <b>{abs(dp_sr):.1f}%</b>: "
                     f"<b>{sr['val_mean']:.1f}</b> vs {sr['base_mean']:.1f} {u_sr} (ср. за 24ч).",
                     color=GREEN if dp_sr>=0 else RED)
        else:
            rec_card(f"Прогноз · {lbl_sr}",
                     f"Ср. за 24ч: <b>{sr['val_mean']:.1f} {u_sr}</b> · "
                     f"Пик: <b>{sr['val_peak']:.1f} {u_sr}</b> в {sr['peak_ts'].strftime('%H:%M')}.",
                     color=TEAL)
        promo_parts = (["акция на топливо"] if sr["sc_pf"] else []) + \
                      (["акция в магазине"] if sr["sc_ps"] else []) + \
                      ([f"реклама «{sr['sc_ch']}»"] if sr["sc_ad"] else [])
        if promo_parts:
            rec_card("Активные стимулы","Включены: <b>"+", ".join(promo_parts)+"</b>.",color=GOLD)
        if sr.get("price_delta") is not None:
            pd2 = sr["price_delta"]
            rec_card("Ценовое воздействие",
                     f"Цена {'повышена' if pd2>0 else 'снижена'} на <b>{abs(pd2):.1f}%</b> "
                     "относительно медианы 2023 г.",
                     color=RED if pd2>0 else GREEN)
        st.divider()

    # Статические рекомендации по EDA-паттернам
    st.markdown("#### Рекомендации по данным 2023 года")

    if metrics_df is not None and sel_target in metrics_df["target"].values:
        tgt_m = metrics_df[metrics_df["target"]==sel_target]
        if sel_sid_rec: tgt_m = tgt_m[tgt_m["station_id"].astype(str)==sel_sid_rec]
        if not tgt_m.empty:
            _mape_c = next((c for c in tgt_m.columns if "MAPE" in c.upper()),None)
            if _mape_c:
                mape = tgt_m[_mape_c].mean()
                if not np.isnan(mape):
                    rel_hint = ("Прогнозы надёжны — используйте для оперативного планирования." if mape<=10 else
                                "Прогноз ориентировочный — закладывайте погрешность." if mape<=20 else
                                "Прогноз нестабилен — применяйте как индикатив.")
                    rec_card(f"Надёжность прогноза · {target_label}",
                             f"MAPE модели <b>{mape:.1f}%</b>. {rel_hint}",
                             color=GREEN if mape<=10 else (GOLD if mape<=20 else RED))

    pcol2 = "promotion_fuel_active" if sel_target in FUEL_COLS else "promotion_shop_active"
    if pcol2 in rd.columns and sel_target in rd.columns:
        on  = rd[rd[pcol2]==1][sel_target].mean()
        off = rd[rd[pcol2]==0][sel_target].mean()
        if off>0 and not np.isnan(on):
            d = (on-off)/off*100
            rec_card("Акционный рычаг",
                     (f"Запустите акцию — по данным 2023 г. это даёт <b>+{d:.1f}%</b> "
                      f"к продажам {target_label} ({on:.1f} vs {off:.1f} {unit})." if d>0 else
                      f"Акция исторически снижает продажи на <b>{abs(d):.1f}%</b> — рассмотрите альтернативы."),
                     color=GREEN if d>0 else RED)

    if "ad_channel" in rd.columns and sel_target in rd.columns:
        aa = rd.groupby("ad_channel")[sel_target].mean()
        if not aa.empty:
            bch = aa.idxmax(); wch = aa.idxmin()
            gain = (aa[bch]-aa[wch])/aa[wch]*100 if aa[wch]>0 else 0
            rec_card("Оптимальный рекламный канал",
                     f"Канал <b>«{bch}»</b> — {aa[bch]:.1f} {unit}, "
                     f"на <b>{gain:.0f}%</b> выше «{wch}».",
                     color=BLUE)

    if sel_target in rd.columns:
        hg3 = rd.groupby(rd["timestamp"].dt.hour)[sel_target].mean()
        if not hg3.empty:
            ph3 = int(hg3.idxmax()); lh3 = int(hg3.idxmin())
            rec_card("Окно высокого спроса",
                     f"Фокус на <b>{ph3}:00–{(ph3+1)%24}:00</b> ({hg3[ph3]:.1f} {unit}). "
                     f"Минимум — {lh3}:00 ({hg3[lh3]:.1f} {unit}).",
                     color=GOLD)

    if sel_target in rd.columns:
        dg3 = rd.groupby(rd["timestamp"].dt.dayofweek)[sel_target].mean()
        if not dg3.empty:
            bd = int(dg3.idxmax()); wd = int(dg3.idxmin())
            lift = (dg3[bd]-dg3[wd])/dg3[wd]*100 if dg3[wd]>0 else 0
            rec_card("Лучший день для акций",
                     f"Планируйте промо на <b>{DAY_NAMES[bd]}</b> — спрос на "
                     f"<b>{lift:.0f}%</b> выше, чем в {DAY_NAMES[wd]} "
                     f"({dg3[bd]:.1f} vs {dg3[wd]:.1f} {unit}).",
                     color=GREEN)

    if "total_traffic" in rd.columns and sel_target in rd.columns:
        r = rd[["total_traffic",sel_target]].corr().iloc[0,1]
        if abs(r) > 0.25:
            rec_card("Трафик как сигнал",
                     (f"Сильная корреляция с трафиком (r={r:.2f}) — рост потока предсказывает рост продаж." if abs(r)>0.5 and r>0 else
                      f"Умеренная связь (r={r:.2f}) — используйте трафик как дополнительный сигнал."),
                     color=TEAL)
