"""
Аналитика продаж АЗС Татнефть — 2023 · Прогнозы TFT на декабрь
Запускать из корня проекта: streamlit run dashboard/forecast_dashboard.py
"""

import glob
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.torch_compat  # noqa: F401
from utils.data_utils import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    TARGET_COLS,
    TEST_END,
    TEST_START,
)

# ── Константы ──────────────────────────────────────────────────
FUEL_COLS = [c for c in TARGET_COLS if c.startswith("sales_")]
SHOP_COLS = [c for c in TARGET_COLS if c.startswith("shop_")]

PRICE_MAP = {
    "sales_AI92":      "price_AI92",
    "sales_AI95":      "price_AI95",
    "sales_AI98":      "price_AI98",
    "sales_DT_EURO":   "price_DT_EURO",
    "sales_DT_TANEKO": "price_DT_TANEKO",
    "sales_DT_SUMMER": "price_DT_SUMMER",
    "sales_DT_WINTER": "price_DT_WINTER",
}

TARGET_LABELS = {
    "sales_AI92": "АИ-92", "sales_AI95": "АИ-95", "sales_AI98": "АИ-98",
    "sales_DT_EURO": "ДТ Евро+", "sales_DT_TANEKO": "ДТ ТАНЕКО",
    "sales_DT_SUMMER": "ДТ Летнее", "sales_DT_WINTER": "ДТ Зимнее",
    "shop_напитки": "Напитки", "shop_закуски": "Закуски",
    "shop_автотовары": "Автотовары", "shop_кофе": "Кофе", "shop_табак": "Табак",
}
UNITS = {**{c: "л/ч" for c in FUEL_COLS}, **{c: "руб/ч" for c in SHOP_COLS}}

STATION_LABELS = {
    "0": "АЗС-001 · трасса М7",
    "1": "АЗС-002 · регион. дорога",
    "2": "АЗС-003 · трасса М7",
    "3": "АЗС-004 · трасса М7",
    "4": "АЗС-005 · регион. дорога",
}

MONTH_NAMES = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
    7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
}
DAY_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

GREEN   = "#00853E"   # только KPI top border и positive рекомендации
GOLD    = "#F59E0B"   # основной цвет данных (area, scatter)
TEAL    = "#2DD4BF"   # "Без акции" / вторичные ряды
RED     = "#F87171"   # пики, негативные, ошибки
BLUE    = "#60A5FA"   # цвет станции в STATION_PAL (не для данных напрямую)
GRAY    = "#8B949E"   # мuted, вторичный текст
TREND_C = "#CBD5E1"   # линия тренда (OLS) на scatter — нейтральный светлый
SEC_LINE= "#334155"   # delimiter под заголовками секций

CARD_BG = "#1C2532"
GRID_C  = "#30363D"
TEXT    = "#E6EDF3"
TEXT_S  = "#8B949E"

STATION_PAL = ["#00853E", "#60A5FA", "#F59E0B", "#F87171", "#A78BFA"]

# ── Конфиг страницы ────────────────────────────────────────────
st.set_page_config(
    page_title="АЗС Татнефть — Аналитика и прогнозирование",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Минимальный CSS — только шрифт и отступы
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
      rel="stylesheet">
<style>
html, body { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 0.75rem !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Компоненты с инлайн-стилями ────────────────────────────────
def kpi(label, value, sub="", accent=GREEN):
    st.markdown(f"""
<div style="background:{CARD_BG};border-radius:12px;padding:18px 20px;
            border-top:3px solid {accent};height:100%;">
  <div style="font-size:0.7rem;color:{TEXT_S};font-weight:600;text-transform:uppercase;
              letter-spacing:0.5px;margin-bottom:5px;">{label}</div>
  <div style="font-size:1.6rem;font-weight:700;color:{TEXT};line-height:1.15;">{value}</div>
  <div style="font-size:0.78rem;color:{accent};font-weight:500;margin-top:4px;">{sub}</div>
</div>""", unsafe_allow_html=True)


def sec(title):
    st.markdown(f"""
<div style="font-size:1rem;font-weight:700;color:{TEXT};padding-bottom:6px;
            border-bottom:1px solid {SEC_LINE};margin:18px 0 12px 0;">{title}</div>
""", unsafe_allow_html=True)


def banner(html):
    st.markdown(f"""
<div style="background:{CARD_BG};border-left:3px solid {SEC_LINE};padding:11px 16px;
            border-radius:0 8px 8px 0;font-size:0.87rem;color:{TEXT_S};
            line-height:1.65;margin-bottom:14px;">{html}</div>
""", unsafe_allow_html=True)


def rec_card(tag, body, color=GREEN):
    st.markdown(f"""
<div style="background:{CARD_BG};border-radius:10px;padding:13px 17px;margin-bottom:9px;
            border-left:4px solid {color};">
  <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.5px;color:{color};margin-bottom:3px;">{tag}</div>
  <div style="font-size:0.9rem;color:{TEXT};">{body}</div>
</div>""", unsafe_allow_html=True)


def chart_layout(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor=CARD_BG, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=GRID_C, color=TEXT_S),
        yaxis=dict(gridcolor=GRID_C, color=TEXT_S),
    )
    return fig


# ── Загрузка данных ────────────────────────────────────────────
@st.cache_data
def load_merged():
    df = pd.read_csv("data/merged_data.csv", parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    return df


@st.cache_data
def load_predictions():
    p = "data/predictions.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    return df


@st.cache_data
def load_metrics():
    p = "data/metrics.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_prepared():
    df = pd.read_csv("data/prepared_data.csv", parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    dec = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    for col in TARGET_COLS:
        if col in df.columns:
            df.loc[dec, col] = np.expm1(df.loc[dec, col].clip(lower=0))
    return df


@st.cache_data
def load_ad_map():
    df = pd.read_csv("data/prepared_data.csv", usecols=["ad_channel", "ad_channel_enc"])
    return (
        df.drop_duplicates().sort_values("ad_channel_enc")
        .set_index("ad_channel_enc")["ad_channel"].to_dict()
    )


@st.cache_data
def load_scalers():
    p = "tft/scalers.pkl"
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_tft():
    try:
        from pytorch_forecasting import TemporalFusionTransformer
        ckpt = "tft/model.ckpt"
        if not os.path.exists(ckpt):
            ckpts = sorted(glob.glob("tft/checkpoints/*.ckpt"))
            if not ckpts:
                return None, None, None, "Чекпоинт не найден"
            ckpt = ckpts[-1]
        with open("tft/training_dataset.pkl", "rb") as f:
            training = pickle.load(f)
        with open("tft/dataset_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
        model.eval()
        return model, training, cfg, None
    except Exception as e:
        return None, None, None, str(e)


# ── Данные ─────────────────────────────────────────────────────
merged_df   = load_merged()
pred_df     = load_predictions()
metrics_df  = load_metrics()
prepared_df = load_prepared()
ad_map      = load_ad_map()
scalers     = load_scalers()

stations    = sorted(merged_df["station_id"].unique())
dec_df      = prepared_df[
    (prepared_df["timestamp"] >= TEST_START) & (prepared_df["timestamp"] <= TEST_END)
].copy()
avail_fuel  = [c for c in FUEL_COLS if c in merged_df.columns]
avail_shop  = [c for c in SHOP_COLS if c in merged_df.columns]

# ── Шапка ──────────────────────────────────────────────────────
st.title("⛽ Аналитика продаж АЗС — ПАО «Татнефть»")
st.caption(f"Данные 2023 · Модель TFT · Горизонт {PREDICTION_LENGTH} ч · Ретроспектива {ENCODER_LENGTH} ч")

# ── Фильтры (вместо сайдбара) ──────────────────────────────────
f1, f2, _ = st.columns([2, 2, 8])
with f1:
    sel_station = st.selectbox(
        "Станция",
        ["Все"] + stations,
        format_func=lambda x: "Все станции" if x == "Все" else STATION_LABELS.get(str(x), f"АЗС-{x}"),
        label_visibility="collapsed",
    )
with f2:
    sel_target = st.selectbox(
        "Показатель",
        TARGET_COLS,
        format_func=lambda x: TARGET_LABELS.get(x, x),
        label_visibility="collapsed",
    )

station_label = (
    "Все станции" if sel_station == "Все"
    else STATION_LABELS.get(str(sel_station), f"АЗС-{sel_station}")
)
target_label = TARGET_LABELS.get(sel_target, sel_target)
unit         = UNITS.get(sel_target, "")

st.caption(f"Фильтр: **{station_label}** · **{target_label}**")

# ── Табы ───────────────────────────────────────────────────────
tab1, tab3, tab4, tab2 = st.tabs([
    "📊 Обзор 2023",
    "🔍 Факторный анализ",
    "🎯 Сценарий & Рекомендации",
    "📈 Прогноз TFT",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 · ОБЗОР 2023
# ═══════════════════════════════════════════════════════════════
with tab1:
    df_ov = merged_df if sel_station == "Все" else merged_df[merged_df["station_id"] == sel_station]

    sec("Ключевые показатели 2023")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total = df_ov[avail_fuel].sum().sum() if avail_fuel else 0
        kpi("Объём реализации топлива", f"{int(total):,}".replace(",", " ") + " л", "сумма за год, все виды")
    with c2:
        s = df_ov[sel_target].sum() if sel_target in df_ov.columns else 0
        u2 = "л" if sel_target in FUEL_COLS else "руб"
        kpi(f"Продажи: {target_label}", f"{s / 1e3:.1f} тыс {u2}", "сумма за год")
    with c3:
        if sel_target in df_ov.columns:
            pm = df_ov.groupby(df_ov["timestamp"].dt.month)[sel_target].sum().idxmax()
            kpi("Пиковый месяц", MONTH_NAMES.get(pm, str(pm)), f"по {target_label}")
        else:
            kpi("Пиковый месяц", "—", "")
    with c4:
        if avail_fuel and sel_target in merged_df.columns:
            bs = merged_df.groupby("station_id")[sel_target].sum().idxmax()
            kpi("Лидер продаж",
                STATION_LABELS.get(str(bs), f"АЗС-{bs}").split("·")[0].strip(),
                f"по {target_label}")
        else:
            kpi("Лидер продаж", "—", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # Сравнение станций + структура
    cl, cr = st.columns([3, 2])
    with cl:
        sec("Объём продаж топлива по станциям")
        if avail_fuel:
            ss = merged_df.groupby("station_id")[avail_fuel].sum().reset_index()
            ss["Станция"] = ss["station_id"].map(lambda x: STATION_LABELS.get(str(x), f"АЗС-{x}"))
            sm = ss.melt(id_vars=["station_id", "Станция"], value_vars=avail_fuel,
                         var_name="Вид топлива", value_name="Объём (л)")
            sm["Вид топлива"] = sm["Вид топлива"].map(TARGET_LABELS)
            fig = px.bar(sm, x="Объём (л)", y="Станция", color="Вид топлива",
                         orientation="h", color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(legend=dict(orientation="h", y=-0.28))
            st.plotly_chart(chart_layout(fig, 320), width="stretch")

    with cr:
        sec("Структура топливных продаж")
        if avail_fuel:
            fs = merged_df[avail_fuel].sum()
            fs.index = [TARGET_LABELS.get(c, c) for c in fs.index]
            fig_p = px.pie(values=fs.values, names=fs.index, hole=0.42,
                           color_discrete_sequence=px.colors.qualitative.Safe)
            fig_p.update_traces(textposition="inside", textinfo="percent+label",
                                textfont_size=11)
            fig_p.update_layout(showlegend=False, margin=dict(t=8, b=8),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p, width="content")

    # Динамика по месяцам
    sec(f"Помесячная динамика — {target_label}")
    if sel_target in df_ov.columns:
        if sel_station == "Все":
            mon = (merged_df.groupby(["station_id", merged_df["timestamp"].dt.month])[sel_target]
                   .mean().reset_index())
            mon.columns = ["station_id", "Месяц", sel_target]
            mon["Станция"] = mon["station_id"].map(lambda x: STATION_LABELS.get(str(x), f"АЗС-{x}"))
            fig_tr = px.line(mon, x="Месяц", y=sel_target, color="Станция",
                             markers=True, color_discrete_sequence=STATION_PAL,
                             labels={"Месяц": "", sel_target: f"Ср. продажи ({unit})"})
            fig_tr.update_layout(legend=dict(orientation="h", y=-0.22))
        else:
            ms = df_ov.groupby(df_ov["timestamp"].dt.month)[sel_target].mean().reset_index()
            ms.columns = ["Месяц", sel_target]
            fig_tr = px.area(ms, x="Месяц", y=sel_target,
                             color_discrete_sequence=[GOLD],
                             labels={"Месяц": "", sel_target: f"Ср. продажи ({unit})"})
            fig_tr.update_traces(fillcolor="rgba(245,158,11,0.10)")
        fig_tr.update_xaxes(tickvals=list(range(1, 13)), ticktext=list(MONTH_NAMES.values()))
        st.plotly_chart(chart_layout(fig_tr, 300), width="stretch")

    # Суточный + недельный паттерн
    cl2, cr2 = st.columns(2)
    with cl2:
        sec(f"Суточный паттерн — {target_label}")
        if sel_target in df_ov.columns:
            hr = pd.to_datetime(df_ov["timestamp"]).dt.hour
            hg = df_ov.groupby(hr)[sel_target].mean().reset_index()
            hg.columns = ["Час", sel_target]
            ph = int(hg.set_index("Час")[sel_target].idxmax())
            fh = px.area(hg, x="Час", y=sel_target, markers=True,
                         color_discrete_sequence=[GOLD],
                         labels={"Час": "Час суток (0–23)", sel_target: f"Ср. продажи ({unit})"})
            fh.update_traces(fillcolor="rgba(245,158,11,0.10)")
            fh.add_vline(x=ph, line_dash="dash", line_color=RED,
                         annotation_text=f"Пик {ph}:00", annotation_position="top right")
            st.plotly_chart(chart_layout(fh, 280), width="stretch")

    with cr2:
        sec(f"Недельный паттерн — {target_label}")
        if sel_target in df_ov.columns:
            dr = pd.to_datetime(df_ov["timestamp"]).dt.dayofweek
            dg = df_ov.groupby(dr)[sel_target].mean().reset_index()
            dg.columns = ["День", sel_target]
            dg["Д"] = dg["День"].map(lambda d: DAY_NAMES[d])
            fd = px.bar(dg, x="Д", y=sel_target,
                        color=sel_target,
                        color_continuous_scale=[[0, "#78350F"], [1, GOLD]],
                        labels={"Д": "", sel_target: f"Ср. продажи ({unit})"})
            fd.update_layout(coloraxis_showscale=False)
            st.plotly_chart(chart_layout(fd, 280), width="stretch")

    # Магазин
    if avail_shop:
        sec("Выручка магазина по категориям и станциям")
        sh = merged_df.groupby("station_id")[avail_shop].sum().reset_index()
        sh["Станция"] = sh["station_id"].map(lambda x: STATION_LABELS.get(str(x), f"АЗС-{x}"))
        shm = sh.melt(id_vars=["station_id", "Станция"], value_vars=avail_shop,
                      var_name="Категория", value_name="Выручка (руб)")
        shm["Категория"] = shm["Категория"].map(TARGET_LABELS)
        fsh = px.bar(shm, x="Станция", y="Выручка (руб)", color="Категория",
                     barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel,
                     labels={"Выручка (руб)": "Выручка, руб (∑ за год)"})
        fsh.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(chart_layout(fsh, 320), width="stretch")


# ═══════════════════════════════════════════════════════════════
# TAB 2 · ПРОГНОЗ TFT
# ═══════════════════════════════════════════════════════════════
with tab2:
    if pred_df is None:
        st.warning("Файл `data/predictions.csv` не найден. Выполните: `python tft/predict.py`")
        st.stop()

    banner(
        "Модель <b>Temporal Fusion Transformer (TFT)</b> обучена на данных января–октября 2023 "
        "и проверена на декабре 2023 (тестовый период). "
        "Горизонт прогноза — <b>24 часа</b>, ретроспектива для контекста — <b>168 часов (7 суток)</b>. "
        "Помимо точечного прогноза модель даёт <b>доверительный интервал q10–q90</b>: "
        "с вероятностью ~80% реальное значение попадёт в эту полосу."
    )

    if metrics_df is not None:
        sub_m = metrics_df if sel_station == "Все" else metrics_df[metrics_df["station_id"].astype(str) == sel_station]
        sub_m = sub_m[sub_m["target"] == sel_target]
        if not sub_m.empty:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi("MAE", f"{sub_m['MAE'].mean():.2f} {unit}", "средняя абс. ошибка")
            with c2:
                kpi("RMSE", f"{sub_m['RMSE'].mean():.2f}", "корень среднеквадр. ошибки")
            with c3:
                mape = sub_m["MAPE_%"].mean()
                acc = GREEN if mape <= 10 else GOLD if mape <= 20 else RED
                kpi("MAPE", f"{mape:.1f}%" if not np.isnan(mape) else "N/A",
                    "≤10% отлично · 10–20% хорошо", accent=acc)
            with c4:
                kpi("Горизонт прогноза", f"{PREDICTION_LENGTH} ч", "шагов на один запуск")

    st.markdown("<br>", unsafe_allow_html=True)
    sec(f"Прогноз vs факт — {target_label} · Декабрь 2023")

    df_plot = pred_df.copy()
    if sel_station != "Все":
        df_plot = df_plot[df_plot["station_id"] == sel_station]

    if not df_plot.empty:
        if sel_station == "Все":
            nc = [c for c in df_plot.columns if c not in ("station_id", "timestamp", "horizon_h")]
            df_plot = df_plot.groupby("timestamp")[nc].mean().reset_index()

        pc, ac = f"{sel_target}_pred", f"{sel_target}_actual"
        q1, q9 = f"{sel_target}_q10", f"{sel_target}_q90"
        fig = go.Figure()
        if ac in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=df_plot[ac],
                                     mode="lines", name="Факт",
                                     line=dict(color="#60A5FA", width=2)))
        if q1 in df_plot.columns and q9 in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df_plot["timestamp"], df_plot["timestamp"][::-1]]),
                y=pd.concat([df_plot[q9], df_plot[q1][::-1]]),
                fill="toself", fillcolor="rgba(0,133,62,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="Интервал q10–q90"))
        if pc in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=df_plot[pc],
                                     mode="lines", name="Прогноз TFT",
                                     line=dict(color=GREEN, width=2, dash="dot")))
        chart_layout(fig, 420)
        fig.update_layout(
            xaxis_title="Дата", yaxis_title=unit,
            paper_bgcolor=CARD_BG,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
            margin=dict(t=60, b=10, l=0, r=0),
        )
        st.plotly_chart(fig, width="stretch")

    cl, cr = st.columns(2)
    with cl:
        sec("Точность по всем переменным")
        if metrics_df is not None:
            disp = metrics_df.copy()
            disp["Переменная"] = disp["target"].map(TARGET_LABELS).fillna(disp["target"])
            if sel_station != "Все":
                disp = disp[disp["station_id"].astype(str) == sel_station]
            ag = disp.groupby("Переменная")[["MAE", "RMSE", "MAPE_%"]].mean().round(2).reset_index()
            ag.columns = ["Переменная", "MAE", "RMSE", "MAPE, %"]
            st.dataframe(ag, width="stretch", height=320, hide_index=True)

    with cr:
        sec("Тепловая карта MAPE: станция × переменная")
        if metrics_df is not None:
            pv = metrics_df.pivot_table(
                index="station_id", columns="target", values="MAPE_%", aggfunc="mean")
            pv.index = [STATION_LABELS.get(str(i), f"АЗС-{i}") for i in pv.index]
            pv.columns = [TARGET_LABELS.get(c, c) for c in pv.columns]
            fhm = px.imshow(pv.round(1), color_continuous_scale="RdYlGn_r",
                            text_auto=True, aspect="auto",
                            labels=dict(color="MAPE %"))
            st.plotly_chart(chart_layout(fhm, 320), width="stretch")

    if pred_df is not None:
        sec("Scatter: прогноз vs факт")
        st.caption("Точки на красной диагонали — идеальный прогноз.")
        pc2, ac2 = f"{sel_target}_pred", f"{sel_target}_actual"
        sc = pred_df[[pc2, ac2]].dropna() if all(c in pred_df.columns for c in [pc2, ac2]) else None
        if sc is not None and not sc.empty:
            fsc = px.scatter(sc, x=ac2, y=pc2, opacity=0.35,
                             color_discrete_sequence=[GREEN],
                             labels={ac2: f"Факт ({unit})", pc2: f"Прогноз ({unit})"})
            mx = max(sc[ac2].max(), sc[pc2].max())
            fsc.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                          line=dict(color=RED, dash="dash"))
            st.plotly_chart(chart_layout(fsc, 340), width="stretch")


# ═══════════════════════════════════════════════════════════════
# TAB 3 · ФАКТОРНЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════════════════════
with tab3:
    fa_df = merged_df.copy()
    if sel_station != "Все":
        fa_df = fa_df[fa_df["station_id"] == sel_station]

    fa1, fa2, fa3, fa4 = st.tabs(["Акции & Реклама", "Трафик", "Погода", "Цены конкурентов"])

    with fa1:
        cl, cr = st.columns(2)
        with cl:
            sec("Эффект акции на продажи")
            pcol = "promotion_fuel_active" if sel_target in FUEL_COLS else "promotion_shop_active"
            if pcol in fa_df.columns:
                bp = fa_df[[pcol, sel_target]].copy()
                bp[pcol] = bp[pcol].map({1: "Акция активна", 0: "Без акции"})
                fb = px.box(bp, x=pcol, y=sel_target, color=pcol,
                            color_discrete_map={"Акция активна": GREEN, "Без акции": TEAL},
                            labels={sel_target: f"{target_label} ({unit})", pcol: ""})
                fb.update_layout(showlegend=False)
                st.plotly_chart(chart_layout(fb, 320), width="stretch")
                on  = fa_df[fa_df[pcol] == 1][sel_target].mean()
                off = fa_df[fa_df[pcol] == 0][sel_target].mean()
                if off > 0 and not np.isnan(on):
                    d = (on - off) / off * 100
                    st.metric("Средние продажи при акции", f"{on:.1f} {unit}",
                              delta=f"{d:+.1f}% vs без акции")
        with cr:
            sec("Продажи по каналу рекламы")
            if "ad_channel" in fa_df.columns:
                aa = (fa_df.groupby("ad_channel")[sel_target].mean()
                      .reset_index().sort_values(sel_target))
                fa = px.bar(aa, x=sel_target, y="ad_channel", orientation="h",
                            color=sel_target,
                            color_continuous_scale=[[0, "#312E81"], [1, "#A78BFA"]],
                            labels={sel_target: f"{target_label} ({unit})", "ad_channel": ""})
                fa.update_layout(coloraxis_showscale=False)
                st.plotly_chart(chart_layout(fa, 320), width="stretch")

    with fa2:
        tc = [c for c in fa_df.columns if c.startswith("traffic_") or c == "total_traffic"]
        cl, cr = st.columns(2)
        with cl:
            sec("Поток автомобилей vs продажи")
            if "total_traffic" in fa_df.columns:
                st_df = fa_df[["total_traffic", sel_target, "timestamp"]].dropna().copy()
                st_df["Час"] = pd.to_datetime(st_df["timestamp"]).dt.hour
                ft = px.scatter(st_df, x="total_traffic", y=sel_target,
                                color="Час", color_continuous_scale="Plasma",
                                opacity=0.4, trendline="ols",
                                labels={"total_traffic": "Трафик (авт/ч)",
                                        sel_target: f"{target_label} ({unit})"})
                st.plotly_chart(chart_layout(ft, 320), width="stretch")
        with cr:
            sec("Корреляция типов трафика")
            if tc:
                tc_av = [c for c in tc if c in fa_df.columns]
                corr = fa_df[tc_av + [sel_target]].corr()[sel_target].drop(sel_target).sort_values()
                fc = px.bar(corr.reset_index(), x=sel_target, y="index", orientation="h",
                            color=sel_target, color_continuous_scale="RdBu",
                            color_continuous_midpoint=0,
                            labels={sel_target: "Корреляция Пирсона", "index": ""})
                fc.update_layout(coloraxis_showscale=False)
                st.plotly_chart(chart_layout(fc, 320), width="stretch")

        sec("Суточный паттерн продаж")
        hr2 = pd.to_datetime(fa_df["timestamp"]).dt.hour
        hg2 = fa_df.groupby(hr2)[sel_target].mean().reset_index()
        hg2.columns = ["Час", sel_target]
        ph2 = int(hg2.set_index("Час")[sel_target].idxmax())
        fh2 = px.area(hg2, x="Час", y=sel_target, markers=True,
                      color_discrete_sequence=[GOLD],
                      labels={"Час": "Час суток (0–23)", sel_target: f"Ср. продажи ({unit})"})
        fh2.update_traces(fillcolor="rgba(245,158,11,0.10)")
        fh2.add_vline(x=ph2, line_dash="dash", line_color=RED,
                      annotation_text=f"Пик: {ph2}:00")
        st.plotly_chart(chart_layout(fh2, 270), width="stretch")

    with fa3:
        wc = ["temperature", "precipitation_mm", "wind_speed_ms", "visibility_km"]
        wl = {"temperature": "Температура, °C", "precipitation_mm": "Осадки, мм/ч",
              "wind_speed_ms": "Скорость ветра, м/с", "visibility_km": "Видимость, км"}
        aw = [c for c in wc if c in fa_df.columns]
        if aw:
            wch = st.selectbox("Погодный фактор", aw, format_func=lambda x: wl.get(x, x))
            sw = fa_df[[wch, sel_target]].dropna()
            fw = px.scatter(sw, x=wch, y=sel_target, trendline="ols", opacity=0.35,
                            color_discrete_sequence=[GOLD],
                            labels={wch: wl.get(wch, wch),
                                    sel_target: f"{target_label} ({unit})"})
            fw.data[-1].line.color = TREND_C
            fw.data[-1].line.width = 2
            st.plotly_chart(chart_layout(fw, 360), width="stretch")
            sec("Корреляция погодных факторов с продажами")
            cw = fa_df[aw + [sel_target]].corr()[sel_target].drop(sel_target)
            cw.index = [wl.get(i, i) for i in cw.index]
            cw_df = cw.round(3).reset_index()
            cw_df.columns = ["Фактор", "r Пирсона"]
            cw_df = cw_df.sort_values("r Пирсона")
            fcw = px.bar(cw_df, x="r Пирсона", y="Фактор", orientation="h",
                         color="r Пирсона",
                         color_continuous_scale=[[0, RED], [0.5, GRAY], [1, BLUE]],
                         color_continuous_midpoint=0, range_x=[-1, 1],
                         labels={"r Пирсона": "Коэффициент корреляции", "Фактор": ""})
            fcw.update_layout(coloraxis_showscale=False)
            st.plotly_chart(chart_layout(fcw, 200), width="stretch")
        else:
            st.info("Погодные данные не найдены.")

    with fa4:
        comp = [c for c in fa_df.columns if c.startswith("competitor_price_")]
        if comp:
            cc = st.selectbox("Конкурент", comp)
            sc2 = fa_df[[cc, sel_target]].dropna()
            if not sc2.empty:
                fcp = px.scatter(sc2, x=cc, y=sel_target, trendline="ols", opacity=0.35,
                                 color_discrete_sequence=[GOLD],
                                 labels={cc: "Цена конкурента, руб",
                                         sel_target: f"{target_label} ({unit})"})
                fcp.data[-1].line.color = TREND_C
                fcp.data[-1].line.width = 2
                st.plotly_chart(chart_layout(fcp, 380), width="stretch")
                st.caption(
                    "Каждая точка — один час 2023 года: цена конкурента в этот час и наши продажи в этот же час. "
                    "Линия тренда показывает общее направление зависимости. "
                    "Отрицательный тренд: рост цены конкурента → покупатели переключаются к нам. "
                    "Положительный: оба показателя реагируют на общий сезонный спрос."
                )
        else:
            st.info("Данные о ценах конкурентов не найдены.")


# ═══════════════════════════════════════════════════════════════
# TAB 4 · СЦЕНАРИЙ & РЕКОМЕНДАЦИИ
# ═══════════════════════════════════════════════════════════════
with tab4:
    sc_col, rec_col = st.columns([1, 1])

    with sc_col:
        sec("Сценарный анализ (What-if)")
        banner(
            "Измените параметры и нажмите <b>Запустить прогноз</b> — TFT пересчитает "
            "прогноз на 24 часа с новыми условиями. "
            "Сравните результат с базовым прогнозом (реальные условия декабря). "
            "Первый запуск загружает модель (~30 с)."
        )

        sc_st = st.selectbox(
            "Станция", stations, key="sc_station",
            format_func=lambda x: STATION_LABELS.get(str(x), f"АЗС-{x}"),
        )
        dec_dates = sorted(dec_df["timestamp"].dt.date.unique())
        sc_date = st.selectbox(
            "Дата начала 24-часового окна",
            dec_dates, format_func=lambda d: d.strftime("%d %B %Y"), key="sc_date",
        )

        ca, cb = st.columns(2)
        with ca:
            sc_pf = st.toggle("Акция на топливо", key="sc_pf")
            sc_ps = st.toggle("Акция в магазине", key="sc_ps")
        with cb:
            sc_ad = st.toggle("Реклама активна", key="sc_ad")

        ad_opts = {v: k for k, v in ad_map.items()}
        sc_ch = st.selectbox("Канал рекламы", list(ad_opts.keys()), key="sc_ch")
        sc_tgt = st.selectbox(
            "Прогнозируемый показатель", TARGET_COLS,
            format_func=lambda x: TARGET_LABELS.get(x, x), key="sc_tgt",
        )

        # Слайдер цены — только для топливных показателей
        sc_price_rub = None
        price_col_sc = PRICE_MAP.get(sc_tgt)
        if price_col_sc and price_col_sc in merged_df.columns:
            st_mask = merged_df["station_id"] == sc_st
            base_p = float(merged_df[st_mask][price_col_sc].median())
            sc_price_rub = st.slider(
                f"Цена {TARGET_LABELS.get(sc_tgt, sc_tgt)}, руб/л",
                min_value=round(base_p * 0.88, 1),
                max_value=round(base_p * 1.12, 1),
                value=round(base_p, 1),
                step=0.5,
                key="sc_price",
            )
            st.caption(f"Медиана 2023 г.: {base_p:.1f} руб/л · диапазон ±12%")

        run = st.button("▶ Запустить прогноз", type="primary")

        if run:
            prog = st.progress(0, text="Инициализация...")
            prog.progress(15, text="Шаг 1/4 — загрузка TFT-модели...")
            model, training, tft_cfg, err = load_tft()

            if err or model is None:
                prog.empty()
                st.error(f"Не удалось загрузить модель: {err}")
            else:
                prog.progress(35, text="Шаг 2/4 — подготовка данных сценария...")
                try:
                    from pytorch_forecasting import TimeSeriesDataSet

                    ps  = pd.Timestamp(sc_date)
                    cs  = ps - pd.Timedelta(hours=ENCODER_LENGTH)
                    pe  = ps + pd.Timedelta(hours=PREDICTION_LENGTH - 1)

                    ctx = prepared_df[
                        (prepared_df["timestamp"] >= cs) & (prepared_df["timestamp"] <= pe)
                    ].copy()

                    fm = (ctx["station_id"] == sc_st) & (ctx["timestamp"] >= ps)
                    ctx.loc[fm, "promotion_fuel_active"] = int(sc_pf)
                    ctx.loc[fm, "promotion_shop_active"] = int(sc_ps)
                    ctx.loc[fm, "ad_active"]             = int(sc_ad)
                    ctx.loc[fm, "ad_channel_enc"]        = ad_opts[sc_ch]

                    # Применяем цену топлива (переводим руб → z-score)
                    if sc_price_rub is not None and price_col_sc and price_col_sc in ctx.columns:
                        sid_key = str(sc_st)
                        scaler_ok = (
                            scalers and sid_key in scalers
                            and isinstance(scalers[sid_key], dict)
                            and price_col_sc in scalers[sid_key]
                        )
                        if scaler_ok:
                            mean_v, std_v = scalers[sid_key][price_col_sc]
                            if std_v > 0:
                                ctx.loc[fm, price_col_sc] = (sc_price_rub - mean_v) / std_v
                        else:
                            st_mask2 = merged_df["station_id"] == sc_st
                            orig_med = float(merged_df[st_mask2][price_col_sc].median())
                            if orig_med > 0:
                                ratio = sc_price_rub / orig_med
                                ctx.loc[fm, price_col_sc] *= ratio

                    for col in tft_cfg["static_cats"] + tft_cfg["known_cats"]:
                        if col in ctx.columns:
                            ctx[col] = ctx[col].astype(str)

                    ds  = TimeSeriesDataSet.from_dataset(training, ctx, stop_randomization=True)
                    ldr = ds.to_dataloader(train=False, batch_size=16, num_workers=0)

                    prog.progress(55, text="Шаг 3/4 — прогон TFT (наиболее долгий шаг)...")

                    res   = model.predict(ldr, mode="quantiles", return_index=True,
                                          trainer_kwargs={"logger": False, "enable_progress_bar": False})
                    preds = res.output if hasattr(res, "output") else res[0]
                    idx   = res.index  if hasattr(res, "index")  else res[1]

                    prog.progress(80, text="Шаг 4/4 — формирование результата...")

                    sid_enc = training.categorical_encoders.get("station_id")
                    if sid_enc is not None:
                        idx = idx.copy()
                        idx["station_id"] = sid_enc.inverse_transform(
                            pd.Series(idx["station_id"].astype(int)))

                    ti    = TARGET_COLS.index(sc_tgt)
                    qm    = len(model.loss.quantiles) // 2
                    arr   = preds[ti].detach().cpu().numpy()

                    ts_lu = (prepared_df[["station_id", "time_idx", "timestamp"]]
                             .drop_duplicates()
                             .set_index(["station_id", "time_idx"])["timestamp"])
                    rows = []
                    for i, (_, row) in enumerate(idx.iterrows()):
                        sid = str(row["station_id"])
                        if sid != sc_st:
                            continue
                        t0 = ts_lu.get((sid, int(row["time_idx"])))
                        if t0 is None:
                            continue
                        for h in range(PREDICTION_LENGTH):
                            ts = pd.Timestamp(t0) + pd.Timedelta(hours=h)
                            if ps <= ts <= pe:
                                rows.append({"timestamp": ts,
                                             "scenario": np.expm1(max(arr[i, h, qm], 0))})

                    sc_ts = pd.DataFrame(rows).groupby("timestamp")["scenario"].mean().reset_index()

                    prog.progress(100, text="Готово!")
                    prog.empty()

                    lbl_s  = TARGET_LABELS.get(sc_tgt, sc_tgt)
                    u_s    = UNITS.get(sc_tgt, "")
                    st_lbl = STATION_LABELS.get(str(sc_st), f"АЗС-{sc_st}")

                    base = None
                    if pred_df is not None:
                        pn = f"{sc_tgt}_pred"
                        bdf = pred_df[(pred_df["station_id"] == sc_st) &
                                      (pred_df["timestamp"] >= ps) &
                                      (pred_df["timestamp"] <= pe)]
                        if pn in bdf.columns:
                            base = bdf[["timestamp", pn]].rename(columns={pn: "baseline"})

                    # Метрика выше графика
                    if base is not None and not sc_ts.empty:
                        ba = base["baseline"].mean()
                        sa = sc_ts["scenario"].mean()
                        if ba > 0:
                            dp = (sa - ba) / ba * 100
                            st.metric(f"Средний прогноз: {lbl_s}",
                                      f"{sa:.2f} {u_s}",
                                      delta=f"{dp:+.1f}% vs базовый")

                    st.markdown("<br>", unsafe_allow_html=True)

                    fr = go.Figure()
                    if base is not None and not base.empty:
                        fr.add_trace(go.Scatter(x=base["timestamp"], y=base["baseline"],
                                                name="Базовый прогноз", mode="lines",
                                                line=dict(color="#60A5FA", dash="dot", width=2)))
                    if not sc_ts.empty:
                        fr.add_trace(go.Scatter(x=sc_ts["timestamp"], y=sc_ts["scenario"],
                                                name="Сценарный прогноз", mode="lines",
                                                line=dict(color=GREEN, width=2.5)))
                    chart_layout(fr, 360)
                    fr.update_layout(
                        title=f"{lbl_s} · {st_lbl} · {sc_date}",
                        xaxis_title="Время", yaxis_title=u_s,
                        legend=dict(orientation="h", y=1.13),
                        margin=dict(t=64, b=10, l=0, r=0),
                    )
                    st.plotly_chart(fr, width="stretch")

                except Exception as e:
                    prog.empty()
                    st.error(f"Ошибка при расчёте: {e}")

    with rec_col:
        sec("Рекомендации")
        st.caption(
            "Паттерны — полный 2023 год. "
            "Надёжность прогноза — метрики TFT на тестовом декабре."
        )

        # Источник: весь год, фильтрация по выбранной станции
        rd = merged_df.copy()
        if sel_station != "Все":
            rd = rd[rd["station_id"] == sel_station]

        # Надёжность прогноза по выбранному показателю (первая карточка — контекст)
        if metrics_df is not None and sel_target in metrics_df["target"].values:
            tgt_m = metrics_df[metrics_df["target"] == sel_target]
            if sel_station != "Все":
                tgt_m = tgt_m[tgt_m["station_id"].astype(str) == sel_station]
            if not tgt_m.empty:
                mape = tgt_m["MAPE_%"].mean()
                if not np.isnan(mape):
                    if mape <= 10:
                        reliability = "высокая"
                        rel_hint = "сценарным прогнозам можно доверять"
                        rel_color = GREEN
                    elif mape <= 20:
                        reliability = "умеренная"
                        rel_hint = "прогноз ориентировочный, учитывайте погрешность"
                        rel_color = GOLD
                    else:
                        reliability = "низкая"
                        rel_hint = "прогноз нестабилен, используйте как индикатив"
                        rel_color = RED
                    rec_card(
                        f"Точность TFT · {target_label}",
                        f"MAPE <b>{mape:.1f}%</b> — надёжность <b>{reliability}</b>. {rel_hint}.",
                        color=rel_color,
                    )

        # Эффект акции
        pcol2 = "promotion_fuel_active" if sel_target in FUEL_COLS else "promotion_shop_active"
        if pcol2 in rd.columns and sel_target in rd.columns:
            on  = rd[rd[pcol2] == 1][sel_target].mean()
            off = rd[rd[pcol2] == 0][sel_target].mean()
            if off > 0 and not np.isnan(on):
                d = (on - off) / off * 100
                rec_card(
                    "Акционные периоды",
                    f"{'↑' if d > 0 else '↓'} <b>{abs(d):.1f}%</b> "
                    f"{'прирост' if d > 0 else 'снижение'} продаж {target_label} при акции "
                    f"({on:.1f} vs {off:.1f} {unit}) — по данным всего 2023 г.",
                    color=GREEN if d > 0 else RED,
                )

        # Лучший рекламный канал
        if "ad_channel" in rd.columns and sel_target in rd.columns:
            aa = rd.groupby("ad_channel")[sel_target].mean()
            wch = rd.groupby("ad_channel")[sel_target].mean()
            if not aa.empty:
                bch = aa.idxmax()
                wch_name = aa.idxmin()
                rec_card(
                    "Рекламный канал",
                    f"Наибольший эффект за год: <b>«{bch}»</b> — ср. {aa[bch]:.1f} {unit}. "
                    f"Наименьший: «{wch_name}» ({aa[wch_name]:.1f} {unit}).",
                    color=BLUE,
                )

        # Пиковые часы
        if sel_target in rd.columns:
            hr3 = pd.to_datetime(rd["timestamp"]).dt.hour
            hg3 = rd.groupby(hr3)[sel_target].mean()
            if not hg3.empty:
                ph3 = int(hg3.idxmax())
                lh3 = int(hg3.idxmin())
                rec_card(
                    "Пиковые часы продаж",
                    f"Пик: <b>{ph3}:00–{(ph3+1)%24}:00</b> ({hg3[ph3]:.1f} {unit}). "
                    f"Минимум: {lh3}:00 ({hg3[lh3]:.1f} {unit}).",
                    color=GOLD,
                )

        # Лучший день недели
        if sel_target in rd.columns:
            dr2 = pd.to_datetime(rd["timestamp"]).dt.dayofweek
            dg2 = rd.groupby(dr2)[sel_target].mean()
            if not dg2.empty:
                bd  = int(dg2.idxmax())
                wd  = int(dg2.idxmin())
                rec_card(
                    "Лучший день недели",
                    f"Максимум: <b>{DAY_NAMES[bd]}</b> ({dg2[bd]:.1f} {unit}). "
                    f"Минимум: {DAY_NAMES[wd]} ({dg2[wd]:.1f} {unit}).",
                    color=GREEN,
                )

        # Зависимость от трафика
        if "total_traffic" in rd.columns and sel_target in rd.columns:
            r = rd[["total_traffic", sel_target]].corr().iloc[0, 1]
            direction = "положительная" if r > 0 else "отрицательная"
            strength  = "сильная" if abs(r) > 0.5 else "умеренная" if abs(r) > 0.25 else "слабая"
            rec_card(
                "Зависимость от трафика",
                f"Корреляция {direction} и {strength} (r = {r:.2f}). "
                + ("Рост трафика → рост продаж." if r > 0 else "Трафик слабо влияет на продажи."),
                color=TEAL,
            )
