"""
Дашборд прогнозов TFT — декабрь 2023.
Вкладки: Прогноз | Качество модели | Факторный анализ | Сценарий & Рекомендации
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

# ── Метки ─────────────────────────────────────────────────────
FUEL_COLS = [c for c in TARGET_COLS if c.startswith("sales_")]
SHOP_COLS = [c for c in TARGET_COLS if c.startswith("shop_")]

TARGET_LABELS = {
    "sales_AI92": "АИ-92",
    "sales_AI95": "АИ-95",
    "sales_AI98": "АИ-98",
    "sales_DT_EURO": "ДТ Евро+",
    "sales_DT_TANEKO": "ДТ ТАНЕКО",
    "sales_DT_SUMMER": "ДТ Летнее",
    "sales_DT_WINTER": "ДТ Зимнее",
    "shop_напитки": "Напитки",
    "shop_закуски": "Закуски",
    "shop_автотовары": "Автотовары",
    "shop_кофе": "Кофе",
    "shop_табак": "Табак",
}
UNITS = {**{c: "л/ч" for c in FUEL_COLS}, **{c: "руб/ч" for c in SHOP_COLS}}

# ── Конфиг страницы ───────────────────────────────────────────
st.set_page_config(
    page_title="TFT Прогнозы — АЗС Татнефть",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Загрузка данных ───────────────────────────────────────────
@st.cache_data
def load_predictions():
    path = "data/predictions.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    return df


@st.cache_data
def load_metrics():
    path = "data/metrics.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_prepared():
    df = pd.read_csv("data/prepared_data.csv", parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    dec_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    for col in TARGET_COLS:
        if col in df.columns:
            df.loc[dec_mask, col] = np.expm1(df.loc[dec_mask, col].clip(lower=0))
    return df


@st.cache_data
def load_ad_channel_map():
    df = pd.read_csv("data/prepared_data.csv", usecols=["ad_channel", "ad_channel_enc"])
    return (
        df.drop_duplicates()
        .sort_values("ad_channel_enc")
        .set_index("ad_channel_enc")["ad_channel"]
        .to_dict()
    )


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
            config = pickle.load(f)

        model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
        model.eval()
        return model, training, config, None
    except Exception as e:
        return None, None, None, str(e)


# ── Данные ────────────────────────────────────────────────────
pred_df = load_predictions()
metrics_df = load_metrics()
prepared_df = load_prepared()
ad_map = load_ad_channel_map()

dec_df = prepared_df[
    (prepared_df["timestamp"] >= TEST_START)
    & (prepared_df["timestamp"] <= TEST_END)
].copy()

stations = sorted(prepared_df["station_id"].unique())

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⛽ TFT Прогнозы")
    st.caption("АЗС Татнефть | Декабрь 2023")
    st.divider()

    sel_station = st.selectbox("Станция", ["Все"] + stations)
    sel_target = st.selectbox(
        "Целевая переменная",
        options=TARGET_COLS,
        format_func=lambda x: TARGET_LABELS.get(x, x),
    )

    st.divider()
    st.caption(f"Encoder: {ENCODER_LENGTH} ч  |  Horizon: {PREDICTION_LENGTH} ч")
    st.caption("TFT — Lim et al., 2020")
    st.divider()
    st.caption("🧠 Интерпретация TFT:")
    st.caption("`streamlit run dashboard/tft_interpretation.py`")


# ── Вспомогательные функции ───────────────────────────────────
def filter_pred(df, station, target):
    if df is None:
        return None
    cols = ["station_id", "timestamp", "horizon_h"]
    for sfx in ["_pred", "_q10", "_q90", "_actual"]:
        c = target + sfx
        if c in df.columns:
            cols.append(c)
    sub = df[cols].copy()
    if station != "Все":
        sub = sub[sub["station_id"] == station]
    return sub.reset_index(drop=True)


def forecast_chart(df, target):
    label = TARGET_LABELS.get(target, target)
    unit = UNITS.get(target, "")
    pred_col, actual_col = f"{target}_pred", f"{target}_actual"
    q10_col, q90_col = f"{target}_q10", f"{target}_q90"

    fig = go.Figure()
    if actual_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[actual_col],
            mode="lines", name="Факт",
            line=dict(color="#4FC3F7", width=2),
        ))
    if pred_col in df.columns:
        if q10_col in df.columns and q90_col in df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df["timestamp"], df["timestamp"][::-1]]),
                y=pd.concat([df[q90_col], df[q10_col][::-1]]),
                fill="toself", fillcolor="rgba(255,167,38,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Доверит. интервал (q10–q90)",
            ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[pred_col],
            mode="lines", name="Прогноз",
            line=dict(color="#FFA726", width=2, dash="dot"),
        ))

    fig.update_layout(
        title=f"{label} — факт vs прогноз ({unit})",
        xaxis_title="Дата",
        yaxis_title=unit,
        legend=dict(orientation="h", y=1.1),
        height=400,
        margin=dict(t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Прогноз декабря",
    "📈 Качество модели",
    "🔍 Факторный анализ",
    "🎯 Сценарий & Рекомендации",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: ПРОГНОЗ ДЕКАБРЯ
# ═══════════════════════════════════════════════════════════════
with tab1:
    if pred_df is None:
        st.warning("Файл data/predictions.csv не найден. Запустите python tft/predict.py")
        st.stop()

    with st.expander("📖 Как читать этот раздел?", expanded=False):
        st.markdown("""
        **Синяя линия** — фактические продажи за декабрь 2023 (исходные данные).

        **Оранжевая пунктирная линия** — прогноз модели TFT на 24 часа вперёд.

        **Оранжевая полупрозрачная область** — доверительный интервал (квантили q10–q90):
        модель утверждает, что с вероятностью ~80% реальное значение попадёт в эту полосу.
        Широкая полоса = модель менее уверена (высокая волатильность данных или редкое событие).

        Метрики в карточках вверху:
        - **MAE** — средняя абсолютная ошибка в единицах измерения (л/ч или руб/ч)
        - **RMSE** — корень среднеквадратичной ошибки (штрафует за крупные промахи сильнее)
        - **MAPE** — относительная ошибка в процентах: ≤10% — отлично, 10–20% — хорошо, >30% — слабо
        """)

    # KPI-строка
    if metrics_df is not None:
        sub_m = metrics_df if sel_station == "Все" else metrics_df[metrics_df["station_id"].astype(str) == sel_station]
        sub_m = sub_m[sub_m["target"] == sel_target]
        if not sub_m.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{sub_m['MAE'].mean():.2f} {UNITS.get(sel_target,'')}")
            c2.metric("RMSE", f"{sub_m['RMSE'].mean():.2f}")
            mape = sub_m["MAPE_%"].mean()
            c3.metric("MAPE", f"{mape:.1f} %" if not np.isnan(mape) else "N/A")
            c4.metric("Горизонт прогноза", f"{PREDICTION_LENGTH} ч")

    st.divider()

    df_plot = filter_pred(pred_df, sel_station, sel_target)
    if df_plot is not None and not df_plot.empty:
        if sel_station == "Все":
            num_cols = [c for c in df_plot.columns if c not in ("station_id", "timestamp", "horizon_h")]
            df_plot = df_plot.groupby("timestamp")[num_cols].mean().reset_index()

        st.plotly_chart(forecast_chart(df_plot, sel_target), width='stretch')

        st.subheader("Среднесуточные значения по декабрю")
        df_day = df_plot.copy()
        df_day["date"] = df_plot["timestamp"].dt.date
        pred_c, act_c = f"{sel_target}_pred", f"{sel_target}_actual"
        agg = {pred_c: "mean"}
        if act_c in df_day.columns:
            agg[act_c] = "mean"
        daily = df_day.groupby("date").agg(agg).reset_index()
        daily.columns = ["Дата", "Прогноз"] + (["Факт"] if act_c in df_day.columns else [])
        st.dataframe(daily.round(2), width='stretch', height=280)
    else:
        st.info("Нет данных для выбранных фильтров.")


# ═══════════════════════════════════════════════════════════════
# TAB 2: КАЧЕСТВО МОДЕЛИ
# ═══════════════════════════════════════════════════════════════
with tab2:
    if metrics_df is None:
        st.warning("Файл data/metrics.csv не найден. Запустите python tft/predict.py")
        st.stop()

    with st.expander("📖 Как читать метрики?", expanded=False):
        st.markdown("""
        **MAE (Mean Absolute Error)** — средняя абсолютная ошибка в единицах измерения.
        Для топлива: если MAE = 5 л/ч, модель в среднем ошибается на 5 литров в час.

        **RMSE (Root Mean Squared Error)** — похож на MAE, но сильнее штрафует за крупные промахи.
        Если RMSE >> MAE — значит, есть редкие часы с большими выбросами ошибки.

        **MAPE (Mean Absolute Percentage Error)** — ошибка в процентах от фактического значения.
        Наиболее интерпретируемая метрика для сравнения разных переменных.
        Ориентир: ≤10% — отлично, 10–20% — хорошо, 20–30% — приемлемо, >30% — требует доработки.

        **Тепловая карта** показывает MAPE по каждой паре (станция × переменная).
        Красный = высокая ошибка, зелёный = низкая. Видно, какие станции или продукты сложнее предсказывать.

        **Scatter "прогноз vs факт"** — в идеале все точки лежат на красной пунктирной линии (y=x).
        Точки выше линии — модель завышает, ниже — занижает прогноз.
        """)

    disp = metrics_df.copy()
    disp["target_label"] = disp["target"].map(TARGET_LABELS).fillna(disp["target"])
    if sel_station != "Все":
        disp = disp[disp["station_id"].astype(str) == sel_station]

    st.subheader("Метрики по всем целевым переменным и станциям")
    st.dataframe(
        disp[["station_id", "target_label", "MAE", "RMSE", "MAPE_%", "n"]]
        .rename(columns={"target_label": "Переменная", "station_id": "Станция"})
        .round(3),
        width='stretch',
        height=320,
    )

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("MAPE по переменным (среднее по станциям)")
        st.caption("Чем короче полоска — тем точнее прогноз по этой переменной.")
        avg_mape = metrics_df.groupby("target")["MAPE_%"].mean().reset_index()
        avg_mape["label"] = avg_mape["target"].map(TARGET_LABELS).fillna(avg_mape["target"])
        avg_mape = avg_mape.sort_values("MAPE_%", na_position="last")
        fig_bar = px.bar(
            avg_mape, x="MAPE_%", y="label", orientation="h",
            color="MAPE_%", color_continuous_scale="RdYlGn_r",
            labels={"MAPE_%": "MAPE, %", "label": ""},
            height=380,
        )
        fig_bar.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_bar, width='stretch')

    with col_right:
        st.subheader("Тепловая карта MAPE: станция × переменная")
        st.caption("Красный — высокая ошибка, зелёный — низкая. Помогает найти проблемные станции и продукты.")
        pivot = metrics_df.pivot_table(
            index="station_id", columns="target", values="MAPE_%", aggfunc="mean"
        )
        pivot.columns = [TARGET_LABELS.get(c, c) for c in pivot.columns]
        fig_hm = px.imshow(
            pivot.round(1),
            color_continuous_scale="RdYlGn_r",
            text_auto=True,
            aspect="auto",
            labels=dict(color="MAPE, %"),
            height=380,
        )
        fig_hm.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_hm, width='stretch')

    if pred_df is not None:
        st.subheader("Прогноз vs факт — scatter")
        st.caption("Идеальная модель: все точки на красной диагональной линии y = x. "
                   "Облако точек вблизи линии — хорошее качество прогноза.")
        pred_c, act_c = f"{sel_target}_pred", f"{sel_target}_actual"
        scatter_df = pred_df[[pred_c, act_c]].dropna() if all(
            c in pred_df.columns for c in [pred_c, act_c]
        ) else None
        if scatter_df is not None and not scatter_df.empty:
            fig_sc = px.scatter(
                scatter_df, x=act_c, y=pred_c,
                labels={act_c: f"Факт ({UNITS.get(sel_target,'')})", pred_c: "Прогноз"},
                opacity=0.4, height=320,
                title=f"{TARGET_LABELS.get(sel_target, sel_target)} — scatter",
            )
            max_val = max(scatter_df[act_c].max(), scatter_df[pred_c].max())
            fig_sc.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                             line=dict(color="red", dash="dash"))
            st.plotly_chart(fig_sc, width='stretch')


# ═══════════════════════════════════════════════════════════════
# TAB 3: ФАКТОРНЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════════════════════
with tab3:
    fa_df = dec_df.copy()
    if sel_station != "Все":
        fa_df = fa_df[fa_df["station_id"] == sel_station]

    target_col = sel_target
    label = TARGET_LABELS.get(target_col, target_col)
    unit = UNITS.get(target_col, "")

    st.info(
        "Анализ влияния внешних факторов на продажи — на основе фактических данных декабря 2023. "
        "Выберите вкладку ниже."
    )

    fa1, fa2, fa3, fa4 = st.tabs(["Акции & Реклама", "Трафик", "Погода", "Цены конкурентов"])

    # ── Акции & Реклама ────────────────────────────────────────
    with fa1:
        st.markdown("""
        **Box plot** — ящик с усами. Линия внутри ящика — медиана. Ящик — межквартильный размах (50% данных).
        Сравниваем распределение продаж *при активной акции* (красный) и *без акции* (синий).
        Если красный ящик выше — акция действительно увеличивает продажи.
        """)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Эффект акций на продажи")
            if "promotion_fuel_active" in fa_df.columns:
                promo_col = "promotion_fuel_active" if target_col in FUEL_COLS else "promotion_shop_active"
                box_df = fa_df[[promo_col, target_col]].copy()
                box_df[promo_col] = box_df[promo_col].map({1: "Акция активна", 0: "Без акции"})
                fig_box = px.box(
                    box_df, x=promo_col, y=target_col,
                    color=promo_col,
                    labels={target_col: f"{label} ({unit})", promo_col: ""},
                    color_discrete_sequence=["#EF5350", "#42A5F5"],
                    height=360,
                )
                fig_box.update_layout(showlegend=False, margin=dict(t=20))
                st.plotly_chart(fig_box, width='stretch')

                on = fa_df[fa_df[promo_col] == 1][target_col].mean()
                off = fa_df[fa_df[promo_col] == 0][target_col].mean()
                if off > 0:
                    delta = (on - off) / off * 100
                    st.metric(
                        "Прирост от акции",
                        f"{on:.1f} {unit}",
                        delta=f"{delta:+.1f}% vs без акции",
                    )

        with c2:
            st.subheader("Продажи по каналу рекламы")
            st.caption("Средние продажи в часы, когда каждый канал рекламы был активен.")
            if "ad_channel" in fa_df.columns:
                ad_avg = (
                    fa_df.groupby("ad_channel")[target_col].mean()
                    .reset_index()
                    .sort_values(target_col, ascending=False)
                )
                fig_ad = px.bar(
                    ad_avg, x=target_col, y="ad_channel", orientation="h",
                    labels={target_col: f"{label} ({unit})", "ad_channel": "Канал"},
                    color=target_col, color_continuous_scale="Blues",
                    height=360,
                )
                fig_ad.update_layout(coloraxis_showscale=False, margin=dict(t=20))
                st.plotly_chart(fig_ad, width='stretch')

    # ── Трафик ────────────────────────────────────────────────
    with fa2:
        st.markdown("""
        **Scatter (левый)** — каждая точка = один час декабря.
        Цвет точки — час суток (тёмный = ночь, светлый = день).
        Если облако точек направлено вверх-вправо — при большем трафике продажи растут.

        **Корреляция (правый)** — коэффициент Пирсона r от −1 до +1.
        r > 0 → совместный рост, r < 0 → обратная зависимость.
        |r| > 0.5 считается сильной связью.
        """)

        traffic_cols = [c for c in fa_df.columns if c.startswith("traffic_") or c == "total_traffic"]
        if traffic_cols and target_col in fa_df.columns:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Общий трафик vs продажи")
                if "total_traffic" in fa_df.columns:
                    # Используем timestamp для получения реального часа
                    sc_df = fa_df[["total_traffic", target_col, "timestamp"]].dropna().copy()
                    sc_df["hour_real"] = pd.to_datetime(sc_df["timestamp"]).dt.hour
                    fig_sc = px.scatter(
                        sc_df, x="total_traffic", y=target_col,
                        color="hour_real", color_continuous_scale="Viridis",
                        opacity=0.4,
                        labels={
                            "total_traffic": "Общий трафик (авт/ч)",
                            target_col: f"{label} ({unit})",
                            "hour_real": "Час суток",
                        },
                        height=360,
                    )
                    fig_sc.update_layout(margin=dict(t=20))
                    st.plotly_chart(fig_sc, width='stretch')

            with c2:
                st.subheader("Корреляция типов трафика с продажами")
                tc_cols = [c for c in traffic_cols if c in fa_df.columns]
                corr = fa_df[tc_cols + [target_col]].corr()[target_col].drop(target_col).sort_values()
                fig_corr = px.bar(
                    corr.reset_index(),
                    x=target_col, y="index", orientation="h",
                    color=target_col,
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    labels={target_col: "Корреляция Пирсона", "index": "Тип трафика"},
                    height=360,
                )
                fig_corr.update_layout(coloraxis_showscale=False, margin=dict(t=20))
                st.plotly_chart(fig_corr, width='stretch')

            st.subheader("Средние продажи по часам суток")
            st.caption("Показывает суточный паттерн: пиковые и спальные часы для выбранной переменной.")
            hour_raw = pd.to_datetime(fa_df["timestamp"]).dt.hour
            hourly = fa_df.groupby(hour_raw)[target_col].mean().reset_index()
            hourly.columns = ["Час", target_col]
            fig_h = px.line(
                hourly, x="Час", y=target_col,
                markers=True,
                labels={"Час": "Час суток (0–23)", target_col: f"{label} ({unit})"},
                height=300,
            )
            fig_h.update_layout(margin=dict(t=10))
            st.plotly_chart(fig_h, width='stretch')

    # ── Погода ────────────────────────────────────────────────
    with fa3:
        st.markdown("""
        **Scatter с линией тренда (OLS)** — каждая точка = один час.
        Красная линия — линейная регрессия: направление показывает характер зависимости.

        Если линия идёт вверх → фактор положительно влияет на продажи.
        Если вниз → отрицательное влияние.
        Угол наклона → сила влияния.

        Таблица корреляций ниже суммирует все погодные факторы разом.
        """)

        weather_cols = ["temperature", "precipitation_mm", "wind_speed_ms", "visibility_km"]
        avail_w = [c for c in weather_cols if c in fa_df.columns]
        if avail_w:
            w_choice = st.selectbox("Погодный фактор", avail_w,
                                    format_func=lambda x: {
                                        "temperature": "Температура (°C, норм.)",
                                        "precipitation_mm": "Осадки (мм/ч, норм.)",
                                        "wind_speed_ms": "Скорость ветра (м/с, норм.)",
                                        "visibility_km": "Видимость (км, норм.)",
                                    }.get(x, x))
            sc_df = fa_df[[w_choice, target_col]].dropna()
            fig_w = px.scatter(
                sc_df, x=w_choice, y=target_col,
                trendline="ols",
                opacity=0.35,
                labels={w_choice: w_choice, target_col: f"{label} ({unit})"},
                height=380,
            )
            st.plotly_chart(fig_w, width='stretch')

            st.subheader("Корреляция всех погодных факторов с продажами")
            corr_w = fa_df[avail_w + [target_col]].corr()[target_col].drop(target_col)
            corr_w.index = [
                {"temperature": "Температура", "precipitation_mm": "Осадки",
                 "wind_speed_ms": "Скорость ветра", "visibility_km": "Видимость"}.get(i, i)
                for i in corr_w.index
            ]
            st.dataframe(
                corr_w.rename("Корреляция с продажами").round(3).to_frame(),
                width='content',
            )

    # ── Цены конкурентов ──────────────────────────────────────
    with fa4:
        st.markdown("""
        **Цены конкурентов** — z-score нормализованные значения (0 = среднее по году).

        Если тренд **отрицательный** (линия идёт вниз) → когда конкурент повышает цены,
        наши продажи растут: покупатели переключаются на нашу АЗС.

        Если тренд **положительный** → цены конкурента и наши продажи движутся синхронно
        (например, оба реагируют на сезонный спрос).
        """)

        comp_cols = [c for c in fa_df.columns if c.startswith("competitor_price_")]
        if comp_cols:
            c_choice = st.selectbox("Цена конкурента", comp_cols)
            sc_df = fa_df[[c_choice, target_col]].dropna()
            if not sc_df.empty:
                fig_cp = px.scatter(
                    sc_df, x=c_choice, y=target_col,
                    trendline="ols", opacity=0.35,
                    labels={c_choice: "Цена конкурента (норм.)", target_col: f"{label} ({unit})"},
                    height=380,
                    title=f"Влияние {c_choice} на {label}",
                )
                st.plotly_chart(fig_cp, width='stretch')
        else:
            st.info("Данные о ценах конкурентов не найдены.")


# ═══════════════════════════════════════════════════════════════
# TAB 4: СЦЕНАРИЙ & РЕКОМЕНДАЦИИ
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.info(
        "Измените будущие условия (акции, реклама) — TFT пересчитает прогноз на 24 часа вперёд. "
        "Это позволяет оценить эффект управленческих решений **до их принятия**. "
        "Первый запуск загружает модель (~30 сек)."
    )

    sc_col, rec_col = st.columns([1, 1])

    # ── Сценарный анализ ──────────────────────────────────────
    with sc_col:
        st.subheader("🎯 Сценарный анализ (What-if)")

        with st.expander("📖 Как работает сценарный анализ?", expanded=False):
            st.markdown("""
            1. Выберите станцию и дату начала 24-часового окна прогноза.
            2. Включите/выключите акции или рекламу.
            3. Нажмите **Запустить прогноз** — TFT пересчитает все 24 шага с новыми условиями.
            4. Сравните **сценарный прогноз** (оранжевый) с **базовым** (синий пунктир).

            Базовый прогноз — это результат из `data/predictions.csv` (реальные условия декабря).
            Разница между ними — **чистый эффект** изменённых параметров.
            """)

        sc_station = st.selectbox("Станция", stations, key="sc_station")

        dec_dates = sorted(dec_df["timestamp"].dt.date.unique())
        valid_dates = [d for d in dec_dates if d >= (TEST_START + pd.Timedelta(hours=ENCODER_LENGTH)).date()]
        sc_date = st.selectbox(
            "Дата прогноза (начало 24-часового окна)",
            options=valid_dates,
            format_func=lambda d: d.strftime("%d %B %Y"),
            key="sc_date",
        )

        st.markdown("**Изменить будущие условия:**")
        sc_promo_fuel = st.toggle("Акция на топливо (promotion_fuel_active)", key="sc_pf")
        sc_promo_shop = st.toggle("Акция на магазин (promotion_shop_active)", key="sc_ps")
        sc_ad_active = st.toggle("Реклама активна (ad_active)", key="sc_ad")

        ad_options = {v: k for k, v in ad_map.items()}
        sc_ad_channel = st.selectbox("Канал рекламы", options=list(ad_options.keys()), key="sc_ch")

        sc_target = st.selectbox(
            "Показать прогноз по",
            TARGET_COLS,
            format_func=lambda x: TARGET_LABELS.get(x, x),
            key="sc_tgt",
        )

        run_btn = st.button("▶ Запустить прогноз", type="primary")

        if run_btn:
            with st.spinner("Загрузка модели и пересчёт (~30 сек при первом запуске)..."):
                model, training, tft_config, err = load_tft()

            if err or model is None:
                st.error(f"Не удалось загрузить модель: {err}")
            else:
                with st.spinner("Вычисление сценарного прогноза..."):
                    try:
                        from pytorch_forecasting import TimeSeriesDataSet

                        pred_start = pd.Timestamp(sc_date)
                        context_start_sc = pred_start - pd.Timedelta(hours=ENCODER_LENGTH)
                        pred_end_sc = pred_start + pd.Timedelta(hours=PREDICTION_LENGTH - 1)

                        ctx = prepared_df[
                            (prepared_df["timestamp"] >= context_start_sc)
                            & (prepared_df["timestamp"] <= pred_end_sc)
                        ].copy()

                        fut_mask = (ctx["station_id"] == sc_station) & (ctx["timestamp"] >= pred_start)
                        ctx.loc[fut_mask, "promotion_fuel_active"] = int(sc_promo_fuel)
                        ctx.loc[fut_mask, "promotion_shop_active"] = int(sc_promo_shop)
                        ctx.loc[fut_mask, "ad_active"] = int(sc_ad_active)
                        ctx.loc[fut_mask, "ad_channel_enc"] = ad_options[sc_ad_channel]

                        for col in tft_config["static_cats"] + tft_config["known_cats"]:
                            if col in ctx.columns:
                                ctx[col] = ctx[col].astype(str)

                        sc_ds = TimeSeriesDataSet.from_dataset(training, ctx, stop_randomization=True)
                        sc_loader = sc_ds.to_dataloader(train=False, batch_size=16, num_workers=0)

                        sc_result = model.predict(
                            sc_loader, mode="quantiles", return_index=True,
                            trainer_kwargs={"logger": False, "enable_progress_bar": False},
                        )
                        sc_preds = sc_result.output if hasattr(sc_result, "output") else sc_result[0]
                        sc_idx = sc_result.index if hasattr(sc_result, "index") else sc_result[1]

                        sid_enc = training.categorical_encoders.get("station_id")
                        if sid_enc is not None:
                            sc_idx = sc_idx.copy()
                            sc_idx["station_id"] = sid_enc.inverse_transform(
                                pd.Series(sc_idx["station_id"].astype(int))
                            )

                        t_i = TARGET_COLS.index(sc_target)
                        q_med = len(model.loss.quantiles) // 2
                        arr = sc_preds[t_i].detach().cpu().numpy()

                        ts_lookup = (
                            prepared_df[["station_id", "time_idx", "timestamp"]]
                            .drop_duplicates()
                            .set_index(["station_id", "time_idx"])["timestamp"]
                        )
                        rows = []
                        for i, (_, row) in enumerate(sc_idx.iterrows()):
                            sid = str(row["station_id"])
                            if sid != sc_station:
                                continue
                            t0 = ts_lookup.get((sid, int(row["time_idx"])))
                            if t0 is None:
                                continue
                            for h in range(PREDICTION_LENGTH):
                                ts = pd.Timestamp(t0) + pd.Timedelta(hours=h)
                                if pred_start <= ts <= pred_end_sc:
                                    rows.append({
                                        "timestamp": ts,
                                        "scenario": np.expm1(max(arr[i, h, q_med], 0)),
                                    })

                        sc_ts = pd.DataFrame(rows).groupby("timestamp")["scenario"].mean().reset_index()

                        baseline = None
                        if pred_df is not None:
                            pred_col_name = f"{sc_target}_pred"
                            base = pred_df[
                                (pred_df["station_id"] == sc_station)
                                & (pred_df["timestamp"] >= pred_start)
                                & (pred_df["timestamp"] <= pred_end_sc)
                            ]
                            if pred_col_name in base.columns:
                                baseline = base[["timestamp", pred_col_name]].rename(
                                    columns={pred_col_name: "baseline"}
                                )

                        fig_sc = go.Figure()
                        if baseline is not None and not baseline.empty:
                            fig_sc.add_trace(go.Scatter(
                                x=baseline["timestamp"], y=baseline["baseline"],
                                name="Базовый прогноз", mode="lines",
                                line=dict(color="#42A5F5", dash="dot"),
                            ))
                        if not sc_ts.empty:
                            fig_sc.add_trace(go.Scatter(
                                x=sc_ts["timestamp"], y=sc_ts["scenario"],
                                name="Сценарный прогноз", mode="lines",
                                line=dict(color="#FFA726", width=2),
                            ))
                        lbl = TARGET_LABELS.get(sc_target, sc_target)
                        unit_sc = UNITS.get(sc_target, "")
                        fig_sc.update_layout(
                            title=f"What-if: {lbl} — {sc_date}",
                            xaxis_title="Время",
                            yaxis_title=unit_sc,
                            legend=dict(orientation="h", y=1.1),
                            height=350,
                        )
                        st.plotly_chart(fig_sc, width='stretch')

                        if baseline is not None and not sc_ts.empty:
                            base_avg = baseline["baseline"].mean()
                            sc_avg = sc_ts["scenario"].mean()
                            if base_avg > 0:
                                delta_pct = (sc_avg - base_avg) / base_avg * 100
                                st.metric(
                                    f"Средний прогноз {lbl}",
                                    f"{sc_avg:.2f} {unit_sc}",
                                    delta=f"{delta_pct:+.1f}% vs базовый",
                                )

                    except Exception as e:
                        st.error(f"Ошибка сценарного анализа: {e}")

    # ── Рекомендации ──────────────────────────────────────────
    with rec_col:
        st.subheader("💡 Автоматические рекомендации")
        st.caption(
            "Вычислены по фактическим данным декабря 2023 и метрикам модели. "
            "Помогают принять решение: когда и как продавать эффективнее."
        )

        recs_df = dec_df.copy()
        if sel_station != "Все":
            recs_df = recs_df[recs_df["station_id"] == sel_station]

        recs = []

        # 1. Эффект акций
        if "promotion_fuel_active" in recs_df.columns and FUEL_COLS[0] in recs_df.columns:
            for fuel in FUEL_COLS[:3]:
                on = recs_df[recs_df["promotion_fuel_active"] == 1][fuel].mean()
                off = recs_df[recs_df["promotion_fuel_active"] == 0][fuel].mean()
                if off > 0 and not np.isnan(on):
                    delta = (on - off) / off * 100
                    sign = "+" if delta > 0 else ""
                    recs.append(
                        f"**Акция на топливо** изменяет продажи {TARGET_LABELS[fuel]} "
                        f"на **{sign}{delta:.1f}%** (ср. {on:.1f} vs {off:.1f} л/ч)"
                    )

        # 2. Лучший канал рекламы
        if "ad_channel" in recs_df.columns and sel_target in recs_df.columns:
            ad_avg = recs_df.groupby("ad_channel")[sel_target].mean()
            if not ad_avg.empty:
                best_ch = ad_avg.idxmax()
                recs.append(
                    f"**Лучший канал рекламы** для {TARGET_LABELS.get(sel_target, sel_target)}: "
                    f"**{best_ch}** (ср. {ad_avg[best_ch]:.1f} {UNITS.get(sel_target,'')})"
                )

        # 3. Пиковые часы (из timestamp — hour z-score нормализован)
        if "timestamp" in recs_df.columns and sel_target in recs_df.columns:
            hour_raw = pd.to_datetime(recs_df["timestamp"]).dt.hour
            hourly = recs_df.groupby(hour_raw)[sel_target].mean()
            if not hourly.empty:
                peak_h = int(hourly.idxmax())
                recs.append(
                    f"**Пиковые продажи** {TARGET_LABELS.get(sel_target, sel_target)}: "
                    f"**{peak_h}:00 – {(peak_h + 2) % 24}:00** "
                    f"(ср. {hourly[peak_h]:.1f} {UNITS.get(sel_target, '')})"
                )

        # 4. Трафик-корреляция
        if "total_traffic" in recs_df.columns and sel_target in recs_df.columns:
            corr_t = recs_df[["total_traffic", sel_target]].corr().iloc[0, 1]
            direction = "положительная" if corr_t > 0 else "отрицательная"
            recs.append(
                f"**Связь трафика** с {TARGET_LABELS.get(sel_target, sel_target)}: "
                f"{direction} корреляция **r = {corr_t:.2f}**"
            )

        # 5. Точность модели
        if metrics_df is not None:
            avg = metrics_df.groupby("target")["MAPE_%"].mean().dropna()
            if not avg.empty:
                best_t = avg.idxmin()
                worst_t = avg.idxmax()
                recs.append(
                    f"**Наиболее точный прогноз**: {TARGET_LABELS.get(best_t, best_t)} "
                    f"(MAPE = {avg[best_t]:.1f}%)"
                )
                recs.append(
                    f"**Наименее предсказуемо**: {TARGET_LABELS.get(worst_t, worst_t)} "
                    f"(MAPE = {avg[worst_t]:.1f}%) — сложная нелинейная динамика"
                )

        # 6. Лучший день недели (из timestamp — day_of_week z-score нормализован)
        if "timestamp" in recs_df.columns and sel_target in recs_df.columns:
            day_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
            dow_raw = pd.to_datetime(recs_df["timestamp"]).dt.dayofweek
            dow_avg = recs_df.groupby(dow_raw)[sel_target].mean()
            if not dow_avg.empty:
                best_dow = int(dow_avg.idxmax())
                recs.append(
                    f"**Лучший день недели** для {TARGET_LABELS.get(sel_target, sel_target)}: "
                    f"**{day_names[best_dow]}** (ср. {dow_avg[best_dow]:.1f} {UNITS.get(sel_target, '')})"
                )

        for i, rec in enumerate(recs, 1):
            st.markdown(f"{i}. {rec}")
            st.divider()
