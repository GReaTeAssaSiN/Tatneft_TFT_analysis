import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_utils import LOG_COLS, STATIC_REALS, TARGET_COLS

st.set_page_config(
    page_title="Татнефть АЗС — EDA Анализ",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Тёмная тема и стили ──────────────────────────────────────
st.markdown("""
<style>
    /* Основной фон */
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stHeader"] { background-color: #13161f; border-bottom: 1px solid #1e2235; }

    /* Убираем отступ сверху */
    .block-container { padding-top: 3.5rem; padding-bottom: 1rem; }

    /* Шапка */
    .dash-header {
        background: #13161f;
        border: 1px solid #1e2235;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: nowrap;
    }
    .dash-logo {
        background: linear-gradient(135deg, #c8a84b, #e8c96b);
        color: #0a0c13;
        font-weight: 700;
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 5px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .dash-title {
        font-size: 13px;
        font-weight: 600;
        color: #e8eaf0;
        margin: 0;
    }
    .dash-sep {
        color: #2a2f45;
        font-size: 16px;
        flex-shrink: 0;
    }
    .dash-sub {
        font-size: 11px;
        color: #8891a8;
        margin: 0;
    }

    /* KPI карточки */
    .kpi-card {
        background: #13161f;
        border: 1px solid #1e2235;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .kpi-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: .6px;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 22px;
        font-weight: 700;
        color: #c8a84b;
    }
    .kpi-sub {
        font-size: 11px;
        color: #8891a8;
        margin-top: 3px;
    }

    /* Карточки секций */
    .section-card {
        background: #13161f;
        border: 1px solid #1e2235;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .section-title {
        font-size: 12px;
        font-weight: 600;
        color: #8891a8;
        text-transform: uppercase;
        letter-spacing: .8px;
        margin-bottom: 12px;
    }

    /* Фильтры */
    .filter-bar {
        background: #13161f;
        border: 1px solid #1e2235;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 14px;
    }
    .filter-label {
        font-size: 11px;
        color: #8891a8;
        margin-bottom: 4px;
    }

    /* Selectbox и мультиселект — фон */
    [data-testid="stSelectbox"] > div,
    [data-testid="stMultiSelect"] > div {
        background-color: #1e2235 !important;
        border: 1px solid #2a2f45 !important;
        border-radius: 6px !important;
    }
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label {
        color: #8891a8 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: .5px;
    }
    /* Теги мультиселекта — убираем красный цвет */
    [data-baseweb="tag"] {
        background-color: #1e2235 !important;
        border: 1px solid #2a2f45 !important;
        border-radius: 4px !important;
    }
    [data-baseweb="tag"] span {
        color: #c8a84b !important;
        font-size: 11px !important;
    }
    [data-baseweb="tag"] [role="button"] {
        color: #8891a8 !important;
    }

    /* Вкладки */
    [data-testid="stTabs"] [role="tablist"] {
        background: #0a0c13;
        border-radius: 8px;
        padding: 3px;
        gap: 2px;
    }
    [data-testid="stTabs"] button[role="tab"] {
        background: transparent !important;
        color: #8891a8 !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        padding: 6px 16px !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #1e2235 !important;
        color: #e8eaf0 !important;
    }

    /* Divider */
    hr { border-color: #1e2235; }

    /* Убираем стандартную боковую панель */
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Загрузка данных ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_data.csv", parse_dates=["timestamp"])
    return df

df = load_data()

FUEL_COLS = [c for c in TARGET_COLS if c.startswith("sales_")]
SHOP_COLS = [c for c in TARGET_COLS if c.startswith("shop_")]
FUEL_LABELS = {
    "sales_AI92": "АИ-92", "sales_AI95": "АИ-95", "sales_AI98": "АИ-98",
    "sales_DT_EURO": "ДТ Евро+", "sales_DT_TANEKO": "ДТ ТАНЕКО",
    "sales_DT_SUMMER": "ДТ Летнее", "sales_DT_WINTER": "ДТ Зимнее",
}
SHOP_LABELS = {
    "shop_напитки": "Магазин: Напитки", "shop_закуски": "Магазин: Закуски",
    "shop_автотовары": "Магазин: Автотовары", "shop_кофе": "Магазин: Кофе",
    "shop_табак": "Магазин: Табак",
}
TARGET_LABELS = {**FUEL_LABELS, **SHOP_LABELS}
PLOTLY_THEME = dict(
    paper_bgcolor="#13161f",
    plot_bgcolor="#13161f",
    font_color="#e8eaf0",
    font_size=12,
)
GOLD = "#c8a84b"
GRID_COLOR = "#1e2235"

def styled_fig(fig):
    fig.update_layout(
        **PLOTLY_THEME,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(bgcolor="#13161f", bordercolor="#1e2235", borderwidth=1),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    )
    return fig

# ── Шапка ────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <span class="dash-logo">ТАТНЕФТЬ</span>
    <span class="dash-title">EDA Анализ</span>
    <span class="dash-sep">|</span>
    <span class="dash-sub">5 АЗС · Почасовые данные · 2023</span>
</div>
""", unsafe_allow_html=True)

# ── Глобальные фильтры ───────────────────────────────────────
stations = sorted(df["station_name"].unique())
date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()

with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([2, 3, 2])
    with fc1:
        selected_stations = st.multiselect(
            "Станции", stations, default=stations, placeholder="Все станции"
        )
    with fc2:
        selected_fuels = st.multiselect(
            "Виды топлива", list(FUEL_LABELS.values()),
            default=list(FUEL_LABELS.values()), placeholder="Все виды топлива"
        )
    with fc3:
        date_range = st.date_input(
            "Период", value=(date_min, date_max),
            min_value=date_min, max_value=date_max
        )
    st.markdown('</div>', unsafe_allow_html=True)

selected_fuel_cols = [k for k, v in FUEL_LABELS.items() if v in selected_fuels]
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

mask = (
    df["station_name"].isin(selected_stations if selected_stations else stations) &
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date)
)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("Нет данных для выбранных фильтров.")
    st.stop()

if selected_fuel_cols:
    fdf["total_sales"] = fdf[selected_fuel_cols].sum(axis=1)
else:
    fdf["total_sales"] = 0

# ── KPI карточки ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Всего продано", f"{fdf['total_sales'].sum():,.0f} л", "за выбранный период"),
    (k2, "Среднее в час", f"{fdf['total_sales'].mean():.1f} л", "по всем станциям"),
    (k3, "Записей", f"{len(fdf):,}", f"{(end_date - start_date).days + 1} дней"),
    (k4, "Станций", str(len(selected_stations) if selected_stations else len(stations)), "выбрано"),
    (k5, "Средний трафик", f"{fdf['total_traffic'].mean():.0f}", "авт/час"),
]
for col, label, val, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Вкладки ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Паттерны продаж",
    "🏪  Сравнение АЗС",
    "🌤️  Погода и трафик",
    "🎯  Акции и реклама",
    "🔗  Корреляции",
    "📊  Статистический анализ",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Паттерны продаж
# ══════════════════════════════════════════════════════════════
with tab1:
    if not selected_fuel_cols:
        st.info("Выберите хотя бы один вид топлива.")
    else:
        # Продажи по месяцам
        monthly = fdf.groupby("month")["total_sales"].mean().reset_index()
        fig_m = px.bar(monthly, x="month", y="total_sales",
                       labels={"month": "Месяц", "total_sales": "Ср. продажи, л/час"},
                       color_discrete_sequence=[GOLD])
        fig_m.update_layout(title="Средние продажи по месяцам", **PLOTLY_THEME,
                            margin=dict(l=8, r=8, t=36, b=8))
        fig_m.update_xaxes(gridcolor=GRID_COLOR, dtick=1)
        fig_m.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_m, width='stretch')

        col_l, col_r = st.columns(2)
        with col_l:
            hourly = fdf.groupby("hour")["total_sales"].mean().reset_index()
            fig_h = px.line(hourly, x="hour", y="total_sales", markers=True,
                            labels={"hour": "Час суток", "total_sales": "Ср. продажи, л/час"},
                            color_discrete_sequence=[GOLD])
            fig_h.update_layout(title="Продажи по часам суток", **PLOTLY_THEME,
                                margin=dict(l=8, r=8, t=36, b=8))
            fig_h.update_xaxes(gridcolor=GRID_COLOR, dtick=2)
            fig_h.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_h, width='stretch')

        with col_r:
            day_map = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
            fdf["day_label"] = fdf["day_of_week"].map(day_map)
            daily = fdf.groupby(["day_of_week", "day_label"])["total_sales"].mean().reset_index()
            daily = daily.sort_values("day_of_week")
            fig_d = px.bar(daily, x="day_label", y="total_sales",
                           labels={"day_label": "День недели", "total_sales": "Ср. продажи, л/час"},
                           color_discrete_sequence=[GOLD])
            fig_d.update_layout(title="Продажи по дням недели", **PLOTLY_THEME,
                                margin=dict(l=8, r=8, t=36, b=8))
            fig_d.update_xaxes(gridcolor=GRID_COLOR)
            fig_d.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_d, width='stretch')

        # Структура по сезонам
        season_map = {"winter": "Зима", "spring": "Весна", "summer": "Лето", "autumn": "Осень"}
        fdf["season_label"] = fdf["season"].map(season_map)
        melt = fdf.groupby("season_label")[selected_fuel_cols].mean().reset_index()
        melt = melt.melt(id_vars="season_label", var_name="fuel", value_name="sales")
        melt["fuel"] = melt["fuel"].map(FUEL_LABELS)
        fig_s = px.bar(melt, x="season_label", y="sales", color="fuel",
                       labels={"season_label": "Сезон", "sales": "Ср. продажи, л/час", "fuel": "Топливо"})
        fig_s.update_layout(title="Структура продаж по сезонам", **PLOTLY_THEME,
                            margin=dict(l=8, r=8, t=36, b=8))
        fig_s.update_xaxes(gridcolor=GRID_COLOR)
        fig_s.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_s, width='stretch')

        # Тепловая карта
        st.markdown("**Тепловая карта: час × день недели**")
        fuel_choice = st.selectbox("Топливо для тепловой карты",
                                   [FUEL_LABELS[c] for c in selected_fuel_cols], key="heat_fuel")
        heat_col = [k for k, v in FUEL_LABELS.items() if v == fuel_choice][0]
        pivot = fdf.pivot_table(index="hour", columns="day_of_week",
                                values=heat_col, aggfunc="mean")
        pivot.columns = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                             labels={"x": "День недели", "y": "Час", "color": "л/час"},
                             title=f"Средние продажи {fuel_choice}")
        fig_heat.update_layout(**PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
        st.plotly_chart(fig_heat, width='stretch')

# ══════════════════════════════════════════════════════════════
# TAB 2 — Сравнение АЗС
# ══════════════════════════════════════════════════════════════
with tab2:
    if not selected_fuel_cols:
        st.info("Выберите хотя бы один вид топлива.")
    else:
        station_total = fdf.groupby("station_name")["total_sales"].sum().reset_index()
        station_total = station_total.sort_values("total_sales", ascending=True)
        fig_st = px.bar(station_total, x="total_sales", y="station_name", orientation="h",
                        labels={"total_sales": "Суммарные продажи, л", "station_name": ""},
                        color_discrete_sequence=[GOLD])
        fig_st.update_layout(title="Суммарные продажи по станциям", **PLOTLY_THEME,
                             margin=dict(l=8, r=8, t=36, b=8))
        fig_st.update_xaxes(gridcolor=GRID_COLOR)
        fig_st.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_st, width='stretch')

        melt_st = fdf.groupby("station_name")[selected_fuel_cols].sum().reset_index()
        melt_st = melt_st.melt(id_vars="station_name", var_name="fuel", value_name="sales")
        melt_st["fuel"] = melt_st["fuel"].map(FUEL_LABELS)
        fig_st2 = px.bar(melt_st, x="station_name", y="sales", color="fuel", barmode="stack",
                         labels={"station_name": "Станция", "sales": "Продажи, л", "fuel": "Топливо"})
        fig_st2.update_layout(title="Структура топлива по АЗС", **PLOTLY_THEME,
                              margin=dict(l=8, r=8, t=36, b=8))
        fig_st2.update_xaxes(gridcolor=GRID_COLOR)
        fig_st2.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_st2, width='stretch')

        fdf["week"] = fdf["timestamp"].dt.isocalendar().week.astype(int)
        weekly = fdf.groupby(["week", "station_name"])["total_sales"].sum().reset_index()
        fig_dyn = px.line(weekly, x="week", y="total_sales", color="station_name",
                          labels={"week": "Неделя года", "total_sales": "Продажи, л", "station_name": "Станция"})
        fig_dyn.update_layout(title="Еженедельная динамика продаж по станциям", **PLOTLY_THEME,
                              margin=dict(l=8, r=8, t=36, b=8))
        fig_dyn.update_xaxes(gridcolor=GRID_COLOR)
        fig_dyn.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_dyn, width='stretch')

        st.divider()
        st.markdown("### Характеристики АЗС (metadata)")

        meta_df = fdf.groupby("station_name").agg(
            avg_sales=("total_sales", "mean"),
            road_type=("road_type", "first"),
            direction=("direction", "first"),
            settlement_size=("settlement_size", "first"),
            distance_to_city_km=("distance_to_city_km", "first"),
            total_pumps=("total_pumps", "first"),
            competitors_within_5km=("competitors_within_5km", "first"),
            customer_loyalty_score=("customer_loyalty_score", "first"),
            staff_quality_score=("staff_quality_score", "first"),
            corporate_customer_ratio=("corporate_customer_ratio", "first"),
            staff_engagement_score=("staff_engagement_score", "first"),
            has_car_wash=("has_car_wash", "first"),
            has_tire_service=("has_tire_service", "first"),
            has_cafe=("has_cafe", "first"),
            has_hotel=("has_hotel", "first"),
            has_shop=("has_shop", "first"),
        ).reset_index()

        col_l, col_r = st.columns(2)
        with col_l:
            if "road_type" in fdf.columns:
                rt = fdf.groupby("road_type")["total_sales"].mean().reset_index()
                rt = rt.sort_values("total_sales", ascending=True)
                fig_rt = px.bar(rt, x="total_sales", y="road_type", orientation="h",
                                labels={"total_sales": "Ср. продажи, л/час", "road_type": ""},
                                color_discrete_sequence=[GOLD])
                fig_rt.update_layout(title="Ср. продажи по типу дороги", **PLOTLY_THEME,
                                     margin=dict(l=8, r=8, t=36, b=8))
                fig_rt.update_xaxes(gridcolor=GRID_COLOR)
                fig_rt.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_rt, width='stretch')
        with col_r:
            if "settlement_size" in fdf.columns:
                ss = fdf.groupby("settlement_size")["total_sales"].mean().reset_index()
                ss = ss.sort_values("total_sales", ascending=True)
                fig_ss = px.bar(ss, x="total_sales", y="settlement_size", orientation="h",
                                labels={"total_sales": "Ср. продажи, л/час", "settlement_size": ""},
                                color_discrete_sequence=["#60a5fa"])
                fig_ss.update_layout(title="Ср. продажи по размеру населённого пункта",
                                     **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
                fig_ss.update_xaxes(gridcolor=GRID_COLOR)
                fig_ss.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_ss, width='stretch')

        col_l, col_r = st.columns(2)
        with col_l:
            if "direction" in fdf.columns:
                dr = fdf.groupby("direction")["total_sales"].mean().reset_index()
                dr = dr.sort_values("total_sales", ascending=True)
                fig_dr = px.bar(dr, x="total_sales", y="direction", orientation="h",
                                labels={"total_sales": "Ср. продажи, л/час", "direction": ""},
                                color_discrete_sequence=["#34d399"])
                fig_dr.update_layout(title="Ср. продажи по направлению дороги", **PLOTLY_THEME,
                                     margin=dict(l=8, r=8, t=36, b=8))
                fig_dr.update_xaxes(gridcolor=GRID_COLOR)
                fig_dr.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_dr, width='stretch')
        with col_r:
            if "distance_to_city_km" in meta_df.columns and "total_pumps" in meta_df.columns:
                fig_dist = px.scatter(meta_df, x="distance_to_city_km", y="avg_sales",
                                      text="station_name", size="total_pumps",
                                      labels={"distance_to_city_km": "Расстояние до города, км",
                                              "avg_sales": "Ср. продажи, л/час",
                                              "total_pumps": "Колонок"},
                                      color_discrete_sequence=[GOLD])
                fig_dist.update_traces(textposition="top center")
                fig_dist.update_layout(title="Расстояние до города vs Продажи (размер = кол-во колонок)",
                                       **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
                fig_dist.update_xaxes(gridcolor=GRID_COLOR)
                fig_dist.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_dist, width='stretch')

        col_l, col_r = st.columns(2)
        with col_l:
            if "competitors_within_5km" in meta_df.columns:
                fig_comp = px.bar(meta_df.sort_values("competitors_within_5km"),
                                  x="station_name", y="competitors_within_5km",
                                  labels={"station_name": "Станция",
                                          "competitors_within_5km": "Конкурентов"},
                                  color_discrete_sequence=["#f87171"])
                fig_comp.update_layout(title="Конкуренты в радиусе 5 км по станциям",
                                       **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
                fig_comp.update_xaxes(gridcolor=GRID_COLOR)
                fig_comp.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_comp, width='stretch')
        with col_r:
            score_cols = ["customer_loyalty_score", "staff_quality_score",
                          "staff_engagement_score", "corporate_customer_ratio"]
            score_labels_map = {
                "customer_loyalty_score": "Лояльность клиентов",
                "staff_quality_score": "Качество персонала",
                "staff_engagement_score": "Вовлечённость персонала",
                "corporate_customer_ratio": "Доля корп. клиентов",
            }
            avail_scores = [c for c in score_cols if c in meta_df.columns]
            if avail_scores:
                scores_melt = meta_df[["station_name"] + avail_scores].melt(
                    id_vars="station_name", var_name="metric", value_name="value"
                )
                scores_melt["metric"] = scores_melt["metric"].map(score_labels_map)
                fig_scores = px.bar(scores_melt, x="station_name", y="value",
                                    color="metric", barmode="group",
                                    labels={"station_name": "Станция",
                                            "value": "Значение", "metric": "Метрика"})
                fig_scores.update_layout(title="Качественные метрики АЗС", **PLOTLY_THEME,
                                         margin=dict(l=8, r=8, t=36, b=8))
                fig_scores.update_xaxes(gridcolor=GRID_COLOR)
                fig_scores.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_scores, width='stretch')

        service_cols_map = {
            "has_car_wash": "Автомойка", "has_tire_service": "Шиномонтаж",
            "has_cafe": "Кафе", "has_hotel": "Отель", "has_shop": "Магазин",
        }
        avail_services = [c for c in service_cols_map if c in meta_df.columns]
        if avail_services:
            svc_df = meta_df[["station_name"] + avail_services].set_index("station_name")
            svc_df = svc_df.rename(columns=service_cols_map)
            fig_svc = px.imshow(svc_df, text_auto=True, aspect="auto",
                                color_continuous_scale=["#1e2235", GOLD],
                                labels={"color": "Наличие"})
            fig_svc.update_layout(title="Услуги АЗС (1 — есть, 0 — нет)",
                                  **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8), height=220)
            st.plotly_chart(fig_svc, width='stretch')

# ══════════════════════════════════════════════════════════════
# TAB 3 — Погода и трафик
# ══════════════════════════════════════════════════════════════
with tab3:
    if not selected_fuel_cols:
        st.info("Выберите хотя бы один вид топлива.")
    else:
        col_l, col_r = st.columns(2)
        sample = fdf.sample(min(3000, len(fdf)), random_state=42)

        with col_l:
            fig_temp = px.scatter(sample, x="temperature", y="total_sales", opacity=0.35,
                                  trendline="lowess",
                                  labels={"temperature": "Температура, °C", "total_sales": "Продажи, л/час"},
                                  color_discrete_sequence=[GOLD])
            fig_temp.update_layout(title="Температура vs Продажи", **PLOTLY_THEME,
                                   margin=dict(l=8, r=8, t=36, b=8))
            fig_temp.update_xaxes(gridcolor=GRID_COLOR)
            fig_temp.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_temp, width='stretch')

        with col_r:
            fig_traf = px.scatter(sample, x="total_traffic", y="total_sales", opacity=0.35,
                                  trendline="lowess",
                                  labels={"total_traffic": "Трафик, авт/час", "total_sales": "Продажи, л/час"},
                                  color_discrete_sequence=["#60a5fa"])
            fig_traf.update_layout(title="Трафик vs Продажи", **PLOTLY_THEME,
                                   margin=dict(l=8, r=8, t=36, b=8))
            fig_traf.update_xaxes(gridcolor=GRID_COLOR)
            fig_traf.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_traf, width='stretch')

        weather_sales = fdf.groupby("weather_condition")["total_sales"].mean().reset_index()
        weather_sales = weather_sales.sort_values("total_sales", ascending=True)
        fig_w = px.bar(weather_sales, x="total_sales", y="weather_condition", orientation="h",
                       labels={"total_sales": "Ср. продажи, л/час", "weather_condition": ""},
                       color_discrete_sequence=[GOLD])
        fig_w.update_layout(title="Средние продажи по типу погоды", **PLOTLY_THEME,
                            margin=dict(l=8, r=8, t=36, b=8))
        fig_w.update_xaxes(gridcolor=GRID_COLOR)
        fig_w.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_w, width='stretch')

        traffic_cols = {
            "traffic_Passengers_cars": "Легковые",
            "traffic_Truck_short": "Грузовые малые",
            "traffic_Truck": "Грузовые",
            "traffic_Truck_long": "Фуры",
            "traffic_Transporter": "Микроавтобусы",
            "traffic_Undefined": "Неопределённые",
        }
        tc = {v: fdf[k].sum() for k, v in traffic_cols.items() if k in fdf.columns}
        fig_pie = px.pie(values=list(tc.values()), names=list(tc.keys()),
                         color_discrete_sequence=px.colors.sequential.Oranges_r)
        fig_pie.update_layout(title="Состав трафика", **PLOTLY_THEME,
                              margin=dict(l=8, r=8, t=36, b=8))
        st.plotly_chart(fig_pie, width='stretch')

        # Осадки, видимость, ветер
        col_l, col_r, col_m = st.columns(3)
        with col_l:
            fig_prec = px.scatter(sample, x="precipitation_mm", y="total_sales", opacity=0.35,
                                  trendline="lowess",
                                  labels={"precipitation_mm": "Осадки, мм", "total_sales": "Продажи, л/час"},
                                  color_discrete_sequence=[GOLD])
            fig_prec.update_layout(title="Осадки vs Продажи", **PLOTLY_THEME,
                                   margin=dict(l=8, r=8, t=36, b=8))
            fig_prec.update_xaxes(gridcolor=GRID_COLOR)
            fig_prec.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_prec, width='stretch')
        with col_r:
            fig_vis = px.scatter(sample, x="visibility_km", y="total_sales", opacity=0.35,
                                 trendline="lowess",
                                 labels={"visibility_km": "Видимость, км", "total_sales": "Продажи, л/час"},
                                 color_discrete_sequence=["#60a5fa"])
            fig_vis.update_layout(title="Видимость vs Продажи", **PLOTLY_THEME,
                                  margin=dict(l=8, r=8, t=36, b=8))
            fig_vis.update_xaxes(gridcolor=GRID_COLOR)
            fig_vis.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_vis, width='stretch')
        with col_m:
            fig_wind = px.scatter(sample, x="wind_speed_ms", y="total_sales", opacity=0.35,
                                  trendline="lowess",
                                  labels={"wind_speed_ms": "Ветер, м/с", "total_sales": "Продажи, л/час"},
                                  color_discrete_sequence=["#34d399"])
            fig_wind.update_layout(title="Ветер vs Продажи", **PLOTLY_THEME,
                                   margin=dict(l=8, r=8, t=36, b=8))
            fig_wind.update_xaxes(gridcolor=GRID_COLOR)
            fig_wind.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_wind, width='stretch')

        # Осадки: дождь, снег, туман
        weather_flags = {"is_rain": "Дождь", "is_snow": "Снег", "is_fog": "Туман"}
        flag_rows = []
        for col_f, label_f in weather_flags.items():
            if col_f in fdf.columns:
                for val, name in [(0, f"Без: {label_f}"), (1, label_f)]:
                    avg = fdf[fdf[col_f] == val]["total_sales"].mean()
                    flag_rows.append({"Условие": name, "Продажи": avg, "Тип": label_f})
        if flag_rows:
            flag_df = pd.DataFrame(flag_rows)
            fig_flags = px.bar(flag_df, x="Условие", y="Продажи", color="Тип",
                               labels={"Продажи": "Ср. продажи, л/час"},
                               color_discrete_sequence=[GOLD, "#60a5fa", "#34d399"])
            fig_flags.update_layout(title="Влияние дождя / снега / тумана на продажи",
                                    **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_flags.update_xaxes(gridcolor=GRID_COLOR)
            fig_flags.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_flags, width='stretch')

# ══════════════════════════════════════════════════════════════
# TAB 4 — Акции и реклама
# ══════════════════════════════════════════════════════════════
with tab4:
    if not selected_fuel_cols:
        st.info("Выберите хотя бы один вид топлива.")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            pf = fdf.groupby("promotion_fuel_active")["total_sales"].mean().reset_index()
            pf["Акция"] = pf["promotion_fuel_active"].map({0: "Без акции", 1: "С акцией"})
            fig_pf = px.bar(pf, x="Акция", y="total_sales",
                            labels={"total_sales": "Ср. продажи, л/час"},
                            color="Акция",
                            color_discrete_map={"Без акции": "#1e2235", "С акцией": GOLD})
            fig_pf.update_layout(title="Акция на топливо", showlegend=False,
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_pf.update_xaxes(gridcolor=GRID_COLOR)
            fig_pf.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_pf, width='stretch')

        with col_r:
            pa = fdf.groupby("ad_active")["total_sales"].mean().reset_index()
            pa["Реклама"] = pa["ad_active"].map({0: "Без рекламы", 1: "С рекламой"})
            fig_pa = px.bar(pa, x="Реклама", y="total_sales",
                            labels={"total_sales": "Ср. продажи, л/час"},
                            color="Реклама",
                            color_discrete_map={"Без рекламы": "#1e2235", "С рекламой": "#60a5fa"})
            fig_pa.update_layout(title="Реклама", showlegend=False,
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_pa.update_xaxes(gridcolor=GRID_COLOR)
            fig_pa.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_pa, width='stretch')

        ch = fdf.groupby("ad_channel")["total_sales"].mean().reset_index()
        ch = ch.sort_values("total_sales", ascending=True)
        fig_ch = px.bar(ch, x="total_sales", y="ad_channel", orientation="h",
                        labels={"total_sales": "Ср. продажи, л/час", "ad_channel": ""},
                        color_discrete_sequence=[GOLD])
        fig_ch.update_layout(title="Средние продажи по каналу рекламы", **PLOTLY_THEME,
                             margin=dict(l=8, r=8, t=36, b=8))
        fig_ch.update_xaxes(gridcolor=GRID_COLOR)
        fig_ch.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_ch, width='stretch')

        hol = fdf.groupby("is_holiday")["total_sales"].mean().reset_index()
        hol["День"] = hol["is_holiday"].map({0: "Обычный день", 1: "Праздник"})
        fig_hol = px.bar(hol, x="День", y="total_sales",
                         labels={"total_sales": "Ср. продажи, л/час"},
                         color="День",
                         color_discrete_map={"Обычный день": "#1e2235", "Праздник": "#34d399"})
        fig_hol.update_layout(title="Праздники vs обычные дни", showlegend=False,
                              **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
        fig_hol.update_xaxes(gridcolor=GRID_COLOR)
        fig_hol.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_hol, width='stretch')

        # Акции магазина и кафе
        col_l, col_r = st.columns(2)
        with col_l:
            ps = fdf.groupby("promotion_shop_active")["total_sales"].mean().reset_index()
            ps["Акция"] = ps["promotion_shop_active"].map({0: "Без акции", 1: "С акцией"})
            fig_ps = px.bar(ps, x="Акция", y="total_sales",
                            labels={"total_sales": "Ср. продажи, л/час"},
                            color="Акция",
                            color_discrete_map={"Без акции": "#1e2235", "С акцией": "#34d399"})
            fig_ps.update_layout(title="Акция на магазин", showlegend=False,
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_ps.update_xaxes(gridcolor=GRID_COLOR)
            fig_ps.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_ps, width='stretch')
        with col_r:
            pc = fdf.groupby("promotion_cafe_active")["total_sales"].mean().reset_index()
            pc["Акция"] = pc["promotion_cafe_active"].map({0: "Без акции", 1: "С акцией"})
            fig_pc = px.bar(pc, x="Акция", y="total_sales",
                            labels={"total_sales": "Ср. продажи, л/час"},
                            color="Акция",
                            color_discrete_map={"Без акции": "#1e2235", "С акцией": "#60a5fa"})
            fig_pc.update_layout(title="Акция на кафе", showlegend=False,
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_pc.update_xaxes(gridcolor=GRID_COLOR)
            fig_pc.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_pc, width='stretch')

        # Пиковые часы, ночные часы, выходные
        col_l, col_r, col_m = st.columns(3)
        with col_l:
            rh = fdf.groupby("is_rush_hour")["total_sales"].mean().reset_index()
            rh["Час"] = rh["is_rush_hour"].map({0: "Обычный час", 1: "Пиковый час"})
            fig_rh = px.bar(rh, x="Час", y="total_sales",
                            labels={"total_sales": "Ср. продажи, л/час"},
                            color="Час",
                            color_discrete_map={"Обычный час": "#1e2235", "Пиковый час": GOLD})
            fig_rh.update_layout(title="Пиковые часы vs обычные", showlegend=False,
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_rh.update_xaxes(gridcolor=GRID_COLOR)
            fig_rh.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_rh, width='stretch')
        with col_r:
            nt = fdf.groupby("is_night")["total_sales"].mean().reset_index()
            nt["Время"] = nt["is_night"].map({0: "День", 1: "Ночь"})
            fig_nt = px.bar(nt, x="Время", y="total_sales",
                            labels={"total_sales": "Ср. продажи, л/час"},
                            color="Время",
                            color_discrete_map={"День": "#1e2235", "Ночь": "#8b5cf6"})
            fig_nt.update_layout(title="День vs Ночь", showlegend=False,
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_nt.update_xaxes(gridcolor=GRID_COLOR)
            fig_nt.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_nt, width='stretch')
        with col_m:
            we = fdf.groupby("is_weekend")["total_sales"].mean().reset_index()
            we["День"] = we["is_weekend"].map({0: "Будни", 1: "Выходные"})
            fig_we = px.bar(we, x="День", y="total_sales",
                            labels={"total_sales": "Ср. продажи, л/час"},
                            color="День",
                            color_discrete_map={"Будни": "#1e2235", "Выходные": "#f59e0b"})
            fig_we.update_layout(title="Будни vs Выходные", showlegend=False,
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_we.update_xaxes(gridcolor=GRID_COLOR)
            fig_we.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_we, width='stretch')

        # Продажи по конкретным праздникам
        if "holiday_name" in fdf.columns:
            hol_named = fdf[fdf["is_holiday"] == 1].groupby("holiday_name")["total_sales"].mean().reset_index()
            hol_named = hol_named.sort_values("total_sales", ascending=True)
            if not hol_named.empty:
                fig_hn = px.bar(hol_named, x="total_sales", y="holiday_name", orientation="h",
                                labels={"total_sales": "Ср. продажи, л/час", "holiday_name": ""},
                                color_discrete_sequence=[GOLD])
                fig_hn.update_layout(title="Средние продажи по праздникам",
                                     **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
                fig_hn.update_xaxes(gridcolor=GRID_COLOR)
                fig_hn.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_hn, width='stretch')

# ══════════════════════════════════════════════════════════════
# TAB 5 — Корреляции
# ══════════════════════════════════════════════════════════════
with tab5:
    if not selected_fuel_cols:
        st.info("Выберите хотя бы один вид топлива.")
    else:
        corr_cols = selected_fuel_cols + [
            "temperature", "precipitation_mm", "wind_speed_ms", "visibility_km",
            "total_traffic", "shop_total_revenue",
            "promotion_fuel_active", "promotion_shop_active", "promotion_cafe_active",
            "ad_active", "is_holiday", "is_weekend", "is_rush_hour", "is_night",
            "competitor_price_AI92", "competitor_price_AI95", "competitor_price_DT",
        ]
        corr_cols = [c for c in corr_cols if c in fdf.columns]
        labels = {**FUEL_LABELS, **{
            "temperature": "Температура", "precipitation_mm": "Осадки",
            "wind_speed_ms": "Ветер", "visibility_km": "Видимость",
            "total_traffic": "Трафик", "shop_total_revenue": "Выручка магазина",
            "promotion_fuel_active": "Акция топливо",
            "promotion_shop_active": "Акция магазин",
            "promotion_cafe_active": "Акция кафе",
            "ad_active": "Реклама", "is_holiday": "Праздник",
            "is_weekend": "Выходной", "is_rush_hour": "Пиковый час",
            "is_night": "Ночь",
            "competitor_price_AI92": "Конк. АИ-92",
            "competitor_price_AI95": "Конк. АИ-95",
            "competitor_price_DT": "Конк. ДТ",
        }}
        corr = fdf[corr_cols].corr().round(2)
        corr.index = [labels.get(c, c) for c in corr.index]
        corr.columns = [labels.get(c, c) for c in corr.columns]

        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_corr.update_layout(title="Матрица корреляций", **PLOTLY_THEME,
                               margin=dict(l=8, r=8, t=36, b=8), height=580)
        st.plotly_chart(fig_corr, width='stretch')

        fdf["total_sales"] = fdf[selected_fuel_cols].sum(axis=1)
        factor_cols = [c for c in corr_cols if c not in selected_fuel_cols]
        top = fdf[factor_cols + ["total_sales"]].corr()["total_sales"].drop("total_sales")
        top = top.abs().sort_values(ascending=True)
        top.index = [labels.get(c, c) for c in top.index]

        fig_top = px.bar(top.reset_index(), x="total_sales", y="index", orientation="h",
                         labels={"total_sales": "|Корреляция| с продажами", "index": ""},
                         color_discrete_sequence=[GOLD])
        fig_top.update_layout(title="Факторы по силе влияния на продажи", **PLOTLY_THEME,
                              margin=dict(l=8, r=8, t=36, b=8))
        fig_top.update_xaxes(gridcolor=GRID_COLOR)
        fig_top.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_top, width='stretch')

# ══════════════════════════════════════════════════════════════
# TAB 6 — Статистический анализ
# ══════════════════════════════════════════════════════════════
with tab6:
    if not selected_fuel_cols:
        st.info("Выберите хотя бы один вид топлива.")
    else:
        # ── БЛОК 1: Распределения до/после log-transform ──────
        st.markdown("### Блок 1 — Распределения целевых переменных")
        target_options = [c for c in selected_fuel_cols + SHOP_COLS if c in fdf.columns]
        target_choice = st.selectbox(
            "Целевая переменная",
            [TARGET_LABELS[c] for c in target_options],
            key="dist_fuel"
        )
        sel_col = [k for k, v in TARGET_LABELS.items() if v == target_choice][0]
        unit_label = "л/час" if sel_col.startswith("sales_") else "руб/час"

        skew_before = fdf[sel_col].skew()
        log_vals    = np.log1p(fdf[sel_col])
        skew_after  = log_vals.skew()

        col_l, col_r = st.columns(2)
        with col_l:
            fig_hist_b = px.histogram(fdf, x=sel_col, nbins=60,
                                      labels={sel_col: f"Значение, {unit_label}"},
                                      color_discrete_sequence=[GOLD])
            fig_hist_b.update_layout(
                title=f"До log-transform | skew = {skew_before:.3f}",
                **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8)
            )
            fig_hist_b.update_xaxes(gridcolor=GRID_COLOR)
            fig_hist_b.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_hist_b, width='stretch')

        with col_r:
            fig_hist_a = px.histogram(x=log_vals, nbins=60,
                                      labels={"x": f"log1p({target_choice})"},
                                      color_discrete_sequence=["#60a5fa"])
            fig_hist_a.update_layout(
                title=f"После log-transform | skew = {skew_after:.3f}",
                **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8)
            )
            fig_hist_a.update_xaxes(gridcolor=GRID_COLOR, title=f"log1p({target_choice})")
            fig_hist_a.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_hist_a, width='stretch')

        st.divider()

        # ── БЛОК 2: Расширенная корреляционная матрица ────────
        st.markdown("### Блок 2 — Корреляционная матрица")
        group_options = {
            "Трафик":  ["traffic_Passengers_cars","traffic_Truck_short","traffic_Truck",
                        "traffic_Truck_long","traffic_Transporter","traffic_Undefined","total_traffic"],
            "Погода":  ["temperature","precipitation_mm","wind_speed_ms","visibility_km",
                        "is_rain","is_snow","is_fog"],
            "Цены":    ["price_AI92","price_AI95","price_AI98",
                        "price_DT_EURO","price_DT_TANEKO","price_DT_SUMMER","price_DT_WINTER",
                        "competitor_price_AI92","competitor_price_AI95","competitor_price_DT"],
            "Акции":   ["promotion_fuel_active","promotion_shop_active","promotion_cafe_active",
                        "ad_active","is_holiday","is_weekend","is_rush_hour","is_night"],
            "Магазин": ["shop_напитки","shop_закуски","shop_автотовары",
                        "shop_кофе","shop_табак","shop_total_revenue"],
        }
        chosen_group = st.selectbox("Группа факторов", list(group_options.keys()), key="corr_group")
        factor_set = [c for c in group_options[chosen_group] if c in fdf.columns]
        corr2_cols = selected_fuel_cols + factor_set
        corr2_cols = [c for c in corr2_cols if c in fdf.columns]

        if corr2_cols:
            all_labels = {**TARGET_LABELS, **{
                "shop_total_revenue": "Выручка магазина (итого)",
                "traffic_Passengers_cars": "Легковые", "traffic_Truck_short": "Груз. малые",
                "traffic_Truck": "Грузовые", "traffic_Truck_long": "Фуры",
                "traffic_Transporter": "Микроавтобусы", "traffic_Undefined": "Неопред.",
                "total_traffic": "Трафик",
                "temperature": "Температура", "precipitation_mm": "Осадки",
                "wind_speed_ms": "Ветер", "visibility_km": "Видимость",
                "is_rain": "Дождь", "is_snow": "Снег", "is_fog": "Туман",
                "price_AI92": "Цена АИ-92", "price_AI95": "Цена АИ-95",
                "price_AI98": "Цена АИ-98",
                "price_DT_EURO": "Цена ДТ Евро+",
                "price_DT_TANEKO": "Цена ДТ ТАНЕКО",
                "price_DT_SUMMER": "Цена ДТ Лето",
                "price_DT_WINTER": "Цена ДТ Зима",
                "competitor_price_AI92": "Конк. АИ-92",
                "competitor_price_AI95": "Конк. АИ-95",
                "competitor_price_DT": "Конк. ДТ",
                "promotion_fuel_active": "Акция топливо",
                "promotion_shop_active": "Акция магазин",
                "promotion_cafe_active": "Акция кафе",
                "ad_active": "Реклама",
                "is_holiday": "Праздник", "is_weekend": "Выходной",
                "is_rush_hour": "Пиковый час", "is_night": "Ночь",
            }}
            corr2 = fdf[corr2_cols].corr().round(2)
            corr2.index   = [all_labels.get(c, c) for c in corr2.index]
            corr2.columns = [all_labels.get(c, c) for c in corr2.columns]
            fig_c2 = px.imshow(corr2, text_auto=True, aspect="auto",
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_c2.update_layout(title=f"Корреляции: топливо × {chosen_group}",
                                 **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8), height=500)
            st.plotly_chart(fig_c2, width='stretch')

            # Топ-5 для выбранного топлива
            if selected_fuel_cols:
                top_fuel = st.selectbox("Топливо для топ-5", [FUEL_LABELS[c] for c in selected_fuel_cols], key="top5_fuel")
                top_fc = [k for k, v in FUEL_LABELS.items() if v == top_fuel][0]
                if top_fc in fdf.columns:
                    top5 = fdf[factor_set + [top_fc]].corr()[top_fc].drop(top_fc).abs()
                    top5 = top5.sort_values(ascending=False).head(5).reset_index()
                    top5.columns = ["Фактор", "|Корреляция|"]
                    top5["Фактор"] = top5["Фактор"].map(lambda x: all_labels.get(x, x))
                    st.markdown(f"**Топ-5 факторов, влияющих на {top_fuel}:**")
                    st.dataframe(top5, width='content', hide_index=True)

        st.divider()

        # ── БЛОК 3: Анализ выбросов ───────────────────────────
        st.markdown("### Блок 3 — Анализ выбросов")
        box_options = selected_fuel_cols + SHOP_COLS + [
            "temperature", "precipitation_mm", "wind_speed_ms", "visibility_km",
            "total_traffic", "shop_total_revenue",
            "competitor_price_AI92", "competitor_price_AI95", "competitor_price_DT",
        ]
        box_options = [c for c in box_options if c in fdf.columns]
        box_labels  = {**TARGET_LABELS, **{
            "temperature": "Температура", "precipitation_mm": "Осадки",
            "wind_speed_ms": "Ветер", "visibility_km": "Видимость",
            "total_traffic": "Трафик", "shop_total_revenue": "Выручка магазина (итого)",
            "competitor_price_AI92": "Конк. цена АИ-92",
            "competitor_price_AI95": "Конк. цена АИ-95",
            "competitor_price_DT": "Конк. цена ДТ",
        }}
        box_choice = st.selectbox("Переменная", [box_labels.get(c, c) for c in box_options], key="box_var")
        box_col    = [c for c in box_options if box_labels.get(c, c) == box_choice][0]

        # Вычисляем трансформацию inline (merged_data — сырые данные без _orig)
        if box_col in LOG_COLS:
            transformed_vals = np.log1p(fdf[box_col].clip(lower=0))
            proc_title = f"{box_choice} — после log1p"
            proc_ylabel = f"log1p({box_choice})"
        else:
            _q1 = fdf[box_col].quantile(0.25)
            _q3 = fdf[box_col].quantile(0.75)
            _iqr = _q3 - _q1
            transformed_vals = fdf[box_col].clip(_q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr)
            proc_title = f"{box_choice} — после winsorization (IQR)"
            proc_ylabel = f"{box_choice} (clip)"

        col_l, col_r = st.columns(2)
        with col_l:
            fig_box_raw = px.box(fdf, x="station_name", y=box_col,
                                 labels={"station_name": "Станция", box_col: "Значение"},
                                 color_discrete_sequence=[GOLD])
            fig_box_raw.update_layout(title=f"{box_choice} — исходные данные",
                                      **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_box_raw.update_xaxes(gridcolor=GRID_COLOR)
            fig_box_raw.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_box_raw, width='stretch')
        with col_r:
            _tmp = fdf[["station_name"]].copy()
            _tmp["_val"] = transformed_vals.values
            fig_box_proc = px.box(_tmp, x="station_name", y="_val",
                                  labels={"station_name": "Станция", "_val": proc_ylabel},
                                  color_discrete_sequence=["#60a5fa"])
            fig_box_proc.update_layout(title=proc_title,
                                       **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_box_proc.update_xaxes(gridcolor=GRID_COLOR)
            fig_box_proc.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_box_proc, width='stretch')

        # Таблица выбросов (df — merged_data, оригинальные значения)
        # STATIC_REALS исключены: у них одно значение на станцию —
        # IQR-метод ошибочно маркировал паспортные данные как выбросы.
        num_c  = df.select_dtypes(include=["number"]).columns
        bin_c  = [c for c in num_c if df[c].dropna().isin([0,1]).all()]
        static_c = set(STATIC_REALS)
        int_c  = [c for c in num_c if c not in bin_c and c not in static_c]
        out_rows = []
        for c in int_c:
            Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            n = ((df[c] < lo) | (df[c] > hi)).sum()
            if n > 0:
                out_rows.append({"Колонка": c, "Выбросов": n,
                                 "IQR нижняя": round(lo, 2), "IQR верхняя": round(hi, 2)})
        if out_rows:
            st.markdown("**Таблица выбросов по всем числовым колонкам:**")
            st.dataframe(pd.DataFrame(out_rows), width='content', hide_index=True)

        st.divider()

        # ── БЛОК 4: Violin plot по станциям ──────────────────
        st.markdown("### Блок 4 — Распределение продаж по станциям (Violin)")
        vio_fuel = st.selectbox("Топливо", [FUEL_LABELS[c] for c in selected_fuel_cols], key="vio_fuel")
        vio_col  = [k for k, v in FUEL_LABELS.items() if v == vio_fuel][0]
        vio_orig = vio_col + "_orig" if vio_col + "_orig" in fdf.columns else vio_col

        fig_vio = px.violin(fdf, x="station_name", y=vio_orig, box=True,
                            labels={"station_name": "Станция", vio_orig: "Продажи, л/час"},
                            color_discrete_sequence=[GOLD])
        fig_vio.update_layout(title=f"Распределение {vio_fuel} по станциям",
                              **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
        fig_vio.update_xaxes(gridcolor=GRID_COLOR)
        fig_vio.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_vio, width='stretch')

        st.divider()

        # ── БЛОК 5: Временные паттерны всех топлив ───────────
        st.markdown("### Блок 5 — Временные паттерны")
        pat_choice = st.radio("Разрез", ["По часам", "По дням недели", "По месяцам"],
                              horizontal=True, key="pat_choice")
        day_map2 = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}

        if pat_choice == "По часам":
            grp = fdf.groupby("hour")[selected_fuel_cols].mean().reset_index()
            grp = grp.melt(id_vars="hour", var_name="fuel", value_name="sales")
            grp["fuel"] = grp["fuel"].map(FUEL_LABELS)
            fig_p = px.line(grp, x="hour", y="sales", color="fuel", markers=True,
                            labels={"hour": "Час суток", "sales": "Ср. продажи, л/час", "fuel": "Топливо"})
            fig_p.update_layout(title="Средние продажи по часу суток",
                                **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_p.update_xaxes(gridcolor=GRID_COLOR, dtick=2)
        elif pat_choice == "По дням недели":
            fdf["day_label2"] = fdf["day_of_week"].map(day_map2)
            grp = fdf.groupby(["day_of_week", "day_label2"])[selected_fuel_cols].mean().reset_index()
            grp = grp.sort_values("day_of_week")
            grp = grp.melt(id_vars=["day_of_week", "day_label2"], var_name="fuel", value_name="sales")
            grp["fuel"] = grp["fuel"].map(FUEL_LABELS)
            fig_p = px.bar(grp, x="day_label2", y="sales", color="fuel", barmode="group",
                           labels={"day_label2": "День недели", "sales": "Ср. продажи, л/час", "fuel": "Топливо"})
            fig_p.update_layout(title="Средние продажи по дням недели",
                                **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_p.update_xaxes(gridcolor=GRID_COLOR)
        else:
            grp = fdf.groupby("month")[selected_fuel_cols].mean().reset_index()
            grp = grp.melt(id_vars="month", var_name="fuel", value_name="sales")
            grp["fuel"] = grp["fuel"].map(FUEL_LABELS)
            fig_p = px.line(grp, x="month", y="sales", color="fuel", markers=True,
                            labels={"month": "Месяц", "sales": "Ср. продажи, л/час", "fuel": "Топливо"})
            fig_p.update_layout(title="Сезонность продаж по месяцам",
                                **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_p.update_xaxes(gridcolor=GRID_COLOR, dtick=1)
        fig_p.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_p, width='stretch')

        st.divider()

        # ── БЛОК 6: Выручка магазина по категориям ───────────
        st.markdown("### Блок 6 — Магазин: структура выручки")
        shop_cats = {
            "shop_напитки": "Напитки",
            "shop_закуски": "Закуски",
            "shop_автотовары": "Автотовары",
            "shop_кофе": "Кофе",
            "shop_табак": "Табак",
        }
        shop_avail = {k: v for k, v in shop_cats.items() if k in fdf.columns}
        if shop_avail:
            col_l, col_r = st.columns(2)
            with col_l:
                shop_means = {v: fdf[k].mean() for k, v in shop_avail.items()}
                fig_shop_pie = px.pie(
                    values=list(shop_means.values()),
                    names=list(shop_means.keys()),
                    color_discrete_sequence=px.colors.sequential.Oranges_r,
                )
                fig_shop_pie.update_layout(title="Доля категорий в выручке магазина",
                                           **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
                st.plotly_chart(fig_shop_pie, width='stretch')
            with col_r:
                shop_by_station = fdf.groupby("station_name")[list(shop_avail.keys())].mean().reset_index()
                shop_by_station = shop_by_station.melt(
                    id_vars="station_name", var_name="category", value_name="revenue"
                )
                shop_by_station["category"] = shop_by_station["category"].map(shop_cats)
                fig_shop_st = px.bar(shop_by_station, x="station_name", y="revenue",
                                     color="category", barmode="stack",
                                     labels={"station_name": "Станция",
                                             "revenue": "Ср. выручка, руб/час",
                                             "category": "Категория"})
                fig_shop_st.update_layout(title="Выручка магазина по категориям и станциям",
                                          **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
                fig_shop_st.update_xaxes(gridcolor=GRID_COLOR)
                fig_shop_st.update_yaxes(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_shop_st, width='stretch')

            # Динамика по часам суток
            shop_hourly = fdf.groupby("hour")[list(shop_avail.keys())].mean().reset_index()
            shop_hourly = shop_hourly.melt(id_vars="hour", var_name="category", value_name="revenue")
            shop_hourly["category"] = shop_hourly["category"].map(shop_cats)
            fig_shop_h = px.line(shop_hourly, x="hour", y="revenue", color="category", markers=False,
                                 labels={"hour": "Час суток",
                                         "revenue": "Ср. выручка, руб/час",
                                         "category": "Категория"})
            fig_shop_h.update_layout(title="Выручка магазина по категориям в течение суток",
                                     **PLOTLY_THEME, margin=dict(l=8, r=8, t=36, b=8))
            fig_shop_h.update_xaxes(gridcolor=GRID_COLOR, dtick=2)
            fig_shop_h.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_shop_h, width='stretch')
