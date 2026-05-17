"""
Дашборд интерпретации TFT-модели — важность переменных и временное внимание.
Запускать из корня проекта: streamlit run dashboard/tft_interpretation.py
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
from utils.data_utils import ENCODER_LENGTH, PREDICTION_LENGTH, TARGET_COLS, TEST_START

st.set_page_config(
    page_title="Интерпретация TFT — АЗС Татнефть",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TARGET_LABELS = {
    "sales_AI92": "АИ-92", "sales_AI95": "АИ-95", "sales_AI98": "АИ-98",
    "sales_DT_EURO": "ДТ Евро+", "sales_DT_TANEKO": "ДТ ТАНЕКО",
    "sales_DT_SUMMER": "ДТ Летнее", "sales_DT_WINTER": "ДТ Зимнее",
    "shop_напитки": "Напитки", "shop_закуски": "Закуски",
    "shop_автотовары": "Автотовары", "shop_кофе": "Кофе", "shop_табак": "Табак",
}


@st.cache_data
def load_prepared():
    df = pd.read_csv("data/prepared_data.csv", parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    return df


# Названия станций из 5stations_metadata.csv (station_id → читаемое имя)
STATION_LABELS = {
    "0": "Татнефть-АЗС-001 (трасса М7)",
    "1": "Татнефть-АЗС-002 (регион. дорога)",
    "2": "Татнефть-АЗС-003 (трасса М7)",
    "3": "Татнефть-АЗС-004 (трасса М7)",
    "4": "Татнефть-АЗС-005 (регион. дорога)",
}


def load_station_names():
    return STATION_LABELS


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


# ── Заголовок ─────────────────────────────────────────────────
st.title("🧠 Интерпретация TFT-модели")
st.caption("АЗС Татнефть | Декабрь 2023")

st.markdown("""
**Temporal Fusion Transformer (TFT)** — это не «чёрный ящик».
В отличие от большинства нейросетей, TFT специально спроектирован для интерпретируемости:

| Механизм | Что показывает |
|---|---|
| **Variable Selection Network (VSN)** | Какие переменные модель считает важными |
| **Temporal Self-Attention** | На какие прошлые моменты модель смотрит при прогнозировании |
| **Static context vectors** | Как характеристики АЗС влияют на всю логику прогноза |

Выберите период и станцию ниже — визуализация пересчитается автоматически.
""")

st.divider()

# ── Загрузка модели ───────────────────────────────────────────
st.info("Загрузка модели и вычисление интерпретации занимает ~30–60 секунд при первом открытии.")

with st.spinner("Загрузка TFT-модели..."):
    model, training, tft_config, err = load_tft()

if err or model is None:
    st.error(f"Не удалось загрузить модель: {err}")
    st.stop()

n_params = sum(p.numel() for p in model.parameters())
st.success(f"Модель загружена. Параметров: {n_params:,}")

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Encoder (ретроспектива)", f"{ENCODER_LENGTH} ч = 7 суток")
col_info2.metric("Horizon (прогноз)", f"{PREDICTION_LENGTH} ч = 1 сутки")
col_info3.metric("Целевых переменных", len(TARGET_COLS))

st.divider()

# ── Данные и настройки интерпретации ──────────────────────────
prepared_df = load_prepared()

# Предопределённые периоды: ключ → (начало периода прогноза, конец)
# Контекст (168 ч энкодера) добавляется автоматически перед началом периода
PERIODS = {
    "Декабрь 2023 — тест":          ("2023-12-01", "2023-12-10"),
    "Ноябрь 2023 — валидация":      ("2023-11-01", "2023-11-30"),
    "Q4 2023 (октябрь–декабрь)":    ("2023-10-01", "2023-12-31"),
    "Q3 2023 (июль–сентябрь)":      ("2023-07-01", "2023-09-30"),
    "Q2 2023 (апрель–июнь)":        ("2023-04-01", "2023-06-30"),
    "Q1 2023 (январь–март)":        ("2023-01-08", "2023-03-31"),  # 8 янв — первый полный контекст
    "Весь 2023 год":                 ("2023-01-08", "2023-12-31"),
}

st.subheader("Настройки интерпретации")
st.caption(
    "VSN-веса — обученные параметры модели, поэтому важность переменных примерно одинакова "
    "на любом периоде. Паттерн внимания (Temporal Self-Attention) может отличаться по сезонам."
)

station_names = load_station_names()
all_stations = sorted(prepared_df["station_id"].unique())

col_p, col_s = st.columns(2)
with col_p:
    sel_period = st.selectbox("Период для интерпретации", list(PERIODS.keys()), index=0)
with col_s:
    sel_station_interp = st.selectbox(
        "Станция",
        ["Все"] + all_stations,
        format_func=lambda x: "Все станции" if x == "Все" else station_names.get(x, f"АЗС {x}"),
    )

period_start, period_end = PERIODS[sel_period]
# Контекст = 7 суток до начала периода
context_start = pd.Timestamp(period_start) - pd.Timedelta(hours=ENCODER_LENGTH)


@st.cache_data(show_spinner=False)
def compute_interpretation(_model, _training, tft_config, period_start, period_end,
                           context_start_str, station):
    """Кешируется по периоду и станции; _model/_training не хешируются (prefix _)."""
    from pytorch_forecasting import TimeSeriesDataSet

    df = load_prepared()
    interp_df = df[
        (df["timestamp"] >= pd.Timestamp(context_start_str))
        & (df["timestamp"] <= pd.Timestamp(period_end + " 23:00:00"))
    ].copy()
    if station != "Все":
        interp_df = interp_df[interp_df["station_id"] == station]

    for col in tft_config["static_cats"] + tft_config["known_cats"]:
        if col in interp_df.columns:
            interp_df[col] = interp_df[col].astype(str)

    ds = TimeSeriesDataSet.from_dataset(_training, interp_df, stop_randomization=True)
    loader = ds.to_dataloader(train=False, batch_size=32, num_workers=0)

    raw = _model.predict(
        loader, mode="raw", return_index=True,
        trainer_kwargs={"logger": False, "enable_progress_bar": False},
    )
    out = raw.output if hasattr(raw, "output") else raw[0]
    interp = _model.interpret_output(out, reduction="mean")
    # Конвертируем тензоры в numpy для сериализации кеша
    return {k: v.detach().cpu().numpy() for k, v in interp.items()}


_cache_key = (period_start, period_end, str(context_start), sel_station_interp)
_already_cached = _cache_key in st.session_state.get("_interp_cache_keys", set())

if not _already_cached:
    st.info(
        "Первый расчёт для этого периода занимает **30–90 секунд** — "
        "модель прогоняет тысячи скользящих окон. "
        "Повторный выбор того же периода — мгновенный (кеш). "
        "Если страница зависла — нажмите **F5** для перезагрузки."
    )

prog = st.progress(0, text="Инициализация...")
prog.progress(10, text="Шаг 1/4 — фильтрация данных по периоду и станции...")

try:
    prog.progress(25, text="Шаг 2/4 — создание временного датасета (скользящие окна)...")
    prog.progress(40, text="Шаг 3/4 — прогон TFT в режиме raw (самый долгий шаг)...")

    interpretation_np = compute_interpretation(
        model, training, tft_config,
        period_start, period_end,
        str(context_start), sel_station_interp,
    )

    prog.progress(90, text="Шаг 4/4 — извлечение весов VSN и внимания...")
    prog.progress(100, text="Готово!")

    keys = st.session_state.get("_interp_cache_keys", set())
    keys.add(_cache_key)
    st.session_state["_interp_cache_keys"] = keys

except Exception as e:
    prog.empty()
    st.error(f"Ошибка вычисления: {e}")
    import traceback
    with st.expander("Подробности"):
        st.code(traceback.format_exc())
    st.stop()

station_label = "Все станции" if sel_station_interp == "Все" else station_names.get(sel_station_interp, sel_station_interp)
st.success(f"Интерпретация: **{sel_period}** | **{station_label}**")

# ═══════════════════════════════════════════════════════════
# БЛОК 1: ВАЖНОСТЬ ПЕРЕМЕННЫХ
# ═══════════════════════════════════════════════════════════
st.subheader("1. Важность переменных (Variable Selection Networks)")

st.markdown("""
**Как это работает:**
TFT содержит три отдельных VSN — для статических признаков, для прошлых наблюдений (энкодер)
и для известных будущих признаков (декодер). Каждый VSN обучает softmax-веса над своими переменными.

> Вес показывает, насколько активно модель использует переменную при принятии решения.
> Веса нормированы: **сумма в каждой группе = 1** (100%). Переменные с весом < 1% практически игнорируются.
""")

tab_static, tab_encoder, tab_decoder = st.tabs([
    "📌 Статические признаки АЗС",
    "🔁 Прошлые наблюдения (Encoder)",
    "🔮 Известные будущие (Decoder)",
])


def _importance_chart(key, names_attr, explanation, n_top=20):
    if key not in interpretation_np:
        st.caption("Данные не доступны для данного чекпоинта.")
        return

    vals = interpretation_np[key]
    names = getattr(model, names_attr)

    if vals.ndim > 1:
        vals = vals.mean(axis=0)

    n = min(len(names), len(vals))
    imp_df = (
        pd.DataFrame({"Переменная": names[:n], "Важность": vals[:n]})
        .sort_values("Важность", ascending=False)
    )
    total = imp_df["Важность"].sum()
    imp_df["Важность, %"] = (imp_df["Важность"] / total * 100).round(2)
    imp_df_top = imp_df.head(n_top)

    st.markdown(explanation)

    c_chart, c_table = st.columns([2, 1])
    with c_chart:
        fig = px.bar(
            imp_df_top, x="Важность, %", y="Переменная",
            orientation="h",
            color="Важность, %", color_continuous_scale="Blues",
            height=max(350, n_top * 22),
            labels={"Важность, %": "Вес, %"},
        )
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10, l=10),
        )
        st.plotly_chart(fig, width='stretch')

    with c_table:
        st.caption(
            f"Топ-{n_top} из {n} переменных. "
            f"Показано {imp_df_top['Важность, %'].sum():.1f}% суммарного веса."
        )
        st.dataframe(
            imp_df_top[["Переменная", "Важность, %"]].reset_index(drop=True),
            hide_index=True,
            height=min(600, n_top * 38),
        )


with tab_static:
    _importance_chart(
        key="static_variables",
        names_attr="static_variables",
        explanation="""
**Статические признаки** не меняются во времени — это «паспорт» АЗС:
тип дороги, расположение, площадь магазина, количество колонок и т. д.

TFT сжимает их в 4 контекстных вектора (c_s, c_e, c_h, c_c), которые
инициализируют LSTM и Variable Selection Networks для временных рядов.

> Высокий вес у статического признака → характеристика АЗС сильно влияет
> на базовый уровень прогнозов для этой станции.
""",
    )

with tab_encoder:
    _importance_chart(
        key="encoder_variables",
        names_attr="encoder_variables",
        explanation="""
**Наблюдаемые прошлые переменные** — данные, которые модель видит только в ретроспективе
(168 часов назад): погода, трафик, цены конкурентов, прошлые продажи (лаговые значения целей).

Encoder обрабатывает эти данные через LSTM + VSN, формируя скрытое состояние,
которое передаётся в Decoder.

> Высокий вес у лаговой целевой переменной (например, `sales_AI92`) означает,
> что модель сильно опирается на «как продавали вчера/неделю назад».
""",
    )

with tab_decoder:
    _importance_chart(
        key="decoder_variables",
        names_attr="decoder_variables",
        explanation="""
**Известные будущие переменные** — данные, которые можно знать заранее:
час суток, день недели, праздники, акции, цены топлива (устанавливаются заранее).

Decoder использует их чтобы «понять», в каких условиях будет происходить прогнозируемый период.

> Высокий вес у `hour_sin`/`hour_cos` → суточная сезонность критична для прогноза.
> Высокий вес у `is_holiday` → праздники заметно меняют паттерн продаж.
""",
    )

st.divider()

# ═══════════════════════════════════════════════════════════
# БЛОК 2: ВРЕМЕННОЕ ВНИМАНИЕ
# ═══════════════════════════════════════════════════════════
st.subheader("2. Временно́е внимание (Temporal Self-Attention)")

st.markdown("""
**Как это работает:**
Self-Attention — это механизм, который позволяет модели «смотреть» на любой момент
из 168 прошлых часов при составлении каждого шага прогноза.

> График показывает: **насколько сильно** модель обращается к каждому прошлому часу
> при прогнозировании (усреднено по всем 24 шагам горизонта и всем выборкам).

**Что искать:**
- **Пик на −24** → модель сильно опирается на вчерашний час (суточная периодичность)
- **Пик на −168** → недельная периодичность (тот же час неделю назад)
- **Равномерный профиль** → модель усредняет прошлое без резких предпочтений
- **Высокие пики в нескольких точках** → модель уловила несколько независимых паттернов

Отрицательные значения на оси X — количество часов до текущего момента прогноза
(−1 = предыдущий час, −168 = неделю назад).
""")

if "attention" in interpretation_np:
    attn = interpretation_np["attention"]

    if attn.ndim == 4:
        attn = attn.mean(axis=(0, 1))
    elif attn.ndim == 3:
        attn = attn.mean(axis=0)

    if attn.ndim == 2:
        st.caption("Тепловая карта: строки = шаги прогноза (1–24 ч вперёд), столбцы = прошлые часы.")
        fig_attn = px.imshow(
            attn,
            color_continuous_scale="Viridis",
            labels=dict(x="Шаг энкодера (часов назад)", y="Шаг прогноза (ч вперёд)", color="Внимание"),
            aspect="auto",
            height=400,
        )
        fig_attn.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_attn, width='stretch')
    else:
        enc_steps = list(range(-len(attn), 0))
        attn_pct = attn / attn.sum() * 100

        fig_attn = go.Figure()
        fig_attn.add_trace(go.Bar(
            x=enc_steps, y=attn_pct,
            marker_color=attn_pct,
            marker_colorscale="Viridis",
            showlegend=False,
        ))
        top3_idx = np.argsort(attn_pct)[-3:][::-1]
        for idx in top3_idx:
            fig_attn.add_annotation(
                x=enc_steps[idx], y=attn_pct[idx],
                text=f"{enc_steps[idx]} ч<br>{attn_pct[idx]:.1f}%",
                showarrow=True, arrowhead=2, arrowsize=1,
                font=dict(size=11, color="white"),
                bgcolor="rgba(80,80,80,0.7)",
                bordercolor="gray", borderwidth=1,
                ay=-40,
            )
        fig_attn.update_layout(
            xaxis_title="Часов назад (от момента прогноза)",
            yaxis_title="Вес внимания, %",
            height=400,
            margin=dict(t=20, b=40),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(-168, 0, 24)),
                ticktext=[f"−{abs(v)}ч (−{abs(v)//24}д)" for v in range(-168, 0, 24)],
            ),
        )
        st.plotly_chart(fig_attn, width='stretch')

        top_h = enc_steps[int(np.argmax(attn_pct))]
        st.markdown(
            f"**Интерпретация:** наибольшее внимание — час **{top_h} ч** "
            f"({abs(top_h) // 24} сут. + {abs(top_h) % 24} ч назад)."
        )
        if abs(top_h - (-24)) <= 3:
            st.info("Доминирует суточная периодичность: модель в первую очередь смотрит на вчерашний час.")
        elif abs(top_h - (-168)) <= 6:
            st.info("Доминирует недельная периодичность: модель смотрит на тот же час неделю назад.")
        else:
            st.info(f"Основной паттерн не совпадает с суточным/недельным — пик на {top_h} ч.")
else:
    st.caption("Данные внимания недоступны для данного чекпоинта.")

st.divider()

# ═══════════════════════════════════════════════════════════
# БЛОК 3: СПРАВКА
# ═══════════════════════════════════════════════════════════
with st.expander("📚 Архитектура TFT — краткая справка"):
    st.markdown("""
    **Temporal Fusion Transformer** (Lim et al., 2020) — архитектура для мультивариантного
    прогнозирования временных рядов с явной интерпретируемостью.

    ```
    Входы:
      Static  ──► Static VSN ──► Context vectors (c_s, c_e, c_h, c_c)
      Past    ──► Encoder VSN ──► LSTM (encoder) ──┐
      Future  ──► Decoder VSN ──► LSTM (decoder) ──┤
                                                   ▼
                                      Temporal Self-Attention
                                                   ▼
                                        Gate + Add & Norm
                                                   ▼
                                  Position-wise Feed-Forward
                                                   ▼
                                  QuantileLoss (7 квантилей)
    ```

    **Ключевые компоненты:**
    - **VSN** — Gated Residual Network с softmax-весами над переменными
    - **LSTM** — захватывает краткосрочную зависимость (локальный контекст)
    - **Self-Attention** — захватывает долгосрочные паттерны (7 суток)
    - **Static context** — инициализирует LSTM и VSN характеристиками каждой АЗС
    - **QuantileLoss** — даёт прогноз в виде доверительного интервала (7 квантилей)

    *Горизонт: 24 ч | Ретроспектива: 168 ч | Целей: 12 | Входов: 93*
    """)
