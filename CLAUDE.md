# FinalWorkDashboard — Спецификация проекта

## Задача

Мультицелевая TFT-модель (Temporal Fusion Transformer, Lim et al. 2020) для
прогнозирования почасовых продаж топлива и товаров магазина сети АЗС Татнефть.
Горизонт: 24 часа. Ретроспектива: 168 часов (7 суток). 12 целевых переменных совместно.

---

## Исходные данные (SourceDataForWork/)

| Файл | Описание |
|---|---|
| `5stations_metadata.csv` | Паспорт 5 АЗС: тип дороги, услуги, кол-во колонок, базовые цены (5 × 32) |
| `5stations_data.csv` | Почасовые данные 2023 г.: погода, трафик, продажи, акции, реклама, цены конкурентов (43800 × 72) |
| `detailed_data.csv` | Те же данные для 25 АЗС (219000 строк) |
| `stations_metadata.csv` | Паспорт 25 АЗС (25 × 32) |
| `TFT_анализ.pdf` | Оригинальная статья TFT (Lim et al., 2020) |

Связь файлов: `metadata LEFT JOIN data ON station_id`.
Результат: 43800 × 89, хранится в `data/merged_data.csv`.

---

## Целевые переменные (TARGET_COLS — 12 штук)

```python
# 7 видов топлива
"sales_AI92", "sales_AI95", "sales_AI98",
"sales_DT_EURO", "sales_DT_TANEKO", "sales_DT_SUMMER", "sales_DT_WINTER"

# 5 категорий магазина
"shop_напитки", "shop_закуски", "shop_автотовары", "shop_кофе", "shop_табак"
```

Категории магазина одновременно являются целевыми и наблюдаемыми прошлыми
(TIME_VARYING_UNKNOWN_REALS) — TFT использует их как авторегрессивные входы энкодера.

---

## Таксономия переменных TFT (единственный источник истины — utils/data_utils.py)

### STATIC_CATS (3 переменных)
Статические категориальные — характеристики АЗС, не меняются во времени:
```
road_type_enc, direction_enc, settlement_size_enc
```

### STATIC_REALS (27 переменных)
Статические вещественные — паспортные данные АЗС. Не winsoriz-уются, не нормализуются:
```
distance_to_city_km, total_pumps, shop_area_m2,
num_pumps_AI92/95/98/DT_EURO/DT_TANEKO/DT_SUMMER/DT_WINTER,
has_car_wash, has_tire_service, has_cafe, has_hotel, has_shop,
competitors_within_5km, customer_loyalty_score, staff_quality_score,
corporate_customer_ratio, staff_engagement_score,
base_price_AI92/95/98/DT_EURO/DT_TANEKO/DT_SUMMER/DT_WINTER
```

### TIME_VARYING_KNOWN_CATS (4 переменных)
Известные будущие категориальные — планируются заранее:
```
season_enc, day_name_enc, ad_channel_enc, holiday_name_enc
```

### TIME_VARYING_KNOWN_REALS (28 переменных)
Известные будущие вещественные — декодер TFT принимает эти переменные для горизонта прогноза.
Только эти переменные можно менять в сценарном what-if анализе:
```
# Циклические временные признаки (raw + sin/cos)
hour, hour_sin, hour_cos,
day_of_week, day_of_week_sin, day_of_week_cos,
week_of_year, week_of_year_sin, week_of_year_cos,
month, month_sin, month_cos,
quarter,
# Бинарные флаги
is_weekend, is_holiday, is_rush_hour, is_night,
# Акции и реклама
promotion_fuel_active, promotion_shop_active, promotion_cafe_active, ad_active,
# Текущие цены топлива (устанавливаются заранее)
price_AI92, price_AI95, price_AI98,
price_DT_EURO, price_DT_TANEKO, price_DT_SUMMER, price_DT_WINTER
```

### TIME_VARYING_UNKNOWN_REALS (19 + 12 = 31 переменная)
Наблюдаемые прошлые — энкодер видит их только за прошлые 168ч, декодер НЕ принимает.
Погода и трафик здесь — what-if по ним невозможен без переобучения:
```
# Погода
temperature, precipitation_mm, visibility_km, wind_speed_ms,
is_snow, is_rain, is_fog, weather_condition_enc,
# Трафик по типам ТС
traffic_Passengers_cars, traffic_Truck_short, traffic_Truck,
traffic_Truck_long, traffic_Transporter, traffic_Undefined, total_traffic,
# Магазин (общая выручка)
shop_total_revenue,
# Цены конкурентов
competitor_price_AI92, competitor_price_AI95, competitor_price_DT,
# + все TARGET_COLS (авторегрессивные входы)
```

---

## Препроцессинг (eda/eda_preprocessing.py)

Порядок шагов строго фиксирован:

1. **Пропуски**: `holiday_name → "нет_праздника"`, `ad_channel → "нет_рекламы"`
2. **Winsorization** выбросов по IQR — строки не удаляются (временной ряд непрерывен).
   Исключены: бинарные переменные, STATIC_REALS (паспортные данные не меняются).
3. **Label Encoding** категориальных → суффикс `_enc`.
   TFT строит эмбеддинги сам через NaNLabelEncoder.
4. **Циклическое кодирование**: `hour/day_of_week/month/week_of_year → _sin/_cos`.
   Решает разрыв 23:00 → 00:00: на числовой оси далеко, на окружности рядом.
5. **time_idx**: монотонный порядковый номер часа per-station через cumcount.
6. **log1p**: все 12 целевых + `shop_total_revenue`. Оригинал сохраняется в `_orig`.
7. **Z-score** нормализация per-station для числовых переменных.
   Исключены: STATIC_REALS, бинарные, `_enc`, `_orig`, LOG_COLS.
   LOG_COLS исключены — TorchNormalizer в TFT нормализует цели отдельно.
8. **Скейлеры**: `tft/scalers.pkl` — `{station_id: {col: (mean, std)}}` + `log1p_cols`.
   Используются в `predict.py` для обратного преобразования.

Результат: `data/prepared_data.csv` (43800 × 119).

---

## Сплиты данных

| Сплит | Период | Доля |
|---|---|---|
| train | Jan–Oct 2023 | ~83.6% |
| val | Nov 2023 | ~8.2% |
| test | Dec 2023 | ~8.2% |

`val_df` включает данные train для encoder context (необходимо для первых окон ноября).
`NaNLabelEncoder` предобучается на полном df (все 43800 строк) → нет KeyError для
праздников, которые встречаются только в ноябре–декабре.

---

## Настройки TFT

| Параметр | CPU | GPU |
|---|---|---|
| `hidden_size` | 64 | 128 |
| `attention_head_size` | 2 | 4 |
| `hidden_continuous_size` | 64 | 64 |
| `BATCH_SIZE` | 32 | 64 |
| `EPOCHS` | 50 | 80 |

Общие параметры:
- `ENCODER_LENGTH = 168` ч, `PREDICTION_LENGTH = 24` ч
- `learning_rate = 3e-4`, `dropout = 0.15`, `gradient_clip = 1.0`
- `EarlyStopping(patience=12, monitor="val_loss")`
- `ReduceLROnPlateau(patience=5)`
- `loss = QuantileLoss()` — 7 квантилей: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
- `target_normalizer = MultiNormalizer([TorchNormalizer(method="robust") × 12])`
- `add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True`

**BestModelSync callback**: кастомный `pl.Callback`, перезаписывает `tft/model.ckpt`
при каждом новом лучшем `val_loss` (через `on_validation_epoch_end`), не ожидая конца
обучения. Обучение авто-возобновляется с последнего чекпоинта через `glob`.

**torch_compat**: патч `torch.load` для PyTorch 2.6 (добавляет `weights_only=False`)
вынесен в `utils/torch_compat.py`, импортируется в `train.py` и `predict.py`.

---

## Структура проекта

```
FinalWorkDashboard/
├── SourceDataForWork/          — исходные файлы задания, не изменять
│   ├── 5stations_metadata.csv
│   ├── 5stations_data.csv
│   ├── detailed_data.csv
│   ├── stations_metadata.csv
│   ├── TFT_анализ.pdf
│   └── Generation/             — материалы диалога с Claude
│
├── data/                       — генерируются скриптами
│   ├── merged_data.csv         — JOIN metadata + data (43800 × 89)
│   ├── prepared_data.csv       — после препроцессинга (43800 × 119)
│   ├── predictions.csv         — прогнозы TFT на декабрь 2023
│   └── metrics.csv             — MAE/RMSE/MAPE по target × station
│
├── eda/
│   ├── eda_column_analysis.py  → reports/column_analysis.md
│   ├── eda_preprocessing.py    → data/prepared_data.csv + tft/scalers.pkl
│   └── tft_report.py           → reports/tft_report.docx
│
├── tft/
│   ├── model.ckpt              — лучшая модель (BestModelSync перезаписывает на лету)
│   ├── prepare_dataset.py      → tft/training_dataset.pkl + tft/dataset_config.pkl
│   ├── train.py                — обучение с BestModelSync + авто-резюм
│   ├── predict.py              → data/predictions.csv + data/metrics.csv
│   ├── checkpoints/            — чекпоинты Lightning (monitor=val_loss, save_top_k=1)
│   └── logs/                   — TensorBoard (tensorboard --logdir tft/logs)
│
├── dashboard/
│   ├── eda_dashboard.py        — EDA по исходным данным (6 вкладок)
│   ├── forecast_dashboard.py   — прогнозы + сценарии (4 вкладки)
│   └── tft_interpretation.py  — VSN-веса + temporal attention (3 вкладки)
│
├── utils/
│   ├── data_utils.py           — ВСЕ константы проекта (единственный источник истины):
│   │                             TARGET_COLS, LOG_COLS, STATIC_CATS, STATIC_REALS,
│   │                             TIME_VARYING_KNOWN_CATS, TIME_VARYING_KNOWN_REALS,
│   │                             TIME_VARYING_UNKNOWN_REALS, CYCLICAL_FEATURES,
│   │                             TRAIN_END, VAL_END, TEST_START, TEST_END,
│   │                             ENCODER_LENGTH, PREDICTION_LENGTH,
│   │                             QUANTILE_LEVELS, Q_MED, Q_LO, Q_HI
│   │                             + load_and_merge(), fill_missing(), add_cyclical_encoding()
│   └── torch_compat.py         — патч torch.load для PyTorch 2.6
│
├── reports/
│   ├── column_analysis.md
│   └── tft_report.docx
│
├── explore_data.py             → data/merged_data.csv
├── requirements.txt
└── CLAUDE.md
```

---

## Запуск (строго по порядку из корня проекта)

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

python explore_data.py               # → data/merged_data.csv
python eda/eda_column_analysis.py    # → reports/column_analysis.md
python eda/eda_preprocessing.py      # → data/prepared_data.csv + tft/scalers.pkl
python eda/tft_report.py             # → reports/tft_report.docx

python tft/prepare_dataset.py        # → tft/training_dataset.pkl + tft/dataset_config.pkl
python tft/train.py                  # → tft/model.ckpt

python tft/predict.py                # → data/predictions.csv + data/metrics.csv

streamlit run dashboard/eda_dashboard.py
streamlit run dashboard/forecast_dashboard.py   # требует predictions.csv + metrics.csv
streamlit run dashboard/tft_interpretation.py   # требует tft/model.ckpt

tensorboard --logdir tft/logs        # в отдельном терминале
```

---

## Дашборд EDA (eda_dashboard.py)

Загружает `data/merged_data.csv`. Если файл отсутствует — `st.error` + `st.stop()`.

Глобальный фильтр: Станция (Все / АЗС-001..005) + Показатель.

**6 вкладок:**

| Вкладка | Содержимое |
|---|---|
| Паттерны продаж | Продажи по месяцам (area), часам суток (area), дням недели (bar); тепловая карта час × день |
| Сравнение АЗС | Суммарные продажи по станциям; структура топлива; динамика по неделям; характеристики (тип дороги, расстояние, сервисы) |
| Погода и трафик | Scatter: температура/осадки vs продажи; типы ТС vs продажи |
| Акции и реклама | Boxplot: с акцией vs без; bar по каналам рекламы; эффект праздников |
| Корреляции | Матрица корреляций факторов с продажами; топ факторов по силе |
| Статистический анализ | Распределения целевых (до/после log1p); корреляции по группам; таблица выбросов |

---

## Дашборд прогнозов TFT (forecast_dashboard.py)

Загружает `merged_data.csv` (обязателен), `prepared_data.csv`, `predictions.csv`,
`metrics.csv`, `tft/scalers.pkl`. Отсутствие любого — информационное сообщение,
функционал деградирует gracefully.

Глобальный фильтр (две колонки вверху страницы): **Станция** + **Показатель**.
Фильтр единый для всех вкладок, включая сценарный анализ (дублирующих selectbox нет).

**4 вкладки:**

### Обзор 2023
- KPI-карточки: суммарные продажи за год, лучший месяц, лидирующая станция.
- **Блок точности модели** (если есть metrics.csv + predictions.csv):
  - Медианная точность = `100 - median(MAPE_%)` по всем целям и станциям
  - R² = медиана по целям, вычислен из pred_df: `1 - SS_res/SS_tot`
  - Кол-во целей с MAPE ≤ 15%
  - Bar-chart точности по каждой цели: NaN (нет продаж) → серый столбик "N/A",
    valid → цвет по порогу (green ≤85%, gold ≤90%, red >90%). hovertemplate явный.
- Временные паттерны: помесячный тренд, почасовой профиль, по дням недели.

### Прогноз TFT
- График прогноз vs факт для декабря 2023 с доверительным интервалом q10–q90.
- KPI по выбранной цели и станции: MAE, RMSE, MAPE.
- Scatter прогноз vs факт + OLS trend line.
- Гистограмма ошибок (pred − actual).
- Тепловая карта MAPE по станциям × целям.

### Факторный анализ
Использует `merged_df` или `pred_df` (декабрь). Под-вкладки:
- **Акции & Реклама**: boxplot с/без акции, bar по каналам рекламы, праздники.
- **Трафик**: scatter трафик vs продажи, корреляция типов ТС.
- **Погода**: scatter температура/осадки vs продажи.
- **Цены конкурентов**: scatter конкурентные цены vs продажи.
- Общий hourly-профиль по выбранной цели.

### Сценарий & Рекомендации
Два столбца равной ширины:

**Левый (sc_col) — What-if анализ:**
- Использует глобальные фильтры (Станция + Показатель) — отдельных selectbox нет.
- При "Все станции": `st.info` + кнопка задизаблена.
- Параметры: дата (Dec 2023 – Jan 2024), час, тумблеры акций (топливо / магазин),
  реклама (toggle + канал), слайдер цены ±12% от медианы (только для топливных целей).
- Дата Jan 2024 или выход за Dec 2023 → синтетический контекст через `build_future_ctx`
  (реальные последние 168ч декабря + синтетический декодер).
- Прогресс-бар 4 шага. Результат: KPI (выбранный час / среднее 24ч / пик) + график.
- Если не синтетический → сравнение со baseline из `predictions.csv`.
- После успешного запуска результаты сохраняются в `st.session_state["sc_result"]`.

**Правый (rec_col) — Рекомендации:**
- Верхний блок "Результат сценария" (из `session_state`) — показывается только если
  `sc_tgt == sel_target` и `sc_st == sel_station` (stale-check при смене фильтра).
  Карточки: Сценарий vs базовый (±%), Активные стимулы, Ценовое воздействие, Пик.
- Нижний блок — статические EDA-рекомендации по данным 2023 г.:
  Надёжность прогноза (MAPE → совет), Акционный рычаг (+ или − %),
  Оптимальный рекламный канал (gain vs худший), Окно высокого спроса (часы),
  Лучший день для акций (lift %), Трафик как сигнал (r → совет).

---

## Дашборд интерпретации TFT (tft_interpretation.py)

Загружает `tft/model.ckpt` + `data/prepared_data.csv`. При отсутствии — `st.error` + `st.stop()`.

Фильтры: период (диапазон дат), станция.

**3 вкладки (VSN-веса):**
- **Статические признаки АЗС** — важность STATIC_CATS + STATIC_REALS
- **Прошлые наблюдения (Encoder)** — важность TIME_VARYING_UNKNOWN_REALS
- **Известные будущие (Decoder)** — важность TIME_VARYING_KNOWN_REALS + KNOWN_CATS

Плюс: temporal self-attention — паттерн внимания на прошлые часы.

---

## Цветовая схема (тёмная тема, inlined CSS)

```python
GREEN   = "#00853E"   # акцент KPI, позитивные карточки
GOLD    = "#F59E0B"   # основные данные (area, scatter)
TEAL    = "#2DD4BF"   # вторичные ряды, нейтральные карточки
RED     = "#F87171"   # ошибки, негативные отклонения
BLUE    = "#60A5FA"   # станции в palette, рекламный канал
GRAY    = "#8B949E"   # muted, вторичный текст, N/A бары
CARD_BG = "#1C2532"   # фон карточек
SEC_LINE= "#334155"   # разделители секций
TEXT    = "#E6EDF3"   # основной текст
```

UI-компоненты: `kpi()`, `sec()`, `banner()`, `rec_card()`, `chart_layout()` —
кастомные функции с inlined HTML/CSS (без внешних CSS-классов).

---

## Ключевые архитектурные решения

1. **Единственный источник констант**: все списки переменных TFT живут только в
   `utils/data_utils.py`. Все скрипты импортируют оттуда. Дублирование запрещено.

2. **What-if ограничения**: изменять для прогноза можно только `TIME_VARYING_KNOWN_*`.
   Погода и трафик — в `TIME_VARYING_UNKNOWN_REALS` (только энкодер), поэтому
   "что если завтра метель" невозможно без переноса в known и переобучения.
   Статические переменные — зашиты в веса модели, менять без переобучения бессмысленно.

3. **Единый фильтр**: forecast_dashboard использует два глобальных selectbox (Станция +
   Показатель) для всех вкладок. Сценарный анализ не имеет своих дублирующих selectbox.

4. **Graceful degradation**: каждый файл данных проверяется при загрузке; отсутствие
   любого файла даёт информативное сообщение, не краш.

5. **Stale check**: динамический блок рекомендаций скрывается при смене глобального
   фильтра (проверка `sc_result["sc_tgt"] == sel_target` и `sc_st == sel_station`).

---

## Стек

```
Python 3.10
pytorch-forecasting 1.7.0   — TFT модель
lightning.pytorch 2.6.1     — обучение
torch 2.6                   — с патчем weights_only в torch_compat.py
streamlit                   — дашборды
plotly                      — графики (graph_objects + express)
pandas, numpy, scikit-learn — препроцессинг
python-docx                 — DOCX отчёты
tensorboard                 — мониторинг обучения
```
