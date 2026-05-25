# FinalWorkDashboard — Спецификация проекта

## Задача

Мультицелевая TFT-модель (Temporal Fusion Transformer, Lim et al. 2020) для
прогнозирования почасовых продаж топлива и товаров магазина сети АЗС Татнефть.
Горизонт: 24 часа (1 сутки). Ретроспектива: 168 часов (7 суток). 12 целевых переменных совместно.

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

### STATIC_REALS (26 переменных)
Статические вещественные — паспортные данные АЗС. Не нормализуются:
```
distance_to_city_km, shop_area_m2,
num_pumps_AI92/95/98/DT_EURO/DT_TANEKO/DT_SUMMER/DT_WINTER,
has_car_wash, has_tire_service, has_cafe, has_hotel, has_shop,
competitors_within_5km, customer_loyalty_score, staff_quality_score,
corporate_customer_ratio, staff_engagement_score,
base_price_AI92/95/98/DT_EURO/DT_TANEKO/DT_SUMMER/DT_WINTER
```
`total_pumps` исключён (= sum(num_pumps_*), дублирует детальные колонки).

### TIME_VARYING_KNOWN_CATS (4 переменных)
Известные будущие категориальные — задаются заранее или в what-if:
```
season_enc, weather_condition_enc, ad_channel_enc, holiday_name_enc
```
`day_name_enc` удалён (дублирует day_of_week, уже есть в KNOWN_REALS с sin/cos).
`weather_condition_enc` перенесён из UNKNOWN → что-если тип погоды.

### TIME_VARYING_KNOWN_REALS (44 переменных)
Известные будущие вещественные — декодер TFT принимает эти переменные для горизонта прогноза.
Всё, что задаётся в what-if сценарии, должно быть здесь:
```
# Циклические временные признаки (raw + sin/cos)
hour, hour_sin, hour_cos,
day_of_week, day_of_week_sin, day_of_week_cos,
week_of_year, week_of_year_sin, week_of_year_cos,
month, month_sin, month_cos,
# Бинарные флаги
is_weekend, is_holiday, is_rush_hour, is_night,
is_shop_open,                 # производный: 1 = 05:00–21:00, 0 = 22:00–04:00
# Акции и реклама
promotion_fuel_active, promotion_shop_active, promotion_cafe_active, ad_active,
# Текущие цены топлива (постоянны per-station в 2023 г.; передаются без Z-score)
price_AI92, price_AI95, price_AI98,
price_DT_EURO, price_DT_TANEKO, price_DT_SUMMER, price_DT_WINTER,
# Погода (по метеопрогнозу; what-if: снег/ясно/дождь/температура)
temperature, precipitation_mm, visibility_km, wind_speed_ms,
is_snow, is_rain, is_fog,
# Трафик по типам ТС (прогнозируется службами дорожного движения)
traffic_Passengers_cars, traffic_Truck_short, traffic_Truck,
traffic_Truck_long, traffic_Transporter, traffic_Undefined,
# Цены конкурентов (мониторинг; what-if: снижение/рост)
competitor_price_AI92, competitor_price_AI95, competitor_price_DT
```
`quarter` удалён (выводится из month). `total_traffic` исключён (несогласован с индивидуальными).

### TIME_VARYING_UNKNOWN_REALS (0 переменных)
Намеренно пусто. Все ковариаты перенесены в KNOWN для поддержки what-if анализа.
Целевые переменные (авторегрессивные входы энкодера) TFT добавляет автоматически
через параметр `target=TARGET_COLS` в TimeSeriesDataSet.

---

## Препроцессинг (eda/eda_preprocessing.py)

Порядок шагов строго фиксирован:

1. **Пропуски**: `holiday_name → "нет_праздника"`, `ad_channel → "нет_рекламы"`
2. **Исключение избыточных колонок**: 7 столбцов удаляются из датафрейма
   (`EXCLUDED_COLS` из `utils/data_utils.py`). Winsorization не применяется —
   не описана в статье TFT; TorchNormalizer(robust) обрабатывает выбросы целей.
   После исключения добавляется производный признак `is_shop_open` (1: 05:00–21:00, 0: 22:00–04:00) —
   стабилизирует нулевые продажи shop_* ночью. Добавляется до вычисления binary_cols,
   чтобы автоматически исключиться из Z-score нормализации.
3. **Label Encoding** категориальных → суффикс `_enc`.
   TFT строит эмбеддинги сам через NaNLabelEncoder.
4. **Циклическое кодирование**: `hour/day_of_week/month/week_of_year → _sin/_cos`.
   Решает разрыв 23:00 → 00:00: на числовой оси далеко, на окружности рядом.
5. **time_idx**: монотонный порядковый номер часа per-station через cumcount.
6. **log1p**: только 12 целевых переменных (LOG_COLS = TARGET_COLS).
   Оригинал сохраняется в `_orig`. `shop_total_revenue` исключена из модели.
7. **Z-score** нормализация per-station для числовых переменных.
   Исключены: STATIC_REALS, бинарные, `_enc`, `_orig`, LOG_COLS (цели),
   а также `NO_ZSCORE_COLS` (`price_*` — константны per-station в 2023 г., std=0).
   LOG_COLS исключены — TorchNormalizer в TFT нормализует цели отдельно.
8. **Скейлеры**: `tft/scalers.pkl` — `{station_id: {col: (mean, std)}}` + `log1p_cols`.
   Используются в `predict.py` для обратного преобразования.

Результат: `data/prepared_data.csv` (43800 × 111).

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
| `hidden_continuous_size` | 32 | 64 |
| `BATCH_SIZE` | 32 | 64 |
| `EPOCHS` | 50 | 80 |

Общие параметры:
- `ENCODER_LENGTH = 168` ч (7 суток), `PREDICTION_LENGTH = 24` ч (1 сутки).
  Горизонт оперативного прогноза: следующие сутки (24 шага).
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
│   ├── prepared_data.csv       — после препроцессинга (43800 × 111)
│   ├── predictions.csv         — скользящие 24ч прогнозы TFT за декабрь 2023
│   │                             (pred + q10 + q90 по 12 целям × 5 АЗС; actual для сравнения)
│   └── metrics.csv             — MAE / RMSE / MAPE по target × station (декабрь 2023)
│
├── eda/
│   └── eda_preprocessing.py    → data/prepared_data.csv + tft/scalers.pkl
│
├── report_generating/          — скрипты генерации отчётов
│   ├── eda_column_analysis.py  → reports/column_analysis.md
│   └── tft_report.py           → reports/tft_report.docx
│
├── tft/
│   ├── model.ckpt              — лучшая модель (BestModelSync перезаписывает на лету)
│   ├── prepare_dataset.py      → tft/training_dataset.pkl + tft/dataset_config.pkl
│   ├── train.py                — обучение с BestModelSync + авто-резюм
│   ├── predict.py              — инференс: скользящие 24ч окна на декабрь →
│   │                             data/predictions.csv + data/metrics.csv
│   ├── checkpoints/            — чекпоинты Lightning (monitor=val_loss, save_top_k=1)
│   └── logs/                   — TensorBoard (tensorboard --logdir tft/logs)
│
├── dashboard/
│   └── app_dashboard.py        — единый дашборд (4 главные вкладки + подвкладки)
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
├── reports/                    — сгенерированные отчёты
│   ├── column_analysis.md
│   ├── merged_data_columns.txt — список колонок merged_data.csv с описанием и примерами
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

python explore_data.py                              # → data/merged_data.csv
python eda/eda_preprocessing.py                     # → data/prepared_data.csv + tft/scalers.pkl
python report_generating/eda_column_analysis.py     # → reports/column_analysis.md
python report_generating/tft_report.py              # → reports/tft_report.docx

python tft/prepare_dataset.py        # → tft/training_dataset.pkl + tft/dataset_config.pkl
python tft/train.py                  # → tft/model.ckpt

python tft/predict.py                # → data/predictions.csv + data/metrics.csv

streamlit run dashboard/app_dashboard.py        # единый дашборд (требует merged_data.csv; остальные файлы graceful)

tensorboard --logdir tft/logs        # в отдельном терминале
```

---

## Единый дашборд (app_dashboard.py)

Загружает `merged_data.csv` (обязателен). `prepared_data.csv`, `predictions.csv`,
`metrics.csv`, `tft/scalers.pkl`, `tft/model.ckpt` — опциональны; отсутствие любого
даёт информативное сообщение, функционал деградирует gracefully.

### Система фильтров (двухуровневая)

**Глобальные фильтры** (вверху страницы, над всеми вкладками):
- `sel_stations_multi` (Станции) + `sel_months` (Период) → формируют `base_mask` и `df_ov`.
- Влияют на все 4 главные вкладки. Колонки `[3, 7]`.
- Оформлены секционным заголовком «🌐 ГЛОБАЛЬНЫЕ ФИЛЬТРЫ» с горизонтальным разделителем.

**Фильтры анализа** (верх вкладки «Анализ данных», только в ней):
- `adv_fuels` + `adv_shops` — строка из двух колонок.
- Expander «🔍 Дополнительные фильтры»: Сезон, Тип дня, Праздники, Акции, Погода,
  Тип дороги, Направление, Расстояние до города, Населённый пункт.
- Формируют `adv_mask` и `fdf`. Не влияют на `df_ov` и другие вкладки.

**Фильтры вкладок Прогноз TFT и Рекомендации** — собственные независимые:
- Станция + Показатель (из TARGET_COLS); работают с `predictions.csv`/`metrics.csv`.

### 4 главные вкладки

**tab1 — Обзор**
- KPI-строка: продажи топлива, выручка магазина, ср. трафик/ч, кол-во станций, пиковый месяц.
- Подвкладки: Паттерны продаж · Сравнение АЗС · Акции & Реклама.
- Паттерны продаж: area-чарты, bar, сезонный bar, структура магазина,
  `st.radio(horizontal=True)` для тепловой карты. Данные — `df_ov`.

**tab2 — Анализ данных**
- Фильтры анализа сверху (см. выше), затем `fdf`.
- Подвкладки: Погода & Трафик · Акции & Реклама · Корреляции · Статистика.
- Корреляции: матрица, топ-5 факторов (`st.radio` горизонтальный для выбора топлива).

**tab3 — Прогноз TFT**
- Фильтры Станция + Показатель вверху таба.
- Подвкладки: Метрики & Точность · Прогноз vs Факт · Сценарий (What-if) · Интерпретация VSN.
- Сценарий: горизонт 1/7/30 дней; `build_future_ctx` для дат за пределами декабря 2023.
  Легенда графика — под осью X (`y=-0.2`, `margin=dict(b=70)`).
  Кнопка «Рассчитать» выровнена через `st.markdown("&nbsp;")` спейсер.
- Интерпретация VSN: собственные фильтры период + станция; VSN-веса, temporal attention.

**tab4 — Рекомендации**
- Фильтры Станция + Показатель (собственные).
- EDA-рекомендации 2023: MAPE-надёжность, акционный рычаг, рекламный канал,
  окно спроса, лучший день, трафик как сигнал.
- Stale-check: блок результатов сценария скрывается при смене фильтра.


---

## Цветовая схема (тёмная тема, inlined CSS)

```python
GOLD        = "#c8a84b"   # основные данные (area, scatter), заголовки акцентов
GREEN       = "#4ECB71"   # позитивные KPI, метрики в норме
RED         = "#E24B4A"   # ошибки, негативные отклонения
BLUE        = "#2E75B6"   # вторичные ряды, рекламный канал
TEAL        = "#1ABC9C"   # нейтральные карточки, заголовок фильтров анализа
GRAY        = "#8B949E"   # muted, вторичный текст, подписи фильтров
GRID_COLOR  = "#1e2235"   # сетка графиков, рамки popover
CARD_BG     = "#13161f"   # фон KPI-карточек и Plotly-графиков
```

UI-компоненты: `kpi_card()`, `sfig()`, `add_regline()` —
кастомные функции с inlined HTML/CSS (без внешних CSS-классов).

---

## Ключевые архитектурные решения

1. **Единственный источник констант**: все списки переменных TFT живут только в
   `utils/data_utils.py`. Все скрипты импортируют оттуда. Дублирование запрещено.

2. **What-if ограничения**: изменять для прогноза можно только `TIME_VARYING_KNOWN_*`.
   Погода и трафик — в `TIME_VARYING_UNKNOWN_REALS` (только энкодер), поэтому
   "что если завтра метель" невозможно без переноса в known и переобучения.
   Статические переменные — зашиты в веса модели, менять без переобучения бессмысленно.

3. **Двухуровневые фильтры**: глобальные (Станции + Период) влияют на все вкладки через
   `df_ov`. Фильтры анализа (топливо, магазин, 9 доп. условий) живут внутри вкладки
   «Анализ данных» и формируют `fdf` — не затрагивая остальные вкладки.

4. **Graceful degradation**: каждый файл данных проверяется при загрузке; отсутствие
   любого файла даёт информативное сообщение, не краш.

5. **Stale check**: динамический блок рекомендаций скрывается при смене глобального
   фильтра (проверка `sc_result["sc_tgt"] == sel_target` и `sc_st == sel_station`).

---

## Стек

```
Python 3.14.3
pytorch-forecasting 1.7.0   — TFT модель
lightning.pytorch 2.6.1     — обучение
torch 2.6                   — с патчем weights_only в torch_compat.py
streamlit                   — дашборды
plotly                      — графики (graph_objects + express)
pandas, numpy, scikit-learn — препроцессинг
python-docx 1.2.0           — DOCX отчёты (не путать с устаревшим пакетом docx 0.2.4)
tensorboard                 — мониторинг обучения
```
