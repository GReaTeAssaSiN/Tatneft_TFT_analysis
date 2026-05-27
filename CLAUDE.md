# FinalWorkDashboard — Спецификация проекта

## Задача

Мультицелевая TFT-модель (Temporal Fusion Transformer, Lim et al. 2020) для
прогнозирования ежедневных продаж топлива и товаров магазина сети АЗС Татнефть.
Горизонт: 7 дней (1 неделя). Ретроспектива: 30 дней (1 месяц). 12 целевых переменных совместно.
Рекурсивное прогнозирование: на 1 день (первый шаг 7-дневного окна) и на 1 месяц (4–5 итераций).

---

## Исходные данные (SourceDataForWork/)

| Файл | Описание |
|---|---|
| `gas_stations_static.csv` | Паспорт 5 АЗС: тип дороги, направление, услуги, кол-во колонок, расположение |
| `gas_stations_temporal_daily_2025.csv` | Ежедневные данные 2025 г.: погода, трафик, продажи, цены, акции, метрики клиентов (1825 × 88) |
| `2.Данные_по_временным_рядам.xlsx` | Исходный Excel с временными рядами |
| `5.Данные_по_временным_рядам_пилотных_АЗС-трафик по типам транспорта.xlsx` | Трафик по типам ТС |
| `_Статистические_данные_пилотных_АЗС_2.xlsx` | Паспортные данные АЗС |

Связь файлов: `gas_stations_static LEFT JOIN gas_stations_temporal_daily_2025 ON station_id`.
Результат: 1825 × 115, хранится в `data/merged_data.csv`.

---

## Целевые переменные (TARGET_COLS — 12 штук)

```python
# 5 видов топлива
"sales_AI92", "sales_AI95", "sales_DT", "sales_DT_bio", "sales_AI100_bio"

# 7 категорий магазина
"shop_напитки_безалкогольные", "shop_кондитерка_снеки", "shop_мороженое",
"shop_автотовары", "shop_кафе_вся_еда", "shop_кофе_все_горячие_напитки", "shop_табак"
```

Все 12 целевых переменных прогнозируются совместно (мультитаргет).
TFT использует их как авторегрессивные входы энкодера через `target=TARGET_COLS`.

---

## Таксономия переменных TFT (единственный источник истины — utils/data_utils.py)

### STATIC_CATS (5 переменных)
Статические категориальные — характеристики АЗС, не меняются во времени.
⚠️ `road_level`, `settlement_size`, `distance_to_city_km` — коды категорий, не числа:
```
road_type_enc, road_level_enc, direction_enc, settlement_size_enc, distance_to_city_km_enc
```

### STATIC_REALS (21 переменная)
Статические вещественные — паспортные данные АЗС. Не нормализуются:
```
shop_area_m2,
num_pumps_AI92/AI92_bio/AI95/AI95_bio/AI100_bio/DT/DT_bio/SUG/KPG/SPG,
has_car_wash, has_tire_service, has_cafe, has_hotel, has_shop,
has_shop_молельная_комната, has_shop_прачечная,
has_shop_электрозарядная_станция, has_shop_подкачка_шин,
competitors_wink
```
`total_pumps` исключён (= sum(num_pumps_*), дублирует детальные колонки).

### TIME_VARYING_KNOWN_CATS (2 переменных)
Известные будущие категориальные — задаются заранее или в what-if:
```
season_enc, holiday_name_enc
```
`weather_condition` — бинарный 0/1, перенесён в KNOWN_REALS.

### TIME_VARYING_KNOWN_REALS (53 переменных)
Известные будущие вещественные — декодер TFT принимает эти переменные для горизонта прогноза.
Всё, что задаётся в what-if сценарии, должно быть здесь.
Данные суточные — `hour` отсутствует:
```
# Циклические временные признаки (raw + sin/cos; 9 колонок)
day_of_week, day_of_week_sin, day_of_week_cos,
week_of_year, week_of_year_sin, week_of_year_cos,
month, month_sin, month_cos,
# Бинарные флаги
is_weekend, is_holiday,
# Тип погоды (бинарный 0/1: 0 = ясно/облачно, 1 = осадки/плохая видимость)
weather_condition,
# Акции
promotion_fuel_active, promotion_shop_active, promotion_cafe_active,
# Текущие цены топлива (меняются ежедневно в 2025 г.; Z-score применяется)
price_AI92, price_AI95, price_DT, price_DT_bio, price_AI100_bio,
# Погода (по метеопрогнозу; what-if: температура, осадки)
temperature, precipitation_mm,
# Трафик-счётчики ТС, попутное направление, полосы 1 и 2 (12 колонок)
traffic_Passengers_cars_1/2_poputn, traffic_Truck_short_1/2_poputn,
traffic_Truck_1/2_poputn, traffic_Truck_long_1/2_poputn,
traffic_Transporter_1/2_poputn, traffic_Undefined_1/2_poputn,
# Трафик-счётчики, встречное направление, полосы 1 и 2 (12 колонок)
traffic_*_1/2_wstrechn (те же 6 типов ТС),
# Цены конкурентов (мониторинг; what-if: снижение/рост)
competitor_price_AI92, competitor_price_AI95, competitor_price_DT,
competitor_price_AI92_brend, competitor_price_AI95_brend,
competitor_price_DT_brend, competitor_price_AI100
```

### TIME_VARYING_UNKNOWN_REALS (18 переменных)
Наблюдаемые прошлые — известны за прошлые периоды, недоступны для горизонта прогноза:
```
# Клиентские метрики (2)
corporate_customer_ratio, customer_loyalty_score,
# Производные метрики трафика (16): скорость, плотность, интенсивность
traffic_scorost_1/2_poputn, traffic_scorost_1/2_wstrechn,
traffic_plotnost_1/2_poputn, traffic_plotnost_1/2_wstrechn,
traffic_intensiv_priv_1/2_poputn, traffic_intensiv_priv_1/2_wstrechn,
traffic_intensiv_fiz_1/2_poputn, traffic_intensiv_fiz_1/2_wstrechn
```
Целевые переменные (авторегрессивные входы энкодера) TFT добавляет автоматически
через параметр `target=TARGET_COLS` в TimeSeriesDataSet.

---

## Препроцессинг (eda/eda_preprocessing.py)

Порядок шагов строго фиксирован. Входные данные: `data/merged_data.csv` (1825 × 115).

1. **Пропуски**: `holiday_name → "нет_праздника"` (единственное поле с NaN).
2. **Исключение избыточных колонок**: 8 столбцов удаляются из датафрейма
   (`EXCLUDED_COLS` из `utils/data_utils.py`): station_name, total_pumps, total_fuel_sales,
   quarter, traffic_uroven_jbslugi_1/2_poputn, traffic_uroven_jbslugi_1/2_wstrechn.
   Winsorization не применяется — не описана в статье TFT; TorchNormalizer(robust)
   обрабатывает выбросы целей. После исключения: 115 − 8 = **107 колонок**.
3. **Label Encoding** категориальных → суффикс `_enc`.
   7 колонок: 4 строковых (road_type, direction, season, holiday_name) +
   3 числовых-кода-категорий (road_level, settlement_size, distance_to_city_km).
   Оригинальные колонки сохраняются рядом с _enc. После: **114 колонок**.
4. **Циклическое кодирование**: `day_of_week/week_of_year/month → _sin/_cos`.
   Данные суточные — `hour` отсутствует. Добавляются 6 колонок. После: **120 колонок**.
5. **time_idx**: монотонный порядковый номер дня per-station через `cumcount()`.
   Значения 0–364 для каждой из 5 станций. После: **121 колонка**.
6. **log1p**: только 12 целевых переменных (LOG_COLS = TARGET_COLS).
   Оригинал сохраняется в `_orig`. После: **133 колонки**.
7. **Z-score** нормализация per-station для числовых переменных.
   Статистика (mean, std) вычисляется ТОЛЬКО по train (Jan–Oct 2025).
   Исключены: STATIC_REALS, бинарные, `_enc`, `_orig`, LOG_COLS (цели).
   `NO_ZSCORE_COLS = []` — в 2025 г. цены меняются ежедневно (std > 0).
   LOG_COLS исключены — TorchNormalizer в TFT нормализует цели отдельно.
   **Скейлеры**: `tft/scalers.pkl` — `{station_id: {col: (mean, std)}}` + `log1p_cols`.

Результат: `data/prepared_data.csv` **(1825 × 133)**.

Счёт колонок: 115 − 8 + 7(_enc) + 6(sin/cos) + 1(time_idx) + 12(_orig) = **133**.

---

## Сплиты данных

| Файл | Период | Строк | Целевой период |
|---|---|---|---|
| `data/train.csv` | Jan–Oct 2025 | 1520 | янв–окт (все окна) |
| `data/val.csv` | Oct 2 – Nov 30 2025 | 300 | ноябрь (30-дн. контекст из окт.) |
| `data/test.csv` | Dec 2025 | 155 | декабрь |

**Два понятия val/test:**
- `data/val.csv` / `data/test.csv` — CSV для EDA (Oct 2 – Nov 30 / Dec 2025).
- `val_df` в `prepare_dataset.py` и `train.py` — `df[(date >= Oct 2) & (date <= Nov 30)]`.
  Начало Oct 2 = TRAIN_END − ENCODER_LENGTH + 1: первое окно энкодер Oct 2–31 → декодер Nov 1–7.
  Все окна декодером в ноябре → val_loss = чистая out-of-sample метрика.
- `test_df` в `prepare_dataset.py` — `df[date >= Nov 1]`, только для проверки батча.
  В `predict.py` test создаётся заново: `df[date >= Dec 1 − 30 дн]` = от Nov 1.

`NaNLabelEncoder` обучается на полном df (все 1825 строк) → нет KeyError для
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
- `ENCODER_LENGTH = 30` дн. (1 месяц), `PREDICTION_LENGTH = 7` дн. (1 неделя).
  Рекурсивный прогноз: 1 день (шаг 0), 7 дней (прямой), 1 месяц (4–5 итераций).
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
├── SourceDataForWork/                      — исходные файлы задания, не изменять
│   ├── gas_stations_static.csv             — паспорт 5 АЗС
│   ├── gas_stations_temporal_daily_2025.csv — ежедневные данные 2025 г. (1825 × 88)
│   ├── 2.Данные_по_временным_рядам.xlsx    — временные ряды (исходный Excel)
│   ├── 5.Данные_по_временным_рядам_пилотных_АЗС-трафик по типам транспорта.xlsx
│   └── _Статистические_данные_пилотных_АЗС_2.xlsx — паспортные данные АЗС
│
├── data/                        — генерируются скриптами, отслеживаются git
│   ├── merged_data.csv          — JOIN metadata + data (1825 × 115)
│   ├── prepared_data.csv        — после препроцессинга (1825 × 133)
│   ├── train.csv                — train-сплит Jan–Oct 2025 (1520 строк, ~83%)
│   ├── val.csv                  — val-сплит Nov 2025 с 30-дневным контекстом (300 строк)
│   ├── test.csv                 — test-сплит Dec 2025 (155 строк)
│   ├── predictions.csv          — скользящие 7-дн. прогнозы TFT за декабрь 2025
│   │                              (pred + q10 + q90 по 12 целям × 5 АЗС; actual для сравнения)
│   └── metrics.csv              — MAE / RMSE / MAPE по target × station (декабрь 2025)
│
├── eda/
│   └── eda_preprocessing.py    → prepared_data.csv + train/val/test.csv + tft/scalers.pkl
│
├── report_generating/           — скрипты генерации отчётов (опциональны)
│   ├── column_descriptions.py  → reports/merged_data_columns.txt
│   ├── eda_column_analysis.py  → reports/column_analysis.md
│   └── tft_report.py           → reports/tft_report.docx
│
├── tft/
│   ├── model.ckpt               — лучшая модель (BestModelSync перезаписывает на лету)
│   ├── train_config.json        — гиперпараметры (создаётся train_dashboard, необязателен)
│   ├── prepare_dataset.py      → tft/training_dataset.pkl + tft/dataset_config.pkl
│   ├── train.py                 — обучение с BestModelSync + авто-резюм
│   ├── predict.py               — инференс: скользящие 7-дн. окна на декабрь 2025 →
│   │                              data/predictions.csv + data/metrics.csv
│   ├── checkpoints/             — чекпоинты Lightning (monitor=val_loss, save_top_k=1)
│   └── logs/                    — TensorBoard (tensorboard --logdir tft/logs)
│
├── dashboard/
│   ├── app_dashboard.py         — итоговый дашборд (4 главные вкладки + подвкладки)
│   └── train_dashboard.py       — дашборд обучения (3 вкладки: гиперпараметры, обучение, результаты)
│
├── utils/
│   ├── data_utils.py            — ВСЕ константы проекта (единственный источник истины):
│   │                              TARGET_COLS, LOG_COLS, STATIC_CATS, STATIC_REALS,
│   │                              TIME_VARYING_KNOWN_CATS, TIME_VARYING_KNOWN_REALS,
│   │                              TIME_VARYING_UNKNOWN_REALS, CYCLICAL_FEATURES,
│   │                              TRAIN_END, VAL_END, TEST_START, TEST_END,
│   │                              ENCODER_LENGTH, PREDICTION_LENGTH,
│   │                              QUANTILE_LEVELS, Q_MED, Q_LO, Q_HI
│   │                              + load_and_merge(), fill_missing(), add_cyclical_encoding()
│   └── torch_compat.py          — патч torch.load для PyTorch 2.6
│
├── reports/                     — сгенерированные отчёты, отслеживаются git
│   ├── merged_data_columns.txt  — описание колонок merged_data.csv
│   ├── column_analysis.md       — анализ переменных по группам TFT
│   └── tft_report.docx          — DOCX: переменные + препроцессинг + TFT
│
├── explore_data.py             → data/merged_data.csv
├── requirements.txt
├── .gitignore
└── CLAUDE.md
```

---

## Запуск (строго по порядку из корня проекта)

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Обязательные шаги
python explore_data.py                              # → data/merged_data.csv
python eda/eda_preprocessing.py                     # → data/prepared_data.csv + train/val/test.csv + tft/scalers.pkl

# (опционально) Отчёты
python report_generating/column_descriptions.py     # → reports/merged_data_columns.txt
python report_generating/eda_column_analysis.py     # → reports/column_analysis.md
python report_generating/tft_report.py              # → reports/tft_report.docx

python tft/prepare_dataset.py        # → tft/training_dataset.pkl + tft/dataset_config.pkl

# Обучение — способ А: консоль
python tft/train.py                  # → tft/model.ckpt
# tft/train_config.json подхватывается автоматически если присутствует

# Обучение — способ Б: UI (гиперпараметры + живой вывод + кривые потерь)
# streamlit run dashboard/train_dashboard.py
# После завершения обучения ЛЮБЫМ способом — обязателен запуск predict.py

python tft/predict.py                # → data/predictions.csv + data/metrics.csv

streamlit run dashboard/app_dashboard.py   # итоговый дашборд

# (опционально) Мониторинг обучения — в отдельном терминале во время train.py
# tensorboard --logdir tft/logs
```

---

## Единый дашборд (app_dashboard.py)

Загружает `merged_data.csv` (обязателен). `prepared_data.csv`, `predictions.csv`,
`metrics.csv`, `tft/scalers.pkl`, `tft/model.ckpt` — опциональны; отсутствие любого
даёт информативное сообщение, функционал деградирует gracefully.

### Система фильтров (двухуровневая)

**Глобальные фильтры** (вверху страницы, над всеми вкладками):
- `sel_stations_multi` (Станции) + `sel_months` (Период) → формируют `base_mask` и `df_ov`.
- Влияют на все 4 главные вкладки. Колонки `[1, 1]`.
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
- Сценарий: горизонт 1/7/30 дней; `build_future_ctx` для дат за пределами декабря 2025.
  Легенда графика — под осью X (`y=-0.2`, `margin=dict(b=70)`).
  Кнопка «Рассчитать» выровнена через `st.markdown("&nbsp;")` спейсер.
- Интерпретация VSN: собственные фильтры период + станция; VSN-веса, temporal attention.

**tab4 — Рекомендации**
- Фильтры Станция + Показатель (собственные).
- EDA-рекомендации 2025: MAPE-надёжность, акционный рычаг, рекламный канал,
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

## Дашборд управления обучением (train_dashboard.py)

Запуск: `streamlit run dashboard/train_dashboard.py` (из корня проекта).
Не зависит от `merged_data.csv` — работает с файлами `tft/` напрямую.

### 3 вкладки

**tab1 — ⚙️ Гиперпараметры**
- Слайдеры/поля: `hidden_size` [32/64/128/256], `attention_head_size` [1/2/4/8],
  `hidden_continuous_size`, `learning_rate`, `dropout`, `gradient_clip`,
  `batch_size` [16/32/64/128], `max_epochs`, `patience`.
- Предупреждение если `hidden_continuous > hidden_size/2`; ошибка если `hidden_size % attn_heads ≠ 0`.
- Пресеты CPU/GPU, сброс в дефолты.
- «Сохранить» → `tft/train_config.json`. «Удалить конфиг» → `train.py` вернётся к auto-дефолтам.

**tab2 — 🚀 Обучение**
- «Запустить обучение»: `subprocess.Popen(sys.executable, "tft/train.py")` с `PYTHONIOENCODING=utf-8`.
- Живой вывод stdout через `threading.Thread` + `queue.Queue` → `st.session_state.train_output`.
- Авто-обновление каждые 2 с через `time.sleep(2); st.rerun()` пока `proc.poll() is None`.
- «Остановить»: `proc.terminate()` + `proc.wait(5)`.
- Отображает статус (обучение / завершено / ошибка) и прошедшее время.
- Предупреждает если `tft/training_dataset.pkl` или `data/prepared_data.csv` не найдены.

**tab3 — 📈 Результаты**
- Кривые потерь из `tft/logs/tft_model/version_*/` через `EventAccumulator`
  (теги `train_loss_step` / `train_loss_epoch`, `val_loss`).
- Маркер лучшего val_loss на графике.
- KPI: размер `tft/model.ckpt`, последний чекпоинт, число TensorBoard версий.
- Таблица активных гиперпараметров (из `tft/train_config.json` или дефолты).

### Интеграция с train.py
`tft/train.py` после блока GPU/CPU дефолтов читает `tft/train_config.json` (если есть)
и переопределяет `BATCH_SIZE`, `EPOCHS`, `HIDDEN_SIZE`, `ATTN_HEADS`, `LEARNING_RATE`,
`DROPOUT`, `GRADIENT_CLIP`, `HIDDEN_CONTINUOUS`, `_PATIENCE`.
Если файла нет — используются auto-дефолты (CPU или GPU в зависимости от устройства).

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
