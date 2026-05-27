# Прогнозирование продаж АЗС Татнефть — TFT модель + EDA / Прогнозы / Интерпретация

Финальная работа по курсу **Технологии и Архитектуры Больших Данных**.  
Разработка мультицелевой модели на базе Temporal Fusion Transformer (TFT, Lim et al. 2020)
для прогнозирования ежедневных продаж топлива и товаров магазина сети АЗС Татнефть.

---

## Задача

Построить мультицелевую модель временных рядов (12 целей совместно):
- **5 видов топлива** (л/день): АИ-92, АИ-95, ДТ, ДТ Bio, АИ-100 Bio
- **7 категорий магазина** (руб/день): напитки безалкогольные, кондитерка/снеки, мороженое, автотовары, кафе/вся еда, кофе/горячие напитки, табак

**Горизонт прогноза:** 7 дней (1 неделя).  
**Ретроспектива:** 30 дней (1 месяц).  
**Прогноз:** роллинговый (наиболее точный 1-шаговый на каждый день декабря) + прямой 7-дневный + рекурсивный на месяц (4–5 итераций).

---

## Данные

5 пилотных АЗС Татнефть, **ежедневные данные за 2025 год** (1825 строк = 5 АЗС × 365 дней):

| Файл | Описание | Колонок |
|---|---|---|
| `SourceDataForWork/gas_stations_static.csv` | Паспорт 5 АЗС: тип дороги, направление, услуги, кол-во колонок | 29 |
| `SourceDataForWork/gas_stations_temporal_daily_2025.csv` | Погода, трафик, продажи, цены, акции, метрики клиентов (ежедневно) | 88 |

JOIN по `station_id`: 29 − 2 (ключи) + 88 = **115 колонок** → `data/merged_data.csv`.

---

## Стек

| Компонент | Технология |
|---|---|
| Язык | Python 3.14.3 |
| Модель | pytorch-forecasting 1.7.0 (TFT) |
| Обучение | lightning.pytorch 2.6.1 |
| Дашборд | Streamlit + Plotly |
| Препроцессинг | pandas, numpy, scikit-learn |
| Отчёты | python-docx 1.2.0 |
| Мониторинг обучения | TensorBoard |

---

## Архитектура TFT

Temporal Fusion Transformer (Lim et al. 2020) принимает 4 типа входов:

| Тип | Переменные | Кол-во |
|---|---|---|
| Статические категориальные (SC) | тип дороги, уровень дороги, направление, населённый пункт, расстояние до города | 5 |
| Статические вещественные (SR) | площадь магазина, количество колонок по видам топлива, наличие услуг, конкуренты | 21 |
| Известные будущие категориальные (KC) | сезон, название праздника | 2 |
| Известные будущие вещественные (KR) | дата (sin/cos), флаги выходных/праздников, погода, акции, цены топлива, трафик по типам ТС и направлениям, цены конкурентов | 53 |
| Наблюдаемые прошлые вещественные (UR) | клиентские метрики, производные метрики трафика (скорость, плотность, интенсивность) | 18 |
| Целевые переменные (авторегрессия) | 5 топлив + 7 категорий магазина | 12 |

**Итого TFT-признаков:** 5 + 21 + 2 + 53 + 18 + 12 = **111**

Горизонт прогноза: **7 дней (1 неделя)**. Ретроспектива: **30 дней (1 месяц)**.  
Функция потерь: `QuantileLoss` — 7 квантилей [0.02, 0.1, 0.25, **0.5**, 0.75, 0.9, 0.98].  
Нормализатор целей: `MultiNormalizer([TorchNormalizer(method="robust") × 12])`.

---

## Структура проекта

```
FinalWorkDashboard/
├── SourceDataForWork/                      — исходные файлы задания, не изменять
│   ├── gas_stations_static.csv             — паспорт 5 АЗС (29 колонок)
│   ├── gas_stations_temporal_daily_2025.csv — ежедневные данные 2025 г. (88 колонок)
│   ├── 2.Данные_по_временным_рядам.xlsx    — временные ряды (исходный Excel)
│   ├── 5.Данные_по_временным_рядам_пилотных_АЗС-трафик по типам транспорта.xlsx
│   └── _Статистические_данные_пилотных_АЗС_2.xlsx — паспортные данные АЗС
│
├── data/                        — генерируются скриптами, отслеживаются git
│   ├── merged_data.csv          — JOIN metadata + data (1825 × 115)
│   ├── prepared_data.csv        — после препроцессинга (1825 × 133)
│   ├── train.csv                — train-сплит Jan–Oct 2025 (1520 строк, ~83%)
│   ├── val.csv                  — val-сплит Oct 2 – Nov 30 2025 (300 строк, с 30-дн. контекстом)
│   ├── test.csv                 — test-сплит Dec 2025 (155 строк)
│   ├── predictions.csv          — скользящие 7-дн. прогнозы TFT за декабрь 2025
│   │                              (медиана + q10/q90 по 12 целям × 5 АЗС; actual для сравнения)
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
│   ├── model.ckpt               — лучшая модель (BestModelSync перезаписывает при новом best val_loss)
│   ├── train_config.json        — гиперпараметры (создаётся train_dashboard, необязателен)
│   ├── prepare_dataset.py      → tft/training_dataset.pkl + tft/dataset_config.pkl
│   ├── train.py                 — обучение с BestModelSync + авто-резюм с чекпоинта
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
│   │                              TARGET_COLS, STATIC_CATS/REALS, KNOWN/UNKNOWN vars,
│   │                              TRAIN_END, VAL_END, TEST_START/END,
│   │                              ENCODER_LENGTH, PREDICTION_LENGTH, QUANTILE_LEVELS
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
└── CLAUDE.md                    — полная спецификация проекта
```

---

## Запуск

```bash
# 1. Установка зависимостей
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Объединение исходных данных
python explore_data.py               # → data/merged_data.csv (1825 × 115)

# 3. Препроцессинг
python eda/eda_preprocessing.py      # → data/prepared_data.csv (1825 × 133) + train/val/test.csv + tft/scalers.pkl

# 4. (опционально) Генерация отчётов
python report_generating/column_descriptions.py    # → reports/merged_data_columns.txt
python report_generating/eda_column_analysis.py    # → reports/column_analysis.md
python report_generating/tft_report.py             # → reports/tft_report.docx

# 5. Подготовка датасета TFT
python tft/prepare_dataset.py        # → tft/training_dataset.pkl + tft/dataset_config.pkl

# 6. Обучение TFT — выберите один из двух способов:

# ── Способ А: через консоль ──────────────────────────────────
python tft/train.py                  # → tft/model.ckpt
# tft/train_config.json подхватывается автоматически, если был сохранён заранее

# ── Способ Б: через дашборд обучения ────────────────────────
streamlit run dashboard/train_dashboard.py
# ⚙️  Гиперпараметры → настроить → «Сохранить конфиг»
# 🚀  Обучение       → «Запустить обучение» (вывод в реальном времени)
# 📈  Результаты     → кривые потерь, метрики по запускам
# ⚠️  После завершения обучения любым способом — обязателен шаг 7

# 7. Инференс (декабрь 2025) — обязателен после обучения
python tft/predict.py                # → data/predictions.csv + data/metrics.csv

# 8. Итоговый дашборд
streamlit run dashboard/app_dashboard.py

# (опционально) Мониторинг обучения — запускать в отдельном терминале во время шага 6
tensorboard --logdir tft/logs
```

---

## Препроцессинг

Входные данные: `data/merged_data.csv` (1825 × 115). Порядок шагов строго фиксирован.

| Шаг | Действие | Δ колонок | Итого |
|---|---|---|---|
| 0 | merged_data.csv | — | 115 |
| 1 | Заполнение пропусков: `holiday_name → "нет_праздника"` | 0 | 115 |
| 2 | Исключение 8 избыточных колонок (station_name, total_pumps, total_fuel_sales, quarter, traffic_uroven_jbslugi_*) | −8 | 107 |
| 3 | Label Encoding 7 категориальных (4 строковых + 3 числовых-кода) → суффикс `_enc` | +7 | 114 |
| 4 | Циклическое sin/cos: day_of_week, week_of_year, month | +6 | 120 |
| 5 | `time_idx` — монотонный порядковый номер дня per-station (0–364) | +1 | 121 |
| 6 | log1p только для 12 целевых переменных; оригинал сохраняется в `_orig` | +12 | 133 |
| 7 | Z-score per-station для числовых переменных (кроме STATIC_REALS, бинарных, `_enc`, `_orig`, целевых) | 0 | 133 |

Результат: `data/prepared_data.csv` **(1825 × 133)**.  
Из 133 колонок: 111 модельных (TFT) + 22 служебных (time_idx, station_id, date + 7 исходных _enc + 12 _orig).

---

## Сплиты данных

| Файл | Период | Строк | Назначение |
|---|---|---|---|
| `data/train.csv` | Jan–Oct 2025 | 1520 | обучение (~83%) |
| `data/val.csv` | Oct 2 – Nov 30 2025 | 300 | валидация с 30-дн. контекстом |
| `data/test.csv` | Dec 2025 | 155 | тестирование |

**Почему val и test включают контекстный период?**  
Для первого окна ноября (Nov 1) энкодер TFT нуждается в 30 днях ретроспективы (октябрь).
`TimeSeriesDataSet.from_dataset` требует непрерывного временного ряда, поэтому
`val_df` передаётся с данными с 1 января, а `test_df` — с 1 ноября.
Окна, перекрывающиеся с train-периодом, не создают утечки данных (модель видела их при обучении).

---

## Настройки TFT

| Параметр | CPU | GPU |
|---|---|---|
| `hidden_size` | 64 | 128 |
| `attention_head_size` | 2 | 4 |
| `hidden_continuous_size` | 32 | 64 |
| `batch_size` | 32 | 64 |
| `max_epochs` | 50 | 80 |

Общие: `encoder_length=30`, `prediction_length=7`, `learning_rate=3e-4`,
`dropout=0.15`, `gradient_clip=1.0`, `EarlyStopping(patience=12)`.

**BestModelSync** — кастомный callback: перезаписывает `tft/model.ckpt` при каждом новом
лучшем `val_loss`, не ожидая конца обучения. Обучение авто-возобновляется с последнего
чекпоинта (через `glob("tft/checkpoints/*.ckpt")`).

---

## Дашборды

### `dashboard/app_dashboard.py` — основной

Единый дашборд. Загружает `merged_data.csv` сразу после шага 2.
Остальные файлы (`prepared_data.csv`, `predictions.csv`, `metrics.csv`, `model.ckpt`) —
опциональны; отсутствие любого даёт информативное сообщение (graceful degradation).

**Фильтры (двухуровневые):**
- **Глобальные** (Станции + Период) — влияют на все 4 вкладки через `df_ov`
- **Фильтры анализа** (Топливо, Магазин, 9 доп. условий) — только вкладка «Анализ данных», формируют `fdf`

| Вкладка | Содержимое |
|---|---|
| **Обзор** | KPI-строка; паттерны продаж (area-чарты, bar, сезонный, тепловая карта); сравнение АЗС; структура магазина; Акции & Реклама |
| **Анализ данных** | Погода & Трафик; Акции & Реклама; Корреляции (матрица, топ-5 факторов); Статистика (распределения, scatter) |
| **Прогноз TFT** | Метрики & Точность (MAPE, R²); Прогноз vs Факт (q10–q90, scatter, гистограмма ошибок); Сценарий What-if (горизонт 1/7/30 дней); Интерпретация VSN + temporal attention |
| **Рекомендации** | Рекомендации по данным 2025: MAPE-надёжность, акционный рычаг, рекламный канал, окно спроса, лучший день, трафик как сигнал |

### `dashboard/train_dashboard.py` — управление обучением

Отдельный дашборд для настройки и запуска обучения без командной строки.
Запуск: `streamlit run dashboard/train_dashboard.py`

| Вкладка | Содержимое |
|---|---|
| **⚙️ Гиперпараметры** | Слайдеры: hidden_size, attention_heads, hidden_continuous, learning_rate, dropout, gradient_clip, batch_size, max_epochs, patience; пресеты CPU/GPU; сохранение в `tft/train_config.json` |
| **🚀 Обучение** | Кнопки Запустить/Остановить; живой stdout `tft/train.py`; статус и прошедшее время |
| **📈 Результаты** | Кривые train/val loss (TensorBoard EventAccumulator); информация о лучшем чекпоинте; таблица активных гиперпараметров |

---

## Авторы

[Горшков Алексей Олегович aka GReaTeAssaSiN](https://github.com/GReaTeAssaSiN), 2026 (студент 1 курса магистратуры КНИТУ-КАИ направления 01.04.02 Математическое моделирование и интеллектуальный анализ данных)  
[Горшков Андрей Олегович aka ImmortalAbode](https://github.com/ImmortalAbode), 2026 (студент 1 курса магистратуры КНИТУ-КАИ направления 01.04.02 Математическое моделирование и интеллектуальный анализ данных)  
Разработка выполнена с использованием [Claude Code](https://claude.ai/code) (Anthropic)
