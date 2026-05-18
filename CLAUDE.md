# FinalWorkDashboard

## Задача
Разработать TFT (Temporal Fusion Transformer) модель для прогнозирования 
продаж топлива сети АЗС Татнефть + интерактивный Streamlit дашборд.

## Файлы данных (папка SourceDataForWork/)
Исходные файлы, предоставленные к заданию:
- 5stations_metadata.csv — паспорт 5 АЗС (5 строк, 32 колонки).
  Статические характеристики: тип дороги, услуги, цены, конкуренты.
- 5stations_data.csv — почасовые данные за 2023 год (43800 строк, 72 колонки).
  Погода, трафик, продажи топлива, акции, реклама, цены конкурентов.
- detailed_data.csv — те же данные но для всех 25 АЗС (219000 строк).
- stations_metadata.csv — паспорт всех 25 АЗС (25 строк, 32 колонки).
- TFT_анализ.pdf — статья с описанием архитектуры TFT-модели (Lim et al., 2020).
- Задание.docx — описание ТЗ к выполнению работы.
- описание данных.docx — описание всех переменных в CSV файлах.

Папка Generation/ — материалы, сгенерированные в промпте и внесённые для учёта:
- tatneft_dashboard.html — пример/прототип дашборда, сгенерированный ранее.
- TFT_задание_шпаргалка.pdf — полное описание задачи и план действий.
- TFT_статья_перевод_RU - перевод статьи TFT_анализ.pdf на русский язык.
- Анализ дашборда исходных данных.txt - анализ дашборда по 5 АЗС (исходные данные).

## Связь файлов
metadata JOIN data по колонке station_id

## Входы TFT
- Статические категориальные: road_type_enc, direction_enc, settlement_size_enc
- Статические вещественные: distance_to_city_km, total_pumps, shop_area_m2,
  num_pumps_AI92/95/98/DT_EURO/DT_TANEKO/DT_SUMMER/DT_WINTER, has_*, competitors_within_5km,
  customer/staff/corporate scores, base_price_AI92/95/98/DT_EURO/DT_TANEKO/DT_SUMMER/DT_WINTER
- Известные будущие категориальные: season_enc, day_name_enc, ad_channel_enc, holiday_name_enc
- Известные будущие вещественные: hour/hour_sin/hour_cos, day_of_week/sin/cos,
  week_of_year/sin/cos, month/sin/cos, quarter,
  is_weekend, is_holiday, is_rush_hour, is_night, promotion_*, ad_active, price_*
- Наблюдаемые прошлые: температура, осадки, трафик, конкурентные цены, shop_total_revenue + TARGET_COLS
  (категории магазина перенесены в TARGET_COLS — они одновременно цели и наблюдаемые прошлые)

## Целевые переменные (что предсказываем)
Топливо (7): sales_AI92, sales_AI95, sales_AI98, sales_DT_EURO,
             sales_DT_TANEKO, sales_DT_SUMMER, sales_DT_WINTER
Магазин (5): shop_напитки, shop_закуски, shop_автотовары, shop_кофе, shop_табак
Итого: 12 целевых переменных (TFT предсказывает все совместно)

## Структура проекта

```
FinalWorkDashboard/
├── SourceDataForWork/          — исходные файлы задания, не изменять
│   ├── 5stations_metadata.csv
│   ├── 5stations_data.csv
│   ├── detailed_data.csv
│   ├── stations_metadata.csv
│   ├── TFT_анализ.pdf
│   ├── Задание.docx
│   ├── описание данных.docx
│   └── Generation/             — материалы из диалога с Claude
│       ├── tatneft_dashboard.html
│       ├── TFT_задание_шпаргалка.pdf
│       └── TFT_статья_перевод_RU.pdf
|       └── Анализ дашборда исходных данных.txt
│
├── data/                       — обработанные данные (генерируются скриптами)
│   ├── merged_data.csv         — JOIN metadata + data (43800 × 89)
│   ├── prepared_data.csv       — после всего препроцессинга (43800 × 119)
│   ├── train.csv               — Jan–Oct 2023 (~83.6%)
│   ├── val.csv                 — Nov 2023 (~8.2%)
│   └── test.csv                — Dec 2023 (~8.2%)
│
├── eda/                        — разведочный анализ и препроцессинг
│   ├── eda_column_analysis.py  — анализ колонок: группы, типы, роли для TFT → reports/column_analysis.md
│   ├── eda_preprocessing.py    — EDA-предобработка (консольный вывод) → data/prepared_data.csv + tft/scalers.pkl
│   └── tft_report.py           — DOCX отчёт: переменные, препроцессинг, TFT → reports/tft_report.docx
│
├── tft/                        — TFT модель
│   ├── scalers.pkl             — {station_id: {col: (mean, std)}} + log1p_cols
│   ├── training_dataset.pkl    — объект TimeSeriesDataSet (для train.py)
│   ├── dataset_config.pkl      — параметры датасета (encoder/prediction length, списки колонок)
│   ├── prepare_dataset.py      — подготовка TimeSeriesDataSet → tft/*.pkl
│   ├── train.py                — обучение TFT → tft/model.ckpt
│   ├── predict.py              — инференс на декабрь 2023 → data/predictions.csv + data/metrics.csv
│   ├── checkpoints/            — чекпоинты (лучший, monitor=val_loss)
│   └── logs/                   — TensorBoard логи (tensorboard --logdir tft/logs)
│
├── dashboard/                  — Streamlit дашборды
│   ├── eda_dashboard.py        — интерактивный EDA дашборд (6 вкладок)
│   ├── forecast_dashboard.py   — дашборд прогнозов TFT (5 вкладок)
│   └── tft_interpretation.py  — интерпретация TFT: важность переменных, внимание
│
├── utils/                      — общие утилиты проекта
│   ├── __init__.py
│   ├── data_utils.py           — load_and_merge, fill_missing, add_cyclical_encoding
│   │                             Константы: TARGET_COLS, LOG_COLS, FILL_MAP, STATIC_REALS,
│   │                             CYCLICAL_FEATURES, TRAIN_END, VAL_END, TEST_START, TEST_END,
│   │                             ENCODER_LENGTH, PREDICTION_LENGTH, QUANTILE_LEVELS, Q_MED/Q_LO/Q_HI
│   └── torch_compat.py         — патч torch.load для PyTorch 2.6 (weights_only=False)
│
├── reports/                    — сгенерированные отчёты
│   ├── column_analysis.md      — анализ колонок по группам TFT (Markdown)
│   └── tft_report.docx         — DOCX: переменные + препроцессинг + TFT
│
├── explore_data.py             — первичное изучение и JOIN исходных CSV
├── requirements.txt            — зависимости проекта
└── CLAUDE.md                   — описание проекта
```

## Стек
- Python, pandas, numpy, scikit-learn
- pytorch-forecasting 1.7.0, lightning.pytorch 2.6.1 (TFT)
- Streamlit + plotly для дашборда
- python-docx для генерации DOCX отчётов
- black, isort (PEP 8 форматирование)
- Виртуальное окружение: venv/

## Запуск
```bash
# EDA + препроцессинг (запускать из корня проекта)
python explore_data.py               # → data/merged_data.csv
python eda/eda_column_analysis.py    # → reports/column_analysis.md
python eda/eda_preprocessing.py      # → data/prepared_data.csv + tft/scalers.pkl
python eda/tft_report.py             # → reports/tft_report.docx

# TFT — подготовка и обучение
python tft/prepare_dataset.py      # → tft/training_dataset.pkl + tft/dataset_config.pkl
python tft/train.py                # → tft/model.ckpt (авто-резюм с последнего чекпоинта)

# EDA дашборд
streamlit run dashboard/eda_dashboard.py

# Дашборд прогнозов TFT (требует data/predictions.csv + data/metrics.csv)
streamlit run dashboard/forecast_dashboard.py

# Интерпретация TFT (требует tft/model.ckpt)
streamlit run dashboard/tft_interpretation.py

# Мониторинг обучения (в отдельном терминале)
tensorboard --logdir tft/logs
```

## Препроцессинг (eda/eda_preprocessing.py)
Порядок шагов:
1. Пропуски: holiday_name → "нет_праздника", ad_channel → "нет_рекламы"
2. Выбросы: winsorization по IQR (строки не удаляются — временной ряд)
   Исключены: бинарные, статические переменные из metadata (STATIC_REALS) — паспортные данные АЗС не меняются
3. Label Encoding: категориальные → _enc (TFT строит эмбеддинги сам)
4. Циклическое кодирование: hour/day_of_week/month/week_of_year → _sin/_cos
   Решает проблему разрыва 23:00 → 00:00 (на числовой оси 23 и 0 далеки, на окружности — рядом)
5. time_idx: монотонный порядковый номер часа per-station (cumcount)
6. log1p: все 12 целевых (7 топливо + 5 магазин) + shop_total_revenue (_orig сохранён)
7. Z-score: per-station нормализация числовых
   Исключены: STATIC_REALS (metadata), бинарные, _enc, _orig, LOG_COLS
   LOG_COLS исключены — TorchNormalizer в TFT нормализует цели отдельно
8. Скейлеры: tft/scalers.pkl — для обратного преобразования прогнозов при инференсе

## Настройки TFT (tft/prepare_dataset.py + tft/train.py)
- ENCODER_LENGTH = 168 ч (7 суток ретроспективы)
- PREDICTION_LENGTH = 24 ч (горизонт прогноза)
- hidden_size: 64 (CPU) / 128 (GPU)
- attention_head_size: 2 (CPU) / 4 (GPU)
- hidden_continuous_size: 64
- dropout = 0.15, gradient_clip = 1.0, lr = 3e-4
- EPOCHS: 50 (CPU) / 80 (GPU), EarlyStopping(patience=12)
- reduce_on_plateau_patience = 5
- target_normalizer: TorchNormalizer(method="robust") per target col
- NaNLabelEncoder предобучается на полном df → нет KeyError при val/test
- torch.load патч для PyTorch 2.6 вынесен в utils/torch_compat.py (импортируется в train.py и predict.py)
- Авто-резюм обучения с последнего чекпоинта (glob tft/checkpoints/*.ckpt)

## План работы
1. ✅ Загрузить и объединить файлы по station_id (explore_data.py)
2. ✅ Анализ колонок + EDA-предобработка с sin/cos + DOCX отчёт (eda/)
3. ✅ EDA дашборд (dashboard/eda_dashboard.py) — 6 вкладок, тёмная тема
4. ✅ Подготовка датасета для TFT с sin/cos признаками (tft/prepare_dataset.py)
5. ✅ Полный аудит кода: переменные, препроцессинг, TFT-конфиг, утилиты, PEP 8
6. ✅ Расширение TARGET_COLS: +5 категорий магазина (итого 12 целей)
7. ✅ Рефакторинг: централизация констант, torch_compat, удаление дублей
8. Переобучение TFT с 12 целями → tft/model.ckpt (запущено с улучшенными гиперпараметрами)
   Порядок: eda_preprocessing.py → prepare_dataset.py → train.py
9. ✅ Инференс (tft/predict.py) → data/predictions.csv + data/metrics.csv
10. ✅ Дашборд прогнозов (dashboard/forecast_dashboard.py) — 5 вкладок: прогноз, метрики,
    факторный анализ, интерпретация TFT, сценарий & рекомендации
    · Сценарный анализ: прогноз на декабрь 2023 и январь 2024 (синтетический контекст)
    · What-if: акции, реклама, цена топлива, выбор АЗС/цели, дата и час начала окна
