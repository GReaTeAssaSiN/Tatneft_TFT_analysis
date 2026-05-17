# Анализ переменных для TFT-модели

**Файл:** `data/merged_data.csv`  
**Строк:** 43800  
**Колонок:** 89  
**Период:** 2023-01-01 — 2023-12-31  
**Станций:** 5  
**Часов/станцию:** 8760

## Сводная таблица: TFT-роль → количество переменных

| TFT-вход | Колонок (исходных) |
|---|---|
| Идентификатор группы (`group_ids`) | 1 |
| Не входят в модель | 3 |
| Статические категориальные (`static_categoricals`) | 3 |
| Статические вещественные (`static_reals`) | 27 |
| Известные будущие категориальные (`known_cats`) | 4 |
| Известные будущие вещественные (`known_reals`) | 20 (+ 8 sin/cos → 28 в модели) |
| Наблюдаемые прошлые вещественные (`unknown_reals`) | 19 |
| Целевые переменные (`target + unknown_reals`) | 12 |

## Описание групп переменных

### Идентификатор группы (`group_ids`)

Ключ группировки в `TimeSeriesDataSet`. TFT обучается отдельно по каждой станции.

**Колонок:** 1

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `station_id` | int64 | 5 | 0 | [0, 4] | Строка. Ключ группировки для TimeSeriesDataSet. |

### Не входят в модель

Читаемые идентификаторы или избыточные суммарные переменные.

**Колонок:** 3

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `station_name` | object | 5 | 0 | Татнефть-АЗС-001, Татнефть-АЗС-002, Татнефть-АЗС-003, ... | Строка. Только для читаемости, в модель не входит. |
| `timestamp` | datetime64[ns] | 8760 | 0 | 2023-01-01 — 2023-12-31 | datetime. Используется для построения time_idx (cumcount). |
| `total_fuel_sales` | float64 | 589 | 0 | [0, 776] | Линейная сумма 7 целевых переменных. Избыточна для модели. |

### Статические категориальные (`static_categoricals`)

Характеристики АЗС, не меняющиеся во времени (тип дороги, направление, размер). TFT кодирует через Entity Embedding (learnable) — Section 4.1 статьи. Предобработка: `LabelEncoder` → целое число → embedding-слой.

**Колонок:** 3

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `road_type` | object | 2 | 0 | федеральная трасса М, региональная дорога | Label Encoding -> road_type_enc. TFT строит embedding-слой. |
| `direction` | object | 3 | 0 | транзит, из_города, в_город | Label Encoding -> direction_enc. |
| `settlement_size` | object | 3 | 0 | сельская местность (, средний город (50-25, мегаполис (>1млн) | Label Encoding -> settlement_size_enc. |

### Статические вещественные (`static_reals`)

Числовые характеристики АЗС из паспорта (metadata). TFT использует через Static Variable Selection Network. Нормализация **не применяется** — паспортные данные не меняются, исключены из Z-score нормализации (`static_skip` в `eda_preprocessing.py`).

**Колонок:** 27

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `distance_to_city_km` | float64 | 5 | 0 | [3.2, 40.6] | Паспортные данные АЗС. Не изменяются. |
| `total_pumps` | int64 | 3 | 0 | [12, 23] | Паспортные данные АЗС. Не изменяются. |
| `shop_area_m2` | int64 | 3 | 0 | [100, 200] | Паспортные данные АЗС. Не изменяются. |
| `num_pumps_AI92` | int64 | 4 | 0 | [2, 6] | Паспортные данные АЗС. Не изменяются. |
| `num_pumps_AI95` | int64 | 4 | 0 | [2, 6] | Паспортные данные АЗС. Не изменяются. |
| `num_pumps_AI98` | int64 | 3 | 0 | [0, 3] | Паспортные данные АЗС. Не изменяются. |
| `num_pumps_DT_EURO` | int64 | 3 | 0 | [1, 4] | Паспортные данные АЗС. Не изменяются. |
| `num_pumps_DT_TANEKO` | int64 | 3 | 0 | [0, 2] | Паспортные данные АЗС. Не изменяются. |
| `num_pumps_DT_SUMMER` | int64 | 3 | 0 | [1, 3] | Паспортные данные АЗС. Не изменяются. |
| `num_pumps_DT_WINTER` | int64 | 3 | 0 | [1, 3] | Паспортные данные АЗС. Не изменяются. |
| `has_car_wash` | int64 | 2 | 0 | [0, 1] | Паспортные данные АЗС. Бинарный 0/1. Не изменяется. |
| `has_tire_service` | int64 | 2 | 0 | [0, 1] | Паспортные данные АЗС. Бинарный 0/1. Не изменяется. |
| `has_cafe` | int64 | 1 | 0 | [1, 1] | Паспортные данные АЗС. Бинарный 0/1. Не изменяется. |
| `has_hotel` | int64 | 2 | 0 | [0, 1] | Паспортные данные АЗС. Бинарный 0/1. Не изменяется. |
| `has_shop` | int64 | 1 | 0 | [1, 1] | Паспортные данные АЗС. Бинарный 0/1. Не изменяется. |
| `competitors_within_5km` | int64 | 3 | 0 | [0, 2] | Паспортные данные АЗС. Не изменяются. |
| `customer_loyalty_score` | float64 | 5 | 0 | [52.1, 79.8] | Паспортные данные АЗС. Не изменяются. |
| `staff_quality_score` | float64 | 5 | 0 | [71.8, 97] | Паспортные данные АЗС. Не изменяются. |
| `staff_engagement_score` | float64 | 5 | 0 | [55.6, 87.3] | Паспортные данные АЗС. Не изменяются. |
| `corporate_customer_ratio` | float64 | 5 | 0 | [0.19, 0.32] | Паспортные данные АЗС. Не изменяются. |
| `base_price_AI92` | float64 | 5 | 0 | [47.6, 50.9] | Паспортные данные АЗС. Не изменяются. |
| `base_price_AI95` | float64 | 5 | 0 | [50.3, 54.5] | Паспортные данные АЗС. Не изменяются. |
| `base_price_AI98` | float64 | 5 | 0 | [55.3, 58.5] | Паспортные данные АЗС. Не изменяются. |
| `base_price_DT_EURO` | float64 | 5 | 0 | [50.5, 52.4] | Паспортные данные АЗС. Не изменяются. |
| `base_price_DT_TANEKO` | float64 | 5 | 0 | [52.1, 54.2] | Паспортные данные АЗС. Не изменяются. |
| `base_price_DT_SUMMER` | float64 | 4 | 0 | [49.7, 51.9] | Паспортные данные АЗС. Не изменяются. |
| `base_price_DT_WINTER` | float64 | 5 | 0 | [51.5, 54.1] | Паспортные данные АЗС. Не изменяются. |

### Известные будущие категориальные (`known_cats`)

Категориальные признаки, известные заранее (сезон, день недели, праздник). TFT подаёт в encoder **и** decoder — позволяет учесть будущий контекст. Предобработка: `NaN` → семантическое значение, `LabelEncoder` → embedding.

**Колонок:** 4

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `season` | object | 4 | 0 | winter, spring, summer, ... | Label Encoding -> season_enc. |
| `day_name` | object | 7 | 0 | Sunday, Monday, Tuesday, ... | Label Encoding -> day_name_enc. |
| `ad_channel` | object | 7 | 0 | ТВ, нет_рекламы, email, ... | NaN -> 'нет_рекламы'. Label Encoding -> ad_channel_enc. |
| `holiday_name` | object | 10 | 0 | Новый год, нет_праздника, Рождество, ... | NaN -> 'нет_праздника'. Label Encoding -> holiday_name_enc. |

### Известные будущие вещественные (`known_reals`)

Числовые признаки, известные заранее (час, цены, акции, флаги дней). TFT подаёт в encoder **и** decoder. Предобработка: Z-score. Циклические (hour, dow, month, woy) → sin/cos. Бинарные (is_weekend, промо-флаги) — без нормализации.

**Колонок:** 20

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `hour` | int64 | 24 | 0 | [0, 23] | Z-score. + sin/cos: hour_sin=sin(2pi*hour/24), hour_cos=cos(2pi*hour/24). Циклическое кодирование устраняет разрыв 23:00->00:00. |
| `day_of_week` | int64 | 7 | 0 | [0, 6] | Z-score. + sin/cos: dow_sin=sin(2pi*dow/7), dow_cos=cos(2pi*dow/7). |
| `week_of_year` | int64 | 52 | 0 | [1, 52] | Z-score. + sin/cos (period=52). |
| `month` | int64 | 12 | 0 | [1, 12] | Z-score. + sin/cos (period=12). |
| `quarter` | int64 | 4 | 0 | [1, 4] | Z-score. |
| `is_weekend` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. Не нормализуется. |
| `is_holiday` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. Не нормализуется. |
| `is_rush_hour` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. Не нормализуется. |
| `is_night` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. Не нормализуется. |
| `promotion_fuel_active` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. Акции планируются заранее. |
| `promotion_shop_active` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. |
| `promotion_cafe_active` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. |
| `ad_active` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. |
| `price_AI92` | float64 | 5 | 0 | [47.6, 50.9] | Z-score per-station. Цена устанавливается заранее. |
| `price_AI95` | float64 | 5 | 0 | [50.3, 54.5] | Z-score per-station. |
| `price_AI98` | float64 | 5 | 0 | [55.3, 58.5] | Z-score per-station. |
| `price_DT_EURO` | float64 | 5 | 0 | [50.5, 52.4] | Z-score per-station. |
| `price_DT_TANEKO` | float64 | 5 | 0 | [52.1, 54.2] | Z-score per-station. |
| `price_DT_SUMMER` | float64 | 4 | 0 | [49.7, 51.9] | Z-score per-station. |
| `price_DT_WINTER` | float64 | 5 | 0 | [51.5, 54.1] | Z-score per-station. |

### Наблюдаемые прошлые вещественные (`unknown_reals`)

Наблюдаемые переменные — доступны только в прошлом (погода, трафик). TFT использует **только в encoder** (168 часов ретроспективы). Предобработка: Winsorization IQR + Z-score per-station. `weather_condition`: LabelEncoder → float (через prescaler в TFT).

**Колонок:** 19

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `temperature` | float64 | 615 | 0 | [-35, 35] | Z-score per-station. Winsorization IQR. |
| `weather_condition` | object | 5 | 0 | снег, ясно, облачно, ... | Label Encoding -> weather_condition_enc (float). Используется как вещественное — TFT обрабатывает через prescaler. |
| `precipitation_mm` | float64 | 113 | 0 | [0, 18.4] | Z-score per-station. Winsorization IQR. |
| `visibility_km` | float64 | 136 | 0 | [0.2, 15] | Z-score per-station. |
| `wind_speed_ms` | float64 | 151 | 0 | [0, 15] | Z-score per-station. Winsorization IQR. |
| `is_snow` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. Не нормализуется. |
| `is_rain` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. |
| `is_fog` | int64 | 2 | 0 | [0, 1] | Бинарный 0/1. |
| `traffic_Passengers_cars` | int64 | 620 | 0 | [4, 735] | Z-score per-station. Winsorization IQR. |
| `traffic_Truck_short` | int64 | 99 | 0 | [1, 106] | Z-score per-station. Winsorization IQR. |
| `traffic_Truck` | int64 | 168 | 0 | [1, 186] | Z-score per-station. Winsorization IQR. |
| `traffic_Truck_long` | int64 | 147 | 0 | [0, 159] | Z-score per-station. Winsorization IQR. |
| `traffic_Transporter` | int64 | 52 | 0 | [0, 53] | Z-score per-station. Winsorization IQR. |
| `traffic_Undefined` | int64 | 27 | 0 | [0, 26] | Z-score per-station. |
| `total_traffic` | int64 | 1014 | 0 | [7, 1.27e+03] | Z-score per-station. Является суммой компонент трафика. VSN автоматически снизит вес при высокой коллинеарности. |
| `shop_total_revenue` | float64 | 1786 | 0 | [0, 3.56e+03] | log1p -> _orig сохраняется. Z-score НЕ применяется (TorchNormalizer). |
| `competitor_price_AI92` | float64 | 83 | 0 | [45.2, 53.4] | Z-score per-station. Winsorization IQR. |
| `competitor_price_AI95` | float64 | 95 | 0 | [47.8, 57.2] | Z-score per-station. Winsorization IQR. |
| `competitor_price_DT` | float64 | 71 | 0 | [48, 55] | Z-score per-station. Winsorization IQR. |

### Целевые переменные (`target + unknown_reals`)

12 переменных (7 топливо + 5 магазин) — **одновременно цель прогноза и observed past**. TFT статья (Section 3): target является частью observed inputs. Предобработка: log1p (skew устранён) + `TorchNormalizer(robust)` внутри TFT. Z-score в `eda_preprocessing.py` **не применяется**. `_orig`-колонки сохраняются для обратного преобразования прогнозов.

**Колонок:** 12

| Переменная | Тип | Уник. | Пропуски | Диапазон / значения | Предобработка |
|---|---|---|---|---|---|
| `shop_напитки` | float64 | 318 | 0 | [0, 444] | log1p -> _orig. TorchNormalizer(robust) внутри TFT. Z-score не применяется. |
| `shop_закуски` | float64 | 586 | 0 | [0, 897] | log1p -> _orig. TorchNormalizer(robust). |
| `shop_автотовары` | float64 | 804 | 0 | [0, 1.32e+03] | log1p -> _orig. TorchNormalizer(robust). |
| `shop_кофе` | float64 | 589 | 0 | [0, 799] | log1p -> _orig. TorchNormalizer(robust). |
| `shop_табак` | float64 | 209 | 0 | [0, 261] | log1p -> _orig. TorchNormalizer(robust). |
| `sales_AI92` | float64 | 234 | 0 | [0, 264] | log1p -> _orig. TorchNormalizer(robust) внутри TFT. Z-score не применяется. |
| `sales_AI95` | float64 | 263 | 0 | [0, 301] | log1p -> _orig. TorchNormalizer(robust). |
| `sales_AI98` | float64 | 73 | 0 | [0, 75] | log1p -> _orig. TorchNormalizer(robust). |
| `sales_DT_EURO` | float64 | 73 | 0 | [0, 75] | log1p -> _orig. TorchNormalizer(robust). |
| `sales_DT_TANEKO` | float64 | 24 | 0 | [0, 23] | log1p -> _orig. TorchNormalizer(robust). |
| `sales_DT_SUMMER` | float64 | 38 | 0 | [0, 38] | log1p -> _orig. TorchNormalizer(robust). |
| `sales_DT_WINTER` | float64 | 24 | 0 | [0, 27] | log1p -> _orig. TorchNormalizer(robust). |

## Циклическое кодирование (решение проблемы 23:00 → 00:00)

Числовой признак `hour=23` и `hour=0` далеки (разница 23), хотя физически соседние. То же для `day_of_week` (6→0) и `month` (12→1).

**Решение:** sin/cos-кодирование проецирует признак на единичную окружность:
```
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```
Евклидово расстояние между (hour=23) и (hour=0) на окружности минимально.

| Признак | Period | Добавляемые колонки |
|---|---|---|
| `hour` | 24 | `hour_sin`, `hour_cos` |
| `day_of_week` | 7 | `day_of_week_sin`, `day_of_week_cos` |
| `month` | 12 | `month_sin`, `month_cos` |
| `week_of_year` | 52 | `week_of_year_sin`, `week_of_year_cos` |

## Коллинеарность и избыточность

| Переменная | Решение |
|---|---|
| `total_fuel_sales` | Исключена — линейная сумма 7 целевых, избыточна |
| `total_traffic` | Оставлена — VSN автоматически снизит вес при высокой коллинеарности |
| `base_price_*` vs `price_*` | Разные: `base_price` из паспорта (константа), `price` — текущая цена с акциями |
| `is_snow/rain/fog` vs `weather_condition` | Частичное перекрытие, но `weather_condition` кодирует градацию (ясно/облачно/туман/дождь/снег) |

## Итог: входы TFT-модели

| Параметр | Значение |
|---|---|
| Encoder length | 168 ч (7 суток ретроспективы) |
| Decoder length | 24 ч (горизонт прогноза) |
| Целевых выходов | 12 |

| TFT-вход | Колонок |
|---|---|
| `static_categoricals` | 3 |
| `static_reals` | 27 |
| `time_varying_known_categoricals` | 4 |
| `time_varying_known_reals` | 20 (20 base + 8 sin/cos) |
| `time_varying_unknown_reals` | 19 |
| `target` | 12 |