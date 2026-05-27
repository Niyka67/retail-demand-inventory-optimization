# Retail Demand Forecasting & Inventory Optimization

Проект демонстрирует полный pipeline для retail/supply chain analytics:

1. генерация демо-данных продаж для сети из 42 филиалов;
2. прогнозирование дневного спроса на уровне `store_id / sku_id / date`;
3. сравнение ML-модели с baseline-прогнозом;
4. расчет рекомендуемого заказа через Newsvendor Model.

Проект сделан как демонстрационная версия. Реальные коммерческие данные не публикуются.

## Бизнес-задача

В розничной сети важно поддерживать баланс между двумя рисками:

- заказать слишком мало и получить дефицит товара на полке;
- заказать слишком много и получить списания скоропортящейся продукции.

Цель проекта - построить прогноз спроса и на его основе рассчитать рекомендуемый объем заказа для магазинов и SKU.

## Данные

Демо-датасет имитирует продажи в сети из 42 филиалов.

Основные поля:

- `date` - дата продажи;
- `store_id` - идентификатор филиала;
- `sku_id` - идентификатор товара;
- `quantity` - дневной спрос;
- `temperature` - температура;
- `is_weekend` - признак выходного дня;
- `category` - товарная категория.

## Pipeline

```text
data/sample_sales.csv
        |
        v
src/forecast_demand.py
        |
        v
outputs/forecast_results.csv
        |
        v
src/optimize_inventory.py
        |
        v
outputs/inventory_recommendations.csv
```

## Модели

В проекте используются два подхода:

- baseline: средний спрос за последние 7 дней по `store_id / sku_id`;
- ML-модель: LightGBM с лаговыми, rolling и календарными признаками.

Если LightGBM не установлен в локальной среде, скрипт автоматически использует простую fallback-модель на NumPy. Это нужно только для удобного запуска демо-проекта.

Качество прогноза оценивается через:

- MAE;
- RMSE;
- WAPE.

## Inventory Optimization

Для расчета заказа используется Newsvendor Model. Оптимальный заказ считается через critical ratio:

```text
critical_ratio = understock_cost / (understock_cost + overstock_cost)
```

Далее точечный ML-прогноз корректируется с учетом ошибки модели и экономических рисков списания/дефицита. Для расчета квантиля нормального распределения используется стандартный модуль Python `statistics`.

## Как запустить

Установить зависимости:

```bash
pip install -r requirements.txt
```

Сгенерировать демо-данные:

```bash
python src/generate_sample_data.py
```

Построить прогноз:

```bash
python src/forecast_demand.py
```

Рассчитать рекомендуемый заказ:

```bash
python src/optimize_inventory.py
```

## Стек

Python, Pandas, NumPy, LightGBM, Matplotlib, Seaborn, Git.
