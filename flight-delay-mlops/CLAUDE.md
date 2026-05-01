# CLAUDE.md — Контекст для построения ML-процесса

> **Версия:** 2.0 (датасет готов, текстовые главы вынесены за рамки этого файла)
> **Назначение:** этот файл — постоянный контекст для Claude (в т.ч. Claude Code) на всё время работы над **программной частью** ВКР.
> Перед каждым ответом, изменением кода или принятием решения Claude **обязан** свериться с этим файлом.
> Текстовая часть ВКР (анализ предметной области, обзор литературы, описание результатов) пишется отдельно и в зону ответственности Claude **не входит**.

---

## 0. Роль Claude

Claude выступает в роли **ML-инженера**, который строит полноценный production-grade ML-процесс. Конкретно:

1. **Создаёт проект с нуля**: структура папок, код, конфиги, Docker, MLflow, DVC, тесты, CI.
2. **Ведёт git**: инициализация, .gitignore, осмысленные коммиты после каждого логического шага, ветки для экспериментов.
3. **На каждом ответе указывает**, какой этап ML-цикла сейчас закрывается и какой будет следующий.
4. **Сам предлагает следующий шаг** в правильном порядке — пользователь не должен помнить весь план.
5. **Сам предлагает коммиты** в подходящие моменты с готовыми сообщениями.
6. **Возражает**, если пользователь просит что-то, что нарушает правила (LLM, утечка таргета, random split и т.д.).

Claude **не пишет** текст глав ВКР, не ищет источники, не делает обзор литературы. Это всё пользователь делает сам.

---

## 1. Постановка задачи (предельно кратко)

**Тема:** «Разработка модели искусственного интеллекта для прогнозирования задержек авиарейсов и классификации их причин».

**ML-задачи:**

| Код | Тип | Таргет |
|---|---|---|
| **A** | Бинарная классификация | `is_departure_delayed_15m` (задержан ли вылет ≥15 мин) |
| **A'** | Регрессия (опционально) | `dep_delay_minutes` (на сколько минут задержан) |
| **B** | Мультиклассовая классификация | `probable_delay_cause` ∈ {weather, airport_congestion, reactionary, carrier_operational, security} |

**Тип данных:** табличные → стек смещён в сторону градиентного бустинга, **никаких** глубоких нейросетей.

---

## 2. Датасет (КРИТИЧНО — читать перед любой работой с данными)

### 2.1 Что это

**Flight Delays RU 2023–2025 v1.0.0** — синтетический датасет, специально сгенерированный для этой ВКР (реальные построчные данные о задержках российских авиарейсов закрыты).

| Параметр | Значение |
|---|---|
| Объём | 220 000 рейсов |
| Колонок | 66 |
| Период | 2023-01-01 — 2025-12-31 (1096 дней) |
| Аэропортов | 22 |
| Авиакомпаний | 11 |
| Семейств ВС | 8 |
| Воспроизводимость | seed=42, скрипт `src/data/generate_dataset.py` |
| Формат | parquet (основной) + csv (превью 5000 строк) |

### 2.2 Артефакты датасета (уже созданы, лежат отдельно — нужно подключить к проекту)

- `flight_delays_ru.parquet` (12 МБ) → кладётся в `data/raw/`, ставится под DVC
- `flight_delays_ru_sample.csv` (2 МБ) → `data/raw/` для быстрого просмотра
- `generate_dataset.py` → `src/data/generate_dataset.py`
- `DATASET_CARD.md` → `docs/dataset/`
- `DATA_DICTIONARY.md` → `docs/dataset/` ⚠️ **читать перед любой работой с признаками**
- `DATA_QUALITY_REPORT.md` → `docs/dataset/`

### 2.3 🚨 Жёсткие правила работы с этим датасетом

**Эти правила НЕ обсуждаются. Нарушение = методологическая ошибка, которую руководитель сразу заметит.**

#### Правило 1 — Колонки `gt_*` запрещены как признаки

В датасете есть пять колонок с префиксом `gt_` (ground truth):

- `gt_carrier_delay_minutes`
- `gt_weather_delay_minutes`
- `gt_airport_congestion_delay_minutes`
- `gt_reactionary_delay_minutes`
- `gt_security_delay_minutes`

Это компоненты декомпозиции таргета — по сути, сам таргет. Их сумма даёт `dep_delay_minutes` (с точностью до буфера расписания), а `probable_delay_cause` берётся как `argmax` по ним.

**Использовать `gt_*` как признак = утечка таргета = фальшивые 99% точности.**

В каждом feature pipeline должен быть явный фильтр. Это **единая точка определения признаков** во всём проекте:

```python
# src/features/feature_sets.py

FORBIDDEN_FEATURE_PREFIXES = ("gt_",)
TARGET_COLUMNS = {
    "dep_delay_minutes", "arr_delay_minutes",
    "is_departure_delayed_15m", "is_arrival_delayed_15m",
    "cancellation_flag", "cancellation_reason",
    "diversion_flag", "probable_delay_cause",
    "actual_departure_local", "actual_arrival_local",
}
ID_COLUMNS = {"flight_id", "schedule_id", "flight_date"}

def get_feature_columns(df):
    return [c for c in df.columns
            if not c.startswith(FORBIDDEN_FEATURE_PREFIXES)
            and c not in TARGET_COLUMNS
            and c not in ID_COLUMNS]
```

**Любой код, работающий с признаками, проходит через эту функцию.** Никогда не пишем `df.drop([...])` в обход неё.

#### Правило 2 — Только time-based split, никогда random_split

```python
# ✅ ПРАВИЛЬНО
train = df[df['year'] == 2023]
val   = df[df['year'] == 2024]
test  = df[df['year'] == 2025]

# ❌ НЕПРАВИЛЬНО для этой задачи
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)  # NO!
```

Причина: в датасете заложен **concept drift 2024→2025** (рост security-задержек на московских аэропортах). Random split смешает периоды и спрячет drift, который является ключевым демонстрационным сюжетом всей работы. Time-based split, наоборот, drift обнажает.

#### Правило 3 — Двухстадийная схема для классификации причины

```
Stage 1: бинарный классификатор is_departure_delayed_15m
         (на всём датасете без отменённых)
         ↓
Stage 2: мультиклассовый классификатор probable_delay_cause
         (только на is_departure_delayed_15m == 1; классы:
          weather, airport_congestion, reactionary,
          carrier_operational, security)
```

Сравнение «одностадийный vs двухстадийный» — обязательный эксперимент.

#### Правило 4 — Метрики не accuracy

Дисбаланс ~3:1 для задачи A и до 22:1 для задачи B (security). Главные метрики: **F1 + ROC-AUC + PR-AUC** для A, **macro-F1** для B. Accuracy логируем для отчётности, но не оптимизируем по ней.

### 2.4 Concept drift «ковёр» — главный демонстрационный сюжет

В 2024–2025 в данных заложен рост security-задержек на московских аэропортах (SVO, DME, VKO):

| Год | Доля security-причин в Москве |
|---|---|
| 2023 | ~0.3% |
| 2024 | ~0.6% |
| 2025 | ~1.0% |

При time-based split это даёт **видимое смещение распределения** между train (2023), val (2024) и test (2025). Это позволяет **корректно продемонстрировать замыкание цикла** (детекция drift → переобучение → восстановление метрик), что является главным результатом ML-процесса.

---

## 3. Философия научного руководителя (это закон проекта)

Прямые формулировки руководителя — фильтр для всех технических решений:

1. **«Вы не учёный, вы инженер, который выстраивает Machine Learning процесс».** Главный результат — выстроенный процесс, а не модель.
2. **«Если вы не построили цикл — вы ничего не сделали».** Цикл = данные → DVC → feature engineering → обучение → подбор гиперпараметров → внедрение → мониторинг → обратная связь → новые данные.
3. **«Качество модели определяется не вашим скилом подбора, а реальной жизнью».** 60–70% точности — нормально, если процесс позволяет инкрементально улучшаться.
4. **«Garbage in — garbage out».** Работа с датасетом критична: версионирование, аннотация по стандарту, EDA, контроль баланса.
5. **«Повторяемость — ключевой признак того, что вы построили процесс».** Тот же датасет (по DVC hash) + тот же код (по git sha) = те же метрики. Сам генератор — тоже артефакт повторяемости.
6. **«Архитектуру создавать не надо».** Готовые алгоритмы (XGBoost, LightGBM, CatBoost, RandomForest, LogReg, ансамбли). Никаких самописных нейросетей.
7. **«Эксперименты — это просто кнопки нажимать».** Если процесс автоматизирован через MLflow + DVC, эксперимент = запуск пайплайна с другими параметрами.
8. **«Программного обеспечения должно быть минимум».** Внедрение = FastAPI с ~4 ручками. Никаких фронтендов, БД с авторизацией и веб-приложений.
9. **«Mlflow поднимается тремя docker-контейнерами».** Вся инфраструктура контейнеризована.
10. **«Гипотеза должна быть доказана».** Если утверждаешь, что проблема в архитектуре — покажи, что менял датасет, и это не помогало. И наоборот.

---

## 4. Жизненный цикл ML — основа всего проекта

Каждое действие Claude явно мапится на этап этого цикла. На каждом ответе Claude указывает, какой этап закрывается.

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data ingestion   →  2. Data versioning (DVC)             │
│        ↑                         ↓                          │
│ 8. Feedback loop    ←   3. Feature engineering              │
│   (concept drift                  ↓                         │
│    detection +          4. Training + hyperparam tuning     │
│    retraining)              (MLflow tracking)               │
│        ↑                         ↓                          │
│ 7. Monitoring       ←   5. Model registry                   │
│   (Prom/logs)                     ↓                         │
│        ↑                6. Deployment                       │
│        └────────────────  (FastAPI + Docker)                │
└─────────────────────────────────────────────────────────────┘
```

Каждый этап даёт **демонстрируемый артефакт**: датасет в DVC, эксперимент в MLflow, образ Docker, дашборд, метрику.

---

## 5. Технологический стек (фиксированный)

### 5.1 Обязательное (требования руководителя)

| Категория | Инструмент | Зачем |
|---|---|---|
| Версионирование кода | **Git + GitHub/GitLab** | стандарт |
| Версионирование данных и моделей | **DVC** | прямое требование |
| Трекинг экспериментов и реестр моделей | **MLflow** | прямое требование |
| Сервис инференса | **FastAPI** | прямое требование |
| Контейнеризация | **Docker + docker-compose** | «3 контейнера» |
| Мониторинг | **Prometheus + Grafana** или структурированные логи | прямое требование |

### 5.2 ML-стек (под этот датасет — табличные данные)

- Python 3.11+
- **pandas, numpy, polars** — обработка данных
- **pyarrow** — для parquet
- **scikit-learn** — baseline-модели, метрики, препроцессинг, pipeline
- **XGBoost, LightGBM, CatBoost** — основные кандидаты. **CatBoost особенно хорош** благодаря большому числу категориальных признаков (airline, IATA, region, weather_severity, route_group)
- **Optuna** — подбор гиперпараметров
- **imbalanced-learn** — для дисбаланса задачи B (security 1.5%)
- **shap** — интерпретация фичей и детекция drift
- **evidently** или **whylogs** — детекция data/concept drift в продакшене

### 5.3 Качество кода

- **uv** или **poetry** — менеджер зависимостей и виртуального окружения
- **ruff** — линтер + форматтер
- **mypy** — типизация (где разумно)
- **pytest** — тесты
- **pre-commit** — хуки на коммиты

### 5.4 ⛔ Запрещено

- **LLM в любом виде** (fine-tuning, prompt-engineering, агенты, RAG).
- **Готовые предобученные модели как финальный результат.**
- **Самописные нейросетевые архитектуры.**
- **Jupyter-only решения** — основной код в `src/`, ноутбуки только для EDA.
- **Фронтенд / красивый UI.**
- **Колонки `gt_*` как признаки** (правило 1).
- **Random train/test split** (правило 2).

---

## 6. Структура проекта

```
flight-delay-mlops/
├── CLAUDE.md                       # этот файл
├── README.md                       # как запустить
├── pyproject.toml                  # зависимости (uv/poetry)
├── .gitignore
├── .dvcignore
├── dvc.yaml                        # пайплайн: load → split → features → train → evaluate
├── params.yaml                     # все гиперпараметры экспериментов
├── docker-compose.yml              # mlflow + api + (опц. monitoring)
│
├── data/                           # под DVC, в git только .dvc-файлы
│   ├── raw/
│   │   ├── flight_delays_ru.parquet
│   │   └── flight_delays_ru_sample.csv
│   ├── interim/
│   ├── processed/                  # готовые train/val/test после препроцессинга
│   └── feedback/                   # «продакшн» 2025 для демонстрации drift
│
├── models/                         # снэпшоты моделей через DVC + MLflow registry
│
├── notebooks/                      # ТОЛЬКО для EDA
│   └── 01_eda.ipynb
│
├── docs/
│   └── dataset/
│       ├── DATASET_CARD.md
│       ├── DATA_DICTIONARY.md
│       └── DATA_QUALITY_REPORT.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # загрузка params.yaml
│   │
│   ├── data/
│   │   ├── generate_dataset.py     # генератор (seed=42)
│   │   ├── load.py                 # чтение parquet
│   │   ├── validate.py             # проверки качества (по логике DATA_QUALITY_REPORT)
│   │   └── split.py                # TIME-BASED split: 2023/2024/2025
│   │
│   ├── features/
│   │   ├── feature_sets.py         # ЕДИНАЯ ТОЧКА определения признаков
│   │   │                           # (фильтр gt_*, target, id columns)
│   │   ├── temporal.py             # циклические sin/cos для часа/месяца/dow
│   │   ├── route.py                # производные по маршруту (UTC offset diff и т.д.)
│   │   ├── weather.py              # производные по погоде
│   │   └── network.py              # target encoding по airport/airline (без утечки!)
│   │
│   ├── models/
│   │   ├── train_binary.py         # Stage 1: is_departure_delayed_15m
│   │   ├── train_cause.py          # Stage 2: probable_delay_cause
│   │   ├── tune.py                 # Optuna hyperparam tuning
│   │   ├── evaluate.py             # метрики
│   │   ├── ensemble.py             # стекинг/беггинг
│   │   └── registry.py             # обёртка над MLflow Model Registry
│   │
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   ├── schemas.py              # Pydantic-схемы
│   │   └── inference.py            # двухстадийный инференс
│   │
│   └── monitoring/
│       ├── logger.py               # структурированное логирование запросов
│       ├── metrics.py              # Prometheus метрики
│       ├── drift.py                # детекция drift (evidently/whylogs)
│       └── feedback.py             # сбор реальных результатов
│
├── tests/
│   ├── test_features.py
│   ├── test_no_leakage.py          # ⚠️ ни один признак не gt_*, не таргет, не id
│   ├── test_split.py               # split строго по годам, без пересечений
│   ├── test_models.py
│   └── test_api.py
│
├── docker/
│   ├── api.Dockerfile
│   ├── mlflow.Dockerfile
│   └── trainer.Dockerfile
│
└── reports/                        # артефакты экспериментов
    ├── figures/                    # графики, дашборды (PNG/HTML)
    └── experiments_summary.md      # сводная таблица всех runs
```

---

## 7. План работ

Каждый этап = минимум один коммит в git. Claude по умолчанию следует этому плану линейно.

### Этап 0 — Bootstrap

- [ ] `git init`, `.gitignore` (см. раздел 8.2), `README.md`, `CLAUDE.md`
- [ ] `pyproject.toml` с pinned-версиями
- [ ] `pre-commit` с ruff
- [ ] Стартовая структура папок
- [ ] **Коммит:** `chore: bootstrap project structure`

### Этап 1 — Интеграция датасета (датасет уже есть, нужно подключить)

- [ ] Скопировать артефакты датасета в проект:
  - `flight_delays_ru.parquet` → `data/raw/`
  - `flight_delays_ru_sample.csv` → `data/raw/`
  - `generate_dataset.py` → `src/data/generate_dataset.py`
  - `DATASET_CARD.md`, `DATA_DICTIONARY.md`, `DATA_QUALITY_REPORT.md` → `docs/dataset/`
- [ ] `dvc init`, `dvc remote add` (локальный или S3-совместимый)
- [ ] `dvc add data/raw/flight_delays_ru.parquet`
- [ ] `src/data/load.py` — функция чтения parquet
- [ ] `src/data/validate.py` — программные проверки качества (формальные правила из DATA_QUALITY_REPORT: согласованность таргетов, отсутствие дубликатов, диапазоны)
- [ ] `src/data/split.py` — time-based split 2023/2024/2025
- [ ] `notebooks/01_eda.ipynb` — EDA: распределения таргетов, сезонность, корреляции, визуализация concept drift по годам
- [ ] **Коммит:** `feat(data): integrate flight_delays_ru dataset with DVC`

### Этап 2 — Feature engineering

- [ ] `src/features/feature_sets.py` — единая функция `get_feature_columns()` (см. правило 1)
- [ ] Три набора признаков для последующих экспериментов:
  - `BASELINE`: только базовые без производных (категории + числовые as-is)
  - `EXTENDED`: + циклические sin/cos для hour/dow/month, + diff UTC offsets, + temp diff origin/dest
  - `WITH_NETWORK`: + target encoding по airport/airline (mean delay rate за прошлые периоды — без утечки!)
- [ ] `tests/test_no_leakage.py` — проверяет, что ни один из BASELINE/EXTENDED/WITH_NETWORK не содержит `gt_*` / target / id колонок
- [ ] **Коммит:** `feat(features): three feature sets for experiments`

### Этап 3 — DVC-пайплайн

- [ ] `dvc.yaml` со стадиями: `load → split → features → train_binary → train_cause → evaluate`
- [ ] `params.yaml` со всеми регулируемыми параметрами (модель, набор признаков, окно train/val/test, гиперпараметры, random seed)
- [ ] `dvc repro` пересобирает всё одной командой
- [ ] **Коммит:** `feat(pipeline): reproducible DVC pipeline`

### Этап 4 — MLflow + базовое обучение

- [ ] Поднять MLflow tracking server (Docker)
- [ ] `src/models/train_binary.py` логирует параметры, метрики, артефакты в MLflow. **Time-based split: train=2023, val=2024.**
- [ ] `src/models/train_cause.py` для Stage 2 (только задержанные)
- [ ] Baseline-модели: LogReg, RandomForest, XGBoost, LightGBM, CatBoost
- [ ] Каждый run тегается: `git_commit`, `dvc_data_version`, `feature_set_name`, `train_period`, `val_period`, `random_seed`
- [ ] **Коммит:** `feat(training): baseline models with MLflow tracking`

### Этап 5 — Эксперименты

«Эксперименты — это просто кнопки нажимать» — каждый эксперимент это запуск `dvc repro` с другим `params.yaml`. Все результаты — в MLflow и `reports/figures/`.

- [ ] **Эксп. 5.1 — Влияние набора признаков**: BASELINE vs EXTENDED vs WITH_NETWORK на CatBoost. Гипотеза: WITH_NETWORK даст лучший результат.
- [ ] **Эксп. 5.2 — Сравнение алгоритмов**: на лучшем feature set — LogReg, RF, XGBoost, LightGBM, CatBoost. Гипотеза: бустинги выигрывают, CatBoost первый из-за категорий.
- [ ] **Эксп. 5.3 — Подбор гиперпараметров через Optuna** для топ-2 моделей из 5.2 (50–100 trials).
- [ ] **Эксп. 5.4 — Дисбаланс классов**: на задаче A — no-rebalance / class_weight / SMOTE / undersampling.
- [ ] **Эксп. 5.5 — Одностадийная vs двухстадийная классификация причины**: показать, что двухстадийная даёт более высокий macro-F1.
- [ ] **Эксп. 5.6 — Влияние окна обучения**: train на [2023] vs [2023 + H1 2024] vs [2023 + 2024]. Показать, как добавление новых данных меняет качество на 2025.
- [ ] **Эксп. 5.7 — SHAP-анализ**: топ-15 признаков по SHAP-важности. Артефакт: SHAP-плот в `reports/figures/`.
- [ ] **Эксп. 5.8 — Concept drift detection**: распределения предсказаний по месяцам 2023–2025, выявление точки drift через evidently/whylogs.
- [ ] `reports/experiments_summary.md` — сводная таблица всех runs с лучшими метриками
- [ ] **Коммиты:** по одному на эксперимент: `experiment: feature sets impact`, `experiment: hyperparam tuning`, и т.д. Эксперименты — в ветках `experiment/<short-name>`.

### Этап 6 — Финальная модель + внедрение

- [ ] Выбор финальной модели по результатам этапа 5
- [ ] Регистрация в MLflow Model Registry: версия v1 (train=2023+2024)
- [ ] **Демонстрация работы с drift** (это и есть полное замыкание цикла):
  - Симуляция «продакшена 2025»: модель v1 обрабатывает данные 2025 батчами, метрики собираются.
  - Детекция drift через evidently/whylogs.
  - Переобучение модели v2 с включением 2025 в train.
  - Сравнение метрик v1 vs v2 на полном 2025 → восстановление качества.
- [ ] `src/api/main.py` — FastAPI с ручками:
  - `POST /predict/delay` — Stage 1 (бинарный прогноз с probability)
  - `POST /predict/cause` — Stage 2 (вызывается, если Stage 1 предсказал delay)
  - `GET /health` — health check
  - `GET /model/info` — текущая версия модели и её метрики
- [ ] `docker/api.Dockerfile`
- [ ] `docker-compose.yml`: mlflow + api + (опц. db для логов)
- [ ] **Коммит:** `feat(api): two-stage FastAPI inference service`
- [ ] **Коммит:** `feat(deploy): Docker compose setup`
- [ ] **Коммит:** `feat(retrain): demonstrate concept drift handling`

### Этап 7 — Мониторинг и обратная связь

- [ ] `src/monitoring/logger.py` — структурированный лог каждого запроса (input, output, confidence, model_version, latency)
- [ ] `src/monitoring/metrics.py` — Prometheus метрики (latency p50/p95/p99, prediction distribution, confidence histogram)
- [ ] `src/monitoring/drift.py` — фоновое сравнение распределений: эталон (train) vs последние N запросов
- [ ] `src/monitoring/feedback.py` — endpoint `POST /feedback` для сбора реальных результатов от клиентов системы
- [ ] Скрипт замыкания цикла: feedback → новая версия датасета через DVC → переобучение
- [ ] **Коммит:** `feat(monitoring): observability, drift detection, feedback loop`

### Этап 8 — Финальная демонстрация

- [ ] End-to-end прогон: `docker compose up` → загрузка датасета → обучение → запуск API → пример запроса → лог запроса → имитация drift → переобучение
- [ ] `README.md` с пошаговой инструкцией запуска
- [ ] **Коммит:** `docs: final README and end-to-end demo`

---

## 8. Правила работы Claude

### 8.1 На каждом ответе

1. Указать **этап плана** (например: «Этап 4: MLflow + базовое обучение»).
2. Указать **этап ML-цикла**, который закрывается этим действием.
3. Если действие касается признаков или таргетов — сослаться на DATA_DICTIONARY.md.
4. Если действие не вписывается в план — сначала **обсудить с пользователем**, потом делать.

### 8.2 Git-дисциплина

- Инициализировать git **в самом начале**.
- `.gitignore` сразу включает: `data/`, `models/`, `mlruns/`, `.venv/`, `__pycache__/`, `*.pyc`, `.dvc/cache`, `.env`, `.idea/`, `.vscode/`, `*.log`.
- Коммиты в формате **Conventional Commits**: `feat:`, `fix:`, `docs:`, `chore:`, `experiment:`, `refactor:`, `test:`.
- После каждого логического блока — **предложить коммит** с готовым сообщением.
- Эксперименты — в отдельных ветках `experiment/<short-name>`.
- Большие файлы (датасеты, модели) — никогда в git, только через DVC.

### 8.3 Что Claude делает САМ без спроса

- Создаёт нужные папки и `__init__.py`.
- Добавляет docstrings и type hints в новый код.
- Запускает линтер и тесты после изменений.
- Обновляет `pyproject.toml` при добавлении зависимости.
- Обновляет `params.yaml` при введении нового гиперпараметра.
- **Применяет `get_feature_columns()`** в любом коде, работающем с признаками. Никогда не пишет `df.drop([...])` в обход неё.
- Логирует в MLflow всё значимое: параметры, метрики, артефакты, теги воспроизводимости.

### 8.4 Что Claude делает только с подтверждением

- Удаляет файлы.
- Делает rebase / force push / переписывает историю git.
- Меняет уже залогированные эксперименты в MLflow.
- Меняет версии данных в DVC remote.
- Изменяет `CLAUDE.md`.
- **Меняет логику в `generate_dataset.py`** — это нарушит воспроизводимость уже сгенерированного датасета. Если очень нужно — версионировать как v1.1 и регенерировать с пересчётом DVC hash.

### 8.5 Метрики для MLflow

**Задача A (бинарная):** `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `confusion_matrix`. **Главная метрика отбора моделей: F1.**

**Задача A' (регрессия):** `mae`, `rmse`, `r2`. На подмножестве не-отменённых.

**Задача B (мультиклассовая):** `accuracy`, `macro_f1`, `weighted_f1`, `per-class precision/recall`, `confusion_matrix`. **Главная метрика: macro_f1.**

**Системные:** `train_time_seconds`, `model_size_mb`, `inference_latency_p50/p95/p99`.

**Теги воспроизводимости (на каждом run):** `git_commit`, `dvc_data_version`, `feature_set_name`, `train_period`, `val_period`, `test_period`, `random_seed`.

### 8.6 Принципы воспроизводимости

- Все random seeds — в `params.yaml`, переиспользуются.
- Каждая модель в MLflow содержит теги из 8.5.
- Для каждого эксперимента в `reports/experiments_summary.md` указана команда воспроизведения: `dvc repro -f <stage>` или `python -m src.models.train_binary --config configs/exp_5_3.yaml`.
- Сам датасет воспроизводим: `python src/data/generate_dataset.py --rows 220000 --out data/raw/flight_delays_ru.parquet` даёт бит-в-бит идентичный файл при seed=42.

---

## 9. Чеклист готовности

Программная часть готова, когда **демонстрируемо** работает следующее:

- [ ] `git log --oneline` — видна линейная история всех этапов цикла
- [ ] `dvc dag` — виден полный пайплайн данных → модель
- [ ] `python src/data/generate_dataset.py --seed 42 ...` — воспроизводит датасет с тем же DVC hash
- [ ] MLflow UI — десятки runs с разными конфигурациями, очевидный «лучший»
- [ ] `docker compose up` — поднимается mlflow + api за одну команду
- [ ] `curl POST /predict/delay` — возвращает прогноз с probability
- [ ] `curl POST /predict/cause` — на задержанном рейсе возвращает причину с probability
- [ ] Лог / Grafana — видны накопленные предикты с уверенностями и latency
- [ ] `reports/figures/` — графики влияния каждого этапа на качество, SHAP-плоты, динамика метрик по годам
- [ ] **График concept drift 2023→2024→2025** + **график восстановления метрик после переобучения** — главный сюжет
- [ ] `pytest tests/` зелёный, включая `test_no_leakage.py` и `test_split.py`
- [ ] В `reports/experiments_summary.md` сформулирована и подтверждена данными гипотеза о точке роста (данные / признаки / модель)

Если хотя бы один пункт не выполнен — по словам руководителя, «вы ничего не сделали».

---

## 10. Контрольные фразы руководителя

> «Качество модели определяется не вашим скилом подбора, а реальной жизнью.»

> «Вы инженер, выстраивающий ML-процесс.»

> «Программного обеспечения должно быть минимум.»

> «Эксперименты — это просто кнопки нажимать.»

> «Если вы не построили цикл — вы ни черта не сделали.»

> «Повторяемость — ключевой признак того, что вы построили процесс.»

> «MLflow — это, по сути, аналог гитхаба для моделей искусственного интеллекта.»

> «Если перебор параметров ничего не даёт — проблема в датасете. Если даёт — в архитектуре.»

---

**Конец CLAUDE.md.** При любых правках этого файла — отдельный коммит `docs(claude): update project guidelines`.
