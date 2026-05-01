# Flight Delay MLOps

Production-grade ML-процесс для прогнозирования задержек авиарейсов и классификации их причин на синтетическом датасете **Flight Delays RU 2023–2025** (220 000 рейсов, 22 аэропорта, 11 авиакомпаний).

Задачи:

- **A — Бинарная классификация:** `is_departure_delayed_15m` (вылет задержан ≥15 мин).
- **B — Мультиклассовая классификация причины:** `probable_delay_cause` ∈ {weather, airport_congestion, reactionary, carrier_operational, security} в двухстадийной схеме.

Полный контекст и правила проекта — в [`CLAUDE.md`](./CLAUDE.md). Документация датасета — в [`docs/dataset/`](./docs/dataset/).

---

## Стек

| Слой | Инструменты |
|---|---|
| Версионирование | git + **DVC** для данных и моделей |
| Обучение | scikit-learn, XGBoost, **LightGBM**, **CatBoost**, Optuna, imbalanced-learn |
| Трекинг | **MLflow** (tracking + Model Registry) |
| Сервис | **FastAPI** + **Docker Compose** |
| Мониторинг | Prometheus + Evidently |
| Качество кода | ruff + mypy + pytest + pre-commit |

---

## Быстрый старт

### 1. Окружение

```bash
cd flight-delay-mlops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
```

То же через Makefile:

```bash
make setup
```

### 2. DVC

```bash
make status
make dag
make repro
```

Датасет уже подключён через DVC:

- `data/raw/flight_delays_ru.parquet.dvc`
- `data/raw/flight_delays_ru_sample.csv.dvc`

Опционально можно настроить remote (локальный или S3/MinIO):

```bash
dvc remote add -d localstore /path/to/dvc-storage
dvc push
```

Если DVC remote не настроен, датасет можно воспроизвести генератором:

```bash
python -m src.data.generate_dataset \
  --out data/raw/flight_delays_ru.parquet \
  --rows 220000 \
  --sample-csv data/raw/flight_delays_ru_sample.csv
```

С тем же `seed=42` файл получается бит-в-бит идентичным.

### 3. Пайплайн и тесты

```bash
make lint
make test
make repro
make verify
```

Обязательные проверки: `tests/test_no_leakage.py` (нет `gt_*` среди признаков), `tests/test_split.py` (time-based split без пересечений).

Текущий DVC-пайплайн:

```text
load -> split -> features -> train_binary -> evaluate
                         \-> train_cause  -> evaluate
             \-> experiments_baseline -> reports/experiments_summary.md
```

Основные train-стадии используют быстрый baseline `logreg` на feature set `EXTENDED`. Экспериментальная стадия 5.1/5.2 сравнивает feature sets и модели `logreg`, `random_forest`, `catboost`; `xgboost` и `lightgbm` подключены как optional backend-и и помечаются `skipped`, если в macOS-окружении не установлен `libomp`.

### 4. Эксперименты

Быстрый baseline-срез этапа 5:

```bash
make experiments-baseline
```

Артефакты:

- `reports/experiments_summary.md` — человекочитаемый отчёт;
- `reports/experiments/baseline_runs.json` — структурированные метрики для DVC;
- `reports/experiments/baseline_runs.csv` — таблица для отчёта/сравнения;
- MLflow experiment `flight-delay-experiments`.

Текущий лучший результат в baseline-срезе: `catboost` + `EXTENDED` для бинарной задачи по F1 и `random_forest` + `EXTENDED` для классификации причины по macro-F1. Для запуска XGBoost/LightGBM на macOS сначала установи OpenMP runtime:

```bash
brew install libomp
make experiments-baseline
```

### 5. MLflow

Локальный режим без Docker:

```bash
make repro
make mlflow-ui
```

Открой `http://127.0.0.1:5000`. Runs логируются в локальную SQLite-базу `mlflow.db`, артефакты — в `mlruns/`; оба пути игнорируются git.

Docker-режим tracking server из трёх контейнеров (Postgres + MinIO + MLflow):

```bash
make mlflow-up
MLFLOW_TRACKING_URI=http://localhost:5000 make repro-force-training
```

Открой:

- MLflow UI: `http://localhost:5000`
- MinIO console: `http://localhost:9001` (`minio` / `minio123`)

Остановить:

```bash
make mlflow-down
```

### 6. Как понять, что требования выполняются

```bash
make verify
```

Скрипт проверяет:

- DVC-стадии `load`, `split`, `features`, `train_binary`, `train_cause`, `evaluate`;
- raw parquet/csv подключены через `.dvc`, а не лежат в git;
- признаки не содержат `gt_*`, таргеты и post-flight поля;
- split строго `train=2023`, `val=2024`, `test=2025`;
- метрики задачи A включают F1, ROC-AUC, PR-AUC;
- метрики задачи B включают macro-F1;
- MLflow tracking настроен;
- отчёт baseline-экспериментов создан;
- большие артефакты данных/моделей не отслеживаются git.

---

## Структура проекта

```
flight-delay-mlops/
├── CLAUDE.md                # контекст проекта (читать перед работой)
├── pyproject.toml           # зависимости, ruff, mypy, pytest
├── params.yaml              # все гиперпараметры пайплайна
├── dvc.yaml                 # стадии: load -> split -> features -> train -> evaluate
├── docker-compose.yml       # postgres + minio + mlflow tracking server
│
├── data/                    # под DVC
│   ├── raw/                 # исходный parquet
│   ├── interim/             # промежуточные артефакты
│   ├── processed/           # train/val/test после препроцессинга
│   └── feedback/            # симулированный продакшн 2025
│
├── docs/dataset/            # DATASET_CARD, DATA_DICTIONARY, DATA_QUALITY_REPORT
├── notebooks/               # ТОЛЬКО EDA
├── src/
│   ├── config.py
│   ├── data/                # generate_dataset, load, validate, split
│   ├── features/            # feature_sets (единая точка), temporal, route, weather
│   ├── models/              # train_binary, train_cause, tune, evaluate, registry
│   ├── api/                 # FastAPI (двухстадийный инференс)
│   └── monitoring/          # logger, metrics, drift, feedback
├── tests/                   # включая test_no_leakage, test_split
├── docker/                  # Dockerfile-ы
└── reports/figures/         # графики экспериментов
```

---

## Жёсткие правила (см. CLAUDE.md §2.3)

1. **Колонки `gt_*` запрещены как признаки** — это компоненты декомпозиции таргета. Все feature-pipeline-ы проходят через `src.features.feature_sets.get_feature_columns()`.
2. **Только time-based split:** train=2023, val=2024, test=2025. Никогда `train_test_split`.
3. **Двухстадийная классификация причины:** Stage 1 (бинарный) → Stage 2 (мультикласс на задержанных).
4. **Метрики не accuracy:** F1 + ROC-AUC + PR-AUC для A, macro-F1 для B.

---

## Текущий статус этапов

- [x] Этап 0 — Bootstrap (структура, конфиги)
- [x] Этап 1 — Интеграция датасета (load/validate/split)
- [x] Этап 2 — Feature engineering (3 feature set + leakage tests)
- [x] Этап 3 — DVC-пайплайн
- [x] Этап 4 — MLflow + базовое обучение
- [ ] Этап 5 — Эксперименты (baseline 5.1/5.2 + CatBoost готовы; дальше Optuna, imbalance, SHAP/drift)
- [ ] Этап 6 — Финальная модель + FastAPI + Docker
- [ ] Этап 7 — Мониторинг и обратная связь
- [ ] Этап 8 — End-to-end демонстрация
