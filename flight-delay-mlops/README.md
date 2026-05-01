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

### 1. Окружение (uv — рекомендуется)

```bash
# установка uv (если ещё нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd flight-delay-mlops
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

Альтернатива через pip:

```bash
cd flight-delay-mlops
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### 2. DVC

```bash
dvc status
dvc dag
dvc repro
```

Датасет уже подключён через DVC:

- `data/raw/flight_delays_ru.parquet.dvc`
- `data/raw/flight_delays_ru_sample.csv.dvc`

Опционально можно настроить remote (локальный или S3/MinIO):

```bash
dvc remote add -d localstore /path/to/dvc-storage
dvc push
```

### 3. Воспроизведение датасета

```bash
python -m src.data.generate_dataset \
  --out data/raw/flight_delays_ru.parquet \
  --rows 220000 \
  --sample-csv data/raw/flight_delays_ru_sample.csv
```

С тем же `seed=42` файл получается бит-в-бит идентичным.

### 4. Пайплайн и тесты

```bash
pytest tests/ -v
dvc repro
```

Обязательные проверки: `tests/test_no_leakage.py` (нет `gt_*` среди признаков), `tests/test_split.py` (time-based split без пересечений).

Текущий DVC-пайплайн:

```text
load -> split -> features -> train_binary -> evaluate
                         \-> train_cause  -> evaluate
```

По умолчанию используется быстрый sklearn-baseline `logreg` на feature set `EXTENDED`. CatBoost/LightGBM/XGBoost выносятся в следующий этап с MLflow-экспериментами.

---

## Структура проекта

```
flight-delay-mlops/
├── CLAUDE.md                # контекст проекта (читать перед работой)
├── pyproject.toml           # зависимости, ruff, mypy, pytest
├── params.yaml              # все гиперпараметры пайплайна
├── dvc.yaml                 # стадии: load -> split -> features -> train -> evaluate
├── docker-compose.yml       # mlflow + api + monitoring
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
- [ ] Этап 4 — MLflow + базовое обучение
- [ ] Этап 5 — Эксперименты (8 шт.)
- [ ] Этап 6 — Финальная модель + FastAPI + Docker
- [ ] Этап 7 — Мониторинг и обратная связь
- [ ] Этап 8 — End-to-end демонстрация
