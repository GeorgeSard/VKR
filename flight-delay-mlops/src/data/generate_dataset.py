"""
generate_dataset.py
===================

Воспроизводимый генератор синтетического датасета задержек авиарейсов
по 22 крупнейшим аэропортам России и 11 авиакомпаниям за 2023-01-01 … 2025-12-31.

Особенности:
- Никаких утечек: компонентные задержки только в "ground truth" блоке колонок.
- Реалистичные паттерны: сезонность, бимодальный час пик, hub-эффекты,
  каскадные задержки (reactionary), погодные шоки.
- Concept drift 2024-2025: учащение security-задержек на московских аэропортах
  (моделируем эффект, описанный в публикациях Росавиации/Forbes).
- Полная воспроизводимость: фиксированный seed=42.

Запуск:
    python generate_dataset.py --out flight_delays_ru.parquet --rows 200000
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# =============================================================================
# 1. СПРАВОЧНИКИ
# =============================================================================

@dataclass(frozen=True)
class Airport:
    iata: str
    city: str
    region: str          # ЦФО, СЗФО и т.д.
    lat: float
    lon: float
    utc_offset: int
    hub_tier: int        # 1 = крупнейший хаб, 2 = крупный, 3 = региональный
    capacity: int        # пропускная способность (рейсов/день, синтет.)
    summer_avg_t: float  # средняя температура июля, °C
    winter_avg_t: float  # средняя температура января, °C


AIRPORTS: list[Airport] = [
    # Москва — три аэропорта, разный профиль (континентальный климат)
    Airport("SVO", "Москва", "ЦФО", 55.972642, 37.414589, 3, 1, 1300, 21, -8),
    Airport("DME", "Москва", "ЦФО", 55.408611, 37.906111, 3, 1, 1100, 21, -8),
    Airport("VKO", "Москва", "ЦФО", 55.591531, 37.261486, 3, 1, 600, 21, -8),
    # Санкт-Петербург (морской климат)
    Airport("LED", "Санкт-Петербург", "СЗФО", 59.800278, 30.262500, 3, 1, 900, 19, -6),
    # Юг
    Airport("AER", "Сочи", "ЮФО", 43.449928, 39.956589, 3, 2, 700, 25, 6),
    Airport("ROV", "Ростов-на-Дону", "ЮФО", 47.493888, 39.924444, 3, 3, 250, 26, -3),
    Airport("MRV", "Минеральные Воды", "СКФО", 44.225158, 43.081889, 3, 3, 200, 25, -2),
    Airport("KRR", "Краснодар", "ЮФО", 45.034689, 39.170539, 3, 2, 350, 27, 1),
    # Поволжье
    Airport("KZN", "Казань", "ПФО", 55.606186, 49.278728, 3, 3, 300, 21, -11),
    Airport("KUF", "Самара", "ПФО", 53.504922, 50.164308, 4, 3, 250, 22, -11),
    Airport("UFA", "Уфа", "ПФО", 54.557506, 55.874417, 5, 3, 280, 21, -13),
    Airport("GOJ", "Нижний Новгород", "ПФО", 56.230050, 43.784167, 3, 3, 220, 20, -10),
    # Урал и Сибирь
    Airport("SVX", "Екатеринбург", "УФО", 56.743108, 60.802744, 5, 2, 450, 19, -14),
    Airport("CEK", "Челябинск", "УФО", 55.305836, 61.503281, 5, 3, 230, 19, -15),
    Airport("OVB", "Новосибирск", "СФО", 55.012622, 82.650656, 7, 2, 500, 20, -16),
    Airport("KJA", "Красноярск", "СФО", 56.172800, 92.493283, 7, 2, 350, 19, -16),
    Airport("IKT", "Иркутск", "СФО", 52.268028, 104.388975, 8, 3, 280, 18, -18),
    Airport("TOF", "Томск", "СФО", 56.380278, 85.208333, 7, 3, 150, 19, -18),
    # Дальний Восток
    Airport("VVO", "Владивосток", "ДФО", 43.398889, 132.148056, 10, 2, 280, 20, -12),
    Airport("KHV", "Хабаровск", "ДФО", 48.528056, 135.188333, 10, 3, 220, 21, -19),
    Airport("YKS", "Якутск", "ДФО", 62.092778, 129.770556, 9, 3, 150, 20, -38),
    # Калининград — анклав, морской климат
    Airport("KGD", "Калининград", "СЗФО", 54.890278, 20.592778, 2, 3, 220, 18, -1),
]

IATA_TO_AIRPORT: dict[str, Airport] = {a.iata: a for a in AIRPORTS}


@dataclass(frozen=True)
class Airline:
    code: str
    name: str
    fleet_avg_age: float       # средний возраст парка (синтет.)
    base_punctuality: float    # коэф. пунктуальности 0..1, чем выше — тем меньше задержек
    base_iata: str             # базовый аэропорт (хаб)


AIRLINES: list[Airline] = [
    Airline("SU", "Аэрофлот",      8.5, 0.92, "SVO"),
    Airline("FV", "Россия",       11.2, 0.90, "LED"),
    Airline("S7", "S7 Airlines",   9.8, 0.91, "OVB"),
    Airline("U6", "Уральские АЛ", 12.5, 0.86, "SVX"),
    Airline("UT", "ЮТэйр",        15.1, 0.83, "VKO"),
    Airline("DP", "Победа",        7.2, 0.94, "VKO"),
    Airline("WZ", "Red Wings",    14.0, 0.82, "DME"),
    Airline("YC", "Якутия",       16.5, 0.78, "YKS"),
    Airline("5N", "Smartavia",    12.8, 0.85, "LED"),
    Airline("A4", "Азимут",        4.5, 0.93, "AER"),
    Airline("HZ", "Аврора",       11.0, 0.86, "VVO"),
]

CODE_TO_AIRLINE: dict[str, Airline] = {a.code: a for a in AIRLINES}

# Семейства ВС: подбираются по дальности маршрута
SHORTHAUL_AC = ["SSJ100", "ATR72", "A319", "A320"]
MIDHAUL_AC = ["A320", "A321", "B737", "SSJ100"]
LONGHAUL_AC = ["B777", "A330", "A321", "B737"]


# =============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def block_minutes_for(distance_km: float) -> int:
    """Эмпирическое плановое время в воздухе + рулёжка."""
    cruise_kmh = 780
    base = 25  # рулёжка туда-сюда
    return int(round(base + (distance_km / cruise_kmh) * 60))


def aircraft_for_route(distance_km: float, rng: random.Random) -> str:
    if distance_km < 1500:
        return rng.choice(SHORTHAUL_AC)
    if distance_km < 4000:
        return rng.choice(MIDHAUL_AC)
    return rng.choice(LONGHAUL_AC)


def is_holiday_window(d: pd.Timestamp) -> int:
    """Окна повышенного спроса: НГ-каникулы, майские, августовский пик отпусков."""
    m, day = d.month, d.day
    if (m == 12 and day >= 28) or (m == 1 and day <= 8):
        return 1
    if m == 5 and day <= 10:
        return 1
    if m == 6 and day >= 25:
        return 1
    if m == 8 and day <= 31:
        return 1
    return 0


def season_multiplier(month: int) -> float:
    """
    Множитель базовой задержки по месяцу:
    - зима (12,1,2): ×1.4 — снег, обледенение
    - весна (3,4,5): ×0.85 — лучший период
    - лето (6,7,8): ×1.8 — грозы + пиковая нагрузка + 2024-2025 БПЛА
    - осень (9,10,11): ×1.0 — норма
    """
    return {
        1: 1.40, 2: 1.30, 3: 0.95, 4: 0.85, 5: 0.85,
        6: 1.65, 7: 1.85, 8: 1.95, 9: 0.95, 10: 0.90,
        11: 1.05, 12: 1.40,
    }[month]


def hour_multiplier(h: int) -> float:
    """Бимодальная нагрузка: утренний и вечерний пик."""
    morning = math.exp(-((h - 8) ** 2) / 6.0)
    evening = math.exp(-((h - 19) ** 2) / 8.0)
    return 0.55 + 1.4 * (0.6 * morning + 1.0 * evening)


def dow_multiplier(dow: int) -> float:
    """0=Пн ... 6=Вс. Пятница и воскресенье хуже."""
    return [1.00, 0.95, 0.95, 1.00, 1.20, 1.05, 1.25][dow]


def kover_effect(date: pd.Timestamp, origin_iata: str, dest_iata: str) -> float:
    """
    Имитация эффекта 'ковёр' на московских аэропортах в 2024-2025:
    пики в мае-сентябре, особенно сильно в 2025.
    """
    moscow_set = {"SVO", "DME", "VKO"}
    if origin_iata not in moscow_set and dest_iata not in moscow_set:
        return 1.0
    if date.year == 2023:
        return 1.0
    seasonal = 1.0
    if 5 <= date.month <= 9:
        seasonal = 1.7 if date.year == 2024 else 2.2
    else:
        seasonal = 1.15 if date.year == 2024 else 1.30
    return seasonal


# =============================================================================
# 3. ГЕНЕРАЦИЯ ПОГОДЫ ПО АЭРОПОРТУ И ДАТЕ
# =============================================================================

def airport_temperature(airport: Airport, doy: int) -> float:
    """Базовая температура по климатологии аэропорта и дню года (синусоида)."""
    summer_peak = 200  # ~19 июля
    seasonal = math.cos(2 * math.pi * (doy - summer_peak) / 365.0)
    mid = (airport.summer_avg_t + airport.winter_avg_t) / 2.0
    amp = (airport.summer_avg_t - airport.winter_avg_t) / 2.0
    return mid + amp * seasonal


def generate_weather(airport: Airport, date: pd.Timestamp, rng: np.random.Generator) -> dict:
    doy = date.dayofyear
    base_t = airport_temperature(airport, doy)
    temp = float(base_t + rng.normal(0, 4))
    precip = float(max(0.0, rng.gamma(0.6, 1.5) - 0.3))
    visibility = float(np.clip(rng.normal(15, 5) - precip * 1.2, 0.3, 25))
    wind = float(max(0.0, rng.gamma(2.0, 2.0)))
    # Severity:
    severity = "normal"
    sev_score = 0
    if precip > 8 or visibility < 1.0 or wind > 18 or temp < -28:
        severity = "severe"
        sev_score = 2
    elif precip > 3 or visibility < 4 or wind > 12 or temp < -18 or temp > 32:
        severity = "moderate"
        sev_score = 1
    return dict(
        temperature_c=round(temp, 1),
        precip_mm=round(precip, 1),
        visibility_km=round(visibility, 1),
        wind_mps=round(wind, 1),
        severity=severity,
        sev_score=sev_score,
    )


# =============================================================================
# 4. ПОСТРОЕНИЕ РАСПИСАНИЯ
# =============================================================================

def build_schedule(rng: random.Random, n_patterns: int = 320) -> list[dict]:
    """
    Шаблоны расписания: рейсы, которые повторяются ежедневно/несколько раз в неделю.
    Каждый шаблон фиксирует: авиакомпания, маршрут, время вылета, дни недели.
    """
    patterns: list[dict] = []
    flight_no_seq = 1000
    iatas = [a.iata for a in AIRPORTS]

    for _ in range(n_patterns):
        airline = rng.choice(AIRLINES)
        # Хабовые рейсы — летят чаще из/в базу авиакомпании
        if rng.random() < 0.55:
            origin_iata = airline.base_iata
            dest_iata = rng.choice([i for i in iatas if i != origin_iata])
        else:
            origin_iata = rng.choice(iatas)
            dest_iata = rng.choice([i for i in iatas if i != origin_iata])

        origin = IATA_TO_AIRPORT[origin_iata]
        dest = IATA_TO_AIRPORT[dest_iata]
        dist = haversine_km(origin.lat, origin.lon, dest.lat, dest.lon)
        block = block_minutes_for(dist)

        # Время вылета: смещение к утреннему/вечернему пику
        if rng.random() < 0.55:
            base_hour = rng.choices([7, 8, 9, 10, 18, 19, 20, 21], k=1)[0]
        else:
            base_hour = rng.randint(5, 23)
        minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])

        # Дни недели: либо 7 (ежедневно), либо 4-6 дней
        if rng.random() < 0.5:
            dows = list(range(7))
        else:
            k = rng.randint(3, 6)
            dows = sorted(rng.sample(range(7), k))

        ac = aircraft_for_route(dist, rng)

        flight_no_seq += rng.randint(1, 4)
        flight_number = f"{airline.code}{flight_no_seq:04d}"

        patterns.append(dict(
            schedule_id=f"SCH{len(patterns)+1:04d}",
            airline_code=airline.code,
            airline_name=airline.name,
            flight_number=flight_number,
            aircraft_family=ac,
            origin_iata=origin.iata,
            dest_iata=dest.iata,
            distance_km=int(round(dist)),
            block_minutes=block,
            sched_hour=base_hour,
            sched_minute=minute,
            dows=dows,
            base_punctuality=airline.base_punctuality,
        ))
    return patterns


# =============================================================================
# 5. МОДЕЛЬ ЗАДЕРЖЕК
# =============================================================================

def airport_congestion(airport: Airport, date: pd.Timestamp, hour: int,
                        rng: np.random.Generator) -> float:
    """
    Индекс загруженности аэропорта в этот час. 0..1.
    """
    # Базовый профиль по часу: пики 7-10 и 18-21
    morning = math.exp(-((hour - 8.5) ** 2) / 4.0)
    evening = math.exp(-((hour - 19.0) ** 2) / 6.0)
    h_load = 0.25 + 0.55 * (0.6 * morning + 1.0 * evening)
    # Hub tier: чем меньше tier, тем больше загруженность
    hub_factor = {1: 1.10, 2: 0.90, 3: 0.70}[airport.hub_tier]
    seasonal = 1.0 + 0.15 * math.cos(2 * math.pi * (date.dayofyear - 200) / 365.0) * (-1)
    # Случайные пиковые дни (отпуска, праздники)
    if is_holiday_window(date):
        h_load *= 1.25
    val = h_load * hub_factor * seasonal + rng.normal(0, 0.04)
    return float(np.clip(val, 0.05, 1.0))


def simulate_flight(
    pattern: dict,
    flight_date: pd.Timestamp,
    weather_origin: dict,
    weather_dest: dict,
    cong_origin: float,
    cong_dest: float,
    inbound_delay: int,
    rng: np.random.Generator,
) -> dict:
    """
    Возвращает компоненты задержки + итоги по одному рейсу.
    """
    origin = IATA_TO_AIRPORT[pattern["origin_iata"]]
    dest = IATA_TO_AIRPORT[pattern["dest_iata"]]
    airline = CODE_TO_AIRLINE[pattern["airline_code"]]

    # ---- 1. Базовые компоненты задержки (минуты) ----

    # carrier_operational: техника / экипажи / турнаранд
    # Целевое распределение: ~25-30% всех задержек как доминирующая причина
    carrier_base = (1 - airline.base_punctuality) * 25.0 + airline.fleet_avg_age * 0.35
    carrier_event = rng.random() < (1 - airline.base_punctuality) * 1.8
    if carrier_event:
        carrier_delay = int(rng.gamma(2.0, 8.0))
    else:
        carrier_delay = int(rng.poisson(carrier_base * 0.05))

    # weather: зависит от severity на обоих концах
    sev_max = max(weather_origin["sev_score"], weather_dest["sev_score"])
    if sev_max == 2:
        weather_delay = int(rng.gamma(4.0, 18.0))   # сильные погодные
    elif sev_max == 1:
        weather_delay = int(rng.gamma(2.0, 9.0)) if rng.random() < 0.55 else 0
    else:
        weather_delay = int(rng.gamma(1.0, 4.0)) if rng.random() < 0.08 else 0

    # airport_congestion: индекс на отправлении в основном
    cong_factor = max(cong_origin, cong_dest * 0.7)
    if cong_factor > 0.75:
        cong_delay = int(rng.gamma(2.0, 7.0))
    elif cong_factor > 0.55:
        cong_delay = int(rng.gamma(1.2, 4.0)) if rng.random() < 0.30 else 0
    else:
        cong_delay = int(rng.gamma(0.5, 2.0)) if rng.random() < 0.05 else 0

    # reactionary: результат каскада от задержки прилёта борта
    # Усиливаем — в реальной авиации это один из крупнейших драйверов
    if inbound_delay >= 30:
        reactionary_delay = max(0, int(inbound_delay * rng.uniform(0.70, 0.95) - 5))
    elif inbound_delay >= 15:
        reactionary_delay = max(0, int(inbound_delay * rng.uniform(0.45, 0.75)))
    elif inbound_delay >= 5:
        reactionary_delay = max(0, int(inbound_delay * rng.uniform(0.20, 0.50)))
    else:
        reactionary_delay = 0

    # security/restrictions: эффект "ковра" на московских аэропортах в 2024-2025
    kover_factor = kover_effect(flight_date, origin.iata, dest.iata)
    base_security_prob = 0.003
    if kover_factor > 1.5:
        base_security_prob *= kover_factor * 3
    if rng.random() < base_security_prob:
        security_delay = int(rng.gamma(3.0, 22.0))
    else:
        security_delay = 0

    # ---- 2. Buffer absorption: часть задержки гасится в расписании ----
    # ВАЖНО: dep_delay_minutes != просто sum компонент. У расписания есть буфер.
    raw_total = (carrier_delay + weather_delay + cong_delay
                 + reactionary_delay + security_delay)

    # Buffer ~5-8 минут абсорбируется (был слишком велик)
    buffer = int(rng.uniform(3, 8))
    dep_delay_minutes = max(0, raw_total - buffer)

    # ---- 3. Arrival delay: dep_delay + полётная вариация ----
    flight_variation = int(rng.normal(0, 4))  # ветер, обходные манёвры
    # Дополнительная задержка на руление в загруженном dest
    if cong_dest > 0.75:
        flight_variation += int(rng.gamma(1.5, 4.0))
    arr_delay_minutes = max(0, dep_delay_minutes + flight_variation)

    # ---- 4. Cancellation / diversion ----
    cancellation_flag = 0
    diversion_flag = 0
    cancellation_reason = None
    if sev_max == 2 and rng.random() < 0.08:
        cancellation_flag = 1
        cancellation_reason = "weather"
    elif security_delay > 90 and rng.random() < 0.15:
        cancellation_flag = 1
        cancellation_reason = "security"
    elif carrier_delay > 30 and rng.random() < 0.005:
        cancellation_flag = 1
        cancellation_reason = "carrier_operational"
    elif rng.random() < 0.0015:
        cancellation_flag = 1
        cancellation_reason = "carrier_operational"

    if not cancellation_flag and arr_delay_minutes > 60 and rng.random() < 0.05:
        diversion_flag = 1

    # ---- 5. Probable delay cause: argmax из компонент ----
    if dep_delay_minutes < 15:
        probable_cause = "none"
    else:
        comps = {
            "weather": weather_delay,
            "airport_congestion": cong_delay,
            "reactionary": reactionary_delay,
            "carrier_operational": carrier_delay,
            "security": security_delay,
        }
        probable_cause = max(comps, key=comps.get)
        if comps[probable_cause] == 0:
            probable_cause = "carrier_operational"

    return dict(
        carrier_delay_minutes=carrier_delay,
        weather_delay_minutes=weather_delay,
        airport_congestion_delay_minutes=cong_delay,
        reactionary_delay_minutes=reactionary_delay,
        security_delay_minutes=security_delay,
        dep_delay_minutes=int(dep_delay_minutes) if not cancellation_flag else None,
        arr_delay_minutes=int(arr_delay_minutes) if not cancellation_flag else None,
        cancellation_flag=cancellation_flag,
        diversion_flag=diversion_flag,
        cancellation_reason=cancellation_reason,
        probable_delay_cause=probable_cause if not cancellation_flag else "cancelled",
        is_departure_delayed_15m=int(dep_delay_minutes >= 15) if not cancellation_flag else None,
        is_arrival_delayed_15m=int(arr_delay_minutes >= 15) if not cancellation_flag else None,
    )


# =============================================================================
# 6. ОСНОВНОЙ ЦИКЛ ГЕНЕРАЦИИ
# =============================================================================

def generate(out_path: Path, target_rows: int) -> pd.DataFrame:
    rng_py = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)

    # Расписание
    n_patterns = 320
    patterns = build_schedule(rng_py, n_patterns=n_patterns)

    # Период
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2025-12-31")
    all_days = pd.date_range(start, end, freq="D")

    # Кэш погоды (по аэропорту, дню) — погода меняется по дням, не по часам
    weather_cache: dict[tuple[str, pd.Timestamp], dict] = {}
    def get_weather(iata: str, day: pd.Timestamp) -> dict:
        key = (iata, day)
        if key not in weather_cache:
            weather_cache[key] = generate_weather(IATA_TO_AIRPORT[iata], day, rng_np)
        return weather_cache[key]

    # Кэш задержки прилёта борта (для reactionary): прокси по аэропорту-времени
    # Идея: прилёт борта на пред. рейсе с этого аэропорта в близком окне формирует
    # inbound_delay для следующего вылета. Упростим: средняя по последним рейсам.
    recent_arrivals: dict[str, list[int]] = {a.iata: [] for a in AIRPORTS}

    # Сначала собираем все (pattern, date) пары
    flight_plan: list[tuple[dict, pd.Timestamp]] = []
    for pat in patterns:
        for d in all_days:
            if d.dayofweek in pat["dows"]:
                flight_plan.append((pat, d))

    # Сэмплируем target_rows
    rng_py.shuffle(flight_plan)
    flight_plan = flight_plan[:target_rows]
    # Сортируем по (date, hour) чтобы reactionary имел осмысленную последовательность
    flight_plan.sort(key=lambda x: (x[1], x[0]["sched_hour"], x[0]["sched_minute"]))

    rows: list[dict] = []
    flight_id_seq = 0

    for pat, d in flight_plan:
        flight_id_seq += 1
        origin = IATA_TO_AIRPORT[pat["origin_iata"]]
        dest = IATA_TO_AIRPORT[pat["dest_iata"]]

        sched_dt_local = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                       hour=pat["sched_hour"], minute=pat["sched_minute"])
        sched_arrival_local = sched_dt_local + pd.Timedelta(minutes=pat["block_minutes"])

        # Погода
        wo = get_weather(origin.iata, d)
        wd = get_weather(dest.iata, d)

        # Загруженность
        cong_o = airport_congestion(origin, d, pat["sched_hour"], rng_np)
        cong_d_ = airport_congestion(dest, d, sched_arrival_local.hour, rng_np)

        # Inbound delay: эмпирический медиан задержки прилётов в этот аэропорт
        # за последние 5 рейсов
        recent = recent_arrivals[origin.iata][-5:]
        if recent:
            inbound_delay = int(np.median(recent))
        else:
            inbound_delay = 0
        # Добавим шум
        inbound_delay = max(0, inbound_delay + int(rng_np.normal(0, 3)))

        result = simulate_flight(pat, d, wo, wd, cong_o, cong_d_,
                                  inbound_delay, rng_np)

        # Логируем arr_delay для следующих рейсов из dest аэропорта
        if not result["cancellation_flag"] and result["arr_delay_minutes"] is not None:
            recent_arrivals[dest.iata].append(result["arr_delay_minutes"])
            if len(recent_arrivals[dest.iata]) > 20:
                recent_arrivals[dest.iata] = recent_arrivals[dest.iata][-20:]

        # Реальное время вылета
        if result["cancellation_flag"]:
            actual_dep = None
            actual_arr = None
        else:
            actual_dep = sched_dt_local + pd.Timedelta(minutes=result["dep_delay_minutes"])
            actual_arr = sched_arrival_local + pd.Timedelta(minutes=result["arr_delay_minutes"])

        row = dict(
            # === ИДЕНТИФИКАТОРЫ ===
            flight_id=f"RU{d.strftime('%Y%m%d')}{flight_id_seq:07d}",
            schedule_id=pat["schedule_id"],
            flight_date=d.date(),

            # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
            year=d.year,
            month=d.month,
            day=d.day,
            day_of_week=d.dayofweek,           # 0=Пн ... 6=Вс
            day_of_year=d.dayofyear,
            week_of_year=d.isocalendar().week,
            is_weekend=int(d.dayofweek >= 5),
            is_holiday_window=is_holiday_window(d),
            quarter=d.quarter,

            # === АВИАКОМПАНИЯ И ВС ===
            airline_code=pat["airline_code"],
            airline_name=pat["airline_name"],
            flight_number=pat["flight_number"],
            aircraft_family=pat["aircraft_family"],
            airline_fleet_avg_age=CODE_TO_AIRLINE[pat["airline_code"]].fleet_avg_age,

            # === МАРШРУТ ===
            origin_iata=origin.iata,
            origin_city=origin.city,
            origin_region=origin.region,
            origin_lat=origin.lat,
            origin_lon=origin.lon,
            origin_utc_offset=origin.utc_offset,
            origin_hub_tier=origin.hub_tier,
            destination_iata=dest.iata,
            destination_city=dest.city,
            destination_region=dest.region,
            destination_lat=dest.lat,
            destination_lon=dest.lon,
            destination_utc_offset=dest.utc_offset,
            destination_hub_tier=dest.hub_tier,
            distance_km=pat["distance_km"],
            route_group=("short" if pat["distance_km"] < 1500
                         else "medium" if pat["distance_km"] < 4000 else "long"),
            planned_block_minutes=pat["block_minutes"],

            # === РАСПИСАНИЕ ===
            scheduled_departure_local=sched_dt_local.isoformat(),
            scheduled_arrival_local=sched_arrival_local.isoformat(),
            scheduled_dep_hour=pat["sched_hour"],
            scheduled_dep_minute=pat["sched_minute"],

            # === ПОГОДА (origin) ===
            origin_temperature_c=wo["temperature_c"],
            origin_precip_mm=wo["precip_mm"],
            origin_visibility_km=wo["visibility_km"],
            origin_wind_mps=wo["wind_mps"],
            origin_weather_severity=wo["severity"],

            # === ПОГОДА (destination) ===
            destination_temperature_c=wd["temperature_c"],
            destination_precip_mm=wd["precip_mm"],
            destination_visibility_km=wd["visibility_km"],
            destination_wind_mps=wd["wind_mps"],
            destination_weather_severity=wd["severity"],

            # === ОПЕРАТИВНЫЕ ПРИЗНАКИ (известны ДО вылета) ===
            origin_congestion_index=round(cong_o, 3),
            destination_congestion_index=round(cong_d_, 3),
            inbound_delay_minutes=inbound_delay,

            # === GROUND TRUTH (НЕ ИСПОЛЬЗОВАТЬ КАК ПРИЗНАКИ!) ===
            # Эти поля — компоненты декомпозиции таргета и используются ТОЛЬКО
            # для вычисления probable_delay_cause и для научного анализа.
            gt_carrier_delay_minutes=result["carrier_delay_minutes"],
            gt_weather_delay_minutes=result["weather_delay_minutes"],
            gt_airport_congestion_delay_minutes=result["airport_congestion_delay_minutes"],
            gt_reactionary_delay_minutes=result["reactionary_delay_minutes"],
            gt_security_delay_minutes=result["security_delay_minutes"],

            # === ТАРГЕТЫ ===
            dep_delay_minutes=result["dep_delay_minutes"],
            arr_delay_minutes=result["arr_delay_minutes"],
            is_departure_delayed_15m=result["is_departure_delayed_15m"],
            is_arrival_delayed_15m=result["is_arrival_delayed_15m"],
            cancellation_flag=result["cancellation_flag"],
            cancellation_reason=result["cancellation_reason"],
            diversion_flag=result["diversion_flag"],
            probable_delay_cause=result["probable_delay_cause"],

            # === ФАКТИЧЕСКОЕ ВРЕМЯ ===
            actual_departure_local=actual_dep.isoformat() if actual_dep is not None else None,
            actual_arrival_local=actual_arr.isoformat() if actual_arr is not None else None,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Сохранение
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {out_path.suffix}")

    return df


# =============================================================================
# 7. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Russian flight delays synthetic dataset.")
    parser.add_argument("--out", type=Path, default=Path("flight_delays_ru.parquet"),
                        help="Output file (.parquet or .csv)")
    parser.add_argument("--rows", type=int, default=200000,
                        help="Target number of rows")
    parser.add_argument("--sample-csv", type=Path, default=None,
                        help="Optional: also save first 5000 rows as CSV for quick inspection")
    args = parser.parse_args()

    print(f"Generating ~{args.rows:,} flights → {args.out}")
    df = generate(args.out, args.rows)
    print(f"Done: {len(df):,} rows × {len(df.columns)} cols")
    print(f"File size: {args.out.stat().st_size / 1024 / 1024:.1f} MB")

    if args.sample_csv:
        df.head(5000).to_csv(args.sample_csv, index=False)
        print(f"Sample saved → {args.sample_csv}")


if __name__ == "__main__":
    main()
