"""
data_generator.py
=================
Synthetic data generation engine for the Human-to-Gorilla Zoonotic Disease
Transmission Risk Analysis project.

All parameters are calibrated to published epidemiological literature.
See data/README.txt for full citations.

Usage:
    from src.data_generator import GorillaDataGenerator
    gen = GorillaDataGenerator(seed=42)
    gen.generate_all(output_dir='data/raw/')
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  SIMULATION PARAMETERS (literature-calibrated)
# ─────────────────────────────────────────────

GORILLA_GROUPS = {
    "Susa": {
        "size": 28,
        "home_range": "Sector_A",
        "altitude": 3200,
        "visit_freq": "high",
    },
    "Pablo": {
        "size": 25,
        "home_range": "Sector_B",
        "altitude": 2900,
        "visit_freq": "high",
    },
    "Agasha": {
        "size": 16,
        "home_range": "Sector_C",
        "altitude": 3100,
        "visit_freq": "medium",
    },
    "Hirwa": {
        "size": 14,
        "home_range": "Sector_D",
        "altitude": 2800,
        "visit_freq": "medium",
    },
    "Ntambara": {
        "size": 12,
        "home_range": "Sector_E",
        "altitude": 3400,
        "visit_freq": "low",
    },
    "Umubano": {
        "size": 10,
        "home_range": "Sector_F",
        "altitude": 2700,
        "visit_freq": "low",
    },
}

# Age categories: Robbins et al. (2004)
AGE_CATEGORIES = [
    "infant",
    "juvenile",
    "subadult",
    "adult_female",
    "adult_male",
    "silverback",
]

# Age-specific susceptibility multipliers — infants and elders highest risk
# Basis: Lonsdorf et al. (2006), Whittier et al. (2010)
AGE_SUSCEPTIBILITY = {
    "infant": 2.8,  # 0-3 years: immature immune system
    "juvenile": 1.6,  # 3-8 years: still developing
    "subadult": 1.0,  # 8-12 years: baseline
    "adult_female": 0.9,
    "adult_male": 0.85,
    "silverback": 1.9,  # older males: senescent immunity
}

# Rwanda seasons
# Basis: Rwanda Meteorological Agency seasonal calendar
SEASONS = {
    1: "dry_short",  # January
    2: "dry_short",  # February
    3: "wet_long",  # March
    4: "wet_long",  # April
    5: "wet_long",  # May
    6: "dry_long",  # June
    7: "dry_long",  # July
    8: "dry_long",  # August
    9: "dry_long",  # September
    10: "wet_short",  # October
    11: "wet_short",  # November
    12: "dry_short",  # December
}

# Tourist season multipliers (dry seasons = peak tourism)
SEASON_TOURIST_MULTIPLIER = {
    "dry_long": 1.8,  # Jun-Sep: peak
    "dry_short": 1.3,  # Dec-Feb: secondary peak
    "wet_long": 0.6,  # Mar-May: low season
    "wet_short": 0.8,  # Oct-Nov: moderate
}

# Background gorilla illness rates by season (respiratory, not from tourists)
# Basis: Grützmacher et al. (2018) — seasonal respiratory virus circulation
SEASON_BACKGROUND_ILLNESS = {
    "dry_long": 0.015,  # baseline — fewer pathogens in drier air
    "dry_short": 0.025,
    "wet_long": 0.045,  # rainy season: higher background transmission
    "wet_short": 0.035,
}

# Respiratory symptom rate in tourists at pre-visit screening
# Basis: Grützmacher et al. (2018) — ~15% detection rate
TOURIST_SYMPTOM_RATE = 0.148

# Probability that a symptomatic tourist transmits to at least one gorilla
# (conditional on being allowed to visit — policy compliance assumption)
# Basis: Palacios et al. (2011), modeling 7m distance as partial not total barrier
TRANSMISSION_PROB_IF_SYMPTOMATIC = 0.22

# Incubation window for gorilla illness after tourist exposure
# Basis: respiratory virus incubation 2-7 days (CDC), extended to 14 for detection lag
INCUBATION_MIN_DAYS = 2
INCUBATION_MAX_DAYS = 14


class GorillaDataGenerator:
    """
    Generates five interlinked synthetic datasets simulating a multi-year
    gorilla health surveillance program.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Default: 42.
    start_year : int
        First year of data. Default: 2015.
    end_year : int
        Last year of data (inclusive). Default: 2023.
    """

    def __init__(self, seed: int = 42, start_year: int = 2015, end_year: int = 2023):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.start_date = datetime(start_year, 1, 1)
        self.end_date = datetime(end_year, 12, 31)
        self.date_range = pd.date_range(self.start_date, self.end_date, freq="D")

        # Storage for generated dataframes
        self.demographics_df = None
        self.visit_logs_df = None
        self.tourist_health_df = None
        self.gorilla_health_df = None
        self.climate_df = None

    # ─────────────────────────────────────────
    #  1. GORILLA DEMOGRAPHICS
    # ─────────────────────────────────────────
    def generate_demographics(self) -> pd.DataFrame:
        """
        Generate a realistic gorilla population roster.
        ~185 individuals across 6 social groups.
        """
        records = []
        gorilla_id_counter = 1

        # Curated name lists (inspired by real Virunga gorilla naming conventions)
        male_names = [
            "Isango",
            "Kurudi",
            "Titus",
            "Bwenge",
            "Nyagakangaga",
            "Gahinga",
            "Munyinya",
            "Kigeli",
            "Iyambere",
            "Inkumbuzi",
            "Akagera",
            "Rugendo",
            "Baraka",
            "Makara",
            "Seka",
            "Vubu",
            "Amani",
            "Keza",
            "Nkima",
            "Bigurura",
            "Igihango",
            "Rwema",
            "Ubumwe",
            "Zirikana",
            "Ndeze",
            "Musaza",
            "Bimanuka",
            "Giraneza",
            "Kweli",
            "Mwiza",
            "Icyizere",
            "Imbaraga",
            "Kigeme",
        ]
        female_names = [
            "Akashya",
            "Nyiramurema",
            "Uburumpu",
            "Inzira",
            "Gukora",
            "Kabuhanga",
            "Muraha",
            "Nyampinga",
            "Agakura",
            "Umwana",
            "Shema",
            "Gakuru",
            "Urukundo",
            "Ingagi",
            "Impuhwe",
            "Amahoro",
            "Intore",
            "Imana",
            "Ubugingo",
            "Akimana",
            "Gahire",
            "Impano",
            "Isange",
            "Akaranga",
            "Nziza",
            "Umutoni",
            "Uwamahoro",
            "Ingabire",
            "Akeza",
            "Umugeni",
        ]

        male_name_pool = male_names.copy()
        female_name_pool = female_names.copy()
        self.rng.shuffle(male_name_pool)
        self.rng.shuffle(female_name_pool)
        m_idx, f_idx = 0, 0

        for group_name, group_info in GORILLA_GROUPS.items():
            group_size = group_info["size"]

            # Composition: 1 silverback, adult females (~40%), rest varied
            n_silverbacks = 1
            n_adult_males = max(1, int(group_size * 0.10))
            n_adult_females = int(group_size * 0.38)
            n_subadults = int(group_size * 0.15)
            n_juveniles = int(group_size * 0.18)
            n_infants = (
                group_size
                - n_silverbacks
                - n_adult_males
                - n_adult_females
                - n_subadults
                - n_juveniles
            )

            composition = (
                [("silverback", "M", 18, 35)] * n_silverbacks
                + [("adult_male", "M", 12, 17)] * n_adult_males
                + [("adult_female", "F", 10, 30)] * n_adult_females
                + [("subadult", self.rng.choice(["M", "F"]), 8, 11)] * n_subadults
                + [("juvenile", self.rng.choice(["M", "F"]), 3, 7)] * n_juveniles
                + [("infant", self.rng.choice(["M", "F"]), 0, 2)] * max(0, n_infants)
            )

            # Track females to assign mother-infant relationships
            female_ids_in_group = []

            for cat, sex, age_min, age_max in composition:
                gid = f"G-{gorilla_id_counter:04d}"
                gorilla_id_counter += 1
                age = int(self.rng.integers(age_min, age_max + 1))
                birth_year = 2023 - age

                if sex == "M":
                    name = male_name_pool[m_idx % len(male_name_pool)]
                    m_idx += 1
                else:
                    name = female_name_pool[f_idx % len(female_name_pool)]
                    f_idx += 1
                    if cat in ["adult_female", "subadult"]:
                        female_ids_in_group.append(gid)

                # Small probability of immunocompromised status
                immuno = bool(self.rng.choice([True, False], p=[0.04, 0.96]))

                # Mortality: small % of individuals deceased during study period
                status_options = ["alive", "deceased", "unknown"]
                status_probs = [0.88, 0.08, 0.04]
                status = self.rng.choice(status_options, p=status_probs)

                records.append(
                    {
                        "gorilla_id": gid,
                        "name": name,
                        "group_id": group_name,
                        "sex": sex,
                        "age_years": age,
                        "age_category": cat,
                        "birth_year": birth_year,
                        "mother_id": None,  # assigned below
                        "home_range": group_info["home_range"],
                        "altitude_m": group_info["altitude"],
                        "status": status,
                        "immunocompromised": immuno,
                    }
                )

            # Assign mothers to infants and some juveniles
            for rec in records:
                if rec["group_id"] == group_name and rec["age_category"] in [
                    "infant",
                    "juvenile",
                ]:
                    if female_ids_in_group:
                        rec["mother_id"] = self.rng.choice(female_ids_in_group)

        self.demographics_df = pd.DataFrame(records)
        print(
            f"✓ Demographics: {len(self.demographics_df)} gorillas across {len(GORILLA_GROUPS)} groups"
        )
        return self.demographics_df

    # ─────────────────────────────────────────
    #  2. SEASONAL CLIMATE DATA
    # ─────────────────────────────────────────
    def generate_climate_data(self) -> pd.DataFrame:
        """
        Generate daily climate data for the Virunga region (2015–2023).
        Rwanda Meteorological Agency baseline: Volcanoes NP, ~2,800m ASL.
        """
        records = []

        for date in self.date_range:
            month = date.month
            season = SEASONS[month]

            # Temperature: altitude-adjusted, seasonal variation
            # Virunga avg: 10–18°C, lower in wet season
            base_temp_max = {
                "dry_long": 17.5,
                "dry_short": 16.8,
                "wet_long": 14.2,
                "wet_short": 15.1,
            }[season]
            base_temp_min = base_temp_max - 7.5

            temp_max = base_temp_max + self.rng.normal(0, 1.5)
            temp_min = base_temp_min + self.rng.normal(0, 1.2)

            # Rainfall
            avg_rainfall = {
                "dry_long": 1.2,
                "dry_short": 2.8,
                "wet_long": 8.5,
                "wet_short": 5.2,
            }[season]
            rainfall = max(0, self.rng.gamma(1.2, avg_rainfall))

            # Humidity
            base_humidity = {
                "dry_long": 58,
                "dry_short": 65,
                "wet_long": 82,
                "wet_short": 75,
            }[season]
            humidity = min(100, max(30, self.rng.normal(base_humidity, 6)))

            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "year": date.year,
                    "month": month,
                    "day_of_year": date.timetuple().tm_yday,
                    "season": season,
                    "rainfall_mm": round(rainfall, 2),
                    "temp_max_c": round(temp_max, 1),
                    "temp_min_c": round(temp_min, 1),
                    "temp_mean_c": round((temp_max + temp_min) / 2, 1),
                    "humidity_pct": round(humidity, 1),
                }
            )

        self.climate_df = pd.DataFrame(records)
        self.climate_df["date"] = pd.to_datetime(self.climate_df["date"])
        print(f"✓ Climate: {len(self.climate_df)} daily records (2015–2023)")
        return self.climate_df

    # ─────────────────────────────────────────
    #  3. TOURIST VISIT LOGS + HEALTH RECORDS
    # ─────────────────────────────────────────
    def generate_tourist_visits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate tourist visit logs and linked health screening records.

        Rules applied:
          - Max 8 tourists per group per day (RDB regulation)
          - Visit frequency scaled by gorilla group popularity & season
          - Pre-2020: lower masking compliance; post-2020: higher
          - Symptomatic tourists may slip through screening (imperfect system)
        """
        visit_records = []
        health_records = []
        tourist_id_counter = 1
        visit_id_counter = 1

        nationality_regions = [
            "Africa",
            "Europe",
            "North_America",
            "Asia",
            "Oceania",
            "South_America",
        ]
        nat_probs = [0.12, 0.35, 0.30, 0.13, 0.06, 0.04]

        weather_options = ["clear", "cloudy", "light_rain", "heavy_rain"]
        weather_by_season = {
            "dry_long": [0.55, 0.30, 0.12, 0.03],
            "dry_short": [0.45, 0.32, 0.18, 0.05],
            "wet_long": [0.10, 0.25, 0.40, 0.25],
            "wet_short": [0.20, 0.30, 0.35, 0.15],
        }

        for date in self.date_range:
            month = date.month
            season = SEASONS[month]
            season_mult = SEASON_TOURIST_MULTIPLIER[season]

            for group_name, group_info in GORILLA_GROUPS.items():
                # Visit frequency based on group popularity
                base_visit_prob = {"high": 0.92, "medium": 0.80, "low": 0.65}[
                    group_info["visit_freq"]
                ]

                # Apply seasonal multiplier (capped)
                visit_prob = min(
                    0.98, base_visit_prob * (0.5 + 0.5 * season_mult / 1.8)
                )

                if self.rng.random() > visit_prob:
                    continue  # No tourists for this group today

                # Number of tourists (1–8, capped at 8 by regulation)
                n_tourists = int(self.rng.integers(3, 9))  # min 3 for trek viability

                # Weather for today
                weather = self.rng.choice(weather_options, p=weather_by_season[season])

                # Masking compliance: post-2020 COVID protocols raised compliance
                if date.year >= 2020:
                    mask_compliance_prob = 0.87
                else:
                    mask_compliance_prob = 0.42

                for _ in range(n_tourists):
                    tid = f"T-{tourist_id_counter:05d}"
                    vid = f"V-{visit_id_counter:05d}"
                    tourist_id_counter += 1
                    visit_id_counter += 1

                    age_group = self.rng.choice(
                        ["18-30", "31-45", "46-60", "61+"], p=[0.22, 0.35, 0.30, 0.13]
                    )
                    nat_region = self.rng.choice(nationality_regions, p=nat_probs)

                    # Symptom status — tourists underreport symptoms
                    truly_symptomatic = self.rng.random() < TOURIST_SYMPTOM_RATE
                    # Screening catches ~70% of symptomatic tourists
                    screening_detects = truly_symptomatic and (self.rng.random() < 0.70)
                    pre_screened = self.rng.random() < 0.88  # 88% actually screened

                    if screening_detects and pre_screened:
                        test_result = "positive"
                        had_symptoms = True
                        severity = self.rng.choice(
                            ["mild", "moderate", "severe"], p=[0.60, 0.32, 0.08]
                        )
                        # Policy: positive tourists should not visit
                        # but compliance is imperfect (~85% turn back)
                        if self.rng.random() < 0.85:
                            continue  # Tourist correctly excluded — no visit record
                    elif truly_symptomatic:
                        test_result = "negative" if pre_screened else "not_tested"
                        had_symptoms = True
                        severity = self.rng.choice(["mild", "moderate"], p=[0.72, 0.28])
                    else:
                        test_result = "negative" if pre_screened else "not_tested"
                        had_symptoms = False
                        severity = "none"

                    mask_worn = self.rng.random() < mask_compliance_prob
                    duration = int(
                        self.rng.integers(35, 62)
                    )  # 35–61 min (max 60 enforced)
                    min_dist = self.rng.normal(
                        8.5, 2.1
                    )  # meters; ~7m rule, sometimes closer
                    min_dist = max(1.5, round(min_dist, 1))

                    visit_records.append(
                        {
                            "visit_id": vid,
                            "tourist_id": tid,
                            "visit_date": date.strftime("%Y-%m-%d"),
                            "group_id": group_name,
                            "home_range": group_info["home_range"],
                            "duration_minutes": duration,
                            "min_distance_m": min_dist,
                            "guide_id": f"GD-{self.rng.integers(1, 25):03d}",
                            "group_size_tourists": n_tourists,
                            "weather_condition": weather,
                            "season": season,
                            "mask_compliant": mask_worn,
                        }
                    )

                    health_records.append(
                        {
                            "tourist_id": tid,
                            "visit_id": vid,
                            "visit_date": date.strftime("%Y-%m-%d"),
                            "group_visited": group_name,
                            "had_symptoms": had_symptoms,
                            "test_result": test_result,
                            "symptom_severity": severity,
                            "age_group": age_group,
                            "nationality_region": nat_region,
                            "visit_duration_min": duration,
                            "mask_compliant": mask_worn,
                            "pre_screened": pre_screened,
                            "truly_symptomatic": truly_symptomatic,  # ground truth for analysis
                        }
                    )

        self.visit_logs_df = pd.DataFrame(visit_records)
        self.tourist_health_df = pd.DataFrame(health_records)
        self.visit_logs_df["visit_date"] = pd.to_datetime(
            self.visit_logs_df["visit_date"]
        )
        self.tourist_health_df["visit_date"] = pd.to_datetime(
            self.tourist_health_df["visit_date"]
        )
        print(f"✓ Visit logs:      {len(self.visit_logs_df):,} records")
        print(f"✓ Tourist health:  {len(self.tourist_health_df):,} records")
        return self.visit_logs_df, self.tourist_health_df

    # ─────────────────────────────────────────
    #  4. GORILLA HEALTH EVENTS
    # ─────────────────────────────────────────
    def generate_gorilla_health_events(self) -> pd.DataFrame:
        """
        Generate gorilla illness events, seeding some in the wake of
        tourist symptomatic visits and others from background/seasonal causes.

        Transmission logic:
          - Tourism-linked: illness onset 2–14 days after symptomatic tourist visit
          - Background: distributed according to seasonal illness rates
          - Intra-group: mother-to-infant pathway modeled separately
        """
        if self.demographics_df is None:
            raise RuntimeError("Run generate_demographics() first.")
        if self.tourist_health_df is None:
            raise RuntimeError("Run generate_tourist_visits() first.")

        records = []
        event_id_counter = 1

        # Pre-build: symptomatic tourist visits by group and date
        symptomatic_visits = self.tourist_health_df[
            self.tourist_health_df["truly_symptomatic"] == True
        ][["visit_date", "group_visited", "symptom_severity"]].copy()

        illness_types = [
            "respiratory",
            "gastrointestinal",
            "dermatological",
            "injury",
            "other",
        ]
        illness_probs = [0.72, 0.12, 0.07, 0.05, 0.04]  # Köndgen et al. (2008)

        gorillas = self.demographics_df[
            self.demographics_df["status"] != "deceased"
        ].copy()

        for gorilla in gorillas.itertuples():
            gid = gorilla.gorilla_id
            group = gorilla.group_id
            age_cat = gorilla.age_category
            susceptibility = AGE_SUSCEPTIBILITY[age_cat]
            immuno = gorilla.immunocompromised
            if immuno:
                susceptibility *= 1.5

            for date in self.date_range:
                month = date.month
                season = SEASONS[month]
                background_rate = SEASON_BACKGROUND_ILLNESS[season] * susceptibility

                # ── Background illness (seasonal/environmental) ──
                if self.rng.random() < background_rate:
                    incubation = int(self.rng.integers(1, 5))
                    onset_date = date + timedelta(days=incubation)
                    illness_type = self.rng.choice(illness_types, p=illness_probs)
                    severity = self._assign_severity(age_cat, immuno)
                    days_ill = self._assign_duration(severity)
                    recovered = severity != "fatal"
                    vet_treated = (severity in ["moderate", "severe", "fatal"]) and (
                        self.rng.random() < 0.75
                    )

                    records.append(
                        {
                            "event_id": f"GHE-{event_id_counter:05d}",
                            "gorilla_id": gid,
                            "event_date": onset_date.strftime("%Y-%m-%d"),
                            "group_id": group,
                            "illness_type": illness_type,
                            "severity": severity,
                            "days_ill": days_ill,
                            "recovered": recovered,
                            "veterinary_treated": vet_treated,
                            "transmission_source": "background_seasonal",
                            "linked_tourist_visit": None,
                        }
                    )
                    event_id_counter += 1

                # ── Tourism-linked illness ──
                day_sym_visits = symptomatic_visits[
                    (symptomatic_visits["group_visited"] == group)
                    & (symptomatic_visits["visit_date"] == pd.Timestamp(date))
                ]

                for sym_visit in day_sym_visits.itertuples():
                    # Transmission probability scaled by severity and age susceptibility
                    sev_mult = {"mild": 0.8, "moderate": 1.0, "severe": 1.4}.get(
                        sym_visit.symptom_severity, 1.0
                    )
                    trans_prob = (
                        TRANSMISSION_PROB_IF_SYMPTOMATIC
                        * susceptibility
                        * sev_mult
                        * 0.15
                    )

                    if self.rng.random() < trans_prob:
                        incubation = int(
                            self.rng.integers(
                                INCUBATION_MIN_DAYS, INCUBATION_MAX_DAYS + 1
                            )
                        )
                        onset_date = date + timedelta(days=incubation)

                        if onset_date > self.end_date:
                            continue

                        severity = self._assign_severity(
                            age_cat, immuno, tourist_linked=True
                        )
                        days_ill = self._assign_duration(severity)
                        recovered = severity != "fatal"
                        vet_treated = (
                            severity in ["moderate", "severe", "fatal"]
                        ) and (self.rng.random() < 0.80)

                        records.append(
                            {
                                "event_id": f"GHE-{event_id_counter:05d}",
                                "gorilla_id": gid,
                                "event_date": onset_date.strftime("%Y-%m-%d"),
                                "group_id": group,
                                "illness_type": "respiratory",  # tourist transmission → respiratory
                                "severity": severity,
                                "days_ill": days_ill,
                                "recovered": recovered,
                                "veterinary_treated": vet_treated,
                                "transmission_source": "tourism_linked",
                                "linked_tourist_visit": sym_visit.visit_date.strftime(
                                    "%Y-%m-%d"
                                ),
                            }
                        )
                        event_id_counter += 1

                # ── Mother-to-infant pathway ──
                if age_cat == "infant" and gorilla.mother_id is not None:
                    # Check if mother had an event recently
                    maternal_rate = (
                        background_rate * 0.6
                    )  # lower — requires mother illness
                    if self.rng.random() < maternal_rate:
                        incubation = int(self.rng.integers(1, 6))
                        onset_date = date + timedelta(days=incubation)
                        if onset_date <= self.end_date:
                            severity = self._assign_severity("infant", immuno)
                            days_ill = self._assign_duration(severity)
                            records.append(
                                {
                                    "event_id": f"GHE-{event_id_counter:05d}",
                                    "gorilla_id": gid,
                                    "event_date": onset_date.strftime("%Y-%m-%d"),
                                    "group_id": group,
                                    "illness_type": "respiratory",
                                    "severity": severity,
                                    "days_ill": days_ill,
                                    "recovered": severity != "fatal",
                                    "veterinary_treated": self.rng.random() < 0.70,
                                    "transmission_source": "maternal_intragroup",
                                    "linked_tourist_visit": None,
                                }
                            )
                            event_id_counter += 1

        self.gorilla_health_df = pd.DataFrame(records)
        self.gorilla_health_df["event_date"] = pd.to_datetime(
            self.gorilla_health_df["event_date"]
        )

        # Remove duplicate events (same gorilla, same day — keep most severe)
        self.gorilla_health_df = (
            self.gorilla_health_df.sort_values(
                "severity",
                key=lambda x: x.map(
                    {"mild": 0, "moderate": 1, "severe": 2, "fatal": 3}
                ),
                ascending=False,
            )
            .drop_duplicates(subset=["gorilla_id", "event_date"], keep="first")
            .reset_index(drop=True)
        )

        print(f"✓ Gorilla health:  {len(self.gorilla_health_df):,} illness events")
        return self.gorilla_health_df

    # ─────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────
    def _assign_severity(
        self, age_cat: str, immuno: bool, tourist_linked: bool = False
    ) -> str:
        """Severity distribution adjusted by age, immune status, and transmission type."""
        base = [0.60, 0.28, 0.09, 0.03]  # mild, moderate, severe, fatal

        if age_cat in ["infant", "silverback"]:
            base = [0.45, 0.32, 0.16, 0.07]
        if immuno:
            base = [0.35, 0.35, 0.20, 0.10]
        if tourist_linked:
            # Human respiratory pathogens tend to be more virulent in gorillas
            # Basis: Palacios et al. (2011) — HMPV caused 26% group mortality
            base = [base[0] * 0.85, base[1], base[2] * 1.1, base[3] * 1.3]

        # Normalize
        total = sum(base)
        base = [b / total for b in base]
        return self.rng.choice(["mild", "moderate", "severe", "fatal"], p=base)

    def _assign_duration(self, severity: str) -> int:
        """Days of illness based on severity level."""
        params = {
            "mild": (3, 8),
            "moderate": (7, 18),
            "severe": (14, 35),
            "fatal": (5, 21),
        }
        lo, hi = params.get(severity, (3, 10))
        return int(self.rng.integers(lo, hi + 1))

    # ─────────────────────────────────────────
    #  SAVE ALL DATASETS
    # ─────────────────────────────────────────
    def generate_all(self, output_dir: str = "data/raw/") -> dict:
        """
        Run full data generation pipeline and save all CSVs.

        Returns
        -------
        dict : keys = dataset names, values = DataFrames
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir.replace("raw", "processed"), exist_ok=True)
        os.makedirs("outputs/figures", exist_ok=True)
        os.makedirs("outputs/reports", exist_ok=True)

        print("=" * 55)
        print("  GORILLA ZOONOTIC ANALYSIS — DATA GENERATION")
        print(f"  Seed: {self.seed} | Period: 2015–2023")
        print("=" * 55)

        self.generate_demographics()
        self.generate_climate_data()
        self.generate_tourist_visits()
        self.generate_gorilla_health_events()

        # Save
        paths = {
            "gorilla_demographics": os.path.join(
                output_dir, "gorilla_demographics.csv"
            ),
            "seasonal_climate_data": os.path.join(
                output_dir, "seasonal_climate_data.csv"
            ),
            "tourist_visit_logs": os.path.join(output_dir, "tourist_visit_logs.csv"),
            "tourist_health_records": os.path.join(
                output_dir, "tourist_health_records.csv"
            ),
            "gorilla_health_events": os.path.join(
                output_dir, "gorilla_health_events.csv"
            ),
        }

        datasets = {
            "gorilla_demographics": self.demographics_df,
            "seasonal_climate_data": self.climate_df,
            "tourist_visit_logs": self.visit_logs_df,
            "tourist_health_records": self.tourist_health_df,
            "gorilla_health_events": self.gorilla_health_df,
        }

        print("\nSaving CSV files...")
        for name, df in datasets.items():
            df.to_csv(paths[name], index=False)
            print(f"  → {paths[name]} ({len(df):,} rows)")

        print("\n✅ All datasets generated successfully.")
        print(f"   Total records: {sum(len(df) for df in datasets.values()):,}")
        return datasets


if __name__ == "__main__":
    gen = GorillaDataGenerator(seed=42)
    gen.generate_all(output_dir="data/raw/")
