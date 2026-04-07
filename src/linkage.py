"""
linkage.py
==========
Temporal proximity linkage functions for identifying likely human-to-gorilla
transmission events.

The core challenge: distinguishing tourism-linked gorilla illness from
background (seasonal, intra-group) transmission. This module implements
date-window linkage with group-location controls.

Usage:
    from src.linkage import TemporalLinkage
    linker = TemporalLinkage(window_days=14)
    linked_df = linker.link_events(tourist_health_df, gorilla_health_df)
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Optional


class TemporalLinkage:
    """
    Performs temporal proximity linkage between tourist respiratory illness
    events and gorilla health events.

    Method
    ------
    For each gorilla illness event:
      1. Identify the gorilla's group.
      2. Look back `window_days` days for symptomatic tourist visits to that group.
      3. Flag the event as 'tourism-proximate' if ≥1 symptomatic visit exists
         in that window, controlling for group location.
      4. Assign a linkage confidence score based on:
           - Number of symptomatic visitors in window
           - Symptom severity of those visitors
           - Gorilla age category (higher susceptibility → higher confidence)
           - Time lag within window (shorter lag → higher confidence)

    Parameters
    ----------
    window_days : int
        Maximum days between tourist visit and gorilla illness onset. Default: 14.
    min_tourist_severity : str
        Minimum tourist symptom severity to count as exposure. Default: 'mild'.
    """

    SEVERITY_WEIGHTS = {'mild': 0.6, 'moderate': 1.0, 'severe': 1.5, 'none': 0.0}

    def __init__(self, window_days: int = 14, min_tourist_severity: str = 'mild'):
        self.window_days = window_days
        self.min_tourist_severity = min_tourist_severity

    def link_events(
        self,
        tourist_df: pd.DataFrame,
        gorilla_df: pd.DataFrame,
        gorilla_demographics: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Main linkage method. Returns gorilla_df augmented with linkage columns.

        Parameters
        ----------
        tourist_df : pd.DataFrame
            tourist_health_records dataframe. Must contain:
            ['visit_date', 'group_visited', 'had_symptoms', 'symptom_severity']
        gorilla_df : pd.DataFrame
            gorilla_health_events dataframe. Must contain:
            ['event_date', 'group_id', 'gorilla_id']
        gorilla_demographics : pd.DataFrame, optional
            Needed to look up age categories for confidence scoring.

        Returns
        -------
        pd.DataFrame
            gorilla_df with added columns:
              - tourism_proximate: bool
              - n_sym_tourists_in_window: int
              - max_severity_in_window: str
              - days_since_last_sym_visit: float (NaN if not tourism-proximate)
              - linkage_confidence: float (0–1)
              - attributed_source: str
        """
        tourist_df = tourist_df.copy()
        gorilla_df = gorilla_df.copy()

        # Ensure datetime
        tourist_df['visit_date'] = pd.to_datetime(tourist_df['visit_date'])
        gorilla_df['event_date'] = pd.to_datetime(gorilla_df['event_date'])

        # Filter to symptomatic tourists only
        sym_tourists = tourist_df[tourist_df['had_symptoms'] == True].copy()

        # Add age category to gorilla events if demographics available
        if gorilla_demographics is not None:
            gorilla_df = gorilla_df.merge(
                gorilla_demographics[['gorilla_id', 'age_category', 'immunocompromised']],
                on='gorilla_id',
                how='left',
            )

        # Pre-compute linkage using SQL for efficiency
        results = self._sql_linkage(sym_tourists, gorilla_df)
        return results

    def _sql_linkage(
        self,
        sym_tourists: pd.DataFrame,
        gorilla_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Uses SQLite in-memory database for efficient temporal range joins.
        This approach mirrors how production epidemiology pipelines work.
        """
        conn = sqlite3.connect(':memory:')

        # Load into SQLite
        sym_tourists.to_sql('sym_visits', conn, index=False, if_exists='replace')
        gorilla_df.to_sql('gorilla_events', conn, index=False, if_exists='replace')

        # Core temporal linkage query
        # Finds all gorilla illness events that occurred within `window_days`
        # after a symptomatic tourist visit to the SAME gorilla group.
        query = f"""
        SELECT
            ge.event_id,
            ge.gorilla_id,
            ge.event_date,
            ge.group_id,
            ge.illness_type,
            ge.severity,
            ge.days_ill,
            ge.recovered,
            ge.veterinary_treated,
            ge.transmission_source,

            -- Linkage columns
            COUNT(sv.tourist_id)                        AS n_sym_tourists_in_window,
            MIN(julianday(ge.event_date)
                - julianday(sv.visit_date))             AS days_since_earliest_exposure,
            MAX(CASE sv.symptom_severity
                    WHEN 'severe'   THEN 3
                    WHEN 'moderate' THEN 2
                    WHEN 'mild'     THEN 1
                    ELSE 0 END)                         AS max_severity_score,
            CASE WHEN COUNT(sv.tourist_id) > 0
                 THEN 1 ELSE 0 END                      AS tourism_proximate

        FROM gorilla_events ge
        LEFT JOIN sym_visits sv
            ON  ge.group_id   = sv.group_visited
            AND julianday(ge.event_date) - julianday(sv.visit_date)
                    BETWEEN 0 AND {self.window_days}
        GROUP BY
            ge.event_id, ge.gorilla_id, ge.event_date, ge.group_id,
            ge.illness_type, ge.severity, ge.days_ill, ge.recovered,
            ge.veterinary_treated, ge.transmission_source
        """

        linked = pd.read_sql(query, conn)
        conn.close()

        # Map severity score back to label
        severity_map = {0: 'none', 1: 'mild', 2: 'moderate', 3: 'severe'}
        linked['max_severity_in_window'] = linked['max_severity_score'].map(severity_map)

        # Compute linkage confidence score (0–1)
        linked['linkage_confidence'] = linked.apply(self._compute_confidence, axis=1)

        # Attributed source (hierarchy: tourism > seasonal > intragroup)
        linked['attributed_source'] = linked.apply(self._attribute_source, axis=1)

        # Clean up
        linked['tourism_proximate'] = linked['tourism_proximate'].astype(bool)
        linked['event_date'] = pd.to_datetime(linked['event_date'])

        # Merge age_category back if it was added
        if 'age_category' in gorilla_df.columns:
            linked = linked.merge(
                gorilla_df[['event_id', 'age_category', 'immunocompromised']],
                on='event_id', how='left'
            )

        return linked

    def _compute_confidence(self, row) -> float:
        """
        Linkage confidence score combining:
          - Whether tourism-proximate at all
          - Number of symptomatic tourists in window
          - Severity of their symptoms
          - Time lag (shorter = higher confidence)
        Score range: 0.0 to 1.0
        """
        if not row['tourism_proximate']:
            return 0.0

        # Base score: starts at 0.3 for any proximate link
        score = 0.30

        # Visitor count contribution (plateaus at 5+)
        n = min(row['n_sym_tourists_in_window'], 5)
        score += 0.10 * n  # up to +0.50

        # Severity contribution
        sev_scores = {0: 0.0, 1: 0.05, 2: 0.10, 3: 0.18}
        score += sev_scores.get(row['max_severity_score'], 0)

        # Time lag: shorter lag → higher confidence
        lag = row['days_since_earliest_exposure']
        if lag is not None and not np.isnan(lag):
            if lag <= 3:
                score += 0.12
            elif lag <= 7:
                score += 0.08
            elif lag <= 10:
                score += 0.04

        return min(1.0, round(score, 3))

    def _attribute_source(self, row) -> str:
        """Rule-based source attribution for attributed transmission."""
        if row['tourism_proximate'] and row['linkage_confidence'] >= 0.50:
            return 'tourism_attributed'
        elif row['tourism_proximate'] and row['linkage_confidence'] >= 0.30:
            return 'tourism_possible'
        else:
            return 'non_tourism'

    def compute_attribution_summary(self, linked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute group-level and overall transmission attribution summary.
        """
        summary = (
            linked_df
            .groupby(['group_id', 'attributed_source'])
            .size()
            .unstack(fill_value=0)
            .assign(
                total_events=lambda df: df.sum(axis=1),
                pct_tourism_attributed=lambda df: (
                    df.get('tourism_attributed', 0) / df['total_events'] * 100
                ).round(1),
                pct_tourism_possible=lambda df: (
                    df.get('tourism_possible', 0) / df['total_events'] * 100
                ).round(1),
                pct_non_tourism=lambda df: (
                    df.get('non_tourism', 0) / df['total_events'] * 100
                ).round(1),
            )
        )
        return summary

    def sensitivity_analysis(
        self,
        tourist_df: pd.DataFrame,
        gorilla_df: pd.DataFrame,
        window_range: list = [7, 10, 14, 21],
    ) -> pd.DataFrame:
        """
        Test sensitivity of attribution results to linkage window choice.
        Returns a dataframe showing how attribution rates change with window_days.
        """
        results = []
        for w in window_range:
            original_window = self.window_days
            self.window_days = w
            linked = self.link_events(tourist_df, gorilla_df)
            n_total = len(linked)
            n_attributed = (linked['attributed_source'] == 'tourism_attributed').sum()
            n_possible = (linked['attributed_source'] == 'tourism_possible').sum()
            results.append({
                'window_days': w,
                'total_events': n_total,
                'tourism_attributed': n_attributed,
                'tourism_possible': n_possible,
                'pct_attributed': round(n_attributed / n_total * 100, 2),
                'pct_possible': round(n_possible / n_total * 100, 2),
            })
            self.window_days = original_window
        return pd.DataFrame(results)
