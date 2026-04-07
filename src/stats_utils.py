"""
stats_utils.py
==============
Statistical testing utilities for the gorilla zoonotic disease analysis.

Includes:
  - Chi-square tests for categorical associations
  - Mann-Whitney U for non-normal distributions
  - Kruskal-Wallis for multi-group comparisons
  - Spearman correlation for seasonal data
  - Effect size calculations (Cramér's V, Cohen's d, eta-squared)
  - Confidence interval bootstrapping

Usage:
    from src.stats_utils import run_age_susceptibility_test, bootstrap_ci
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
#  EFFECT SIZES
# ─────────────────────────────────────────────────────────────

def cramers_v(contingency_table: np.ndarray) -> float:
    """
    Compute Cramér's V effect size for chi-square tests.
    Range: 0 (no association) to 1 (perfect association).
    """
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum()
    k = min(contingency_table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d effect size for two independent groups.
    Interpretation: 0.2=small, 0.5=medium, 0.8=large.
    """
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def eta_squared(groups: list) -> float:
    """
    Eta-squared (η²) effect size for one-way ANOVA / Kruskal-Wallis.
    Interpretation: 0.01=small, 0.06=medium, 0.14=large.
    """
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


# ─────────────────────────────────────────────────────────────
#  CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for any statistic.

    Parameters
    ----------
    data : array-like
    statistic : callable — function applied to each bootstrap sample
    n_boot : int — number of bootstrap iterations
    ci : float — confidence level (0–1)

    Returns
    -------
    (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(seed)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def proportion_ci(successes: int, trials: int, ci: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    More accurate than normal approximation for small samples.
    """
    p = successes / trials if trials > 0 else 0
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0, center - margin), min(1, center + margin)


# ─────────────────────────────────────────────────────────────
#  CORE STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────

def run_chi_square_test(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    label: str = '',
    print_results: bool = True,
) -> dict:
    """
    Run chi-square test of independence between two categorical variables.

    Returns
    -------
    dict with chi2, p_value, df, cramers_v, significant (bool), contingency_table
    """
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    cv = cramers_v(contingency.values)

    result = {
        'test':           'chi_square',
        'label':          label,
        'chi2':           round(chi2, 4),
        'p_value':        round(p, 6),
        'degrees_of_freedom': dof,
        'cramers_v':      round(cv, 4),
        'significant':    p < 0.05,
        'contingency_table': contingency,
    }

    if print_results:
        _print_test_result(result)

    return result


def run_kruskal_wallis_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    label: str = '',
    print_results: bool = True,
) -> dict:
    """
    Kruskal-Wallis H-test (non-parametric ANOVA).
    Tests whether multiple groups have the same distribution.

    Appropriate for non-normal illness duration / count data.
    """
    groups = [
        df[df[group_col] == g][value_col].dropna().values
        for g in df[group_col].unique()
    ]
    groups = [g for g in groups if len(g) > 0]

    h_stat, p = stats.kruskal(*groups)
    eta2 = eta_squared(groups)

    result = {
        'test':           'kruskal_wallis',
        'label':          label,
        'h_statistic':    round(h_stat, 4),
        'p_value':        round(p, 6),
        'eta_squared':    round(eta2, 4),
        'significant':    p < 0.05,
        'n_groups':       len(groups),
        'group_medians':  {
            g: round(np.median(df[df[group_col] == g][value_col].dropna()), 2)
            for g in df[group_col].unique()
        }
    }

    if print_results:
        _print_test_result(result)

    return result


def run_mann_whitney_test(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_label: str = 'Group 1',
    group2_label: str = 'Group 2',
    label: str = '',
    alternative: str = 'two-sided',
    print_results: bool = True,
) -> dict:
    """
    Mann-Whitney U test (non-parametric equivalent of independent t-test).
    """
    u_stat, p = stats.mannwhitneyu(group1, group2, alternative=alternative)
    d = cohens_d(group1, group2)

    result = {
        'test':           'mann_whitney_u',
        'label':          label,
        'u_statistic':    round(u_stat, 4),
        'p_value':        round(p, 6),
        'cohens_d':       round(d, 4),
        'significant':    p < 0.05,
        'alternative':    alternative,
        f'{group1_label}_median': round(np.median(group1), 3),
        f'{group2_label}_median': round(np.median(group2), 3),
    }

    if print_results:
        _print_test_result(result)

    return result


def run_spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = 'X',
    y_label: str = 'Y',
    label: str = '',
    print_results: bool = True,
) -> dict:
    """
    Spearman rank correlation — robust to outliers, appropriate for
    non-normal environmental and epidemiological data.
    """
    rho, p = stats.spearmanr(x, y)

    result = {
        'test':       'spearman_correlation',
        'label':      label,
        'x_label':    x_label,
        'y_label':    y_label,
        'rho':        round(rho, 4),
        'p_value':    round(p, 6),
        'significant': p < 0.05,
        'n':          len(x),
    }

    if print_results:
        _print_test_result(result)

    return result


# ─────────────────────────────────────────────────────────────
#  DOMAIN-SPECIFIC TESTS
# ─────────────────────────────────────────────────────────────

def run_age_susceptibility_test(
    gorilla_health_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    illness_type: str = 'respiratory',
    print_results: bool = True,
) -> dict:
    """
    Test whether age category is a significant predictor of gorilla disease
    susceptibility (incidence rate ratio across age categories).

    Method: Kruskal-Wallis on illness rates, then pairwise Mann-Whitney
    with Bonferroni correction for multiple comparisons.
    """
    # Merge demographics
    merged = gorilla_health_df.merge(
        demographics_df[['gorilla_id', 'age_category']],
        on='gorilla_id', how='left'
    )
    if illness_type != 'all':
        merged = merged[merged['illness_type'] == illness_type]

    # Compute per-gorilla illness counts
    illness_counts = (
        merged.groupby(['gorilla_id', 'age_category'])
        .size()
        .reset_index(name='n_illness_events')
    )

    # Add gorillas with 0 events
    all_gorillas = demographics_df[['gorilla_id', 'age_category']].copy()
    illness_counts = all_gorillas.merge(illness_counts, on=['gorilla_id', 'age_category'], how='left')
    illness_counts['n_illness_events'] = illness_counts['n_illness_events'].fillna(0)

    # Overall Kruskal-Wallis
    kw_result = run_kruskal_wallis_test(
        illness_counts, 'age_category', 'n_illness_events',
        label=f'Age category vs {illness_type} illness events',
        print_results=print_results
    )

    # Pairwise comparisons with Bonferroni correction
    age_categories = illness_counts['age_category'].unique()
    n_comparisons = len(age_categories) * (len(age_categories) - 1) / 2
    alpha_bonferroni = 0.05 / n_comparisons

    pairwise = []
    for i, cat1 in enumerate(age_categories):
        for cat2 in age_categories[i+1:]:
            g1 = illness_counts[illness_counts['age_category'] == cat1]['n_illness_events'].values
            g2 = illness_counts[illness_counts['age_category'] == cat2]['n_illness_events'].values
            if len(g1) > 1 and len(g2) > 1:
                _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                pairwise.append({
                    'comparison': f'{cat1} vs {cat2}',
                    'p_raw': round(p, 6),
                    'p_bonferroni': round(min(1.0, p * n_comparisons), 6),
                    'significant_bonferroni': (p * n_comparisons) < 0.05,
                    'median_g1': round(np.median(g1), 3),
                    'median_g2': round(np.median(g2), 3),
                })

    return {
        'overall_kruskal_wallis': kw_result,
        'pairwise_comparisons': pd.DataFrame(pairwise),
        'alpha_bonferroni': round(alpha_bonferroni, 5),
        'illness_counts': illness_counts,
    }


def run_seasonal_attribution_test(
    gorilla_health_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    print_results: bool = True,
) -> dict:
    """
    Test whether seasonal/environmental factors independently drive gorilla illness
    (evidence against all illness being tourism-attributed).
    """
    # Merge illness events with climate data
    climate_df = climate_df.copy()
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    gorilla_health_df = gorilla_health_df.copy()
    gorilla_health_df['event_date'] = pd.to_datetime(gorilla_health_df['event_date'])

    daily_counts = (
        gorilla_health_df
        .groupby('event_date')
        .size()
        .reset_index(name='daily_illness_count')
        .rename(columns={'event_date': 'date'})
    )

    merged = climate_df.merge(daily_counts, on='date', how='left')
    merged['daily_illness_count'] = merged['daily_illness_count'].fillna(0)

    # Test 1: Rainfall vs illness count (Spearman)
    rain_corr = run_spearman_correlation(
        merged['rainfall_mm'].values,
        merged['daily_illness_count'].values,
        x_label='Rainfall (mm)',
        y_label='Daily illness count',
        label='Rainfall ~ Gorilla illness',
        print_results=print_results,
    )

    # Test 2: Season vs illness rate (Kruskal-Wallis)
    season_kw = run_kruskal_wallis_test(
        merged, 'season', 'daily_illness_count',
        label='Season vs daily gorilla illness count',
        print_results=print_results,
    )

    # Test 3: Humidity vs illness count
    humidity_corr = run_spearman_correlation(
        merged['humidity_pct'].values,
        merged['daily_illness_count'].values,
        x_label='Humidity (%)',
        y_label='Daily illness count',
        label='Humidity ~ Gorilla illness',
        print_results=print_results,
    )

    return {
        'rainfall_correlation': rain_corr,
        'season_kruskal_wallis': season_kw,
        'humidity_correlation': humidity_corr,
        'merged_data': merged,
    }


# ─────────────────────────────────────────────────────────────
#  FORMATTING
# ─────────────────────────────────────────────────────────────

def _print_test_result(result: dict):
    """Pretty-print a test result dictionary."""
    sig_str = "✓ SIGNIFICANT" if result.get('significant') else "✗ not significant"
    alpha = "(α=0.05)"

    print(f"\n{'─'*55}")
    print(f"  {result.get('test', 'test').upper().replace('_', ' ')}")
    if result.get('label'):
        print(f"  {result['label']}")
    print(f"{'─'*55}")

    skip_keys = {'test', 'label', 'significant', 'contingency_table',
                 'group_medians', 'n_groups', 'alternative'}

    for k, v in result.items():
        if k in skip_keys:
            continue
        if isinstance(v, pd.DataFrame):
            continue
        print(f"  {k:<28} {v}")

    print(f"\n  {sig_str} {alpha}")
    print(f"{'─'*55}")


def compile_results_table(results_list: list) -> pd.DataFrame:
    """
    Compile multiple test results into a publishable summary table.
    """
    rows = []
    for r in results_list:
        row = {
            'Test':       r.get('test', '').replace('_', ' ').title(),
            'Variable':   r.get('label', ''),
            'Statistic':  r.get('chi2', r.get('h_statistic', r.get('rho', r.get('u_statistic', '')))),
            'p-value':    r.get('p_value', ''),
            'Effect size': r.get('cramers_v', r.get('eta_squared', r.get('rho', r.get('cohens_d', '')))),
            'Significant': '✓' if r.get('significant') else '✗',
        }
        rows.append(row)
    return pd.DataFrame(rows)
