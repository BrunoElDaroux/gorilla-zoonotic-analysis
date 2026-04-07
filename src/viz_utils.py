"""
viz_utils.py
============
Reusable, publication-quality visualization functions for the gorilla
zoonotic disease transmission analysis.

All functions return (fig, ax) or fig objects for further customization.

Usage:
    from src.viz_utils import set_style, plot_age_susceptibility
    set_style()
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────────────────────

PALETTE = {
    'tourism_attributed': '#C0392B',   # deep red — clear risk
    'tourism_possible':   '#E67E22',   # amber — uncertain
    'non_tourism':        '#27AE60',   # green — background
    'maternal_intragroup':'#8E44AD',   # purple — intra-group
    'dry_long':           '#F39C12',   # gold
    'dry_short':          '#F1C40F',   # yellow
    'wet_long':           '#2980B9',   # blue
    'wet_short':          '#85C1E9',   # light blue
    'infant':             '#E74C3C',
    'juvenile':           '#E67E22',
    'subadult':           '#F1C40F',
    'adult_female':       '#2ECC71',
    'adult_male':         '#3498DB',
    'silverback':         '#8E44AD',
}

AGE_ORDER = ['infant', 'juvenile', 'subadult', 'adult_female', 'adult_male', 'silverback']
SEASON_ORDER = ['dry_short', 'wet_long', 'dry_long', 'wet_short']


def set_style():
    """Apply consistent publication-quality matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor':    'white',
        'axes.facecolor':      '#FAFAFA',
        'axes.grid':           True,
        'grid.color':          '#E0E0E0',
        'grid.linestyle':      '-',
        'grid.alpha':          0.7,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'font.family':         'sans-serif',
        'font.size':           11,
        'axes.titlesize':      14,
        'axes.titleweight':    'bold',
        'axes.labelsize':      12,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'legend.fontsize':     10,
        'figure.titlesize':    16,
        'figure.titleweight':  'bold',
    })


def save_fig(fig, filename: str, output_dir: str = 'outputs/figures/', dpi: int = 150):
    """Save figure with consistent settings."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  → Saved: {path}")


# ─────────────────────────────────────────────────────────────
#  1. AGE CATEGORY SUSCEPTIBILITY
# ─────────────────────────────────────────────────────────────

def plot_age_susceptibility(
    illness_counts: pd.DataFrame,
    figsize: tuple = (12, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart: mean illness events per gorilla by age category.
    With 95% bootstrap CI error bars.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Mean illness events per gorilla
    summary = (
        illness_counts
        .groupby('age_category')['n_illness_events']
        .agg(['mean', 'std', 'count'])
        .reindex([c for c in AGE_ORDER if c in illness_counts['age_category'].unique()])
        .reset_index()
    )
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    colors = [PALETTE.get(cat, '#95A5A6') for cat in summary['age_category']]
    bars = axes[0].bar(
        summary['age_category'],
        summary['mean'],
        color=colors,
        yerr=summary['se'] * 1.96,
        capsize=5,
        edgecolor='white',
        linewidth=0.8,
        error_kw={'linewidth': 1.5},
    )

    axes[0].set_title('Mean Illness Events per Gorilla\nby Age Category (±95% CI)')
    axes[0].set_xlabel('Age Category')
    axes[0].set_ylabel('Mean Number of Illness Events')
    axes[0].set_xticklabels(summary['age_category'], rotation=30, ha='right')

    # Annotate with sample sizes
    for bar, n in zip(bars, summary['count']):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar.get_yerr().max() * 0.1 if hasattr(bar, 'get_yerr') else bar.get_height(),
            f'n={int(n)}',
            ha='center', va='bottom', fontsize=9, color='gray'
        )

    # Right: Box plots for distribution
    age_groups_present = [c for c in AGE_ORDER if c in illness_counts['age_category'].values]
    box_data = [
        illness_counts[illness_counts['age_category'] == cat]['n_illness_events'].values
        for cat in age_groups_present
    ]
    bp = axes[1].boxplot(
        box_data,
        labels=age_groups_present,
        patch_artist=True,
        medianprops={'color': 'white', 'linewidth': 2.5},
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4},
    )
    for patch, cat in zip(bp['boxes'], age_groups_present):
        patch.set_facecolor(PALETTE.get(cat, '#95A5A6'))
        patch.set_alpha(0.8)

    axes[1].set_title('Distribution of Illness Events\nby Age Category')
    axes[1].set_xlabel('Age Category')
    axes[1].set_ylabel('Number of Illness Events')
    axes[1].set_xticklabels(age_groups_present, rotation=30, ha='right')

    fig.suptitle('Gorilla Disease Susceptibility by Age Category', y=1.02)
    plt.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────
#  2. TEMPORAL LINKAGE TIMELINE
# ─────────────────────────────────────────────────────────────

def plot_linkage_timeline(
    linked_df: pd.DataFrame,
    tourist_df: pd.DataFrame,
    group_filter: Optional[str] = None,
    year_filter: Optional[int] = None,
    figsize: tuple = (16, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Timeline plot showing symptomatic tourist visits and subsequent gorilla
    illness events for a given group/period.
    """
    set_style()

    ldf = linked_df.copy()
    tdf = tourist_df[tourist_df['had_symptoms'] == True].copy()
    ldf['event_date'] = pd.to_datetime(ldf['event_date'])
    tdf['visit_date'] = pd.to_datetime(tdf['visit_date'])

    if group_filter:
        ldf = ldf[ldf['group_id'] == group_filter]
        tdf = tdf[tdf['group_visited'] == group_filter]
    if year_filter:
        ldf = ldf[ldf['event_date'].dt.year == year_filter]
        tdf = tdf[tdf['visit_date'].dt.year == year_filter]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'height_ratios': [1, 2]})

    # Top: Symptomatic tourist visits
    sym_counts = tdf.groupby('visit_date').size().reset_index(name='count')
    ax1.bar(sym_counts['visit_date'], sym_counts['count'],
            color='#E74C3C', alpha=0.7, width=1.0)
    ax1.set_ylabel('Symptomatic\nTourists', fontsize=10)
    ax1.set_title(
        f'Symptomatic Tourist Visits & Gorilla Illness Events'
        + (f' — {group_filter}' if group_filter else ' — All Groups')
        + (f' ({year_filter})' if year_filter else ''),
    )
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Bottom: Gorilla illness events by attribution
    source_colors = {
        'tourism_attributed': '#C0392B',
        'tourism_possible':   '#E67E22',
        'non_tourism':        '#27AE60',
    }

    for source, color in source_colors.items():
        mask = ldf.get('attributed_source', pd.Series(dtype=str)) == source
        subset = ldf[mask] if 'attributed_source' in ldf.columns else pd.DataFrame()
        if len(subset) > 0:
            daily = subset.groupby('event_date').size().reset_index(name='count')
            ax2.bar(daily['event_date'], daily['count'],
                    color=color, alpha=0.8, width=1.0, label=source.replace('_', ' ').title())

    # Linkage window shading example
    ax2.set_ylabel('Gorilla Illness\nEvents', fontsize=10)
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    return fig, np.array([ax1, ax2])


# ─────────────────────────────────────────────────────────────
#  3. SEASONAL PATTERNS
# ─────────────────────────────────────────────────────────────

def plot_seasonal_patterns(
    merged_df: pd.DataFrame,
    figsize: tuple = (14, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Multi-panel seasonal analysis: tourism counts, illness counts, rainfall.
    """
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    merged_df = merged_df.copy()
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['month'] = merged_df['date'].dt.to_period('M').dt.to_timestamp()

    monthly = merged_df.groupby('month').agg(
        total_rainfall=('rainfall_mm', 'sum'),
        daily_illness_mean=('daily_illness_count', 'mean'),
        humidity_mean=('humidity_pct', 'mean'),
    ).reset_index()

    # Panel 1: Rainfall
    axes[0].bar(monthly['month'], monthly['total_rainfall'],
                color='#2980B9', alpha=0.7, width=20)
    axes[0].set_ylabel('Monthly Rainfall (mm)')
    axes[0].set_title('Environmental Drivers of Gorilla Disease Incidence')

    # Panel 2: Mean daily gorilla illness count
    axes[1].plot(monthly['month'], monthly['daily_illness_mean'],
                 color='#C0392B', linewidth=2.0, marker='o', markersize=4)
    axes[1].fill_between(monthly['month'], monthly['daily_illness_mean'],
                          alpha=0.2, color='#C0392B')
    axes[1].set_ylabel('Mean Daily\nIllness Events')

    # Panel 3: Humidity
    axes[2].plot(monthly['month'], monthly['humidity_mean'],
                 color='#8E44AD', linewidth=1.8, marker='s', markersize=3)
    axes[2].set_ylabel('Mean Humidity (%)')
    axes[2].set_xlabel('Month')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────
#  4. ATTRIBUTION PIE / BAR
# ─────────────────────────────────────────────────────────────

def plot_attribution_summary(
    linked_df: pd.DataFrame,
    figsize: tuple = (14, 6),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side: overall attribution pie chart + per-group stacked bar.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Pie chart — overall
    source_counts = linked_df['attributed_source'].value_counts()
    colors = [PALETTE.get(s, '#95A5A6') for s in source_counts.index]
    wedges, texts, autotexts = ax1.pie(
        source_counts.values,
        labels=[s.replace('_', ' ').title() for s in source_counts.index],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax1.set_title('Overall Attribution of\nGorilla Illness Events')

    # Stacked bar — per group
    if 'attributed_source' in linked_df.columns:
        group_attr = pd.crosstab(
            linked_df['group_id'],
            linked_df['attributed_source'],
            normalize='index'
        ) * 100

        source_order = ['tourism_attributed', 'tourism_possible', 'non_tourism']
        for col in source_order:
            if col not in group_attr.columns:
                group_attr[col] = 0
        group_attr = group_attr[source_order]

        bottom = np.zeros(len(group_attr))
        for col in source_order:
            ax2.bar(
                group_attr.index,
                group_attr[col],
                bottom=bottom,
                color=PALETTE.get(col, '#95A5A6'),
                label=col.replace('_', ' ').title(),
                alpha=0.85,
            )
            bottom += group_attr[col].values

    ax2.set_title('Attribution by Gorilla Group\n(% of Events)')
    ax2.set_ylabel('Percentage of Events (%)')
    ax2.set_xlabel('Gorilla Group')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, 110)

    plt.tight_layout()
    return fig, np.array([ax1, ax2])


# ─────────────────────────────────────────────────────────────
#  5. ROC CURVE (Risk Model)
# ─────────────────────────────────────────────────────────────

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = 'Logistic Regression',
    figsize: tuple = (7, 7),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ROC curve for the risk model."""
    set_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='#C0392B', lw=2.5,
            label=f'{model_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#C0392B')

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve — Tourism-Linked Gorilla Illness Risk Model')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────
#  6. HEATMAP — Group × Month Illness Intensity
# ─────────────────────────────────────────────────────────────

def plot_group_month_heatmap(
    gorilla_health_df: pd.DataFrame,
    figsize: tuple = (14, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of illness event counts: gorilla groups × months.
    Reveals both group-level differences and seasonal peaks.
    """
    set_style()
    df = gorilla_health_df.copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['month_name'] = df['event_date'].dt.strftime('%b')
    df['month_num'] = df['event_date'].dt.month

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    pivot = (
        df.groupby(['group_id', 'month_name'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=month_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap='YlOrRd',
        annot=True,
        fmt='d',
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Number of Illness Events'},
    )
    ax.set_title('Gorilla Illness Events: Group × Month Heatmap\n(All Years Combined)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Gorilla Group')
    plt.tight_layout()
    return fig, ax
