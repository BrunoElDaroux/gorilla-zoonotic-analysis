# Human-to-Gorilla Zoonotic Disease Transmission Risk Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BrunoElDaroux/gorilla-zoonotic-analysis/main)

## Project Overview

This project analyzes the risk and patterns of **human-to-gorilla zoonotic disease transmission** in mountain gorilla (*Gorilla beringei beringei*) habitats, focusing on respiratory pathogen spillover events linked to ecotourism. The genomic similarity (~98.4% shared DNA, Scally et al., 2012) between humans and mountain gorillas makes gorillas exceptionally susceptible to human respiratory pathogens.

---

## Research Questions

1. What is the temporal relationship between tourist respiratory illness events and subsequent gorilla illness onset?
2. Which gorilla age categories carry the highest disease susceptibility?
3. What proportion of gorilla respiratory events are attributable to tourism vs. seasonal/environmental factors?
4. What evidence supports the Dian Fossey Gorilla Fund's 7-meter distance rule and pre-visit screening protocols?

---

## Research Findings

1. `Age Category is the Strongest Predictor of Respiratory Illness Susceptibility`
A Kruskal-Wallis test confirmed that illness rates differ significantly across age categories (H = 55.81, p < 0.001, η² = 0.65 ,large effect). Infants and silverbacks carried the highest illness burden. Pairwise Bonferroni-corrected comparisons identified the specific age groups driving this effect. This finding is consistent with immunological theory: infants lack acquired immunity while older silverbacks face immunosenescence compounding injury costs from dominance contests. 

2. `Illness Severity is Significantly Associated with Age Category`
A chi-square test confirmed that severe and fatal respiratory events are not distributed randomly across age classes (χ² = 294.09, p < 0.001, Cramér's V = 0.14). While the effect size is modest; indicating that age alone does not fully determine outcome; the association is robust and replicates the pattern reported by Palacios et al. (2011) following the fatal HMPV outbreak in mountain gorillas.

3. `Tourism-Linked Events Are More Severe Than Background Illness`
Tourism-attributed respiratory events showed significantly higher severity scores than background seasonal illness (Mann-Whitney U = 24,904,593.5, p < 0.001, r = 0.16). Though the effect size is small, the directional finding is consistent with Grützmacher et al. (2018): human respiratory pathogens, to which gorillas have no prior immunity, produce more severe clinical outcomes than endemic seasonal pathogens.

4. `Same-Day Tourist Symptoms Show a Counterintuitive Negative Correlation, the 7-Day Lag is the Biologically Meaningful Signal`
Spearman correlation between daily symptomatic tourist counts and same-day gorilla illness events was negative (ρ = −0.118, p < 0.001), likely reflecting seasonal confounding, tourist peaks coincide with dry seasons when gorilla background illness is naturally lower. The biologically meaningful window is the 7-14 day incubation lag. Lag-window analysis in Notebook 04 identifies the peak transmission lag, providing the correct temporal signal for conservation health policy.

5. `Masking Compliance Shows No Statistically Significant Association with Gorilla Illness Rates`
Annual masking compliance rate was not significantly correlated with gorilla respiratory event counts (ρ = 0.25, p = 0.517). This null result is expected given the limited statistical power of 9 years of annual data. It does not disprove the protective effect of masking — it reflects that annual aggregate data is too coarse to detect a protocol-level signal that operates at the individual visit level.

6. `Logistic Regression Model Achieves Near-Perfect Discrimination (AUC = 0.984)`
A balanced logistic regression classifier trained on 14,877 events and evaluated on 4,959 held-out events achieved a test AUC of 0.984 (5-fold CV: 0.984 ± 0.0004), indicating near-perfect discrimination between tourism-attributed and background illness events. The dominant predictor was the number of symptomatic tourists within the exposure window (OR = 339,305), confirming that direct symptomatic contact; not background tourist volume, drives attribution risk. Annual masking compliance was protective (OR = 0.85), and wet long-season events showed slightly reduced attribution risk (OR = 0.87), consistent with reduced tourism pressure during heavy rainfall periods.

> ⚠️ **Model Limitation:** Policy scenario simulations were uninformative due to the extreme dominance of the symptomatic tourist window feature. All scenarios returned attribution probabilities indistinguishable from 100%, indicating that the logistic regression is not an appropriate tool for marginal policy effect estimation in this context. Future work should apply causal inference methods (e.g., do-calculus or counterfactual simulation) to properly quantify intervention impacts.

---

## Project Structure

```
gorilla-zoonotic-analysis/
│
├── README.md                        ← You are here
├── .gitignore                       ← Git ignore rules
├── requirements.txt                 ← Python package dependencies
├── environment.yml                  ← Conda environment spec
│
├── data/
│   ├── README.txt                   ← Data sources, parameters, references
│   ├── raw/                         
│   │   ├── tourist_health_records.csv
│   │   ├── gorilla_health_events.csv
│   │   ├── gorilla_demographics.csv
│   │   ├── tourist_visit_logs.csv
│   │   └── seasonal_climate_data.csv
│   └── processed/                   ← Cleaned/merged outputs 
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb         ← Exploratory data analysis
│   ├── 02_date_proximity_linkage.ipynb        ← Core temporal linkage analysis
│   ├── 03_statistical_analysis.ipynb          ← SciPy hypothesis testing
│   ├── 04_seasonal_analysis.ipynb             ← Seasonal & climate patterns
│   ├── 05_risk_modeling.ipynb                 ← Logistic regression risk model
│   └── 06_final_dashboard.ipynb               ← Publication-quality visualizations
│
├── src/
│   ├── __init__.py
│   ├── linkage.py                   ← Temporal linkage analysis functions
│   ├── stats_utils.py               ← Statistical testing utilities
│   └── viz_utils.py                 ← Reusable visualization functions
│
└── outputs/
    ├── figures/                     ← Saved plots (PNG/SVG)
    └── reports/                     ← Summary statistics exports
```

---


** NOTEBOOKS:**

| Notebook | Purpose | Output |
|----------|---------|--------|
| `01` | EDA, cleaning, preprocessing | `data/processed/*.csv` |
| `02` | Temporal linkage analysis | Linked transmission events |
| `03` | Statistical hypothesis testing | p-values, effect sizes |
| `04` | Seasonal patterns & climate correlation | Seasonal decomposition |
| `05` | Risk modeling (logistic regression) | Model coefficients, ROC |
| `06` | Final dashboard visualizations | `outputs/figures/*.png` |

---

## Key Literature References

| Citation | Relevance |
|----------|-----------|
| Grützmacher et al. (2018) *Sci Rep* | Human-gorilla respiratory pathogen sharing |
| Palacios et al. (2011) *Emerg Infect Dis* | Fatal HMPV outbreak in mountain gorillas |
| Scally et al. (2012) *Nature* | Gorilla genome, ~98.4% human DNA similarity |
| Whittier et al. (2010) *Am J Primatol* | Disease risk framework for great apes |
| Rwego et al. (2008) *Emerg Infect Dis* | E. coli transmission, gorilla-human interface |
| Lonsdorf et al. (2006) *Anim Conserv* | Age/sex disease susceptibility in primates |
| DFGFI Health Protocols (2019) | 7-meter rule, masking, pre-visit screening |

---


## Skills Demonstrated

`Python` · `Pandas` · `SQL (SQLite)` · `SciPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Plotly` · `Epidemiological Analysis` · `Temporal Linkage` · `Logistic Regression` · `Jupyter Notebooks`

---


## Author

*Data Technician with research experience at the Dian Fossey Gorilla Fund, building end-to-end computational pipelines across four domains: spatial movement ecology (GeoPandas, KDE, permutation testing), population genetics (CERVUS microsatellite LOD scoring, Queller-Goodnight kinship estimation), machine learning survival analysis (Random Forest, temporal cross-validation), and conservation epidemiology (logistic regression, SciPy hypothesis testing, temporal linkage). Technical stack: Python · R · SQL · scikit-learn · SciPy · GeoPandas · Git. All work is grounded in longitudinal biological datasets with direct conservation policy implications across the Virunga Massif: Rwanda, Uganda, and DRC.*
