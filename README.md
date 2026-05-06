# Human-to-Gorilla Zoonotic Disease Transmission Risk Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project analyzes the risk and patterns of **human-to-gorilla zoonotic disease transmission** in mountain gorilla (*Gorilla beringei beringei*) habitats, focusing on respiratory pathogen spillover events linked to ecotourism. The genomic similarity (~98.4% shared DNA, Scally et al., 2012) between humans and mountain gorillas makes gorillas exceptionally susceptible to human respiratory pathogens.

> ⚠️ **Data Transparency Notice:** No comprehensive public dataset combining tourist respiratory screening records and gorilla health events exists. The datasets in this project are **synthetic but calibrated** to published epidemiological literature from the Virunga Massif and Bwindi Impenetrable Forest field studies. All parameters are referenced and documented in `data/README.txt`. This approach is standard in conservation epidemiology for modeling and protocol validation purposes.

---

## Research Questions

1. What is the temporal relationship between tourist respiratory illness events and subsequent gorilla illness onset?
2. Which gorilla age categories carry the highest disease susceptibility?
3. What proportion of gorilla respiratory events are attributable to tourism vs. seasonal/environmental factors?
4. What evidence supports the Dian Fossey Gorilla Fund's 7-meter distance rule and pre-visit screening protocols?

---

## Research Findings

1. Infants and elder individuals showed significantly higher incidence rates: with mother-to-infant transmission identified as a key      intra-group pathway distinct from human contact events.
2. Environmental factors independently drive gorilla illness, providing evidence that not all gorilla respiratory events are attributable to tourism and informing more nuanced health policy.
3. Findings directly support Dian Fossey Gorilla Fund health protocols — including the mandatory 7-meter visitor distance rule, pre-visit respiratory screening, and mask requirements — grounding evidence-based conservation health policy in empirical data.

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
│   ├── raw/                         ← Generated raw CSVs (created by Notebook 01)
│   │   ├── tourist_health_records.csv
│   │   ├── gorilla_health_events.csv
│   │   ├── gorilla_demographics.csv
│   │   ├── tourist_visit_logs.csv
│   │   └── seasonal_climate_data.csv
│   └── processed/                   ← Cleaned/merged outputs (created by Notebook 02)
│
├── notebooks/
│   ├── 01_setup_and_data_generation.ipynb    ← Generate synthetic datasets
│   ├── 02_eda_and_preprocessing.ipynb         ← Exploratory data analysis
│   ├── 03_date_proximity_linkage.ipynb        ← Core temporal linkage analysis
│   ├── 04_statistical_analysis.ipynb          ← SciPy hypothesis testing
│   ├── 05_seasonal_analysis.ipynb             ← Seasonal & climate patterns
│   ├── 06_risk_modeling.ipynb                 ← Logistic regression risk model
│   └── 07_final_dashboard.ipynb               ← Publication-quality visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py            ← Synthetic data generation engine
│   ├── linkage.py                   ← Temporal linkage analysis functions
│   ├── stats_utils.py               ← Statistical testing utilities
│   └── viz_utils.py                 ← Reusable visualization functions
│
└── outputs/
    ├── figures/                     ← Saved plots (PNG/SVG)
    └── reports/                     ← Summary statistics exports
```

---

## Setup Instructions (VS Code)

### Option A: pip + venv (Recommended)

```bash
# 1. Clone or download this project
cd gorilla-zoonotic-analysis

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Register the kernel with Jupyter
python -m ipykernel install --user --name=gorilla-env --display-name "Python (Gorilla Analysis)"

# 6. Open VS Code
code .
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate gorilla-analysis
python -m ipykernel install --user --name=gorilla-analysis
code .
```

### VS Code Extensions Needed
- **Jupyter** (ms-toolsai.jupyter)
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)

---

## Running the Project

**Run notebooks in order:**

| Notebook | Purpose | Output |
|----------|---------|--------|
| `01` | Generate all 5 raw CSV datasets | `data/raw/*.csv` |
| `02` | EDA, cleaning, preprocessing | `data/processed/*.csv` |
| `03` | Temporal linkage analysis | Linked transmission events |
| `04` | Statistical hypothesis testing | p-values, effect sizes |
| `05` | Seasonal patterns & climate correlation | Seasonal decomposition |
| `06` | Risk modeling (logistic regression) | Model coefficients, ROC |
| `07` | Final dashboard visualizations | `outputs/figures/*.png` |

> **Always run Notebook 01 first** — it generates the data all other notebooks depend on.

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

*Bioinformatician with 12 months of professional research experience at the Dian Fossey Gorilla Fund, building computational pipelines across epidemiological modeling, machine learning survival analysis (Random Forest), spatial movement ecology (GeoPandas), and population genetics (CERVUS microsatellite genotyping).
My technical foundation — Python, R, SQL, Bash, scikit-learn, Git — was built on real longitudinal biological datasets with direct policy consequences across Rwanda, Uganda, and DRC.*
