# Comparative Analysis of Metrics for Model Drift

**Author:** Vignesh Narayanaswamy  
**Project:** Model Risk Management - Drift Detection Metrics

## Overview

This repository contains a comprehensive analysis comparing **Hellinger Distance** and **Population Stability Index (PSI)** for monitoring model drift in machine learning applications. The analysis provides practical guidance for model owners to make informed decisions regarding metric choice for their specific monitoring needs.

## 📊 Key Components

### Analysis Files
- **`Drift Metrics.Rmd`** - Main R Markdown analysis file with comprehensive comparison
- **`Drift Metrics.R`** - Standalone R script version of the analysis
- **`Hellinger_vs_PSI.ipynb`** - Jupyter notebook with Python implementation

### Generated Reports
- **`Drift-Metrics.html`** - Interactive HTML report with embedded visualizations
- **`Drift-Metrics.pdf`** - Print-ready PDF version 
- **`Drift-Metrics.docx`** - Editable Word document version

### Visualizations
- **`plot_0.png` to `plot_6.png`** - Generated visualization outputs showing metric comparisons

## 🎯 Analysis Scope

### Types of Drift Examined
1. **Mean Shift** - Distribution mean changes from 0 to 3
2. **Standard Deviation Shift** - Standard deviation changes from 1 to 2  
3. **Distributional Shift** - Transition from normal to Poisson distribution
4. **Transitioning Series** - Gradual transition from mean 0 to mean 3
5. **Stationary Series** - Control case with no drift
6. **Skewness Shift** - Changes in distribution shape
7. **Noisy Signal** - High variance scenarios

### Key Findings
- **PSI** shows higher sensitivity to distributional shifts, especially with small proportions
- **Hellinger Distance** provides more robust performance against outliers and noise
- Bin selection significantly impacts both metrics
- PSI tends to be more reactive but may produce false positives

## 🔧 Technical Requirements

### R Dependencies
```r
# Required packages
library(reticulate)
library(knitr)
library(rmarkdown)
```

### Python Dependencies
```python
# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import skewnorm
import seaborn as sns
```

## 🚀 Usage

### Running the Analysis

1. **R Markdown Version:**
   ```r
   # Render to HTML
   rmarkdown::render("Drift Metrics.Rmd", output_format = "html_document")
   
   # Render to PDF  
   rmarkdown::render("Drift Metrics.Rmd", output_format = "pdf_document")
   
   # Render to Word
   rmarkdown::render("Drift Metrics.Rmd", output_format = "word_document")
   ```

2. **R Script Version:**
   ```r
   source("Drift Metrics.R")
   ```

3. **Python/Jupyter Version:**
   ```bash
   jupyter notebook Hellinger_vs_PSI.ipynb
   ```

### System Requirements
- R 4.3.1 or higher
- Python 3.9+ with required packages
- Pandoc (for R Markdown rendering)

## 📈 Methodology

The analysis uses simulated data with:
- **Training Data:** 1,000 observations from standard normal distributions
- **Live Data:** 3,000 observations with introduced drift at observation 1,000
- **Window Size:** 150 observations for drift detection
- **Step Size:** 1 observation sliding window
- **Binning:** 10, 20, and 50 bins tested for sensitivity analysis

## 🎯 Applications

This analysis is particularly relevant for:
- **Model Risk Management** teams implementing drift monitoring
- **Data Scientists** designing model monitoring frameworks  
- **Compliance** teams ensuring model performance tracking
- **ML Engineers** building automated drift detection systems

## 📚 References

This work integrates principles from:
- Model Risk Management best practices
- Statistical drift detection literature

## 📄 License

This project is for research and educational purposes. Please cite appropriately if used in academic or commercial applications.

---

*For questions or collaboration opportunities, please reach out to the author.*
