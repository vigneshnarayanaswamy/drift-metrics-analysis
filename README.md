# Drift Metrics Analysis: Hellinger Distance vs PSI

**Author:** Vignesh Narayanaswamy

A comparative analysis of Hellinger Distance and Population Stability Index (PSI) for detecting model drift in machine learning applications.

## Files

- **`Drift Metrics.Rmd`** - Main R Markdown analysis
- **`Drift Metrics.R`** - R script version  
- **`Hellinger_vs_PSI.ipynb`** - Python/Jupyter implementation
- **`Drift-Metrics.html`** - Generated HTML report
- **`Drift-Metrics.pdf`** - Generated PDF report

## Usage

**R Markdown:**
```r
rmarkdown::render("Drift Metrics.Rmd")
```

**R Script:**
```r
source("Drift Metrics.R")
```

**Python:**
```bash
jupyter notebook Hellinger_vs_PSI.ipynb
```

## Requirements

- R with `reticulate`, `knitr`, `rmarkdown`
- Python with `pandas`, `numpy`, `matplotlib`, `scipy`, `seaborn`
- Pandoc (for R Markdown)

## Summary

The analysis compares both metrics across different types of distributional shifts using simulated data. PSI shows higher sensitivity but may produce false positives, while Hellinger Distance provides more robust performance against outliers.
