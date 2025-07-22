# Drift Metrics Analysis: Hellinger Distance vs PSI

**Author:** Vignesh Narayanaswamy

A comparative analysis of Hellinger Distance and Population Stability Index (PSI) for detecting model drift in machine learning applications.

## Generated Reports

- **[Interactive HTML Report](reports/Drift-Metrics.html)** - Full analysis with visualizations
- **[PDF Report](reports/Drift-Metrics.pdf)** - Print-ready version
- **[Word Document](reports/Drift-Metrics.docx)** - Editable version

## Usage

**R Markdown:**
```r
rmarkdown::render("src/Drift Metrics.Rmd")
```

**Python:**
```bash
jupyter notebook src/Hellinger_vs_PSI.ipynb
```

## Requirements

- R with `reticulate`, `knitr`, `rmarkdown`
- Python with `pandas`, `numpy`, `matplotlib`, `scipy`, `seaborn`
- Pandoc (for R Markdown)

## Summary

The analysis compares both metrics across different types of distributional shifts using simulated data. PSI shows higher sensitivity but may produce false positives, while Hellinger Distance provides more robust performance against outliers.
