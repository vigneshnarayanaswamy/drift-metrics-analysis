# Drift Metrics Analysis: Hellinger Distance vs PSI

**Author:** Vignesh Narayanaswamy

A comparative analysis of Hellinger Distance and Population Stability Index (PSI) for detecting model drift in machine learning applications.

## Repository Structure

```
├── src/                    # Source code
├── reports/                # Generated reports
├── figures/                # Visualizations
└── archive/                # Legacy files
```

### 📂 Source Code
- **[Drift Metrics.Rmd](src/Drift%20Metrics.Rmd)** - Main R Markdown analysis
- **[Drift Metrics.R](src/Drift%20Metrics.R)** - R script version
- **[Hellinger_vs_PSI.ipynb](src/Hellinger_vs_PSI.ipynb)** - Python/Jupyter implementation

### 📊 Generated Reports
- **[Drift-Metrics.html](reports/Drift-Metrics.html)** - Interactive HTML report
- **[Drift-Metrics.pdf](reports/Drift-Metrics.pdf)** - PDF report  
- **[Drift-Metrics.docx](reports/Drift-Metrics.docx)** - Word document

### 📈 Visualizations
- **[Plot 0](figures/plot_0.png)** - Time series visualization
- **[Plot 1](figures/plot_1.png)** - Distribution comparison
- **[Plot 2](figures/plot_2.png)** - Metric comparison
- **[Plot 3](figures/plot_3.png)** - Statistical analysis
- **[Plot 4](figures/plot_4.png)** - Drift detection
- **[Plot 5](figures/plot_5.png)** - Bin size analysis
- **[Plot 6](figures/plot_6.png)** - Performance metrics

## Usage

**R Markdown:**
```r
rmarkdown::render("src/Drift Metrics.Rmd")
```

**R Script:**
```r
source("src/Drift Metrics.R")
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
