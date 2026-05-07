# ML.Classification.FeatureSelection

`ML.Classification.FeatureSelection` provides an R interface to a Python-based machine learning pipeline for binary and multiclass classification, feature selection, model comparison, and feature stability analysis.

The package uses [`reticulate`](https://rstudio.github.io/reticulate/) to manage Python dependencies and run the bundled Python pipeline from R.

## Main Features

- Binary and multiclass classification
- Bootstrap-based model evaluation
- Supervised feature filtering
- Multiple machine learning models:
  - Random Forest
  - Support Vector Machine
  - Logistic LASSO
  - Logistic Elastic Net
  - LightGBM, if available
- Classification performance summaries
- Variable selection summaries
- Bootstrap-level results
- Feature stability heatmap
- Classification performance plot

## Installation

Install the package from a local source package:

```r
install.packages(
  "ML.Classification.FeatureSelection_0.0.0.9000.tar.gz",
  repos = NULL,
  type = "source"
)
```

Or, during development, install from the package folder:

```r
devtools::install()
```

## Python Dependency Setup

This package is designed to let `reticulate` manage the Python environment.

Start a fresh R session, then run:

```r
library(ML.Classification.FeatureSelection)

use_ml_python_requirements()
check_python_setup()
```

Do **not** call `reticulate::use_python()` if you want `reticulate` to manage Python automatically.

The package currently declares the following Python dependencies:

```text
numpy==1.26.4
pandas==2.3.3
matplotlib==3.7.2
seaborn==0.12.2
statsmodels==0.14.0
scipy==1.11.1
scikit-learn==1.6.1
joblib==1.2.0
lightgbm==4.3.0
```

## Basic Usage

```r
library(ML.Classification.FeatureSelection)

use_ml_python_requirements()
check_python_setup()

df <- read.csv("MCI-Convert-training.csv")

res <- run_ml_pipeline(
  training_data = df,
  outcome_col = "y",
  demo_cols = c("AGE", "PTGENDER", "APOE4"),
  id_cols = c("RID"),
  n_bootstrap = 10,
  pvalue_threshold = 0.05,
  min_features_after_filter = 5,
  n_iter_search = 5,
  random_state = 10,
  n_jobs = 1,
  make_performance_plot = TRUE,
  save_plot_path = "figures/classification_performance.png",
  plot_title = "Classification Performance",
  make_stability_heatmap = TRUE,
  stability_heatmap_path = "figures/feature_stability_heatmap.png",
  heatmap_top_n_features = 30
)
```

## Required Data Format

The input dataset should be an R `data.frame`.

It should contain:

- one outcome column
- optional ID columns
- optional demographic covariates
- numeric feature columns

Example structure:

```text
RID   y        AGE   PTGENDER   APOE4   feature1   feature2   feature3
1     Class_1  72    0          1       0.52       -1.20      0.88
2     Class_2  68    1          0       1.14        0.34     -0.41
3     Class_1  75    0          2      -0.73        1.62      0.25
```

Example call:

```r
res <- run_ml_pipeline(
  training_data = df,
  outcome_col = "y",
  demo_cols = c("AGE", "PTGENDER", "APOE4"),
  id_cols = c("RID"),
  n_bootstrap = 10,
  n_jobs = 1
)
```

## Output

The returned object is a list.

Common outputs include:

```r
names(res)
```

```text
classification_summary
variable_selection_summary
variable_selection_top_features
best_bootstrap_variable_sets
bootstrap_results
performance_plot_data
performance_plot_long
performance_plot_path
stability_heatmap_data
stability_heatmap_path
label_mapping
reverse_label_mapping
```

### Classification Summary

```r
res$classification_summary
```

This table summarizes model performance across bootstrap samples, including metrics such as:

- AUC
- balanced accuracy
- weighted F1 score
- Matthews correlation coefficient
- number of selected features

### Variable Selection Summary

```r
res$variable_selection_summary
```

This table contains feature-level variable selection results, including:

- average feature rank
- selection frequency
- model-specific average ranks

### Top Features

```r
res$variable_selection_top_features
```

This table contains features that pass the default stability threshold used by the Python pipeline.

If this table is empty during small test runs, check:

```r
head(res$variable_selection_summary)
```

Small bootstrap numbers may not produce stable features above the threshold.

### Bootstrap Results

```r
res$bootstrap_results
```

This table contains bootstrap-level model results, including selected features, best parameters, confusion matrices, and performance metrics.

## Saving Output Tables

Example:

```r
dir.create("output", showWarnings = FALSE)

write.csv(
  res$classification_summary,
  "output/classification_summary.csv",
  row.names = FALSE
)

write.csv(
  res$variable_selection_summary,
  "output/variable_selection_summary.csv",
  row.names = FALSE
)

write.csv(
  res$variable_selection_top_features,
  "output/top_features.csv",
  row.names = FALSE
)
```

Some output tables, such as `bootstrap_results`, may contain list columns. To save them as CSV files, convert list columns first:

```r
make_csv_safe <- function(df) {
  df[] <- lapply(df, function(x) {
    if (is.list(x)) {
      vapply(x, function(z) {
        if (is.null(z) || length(z) == 0) {
          return("")
        }
        paste(unlist(z), collapse = "; ")
      }, character(1))
    } else {
      x
    }
  })

  df
}

bootstrap_results_csv <- make_csv_safe(res$bootstrap_results)

write.csv(
  bootstrap_results_csv,
  "output/bootstrap_results.csv",
  row.names = FALSE
)
```

## Saving Plots

The pipeline can save plots directly:

```r
res <- run_ml_pipeline(
  training_data = df,
  outcome_col = "y",
  id_cols = c("RID"),
  n_bootstrap = 10,
  make_performance_plot = TRUE,
  save_plot_path = "figures/classification_performance.png",
  make_stability_heatmap = TRUE,
  stability_heatmap_path = "figures/feature_stability_heatmap.png"
)
```

Generated files:

```text
figures/classification_performance.png
figures/feature_stability_heatmap.png
```

## Testing the Package

During development, run:

```r
devtools::document()
devtools::load_all()
```

Then, from a fresh R session:

```r
use_ml_python_requirements()
check_python_setup()

source("test_run_pipeline.R")
```

After the test works, install the package:

```r
devtools::install()
```

Restart R, then test the installed version:

```r
library(ML.Classification.FeatureSelection)

use_ml_python_requirements()
check_python_setup()

source("test_run_pipeline.R")
```

## Building the Package

Run:

```r
devtools::document()
devtools::check()
devtools::build()
```

To build the PDF manual:

```r
devtools::build_manual()
```

A LaTeX installation, such as TinyTeX, is required for the PDF manual.

## Notes for ShinyApps.io

For memory-limited environments such as ShinyApps.io, use conservative parallel settings:

```r
n_jobs = 1
```

or at most:

```r
n_jobs = 2
```

Avoid using:

```r
n_jobs = -1
```

on ShinyApps.io unless the app has sufficient CPU and memory resources.

## Important Notes

- Call `use_ml_python_requirements()` before Python is initialized.
- Do not call `reticulate::use_python()` if using reticulate-managed Python.
- Restart R before switching Python environments.
- Keep the bundled Python file under `inst/python/`.
- Do not manually edit `NAMESPACE`; regenerate it using `devtools::document()`.

## License

This package is licensed under the MIT License.
