# ============================================================
# test_run_pipeline.R
# Local test script for ML.Classification.FeatureSelection
# using custom generateTestData()
# Reticulate-managed Python version
# ============================================================

# IMPORTANT:
# 1. Start from a fresh R session before running this script.
# 2. Do NOT call reticulate::use_python().
# 3. If testing after devtools::load_all(), do not need library(package).
# 4. If testing installed package, uncomment library(ML.Classification.FeatureSelection).

library(reticulate)

# Uncomment this only after devtools::install()
# library(ML.Classification.FeatureSelection)

# ------------------------------------------------------------
# 1. Declare Python requirements before Python is initialized
# ------------------------------------------------------------

use_ml_python_requirements()
check_python_setup()

# ------------------------------------------------------------
# 2. Load custom simulation function
# ------------------------------------------------------------

source("generateData.R")

# ------------------------------------------------------------
# 3. Generate binary two-view simulation data
# ------------------------------------------------------------

mydata <- generateTestData(
  myseed = 1234L,
  n1 = 50L,
  n2 = 50L,
  n3 = NULL,
  p1 = 200L,
  p2 = 200L,
  n_signal = 50L,
  signal_strength = 1.0,
  test_fraction = 0.3,
  nReplicate = 1L
)

df <- makePipelineDataFrame(
  mydata = mydata,
  replicate_id = 1L,
  use_test_data = FALSE
)

cat("\nSimulated training data dimension:\n")
print(dim(df))

cat("\nOutcome distribution:\n")
print(table(df$y))

cat("\nFirst few columns:\n")
print(head(df[, 1:8]))

# ------------------------------------------------------------
# 4. Run the ML pipeline
# ------------------------------------------------------------

res <- run_ml_pipeline(
  training_data = df,
  outcome_col = "y",
  demo_cols = NULL,
  id_cols = c("RID"),
  n_bootstrap = 10,
  pvalue_threshold = 0.05,
  min_features_after_filter = 5,
  n_iter_search = 2,
  random_state = 123,
  n_jobs = 1,
  make_performance_plot = TRUE,
  save_plot_path = "test_output/classification_performance.png",
  show_plot = FALSE,
  plot_title = "Custom Simulation Classification Performance",
  verbose = TRUE,
  progress_bar = FALSE,
  print_every_bootstrap = 1,
  make_stability_heatmap = TRUE,
  stability_heatmap_path = "test_output/feature_stability_heatmap.png",
  show_stability_heatmap = FALSE,
  heatmap_top_n_features = 20,
  heatmap_title = "Custom Simulation Feature Stability Heatmap"
)

# ------------------------------------------------------------
# 5. Save summary tables
# ------------------------------------------------------------

# Create output folder if it does not exist
dir.create("test_output", showWarnings = FALSE, recursive = TRUE)


make_csv_safe <- function(df) {
  df[] <- lapply(df, function(x) {
    if (is.list(x)) {
      vapply(x, function(z) {
        if (is.null(z)) {
          return("")
        }

        if (length(z) == 0) {
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

# Classification Summary Table
classification_summary <- res$classification_summary

write.csv(
  classification_summary,
  file = "test_output/classification_summary.csv",
  row.names = FALSE
)

cat("\nClassification Summary Table saved to:\n")
cat("test_output/classification_summary.csv\n")

cat("\nClassification Summary Table:\n")
print(classification_summary)


# Top Features Table
top_features <- res$variable_selection_top_features

write.csv(
  top_features,
  file = "test_output/top_features.csv",
  row.names = FALSE
)

cat("\nTop Features Table saved to:\n")
cat("test_output/top_features.csv\n")

cat("\nTop Features Table:\n")
print(top_features)


# Save full variable selection summary
write.csv(
  res$variable_selection_summary,
  file = "test_output/variable_selection_summary_full.csv",
  row.names = FALSE
)

cat("\nFull Variable Selection Summary saved to:\n")
cat("test_output/variable_selection_summary_full.csv\n")


# Save bootstrap-level results
bootstrap_results_csv <- make_csv_safe(res$bootstrap_results)

write.csv(
  bootstrap_results_csv,
  file = "test_output/bootstrap_results.csv",
  row.names = FALSE
)

cat("\nBootstrap Results saved to:\n")
cat("test_output/bootstrap_results.csv\n")

# ------------------------------------------------------------
# 6. Basic checks
# ------------------------------------------------------------

stopifnot(is.list(res))
stopifnot("classification_summary" %in% names(res))
stopifnot("variable_selection_summary" %in% names(res))
stopifnot("variable_selection_top_features" %in% names(res))
stopifnot("bootstrap_results" %in% names(res))

stopifnot(nrow(res$classification_summary) > 0)
stopifnot(nrow(res$bootstrap_results) > 0)

stopifnot(file.exists("test_output/classification_performance.png"))
stopifnot(file.exists("test_output/feature_stability_heatmap.png"))

stopifnot(file.exists("test_output/classification_summary.csv"))
stopifnot(file.exists("test_output/top_features.csv"))
stopifnot(file.exists("test_output/variable_selection_summary_full.csv"))
stopifnot(file.exists("test_output/bootstrap_results.csv"))

cat("\nAll local tests passed.\n")
