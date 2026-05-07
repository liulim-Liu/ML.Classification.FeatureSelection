#' Run the Python machine-learning classification pipeline
#'
#' Runs the bundled Python machine-learning pipeline from R using
#' [reticulate::source_python()]. The pipeline supports binary and multiclass
#' classification and returns classification summaries, variable-selection
#' summaries, bootstrap-level results, plot data, heatmap data, and label
#' mappings.
#'
#' @param training_data A data frame containing the training dataset.
#' @param outcome_col Character string giving the outcome column name.
#' @param demo_cols Optional character vector of demographic covariates used
#'   in the supervised filtering step. These columns are not treated as model
#'   features in the final variable-selection summary.
#' @param id_cols Optional character vector of ID columns to exclude from model
#'   fitting.
#' @param n_bootstrap Integer number of bootstrap iterations.
#' @param pvalue_threshold Numeric p-value threshold used in supervised
#'   filtering.
#' @param min_features_after_filter Integer minimum number of features retained
#'   after supervised filtering.
#' @param n_iter_search Integer number of random-search iterations for
#'   hyperparameter tuning.
#' @param random_state Integer random seed passed to the Python pipeline.
#' @param n_jobs Integer number of parallel jobs used by scikit-learn. For
#'   memory-limited environments such as ShinyApps.io, use `1` or `2`.
#' @param make_performance_plot Logical; whether to create a classification
#'   performance plot.
#' @param save_plot_path Optional file path for saving the classification
#'   performance plot.
#' @param show_plot Logical; whether to display the performance plot.
#' @param plot_title Optional character string for the performance plot title.
#' @param verbose Logical; whether to print progress messages.
#' @param progress_bar Logical; whether to show a Python progress bar if
#'   `tqdm` is available.
#' @param print_every_bootstrap Integer frequency for printing bootstrap
#'   progress updates.
#' @param make_stability_heatmap Logical; whether to create a feature-stability
#'   heatmap.
#' @param stability_heatmap_path Optional file path for saving the stability
#'   heatmap.
#' @param show_stability_heatmap Logical; whether to display the heatmap.
#' @param heatmap_top_n_features Integer number of top features to show in the
#'   stability heatmap.
#' @param heatmap_title Optional character string for the heatmap title.
#'
#' @return A list returned by the Python pipeline. Common elements include:
#' \describe{
#'   \item{classification_summary}{Per-model performance summary.}
#'   \item{variable_selection_summary}{Feature-ranking and selection-frequency table.}
#'   \item{variable_selection_top_features}{Top selected features filtered by frequency.}
#'   \item{best_bootstrap_variable_sets}{Selected features from the best bootstrap for each model.}
#'   \item{bootstrap_results}{Bootstrap-level model results.}
#'   \item{performance_plot_data}{Wide-format data used for the performance plot.}
#'   \item{performance_plot_long}{Long-format data used for the performance plot.}
#'   \item{stability_heatmap_data}{Feature-by-model stability table.}
#'   \item{label_mapping}{Mapping from original outcome classes to encoded labels.}
#' }
#'
#' @examples
#' \dontrun{
#' library(reticulate)
#' use_python("...", required = TRUE)
#'
#' df <- read.csv("MCI-Convert-training.csv")
#'
#' res <- run_ml_pipeline(
#'   training_data = df,
#'   outcome_col = "y",
#'   demo_cols = c("AGE", "PTGENDER", "APOE4"),
#'   id_cols = c("RID"),
#'   n_bootstrap = 10,
#'   n_jobs = 1,
#'   save_plot_path = "classification.png",
#'   stability_heatmap_path = "feature_stability_heatmap.png"
#' )
#'
#' res$classification_summary
#' res$variable_selection_summary
#' res$bootstrap_results
#' }
#'
#' @export
run_ml_pipeline <- function(
    training_data,
    outcome_col,
    demo_cols = NULL,
    id_cols = NULL,
    n_bootstrap = 100,
    pvalue_threshold = 0.05,
    min_features_after_filter = 5,
    n_iter_search = 20,
    random_state = 10,
    n_jobs = -1,
    make_performance_plot = TRUE,
    save_plot_path = NULL,
    show_plot = FALSE,
    plot_title = NULL,
    verbose = TRUE,
    progress_bar = TRUE,
    print_every_bootstrap = 1,
    make_stability_heatmap = TRUE,
    stability_heatmap_path = NULL,
    show_stability_heatmap = FALSE,
    heatmap_top_n_features = 30,
    heatmap_title = NULL
) {
  if (!is.data.frame(training_data)) {
    stop("'training_data' must be a data frame.", call. = FALSE)
  }

  if (!is.character(outcome_col) || length(outcome_col) != 1) {
    stop("'outcome_col' must be a single character string.", call. = FALSE)
  }

  if (!outcome_col %in% names(training_data)) {
    stop("'outcome_col' is not a column in 'training_data'.", call. = FALSE)
  }

  if (!is.null(demo_cols) && !all(demo_cols %in% names(training_data))) {
    missing_demo <- demo_cols[!demo_cols %in% names(training_data)]
    stop(
      "The following 'demo_cols' are not in 'training_data': ",
      paste(missing_demo, collapse = ", "),
      call. = FALSE
    )
  }

  if (!is.null(id_cols) && !all(id_cols %in% names(training_data))) {
    missing_id <- id_cols[!id_cols %in% names(training_data)]
    stop(
      "The following 'id_cols' are not in 'training_data': ",
      paste(missing_id, collapse = ", "),
      call. = FALSE
    )
  }

  myMain <- load_python_pipeline()

  if (is.null(demo_cols)) {
    demo_cols_py <- NULL
  } else {
    demo_cols_py <- reticulate::r_to_py(as.list(demo_cols))
  }

  if (is.null(id_cols)) {
    id_cols_py <- NULL
  } else {
    id_cols_py <- reticulate::r_to_py(as.list(id_cols))
  }

  res <- myMain(
    training_data = training_data,
    outcome_col = outcome_col,
    demo_cols = demo_cols_py,
    id_cols = id_cols_py,
    n_bootstrap = as.integer(n_bootstrap),
    pvalue_threshold = pvalue_threshold,
    min_features_after_filter = as.integer(min_features_after_filter),
    n_iter_search = as.integer(n_iter_search),
    random_state = as.integer(random_state),
    n_jobs = as.integer(n_jobs),
    make_performance_plot = make_performance_plot,
    save_plot_path = save_plot_path,
    show_plot = show_plot,
    plot_title = plot_title,
    verbose = verbose,
    progress_bar = progress_bar,
    print_every_bootstrap = as.integer(print_every_bootstrap),
    make_stability_heatmap = make_stability_heatmap,
    stability_heatmap_path = stability_heatmap_path,
    show_stability_heatmap = show_stability_heatmap,
    heatmap_top_n_features = as.integer(heatmap_top_n_features),
    heatmap_title = heatmap_title
  )

  return(res)
}
