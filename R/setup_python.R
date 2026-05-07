#' Check Python setup for the machine-learning pipeline
#'
#' This function prints the active Python configuration and checks whether the
#' Python modules required by the bundled machine-learning pipeline are available.
#'
#' The required modules include `numpy`, `pandas`, `sklearn`, `statsmodels`,
#' `scipy`, `matplotlib`, and `seaborn`. The modules `lightgbm` and `tqdm`
#' are treated as optional because the Python pipeline can skip LightGBM if it
#' is not installed.
#'
#' @return Invisibly returns a data frame with module availability.
#'
#' @examples
#' \dontrun{
#' library(reticulate)
#' use_python("C:/Users/llm/anaconda3/python.exe", required = TRUE)
#'
#' check_python_setup()
#' }
#'
#' @export
check_python_setup <- function() {
  required_modules <- c(
    "numpy",
    "pandas",
    "sklearn",
    "statsmodels",
    "scipy",
    "matplotlib",
    "seaborn"
  )

  optional_modules <- c(
    "lightgbm",
    "tqdm"
  )

  cat("Python configuration:\n")
  print(reticulate::py_config())

  required_df <- data.frame(
    module = required_modules,
    required = TRUE,
    available = vapply(
      required_modules,
      reticulate::py_module_available,
      logical(1)
    )
  )

  optional_df <- data.frame(
    module = optional_modules,
    required = FALSE,
    available = vapply(
      optional_modules,
      reticulate::py_module_available,
      logical(1)
    )
  )

  out <- rbind(required_df, optional_df)

  cat("\nPython module availability:\n")
  print(out, row.names = FALSE)

  invisible(out)
}
