#' Declare Python dependencies for the machine-learning pipeline
#'
#' Declares the Python packages and versions required by the bundled Python
#' pipeline. This should be called before Python is initialized by reticulate.
#'
#' @return Invisibly returns `NULL`.
#'
#' @examples
#' \dontrun{
#' use_ml_python_requirements()
#' }
#'
#' @export
use_ml_python_requirements <- function() {
  reticulate::py_require(
      packages = c(
        "numpy==1.26.4",
        "pandas==2.3.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "statsmodels==0.14.0",
        "scipy==1.11.1",
        "scikit-learn==1.6.1",
        "joblib==1.2.0",
        "lightgbm==4.3.0"
      ),
      python_version = "3.9.18"
    )
  invisible(NULL)
}
