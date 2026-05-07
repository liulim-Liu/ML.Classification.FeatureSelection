#' Load the Python machine-learning pipeline
#'
#' Internal helper function used by run_ml_pipeline() to locate and source
#' the bundled Python pipeline file.
#'
#' @return The Python function `myMain()` loaded through reticulate.
#'
#' @keywords internal
load_python_pipeline <- function() {
  py_file <- system.file(
    "python",
    "myMain_pipeline_multiclass.py",
    package = "ML.Classification.FeatureSelection"
  )

  if (identical(py_file, "")) {
    stop(
      "Cannot find 'myMain_pipeline_multiclass.py' under inst/python/. ",
      "Please check that the Python file is included in the package.",
      call. = FALSE
    )
  }

  env <- new.env(parent = emptyenv())

  reticulate::source_python(
    file = py_file,
    envir = env
  )

  if (!exists("myMain", envir = env, inherits = FALSE)) {
    stop(
      "The Python file was loaded, but function 'myMain' was not found. ",
      "Please check that myMain() is defined in myMain_pipeline_multiclass.py.",
      call. = FALSE
    )
  }

  env$myMain
}
