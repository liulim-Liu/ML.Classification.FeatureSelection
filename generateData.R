# ============================================================
# generateTestData.R
# Simple simulation data generator for testing
# ML.Classification.FeatureSelection
# ============================================================

#' Generate simple two-view classification data for package testing
#'
#' This function creates simulated two-view data with either binary or
#' multiclass categorical outcomes. It is designed only for testing the
#' reticulate-based ML pipeline package.
#'
#' @param myseed Integer seed.
#' @param n1 Number of samples in class 1.
#' @param n2 Number of samples in class 2.
#' @param n3 Optional number of samples in class 3. If NULL, binary data are generated.
#' @param p1 Number of variables in view 1.
#' @param p2 Number of variables in view 2.
#' @param n_signal Number of signal variables per view.
#' @param signal_strength Mean shift for signal variables.
#' @param test_fraction Fraction of samples used as test data.
#' @param nReplicate Number of simulated replicates.
#'
#' @return A list with TrainData and TestData. Each replicate contains X and Y.
#'
generateTestData <- function(
    myseed = 1234L,
    n1 = 50L,
    n2 = 50L,
    n3 = NULL,
    p1 = 20L,
    p2 = 20L,
    n_signal = 5L,
    signal_strength = 1.0,
    test_fraction = 0.3,
    nReplicate = 1L
) {
  if (!is.null(myseed)) {
    set.seed(myseed)
  }

  simulate_one <- function(seed_offset = 0L) {
    set.seed(myseed + seed_offset)

    if (is.null(n3)) {
      class_sizes <- c(n1, n2)
      y <- c(rep("Class_1", n1), rep("Class_2", n2))
    } else {
      class_sizes <- c(n1, n2, n3)
      y <- c(rep("Class_1", n1), rep("Class_2", n2), rep("Class_3", n3))
    }

    n <- sum(class_sizes)
    k <- length(class_sizes)

    X1 <- matrix(rnorm(n * p1), nrow = n, ncol = p1)
    X2 <- matrix(rnorm(n * p2), nrow = n, ncol = p2)

    n_signal1 <- min(n_signal, p1)
    n_signal2 <- min(n_signal, p2)

    start_idx <- 1L

    for (class_id in seq_len(k)) {
      end_idx <- start_idx + class_sizes[class_id] - 1L
      idx <- start_idx:end_idx

      # Add class-specific signal to early features.
      # This makes the classification problem learnable.
      if (class_id == 1L) {
        X1[idx, seq_len(n_signal1)] <- X1[idx, seq_len(n_signal1)] + signal_strength
        X2[idx, seq_len(n_signal2)] <- X2[idx, seq_len(n_signal2)] + signal_strength
      }

      if (class_id == 2L) {
        X1[idx, seq_len(n_signal1)] <- X1[idx, seq_len(n_signal1)] - signal_strength
        X2[idx, seq_len(n_signal2)] <- X2[idx, seq_len(n_signal2)] - signal_strength
      }

      if (class_id == 3L) {
        X1[idx, seq_len(n_signal1)] <- X1[idx, seq_len(n_signal1)] + 0.5 * signal_strength
        X2[idx, seq_len(n_signal2)] <- X2[idx, seq_len(n_signal2)] - 0.5 * signal_strength
      }

      start_idx <- end_idx + 1L
    }

    colnames(X1) <- paste0("view1_X", seq_len(p1))
    colnames(X2) <- paste0("view2_X", seq_len(p2))

    # Shuffle rows
    perm <- sample(seq_len(n))
    X1 <- X1[perm, , drop = FALSE]
    X2 <- X2[perm, , drop = FALSE]
    y <- y[perm]

    # Train/test split stratified approximately by random split
    test_n <- max(1L, floor(test_fraction * n))
    test_idx <- sample(seq_len(n), size = test_n)
    train_idx <- setdiff(seq_len(n), test_idx)

    train <- list(
      X = list(
        X1[train_idx, , drop = FALSE],
        X2[train_idx, , drop = FALSE]
      ),
      Y = y[train_idx]
    )

    test <- list(
      X = list(
        X1[test_idx, , drop = FALSE],
        X2[test_idx, , drop = FALSE]
      ),
      Y = y[test_idx]
    )

    list(TrainData = train, TestData = test)
  }

  TrainData <- vector("list", nReplicate)
  TestData <- vector("list", nReplicate)

  for (j in seq_len(nReplicate)) {
    one <- simulate_one(seed_offset = j)
    TrainData[[j]] <- one$TrainData
    TestData[[j]] <- one$TestData
  }

  list(
    TrainData = TrainData,
    TestData = TestData
  )
}


#' Convert generated data to a flat data frame using View 1 only
#'
#' This converts the output of generateTestData() into the format required by
#' run_ml_pipeline(), using only the first view.
#'
#' @param mydata Output from generateTestData().
#' @param replicate_id Which replicate to use.
#' @param use_test_data Logical; if FALSE, uses TrainData. If TRUE, uses TestData.
#'
#' @return A data frame with RID, view 1 features, and outcome y.
#'
makePipelineDataFrame <- function(
    mydata,
    replicate_id = 1L,
    use_test_data = FALSE
) {
  data_name <- if (use_test_data) "TestData" else "TrainData"

  X1 <- mydata[[data_name]][[replicate_id]][["X"]][[1]]
  Y  <- mydata[[data_name]][[replicate_id]][["Y"]]

  X1 <- as.data.frame(X1)

  if (is.null(colnames(X1))) {
    colnames(X1) <- paste0("view1_X", seq_len(ncol(X1)))
  }

  df <- X1
  df$RID <- seq_len(nrow(df))
  df$y <- as.factor(Y)

  # Put RID and y first
  df <- df[, c("RID", "y", setdiff(names(df), c("RID", "y")))]

  df
}
