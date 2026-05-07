from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except Exception:
    _HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")


def _log_message(message: str, verbose: bool = True) -> None:
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")




def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def _progress_time_message(start_time: float, completed: int, total: int) -> str:
    elapsed = time.time() - start_time
    if completed <= 0:
        return f"elapsed={_format_seconds(elapsed)}, remaining=estimating..."
    avg_per_unit = elapsed / completed
    remaining = avg_per_unit * max(total - completed, 0)
    return f"elapsed={_format_seconds(elapsed)}, est. remaining={_format_seconds(remaining)}"

def _iter_bootstraps(n_bootstrap: int, progress_bar: bool = True, progress_desc: str = "Bootstraps"):
    iterator = range(n_bootstrap)
    if progress_bar and _HAS_TQDM:
        return tqdm(iterator, total=n_bootstrap, desc=progress_desc)
    return iterator


DEFAULT_RF_GRID = {
    "n_estimators": [200, 400, 600, 800, 1200],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [None, 10, 30, 50, 80],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

DEFAULT_SVM_GRID = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4],
    "gamma": ["scale", "auto"],
}

DEFAULT_LGB_GRID = {
    "learning_rate": [0.05, 0.075, 0.1],
    "n_estimators": [100, 200, 300],
    "min_child_samples": [20, 50, 100],
}

DEFAULT_LASSO_GRID = {
    "C": np.logspace(-2, 2, 10),
}

DEFAULT_ENET_GRID = {
    "C": np.logspace(-2, 2, 10),
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}


@dataclass
class BootstrapResult:
    model_name: str
    bootstrap_id: int
    best_params: Dict[str, Any]
    n_supervised_features: int
    n_model_selected_features: int
    selected_features: List[str]
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
    auc: float
    balanced_accuracy: float
    f1_weighted: float
    mcc: float



def _encode_target(y: pd.Series) -> Tuple[pd.Series, Dict[Any, int], Dict[int, Any]]:
    y = pd.Series(y).copy()
    classes = sorted(y.dropna().unique().tolist())
    if len(classes) < 2:
        raise ValueError("myMain requires at least 2 outcome classes.")
    forward = {cls: i for i, cls in enumerate(classes)}
    reverse = {i: cls for cls, i in forward.items()}
    return y.map(forward).astype(int), forward, reverse



def _safe_feature_pvalue(y: pd.Series, demo_df: pd.DataFrame, var_series: pd.Series) -> float:
    reduced = sm.add_constant(demo_df.copy(), has_constant="add")
    full = reduced.copy()
    full["var"] = var_series.values

    try:
        n_classes = int(pd.Series(y).nunique())
        if n_classes == 2:
            fit_reduced = sm.Logit(y, reduced).fit(disp=False)
            fit_full = sm.Logit(y, full).fit(disp=False)
        else:
            fit_reduced = sm.MNLogit(y, reduced).fit(disp=False)
            fit_full = sm.MNLogit(y, full).fit(disp=False)

        lr_stat = 2.0 * (fit_full.llf - fit_reduced.llf)
        df_diff = max(int(fit_full.df_model - fit_reduced.df_model), 1)
        return float(chi2.sf(lr_stat, df_diff))
    except Exception:
        return np.nan



def _supervised_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    demo_cols: Optional[List[str]],
    pvalue_threshold: float,
    min_features: int,
) -> List[str]:
    feature_cols = [c for c in X_train.columns if c not in (demo_cols or [])]

    if not demo_cols:
        return feature_cols

    demo_df = X_train[demo_cols].copy()
    pvals: List[Tuple[str, float]] = []
    for col in feature_cols:
        pval = _safe_feature_pvalue(y_train, demo_df, X_train[col])
        pvals.append((col, pval))

    selected = [name for name, p in pvals if pd.notna(p) and p <= pvalue_threshold]

    if len(selected) >= min_features:
        return selected

    ranked = sorted(
        pvals,
        key=lambda x: np.inf if pd.isna(x[1]) else x[1],
    )
    fallback = [name for name, _ in ranked[: max(min_features, 1)]]
    return fallback



def _make_bootstrap_split(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    all_classes = set(pd.Series(y).unique().tolist())

    for _ in range(100):
        train_idx = rng.integers(0, n, size=n)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[train_idx] = True
        test_idx = np.where(~in_bag)[0]

        if len(test_idx) == 0:
            continue

        train_classes = set(pd.Series(y.iloc[train_idx]).unique().tolist())
        test_classes = set(pd.Series(y.iloc[test_idx]).unique().tolist())

        if train_classes == all_classes and len(test_classes) >= 2:
            return train_idx, test_idx

    train_idx = rng.integers(0, n, size=n)
    in_bag = np.zeros(n, dtype=bool)
    in_bag[train_idx] = True
    test_idx = np.where(~in_bag)[0]

    if len(test_idx) == 0:
        test_idx = rng.choice(np.arange(n), size=max(1, n // 4), replace=False)

    return train_idx, test_idx



def _fit_search(
    estimator,
    param_grid: Optional[Dict[str, Iterable[Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter_search: int,
    random_state: int,
    n_jobs: int,
):
    if not param_grid:
        est = clone(estimator)
        est.fit(X_train, y_train)
        return est, {}

    min_class_count = int(pd.Series(y_train).value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=param_grid,
        n_iter=min(n_iter_search, int(np.prod([len(list(v)) for v in param_grid.values()]))),
        cv=cv,
        scoring="balanced_accuracy",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_



def _extract_importance(model_name: str, fitted_model, X_train_df: pd.DataFrame, X_test_scaled: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    feature_names = X_train_df.columns.tolist()

    if model_name == "RandomForest":
        values = fitted_model.feature_importances_
    elif model_name == "SVM":
        values = permutation_importance(
            fitted_model,
            X_test_scaled,
            y_test,
            n_repeats=10,
            random_state=1,
            scoring="balanced_accuracy",
        ).importances_mean
    elif model_name == "LightGBM":
        values = fitted_model.feature_importances_
    elif model_name in {"Lasso", "ElasticNet"}:
        coef = np.asarray(fitted_model.coef_)
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.abs(coef).mean(axis=0)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return dict(zip(feature_names, map(float, values)))



def _selected_features_from_importance(importance_dict: Dict[str, float], tol: float = 1e-12) -> List[str]:
    return [k for k, v in importance_dict.items() if np.abs(v) > tol]



def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        scores = np.asarray(scores)
        classes = np.unique(y_true)

        if scores.ndim == 1:
            return float(roc_auc_score(y_true, scores))

        if scores.ndim == 2 and scores.shape[1] == 2 and len(classes) == 2:
            return float(roc_auc_score(y_true, scores[:, 1]))

        return float(
            roc_auc_score(
                y_true,
                scores,
                multi_class="ovr",
                average="weighted",
            )
        )
    except Exception:
        return np.nan



def _summarize_model_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, d in results_df.groupby("model_name"):
        rows.append(
            {
                "model": model_name,
                "n_bootstrap": len(d),
                "auc_mean": d["auc"].mean(),
                "auc_sd": d["auc"].std(ddof=1),
                "balanced_accuracy_mean": d["balanced_accuracy"].mean(),
                "balanced_accuracy_sd": d["balanced_accuracy"].std(ddof=1),
                "f1_weighted_mean": d["f1_weighted"].mean(),
                "f1_weighted_sd": d["f1_weighted"].std(ddof=1),
                "mcc_mean": d["mcc"].mean(),
                "mcc_sd": d["mcc"].std(ddof=1),
                "n_supervised_features_mean": d["n_supervised_features"].mean(),
                "n_model_selected_features_mean": d["n_model_selected_features"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)



def _summarize_variable_selection(
    feature_names: List[str],
    bootstrap_results: List[BootstrapResult],
    model_names: List[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=feature_names)

    for model_name in model_names:
        model_res = [r for r in bootstrap_results if r.model_name == model_name]
        rank_map = {f: [] for f in feature_names}

        for r in model_res:
            items = sorted(r.feature_importance.items(), key=lambda x: x[1], reverse=True)
            selected = [(k, v) for k, v in items if np.abs(v) > 1e-12]
            rank_lookup = {name: rank + 1 for rank, (name, _) in enumerate(selected)}
            for f in feature_names:
                rank_map[f].append(rank_lookup.get(f, np.nan))

        out[f"{model_name}_avg_rank"] = [np.nanmean(rank_map[f]) for f in feature_names]
        out[f"{model_name}_select_freq"] = [np.sum(~np.isnan(rank_map[f])) for f in feature_names]

    rank_cols = [c for c in out.columns if c.endswith("_avg_rank")]
    freq_cols = [c for c in out.columns if c.endswith("_select_freq")]
    out["overall_avg_rank"] = out[rank_cols].mean(axis=1, skipna=True)
    out["Freq"] = np.floor(out[freq_cols].mean(axis=1, skipna=True) / max(len([r for r in bootstrap_results if r.model_name == model_names[0]]), 1) * 100).astype(int)
    keep_cols = ["overall_avg_rank", "Freq"] + rank_cols
    out = out[keep_cols]
    return out.sort_values(["overall_avg_rank", "Freq"], ascending=[True, False]).reset_index(names="feature")


def _make_variable_selection_filtered(variable_selection_summary: pd.DataFrame, min_freq: int = 70) -> pd.DataFrame:
    filtered = variable_selection_summary.loc[
        variable_selection_summary["Freq"] > min_freq,
        ["feature", "Freq", "overall_avg_rank"],
    ].copy()
    return filtered.sort_values(["overall_avg_rank", "Freq", "feature"], ascending=[True, False, True]).reset_index(drop=True)


def _best_bootstrap_variable_sets(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    metric_priority = ["balanced_accuracy", "auc", "mcc", "f1_weighted"]
    rows = []

    for model_name, d in bootstrap_df.groupby("model_name", sort=False):
        best_row = d.sort_values(metric_priority + ["bootstrap_id"], ascending=[False, False, False, False, True]).iloc[0]
        rows.append(
            {
                "model": model_name,
                "best_bootstrap_id": int(best_row["bootstrap_id"]),
                "balanced_accuracy": float(best_row["balanced_accuracy"]),
                "auc": float(best_row["auc"]),
                "mcc": float(best_row["mcc"]),
                "f1_weighted": float(best_row["f1_weighted"]),
                "n_supervised_features": int(best_row["n_supervised_features"]),
                "n_model_selected_features": int(best_row["n_model_selected_features"]),
                "selected_features": list(best_row["selected_features"]),
                "selected_features_string": ";".join(list(best_row["selected_features"])),
                "best_params": best_row["best_params"],
                "best_params_string": str(best_row["best_params"]),
            }
        )

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)




def _performance_plot_data(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    model_label_map = {
        "RandomForest": "Random Forest",
        "SVM": "Support Vector Machine",
        "LightGBM": "Light GBM",
        "Lasso": "Lasso",
        "ElasticNet": "Elastic Net",
    }

    plot_df = bootstrap_df[["model_name", "balanced_accuracy", "f1_weighted", "mcc", "auc"]].copy()
    plot_df["model"] = plot_df["model_name"].map(model_label_map).fillna(plot_df["model_name"])
    plot_df = plot_df.drop(columns=["model_name"])
    plot_df = plot_df.rename(
        columns={
            "balanced_accuracy": "Balanced Acc",
            "f1_weighted": "F1",
            "mcc": "MCC",
            "auc": "AUC",
        }
    )
    return plot_df[["Balanced Acc", "F1", "MCC", "AUC", "model"]]


def _plot_classification_performance(
    plot_df: pd.DataFrame,
    save_plot_path: Optional[str] = None,
    show_plot: bool = False,
    title: Optional[str] = None,
):
    df_melted = plot_df.melt(id_vars=["model"], var_name="Metric", value_name="Value")

    metric_order = ["Balanced Acc", "F1", "MCC", "AUC"]
    model_order = [
        m for m in [
            "Random Forest",
            "Support Vector Machine",
            "Light GBM",
            "Lasso",
            "Elastic Net",
        ]
        if m in df_melted["model"].unique()
    ]

    colors = ["#e94c88", "#ffb602", "#5dade2", "#aa6f73"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="model",
        y="Value",
        hue="Metric",
        data=df_melted,
        order=model_order,
        hue_order=metric_order,
        palette=sns.color_palette(colors),
        ax=ax,
    )

    ax.set_title(
        title
        or "Classification Performance Over Bootstrap Samples",
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Metrics")
    fig.tight_layout()

    if save_plot_path:
        save_path = Path(save_plot_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, df_melted


def _stability_heatmap_data(
    bootstrap_results: List[BootstrapResult],
    model_names: List[str],
    top_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    feature_names = sorted({f for r in bootstrap_results for f in r.feature_importance.keys()})
    out = pd.DataFrame(index=feature_names)

    for model_name in model_names:
        model_res = [r for r in bootstrap_results if r.model_name == model_name]
        denom = max(len(model_res), 1)
        freq = {}
        for f in feature_names:
            count = 0
            for r in model_res:
                if abs(r.feature_importance.get(f, 0.0)) > 1e-12:
                    count += 1
            freq[f] = np.floor(count / denom * 100)
        out[model_name] = pd.Series(freq)

    out = out.fillna(0).astype(int)
    if top_features is not None:
        keep = [f for f in top_features if f in out.index]
        out = out.loc[keep]
    return out


def _plot_stability_heatmap(
    heatmap_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    title: Optional[str] = None,
):
    model_label_map = {
        "RandomForest": "Random Forest",
        "SVM": "Support Vector Machine",
        "LightGBM": "Light GBM",
        "Lasso": "Lasso",
        "ElasticNet": "Elastic Net",
    }
    plot_df = heatmap_df.copy()
    plot_df = plot_df.rename(columns=model_label_map)

    fig_h = max(4, 0.35 * max(len(plot_df), 1) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.heatmap(plot_df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Selection frequency (%)'}, ax=ax)
    ax.set_title(title or 'Feature Stability Heatmap Across Models')
    ax.set_xlabel('Model')
    ax.set_ylabel('Feature')
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def myMain(
    training_data: pd.DataFrame,
    outcome_col: str,
    demo_cols: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
    n_bootstrap: int = 100,
    pvalue_threshold: float = 0.05,
    min_features_after_filter: int = 5,
    n_iter_search: int = 20,
    random_state: int = 10,
    n_jobs: int = -1,
    make_performance_plot: bool = True,
    save_plot_path: Optional[str] = None,
    show_plot: bool = False,
    plot_title: Optional[str] = None,
    verbose: bool = True,
    progress_bar: bool = True,
    print_every_bootstrap: int = 1,
    make_stability_heatmap: bool = True,
    stability_heatmap_path: Optional[str] = None,
    show_stability_heatmap: bool = False,
    heatmap_top_n_features: int = 30,
    heatmap_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the model pipeline on a training dataset only.

    Parameters
    ----------
    training_data : pd.DataFrame
        Full training dataset. Testing data should stay outside this function.
    outcome_col : str
        Outcome column name. Supports binary and multiclass classification.
    demo_cols : list[str], optional
        Demographic covariates used only in the supervised filtering step.
        Example: ["AGE", "PTGENDER", "APOE4"]
    id_cols : list[str], optional
        Columns to exclude from model fitting, such as ["RID"].
    n_bootstrap : int, default=100
        Number of bootstrap iterations.
    make_performance_plot : bool, default=True
        Whether to create a grouped boxplot of classification metrics across bootstraps.
    save_plot_path : str, optional
        File path for saving the performance plot, for example "figures/mci.png".
    show_plot : bool, default=False
        Whether to display the plot immediately.
    plot_title : str, optional
        Custom title for the performance plot.
    verbose : bool, default=True
        Whether to print progress messages during execution.
    progress_bar : bool, default=True
        Whether to show a tqdm progress bar over bootstrap iterations.
    print_every_bootstrap : int, default=1
        Frequency for printing bootstrap progress updates. Use 5 to print every 5 bootstraps.
    make_stability_heatmap : bool, default=True
        Whether to create a feature-stability heatmap across models.
    stability_heatmap_path : str, optional
        File path for saving the stability heatmap.
    show_stability_heatmap : bool, default=False
        Whether to display the heatmap immediately.
    heatmap_top_n_features : int, default=30
        Number of top features (by overall_avg_rank) to show in the heatmap.
    heatmap_title : str, optional
        Custom title for the stability heatmap.

    Returns
    -------
    dict with:
        - classification_summary: per-model summary table
        - variable_selection_summary: variable-ranking table with Freq
        - variable_selection_top_features: 3-column table filtered to Freq > 75
        - best_bootstrap_variable_sets: selected-variable set from the best bootstrap of each model
        - bootstrap_results: per-bootstrap results table
        - performance_plot_data: wide-format data used for plotting
        - performance_plot_long: long-format data used for plotting
        - performance_plot_path: saved figure path if requested
        - stability_heatmap_data: feature-by-model stability table in %
        - stability_heatmap_path: saved heatmap path if requested
        - label_mapping: original class to integer mapping
    """
    if outcome_col not in training_data.columns:
        raise ValueError(f"{outcome_col!r} is not a column in training_data.")

    demo_cols = demo_cols or []
    id_cols = id_cols or []

    missing_demo = [c for c in demo_cols if c not in training_data.columns]
    missing_ids = [c for c in id_cols if c not in training_data.columns]
    if missing_demo:
        raise ValueError(f"demo_cols not found: {missing_demo}")
    if missing_ids:
        raise ValueError(f"id_cols not found: {missing_ids}")

    data = training_data.dropna(axis=0, how="any").copy()
    y_encoded, label_mapping, reverse_mapping = _encode_target(data[outcome_col])

    excluded = set([outcome_col] + id_cols)
    X = data.drop(columns=list(excluded)).copy()
    all_model_features = [c for c in X.columns if c not in demo_cols]

    if len(all_model_features) == 0:
        raise ValueError("No model features remain after removing outcome/id/demo columns.")

    _log_message(f"myMain started: {len(data)} complete-case samples, {len(all_model_features)} candidate model features, {n_bootstrap} bootstraps.", verbose=verbose)

    if not _HAS_LIGHTGBM:
        warnings.warn("lightgbm is not installed, so the LightGBM model will be skipped.")

    n_classes = int(y_encoded.nunique())
    multi_class_mode = "ovr" if n_classes == 2 else "multinomial"

    model_specs = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": DEFAULT_RF_GRID,
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=random_state),
            "param_grid": DEFAULT_SVM_GRID,
        },
        "Lasso": {
            # using logistic lasso for classification instead of linear Lasso regression
            "estimator": LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=5000,
                random_state=random_state,
                multi_class=multi_class_mode,
            ),
            "param_grid": DEFAULT_LASSO_GRID,
        },
        "ElasticNet": {
            # using logistic elastic net for classification instead of linear ElasticNet regression
            "estimator": LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                random_state=random_state,
                multi_class=multi_class_mode,
            ),
            "param_grid": DEFAULT_ENET_GRID,
        },
    }

    if _HAS_LIGHTGBM:
        lgb_kwargs = {"random_state": random_state, "verbosity": -1}
        if n_classes > 2:
            lgb_kwargs.update({"objective": "multiclass", "num_class": n_classes})
        model_specs["LightGBM"] = {
            "estimator": lgb.LGBMClassifier(**lgb_kwargs),
            "param_grid": DEFAULT_LGB_GRID,
        }

    rng = np.random.default_rng(random_state)
    bootstrap_results: List[BootstrapResult] = []

    pipeline_start_time = time.time()
    total_models = len(model_specs)
    _log_message(f"Models to fit in each bootstrap: {', ' .join(model_specs.keys())}.", verbose=verbose)

    for b in _iter_bootstraps(n_bootstrap, progress_bar=progress_bar, progress_desc="Bootstraps"):
        bootstrap_id = b + 1
        should_print = verbose and (bootstrap_id == 1 or bootstrap_id % max(print_every_bootstrap, 1) == 0 or bootstrap_id == n_bootstrap)
        if should_print:
            timing_msg = _progress_time_message(pipeline_start_time, bootstrap_id - 1, n_bootstrap)
            _log_message(f"Bootstrap {bootstrap_id}/{n_bootstrap}: generating in-bag and out-of-bag samples ({timing_msg}).", verbose=verbose)

        train_idx, test_idx = _make_bootstrap_split(X, y_encoded, rng)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y_encoded.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y_encoded.iloc[test_idx].reset_index(drop=True)

        selected_by_filter = _supervised_filter(
            X_train=X_train,
            y_train=y_train,
            demo_cols=demo_cols,
            pvalue_threshold=pvalue_threshold,
            min_features=min_features_after_filter,
        )

        if should_print:
            timing_msg = _progress_time_message(pipeline_start_time, bootstrap_id - 1, n_bootstrap)
            _log_message(
                f"Bootstrap {bootstrap_id}/{n_bootstrap}: supervised filtering kept {len(selected_by_filter)} features; fitting {total_models} models ({timing_msg}).",
                verbose=verbose,
            )

        X_train_f = X_train[selected_by_filter].copy()
        X_test_f = X_test[selected_by_filter].copy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_test_scaled = scaler.transform(X_test_f)

        for model_idx, (model_name, spec) in enumerate(model_specs.items(), start=1):
            if should_print:
                timing_msg = _progress_time_message(pipeline_start_time, bootstrap_id - 1, n_bootstrap)
                _log_message(
                    f"Bootstrap {bootstrap_id}/{n_bootstrap}: fitting {model_name} ({model_idx}/{total_models}; {timing_msg}).",
                    verbose=verbose,
                )
            model_seed = random_state + b + hash(model_name) % 10000
            fitted_model, best_params = _fit_search(
                estimator=spec["estimator"],
                param_grid=spec["param_grid"],
                X_train=X_train_scaled,
                y_train=y_train.to_numpy(),
                n_iter_search=n_iter_search,
                random_state=model_seed,
                n_jobs=n_jobs,
            )

            y_pred = fitted_model.predict(X_test_scaled)

            if hasattr(fitted_model, "predict_proba"):
                y_score = fitted_model.predict_proba(X_test_scaled)
            elif hasattr(fitted_model, "decision_function"):
                y_score = fitted_model.decision_function(X_test_scaled)
            else:
                y_score = y_pred

            importance = _extract_importance(
                model_name=model_name,
                fitted_model=fitted_model,
                X_train_df=X_train_f,
                X_test_scaled=X_test_scaled,
                y_test=y_test.to_numpy(),
            )
            selected_features = _selected_features_from_importance(importance)

            bootstrap_results.append(
                BootstrapResult(
                    model_name=model_name,
                    bootstrap_id=bootstrap_id,
                    best_params=best_params,
                    n_supervised_features=len(selected_by_filter),
                    n_model_selected_features=len(selected_features),
                    selected_features=selected_features,
                    feature_importance=importance,
                    confusion_matrix=confusion_matrix(
                        y_test,
                        y_pred,
                        labels=np.arange(n_classes),
                    ).tolist(),
                    auc=_safe_auc(y_test.to_numpy(), np.asarray(y_score)),
                    balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                    f1_weighted=float(f1_score(y_test, y_pred, average="weighted")),
                    mcc=float(matthews_corrcoef(y_test, y_pred)),
                )
            )

    bootstrap_df = pd.DataFrame([
        {
            "model_name": r.model_name,
            "bootstrap_id": r.bootstrap_id,
            "best_params": r.best_params,
            "best_params_string": str(r.best_params),
            "n_supervised_features": r.n_supervised_features,
            "n_model_selected_features": r.n_model_selected_features,
            "selected_features": r.selected_features,
            "selected_features_string": ";".join(r.selected_features),
            "feature_importance": r.feature_importance,
            "confusion_matrix": r.confusion_matrix,
            "auc": r.auc,
            "balanced_accuracy": r.balanced_accuracy,
            "f1_weighted": r.f1_weighted,
            "mcc": r.mcc,
        }
        for r in bootstrap_results
    ])

    _log_message(
        f"Bootstrap fitting complete. Summarizing classification and variable-selection results ({_progress_time_message(pipeline_start_time, n_bootstrap, n_bootstrap)}).",
        verbose=verbose,
    )

    classification_summary = _summarize_model_results(bootstrap_df)
    variable_selection_summary = _summarize_variable_selection(
        feature_names=all_model_features,
        bootstrap_results=bootstrap_results,
        model_names=list(model_specs.keys()),
    )
    variable_selection_top_features = _make_variable_selection_filtered(
        variable_selection_summary=variable_selection_summary,
        min_freq=75,
    )
    best_bootstrap_variable_sets = _best_bootstrap_variable_sets(bootstrap_df)

    performance_plot_data = _performance_plot_data(bootstrap_df)
    performance_plot_long = performance_plot_data.melt(
        id_vars=["model"], var_name="Metric", value_name="Value"
    )

    performance_plot_path = None
    if make_performance_plot:
        _log_message(
            f"Creating classification performance visualization ({_progress_time_message(pipeline_start_time, n_bootstrap, n_bootstrap)}).",
            verbose=verbose,
        )
        _fig, _ax, performance_plot_long = _plot_classification_performance(
            plot_df=performance_plot_data,
            save_plot_path=save_plot_path,
            show_plot=show_plot,
            title=plot_title,
        )
        performance_plot_path = save_plot_path

    stability_heatmap_data = None
    saved_stability_heatmap_path = None
    if make_stability_heatmap:
        _log_message(
            f"Creating feature stability heatmap ({_progress_time_message(pipeline_start_time, n_bootstrap, n_bootstrap)}).",
            verbose=verbose,
        )
        top_features_for_heatmap = variable_selection_summary.sort_values(["overall_avg_rank", "Freq"], ascending=[True, False])["feature"].head(max(heatmap_top_n_features, 1)).tolist()
        stability_heatmap_data = _stability_heatmap_data(
            bootstrap_results=bootstrap_results,
            model_names=list(model_specs.keys()),
            top_features=top_features_for_heatmap,
        )
        if stability_heatmap_path is None:
            stability_heatmap_path = "figures/feature_stability_heatmap.png"
        _fig_hm, _ax_hm = _plot_stability_heatmap(
            heatmap_df=stability_heatmap_data,
            save_path=stability_heatmap_path,
            show_plot=show_stability_heatmap,
            title=heatmap_title,
        )
        saved_stability_heatmap_path = stability_heatmap_path

    _log_message(
        f"myMain finished in {_format_seconds(time.time() - pipeline_start_time)}.",
        verbose=verbose,
    )

    return {
        "classification_summary": classification_summary,
        "variable_selection_summary": variable_selection_summary,
        "variable_selection_top_features": variable_selection_top_features,
        "best_bootstrap_variable_sets": best_bootstrap_variable_sets,
        "bootstrap_results": bootstrap_df,
        "performance_plot_data": performance_plot_data,
        "performance_plot_long": performance_plot_long,
        "performance_plot_path": performance_plot_path,
        "stability_heatmap_data": stability_heatmap_data,
        "stability_heatmap_path": saved_stability_heatmap_path,
        "label_mapping": label_mapping,
        "reverse_label_mapping": reverse_mapping,
    }


if __name__ == "__main__":
    print("This file defines myMain(training_data, outcome_col, ...).")
