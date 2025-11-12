"""
Stacked baseline for the bio-cybernetics task.
- Robust to sklearn version differences (OneHotEncoder sparse/sparse_output).
- 5-fold OOF stacking of multiple models + logistic meta-learner.
- Writes answers.csv compatible with example.csv if present.
"""

import os
import sys
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
# (Можно добавить RandomForest/ExtraTrees при желании, но они медленнее и не всегда дают прирост)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ---------- Utils: robust encoders / preprocessors ----------

def detect_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical by dtype."""
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def make_ohe_compatible() -> OneHotEncoder:
    """Create OneHotEncoder that works with both old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocessor_std_ohe(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Numeric: impute+scale; Categorical: impute+OHE+to dense."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ohe = make_ohe_compatible()
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def preprocessor_ohe_no_scale(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Numeric: impute only; Categorical: impute+OHE; No scaling (удобно для бустингов)."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    ohe = make_ohe_compatible()
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def preprocessor_quantile_then_scale(num_cols: List[str], cat_cols: List[str], n_samples: int) -> ColumnTransformer:
    """Numeric: impute + quantile->normal + scale (часто помогает логрегу); Categorical: impute+OHE."""
    n_quant = min(1000, max(10, n_samples))
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=n_quant, subsample=int(1e6), random_state=RANDOM_STATE)),
        ("scale", StandardScaler()),
    ])
    ohe = make_ohe_compatible()
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def try_histgb(**kwargs):
    """Robust creator for HistGradientBoosting (some params may be missing in very old sklearn)."""
    try:
        return HistGradientBoostingClassifier(**kwargs)
    except TypeError:
        # drop maybe-unknown keys
        safe = {}
        for k, v in kwargs.items():
            try:
                HistGradientBoostingClassifier(**{k: v})
                safe[k] = v
            except TypeError:
                pass
        return HistGradientBoostingClassifier(**safe)


# ---------- Stacking core ----------

def oof_stacking(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
    base_specs: List[Tuple[str, Pipeline]], n_splits: int = 5, random_state: int = 42
):
    """
    base_specs: list of (name, pipeline_with_model)
    Returns:
        oof_pred: (n_samples, n_models) OOF probabilities
        test_pred: (n_test, n_models) test probabilities from full-fit models
        cv_scores: dict name -> fold AUC list
        fitted_full: dict name -> fitted pipeline on full train
    """
    n = len(X)
    k = len(base_specs)
    oof_pred = np.zeros((n, k), dtype=float)
    test_pred = np.zeros((len(X_test), k), dtype=float)
    cv_scores: Dict[str, List[float]] = {name: [] for name, _ in base_specs}
    fitted_full = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # OOF
    for m_idx, (name, pipe) in enumerate(base_specs):
        fold_preds = np.zeros(n, dtype=float)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]
            pipe.fit(X_tr, y_tr)
            if hasattr(pipe, "predict_proba"):
                pv = pipe.predict_proba(X_va)[:, 1]
            else:
                # fallback to decision_function -> sigmoid
                scores_raw = pipe.decision_function(X_va)
                pv = 1.0 / (1.0 + np.exp(-scores_raw))
            fold_preds[va] = pv
            auc = roc_auc_score(y_va, pv)
            cv_scores[name].append(float(auc))
            print(f"[OOF] {name} fold {fold}/{n_splits}: AUC={auc:.5f}")
        oof_pred[:, m_idx] = fold_preds
        print(f"[OOF] {name} mean AUC: {np.mean(cv_scores[name]):.5f} ± {np.std(cv_scores[name]):.5f}")

        # full fit for test
        pipe_full = Pipeline(pipe.steps)  # clone-ish (simple copy of steps)
        pipe_full.fit(X, y)
        if hasattr(pipe_full, "predict_proba"):
            pt = pipe_full.predict_proba(X_test)[:, 1]
        else:
            scores_raw = pipe_full.decision_function(X_test)
            pt = 1.0 / (1.0 + np.exp(-scores_raw))
        test_pred[:, m_idx] = pt
        fitted_full[name] = pipe_full

    return oof_pred, test_pred, cv_scores, fitted_full


def write_submission(proba: np.ndarray, test: pd.DataFrame, out_path: str = "answers.csv") -> None:
    """Write answers.csv; try to mimic example.csv format if present."""
    if os.path.exists("example.csv"):
        example = pd.read_csv("example.csv")
        if len(example) == len(proba):
            tgt_cols = [c for c in example.columns if c.lower() == "target"]
            if tgt_cols:
                example[tgt_cols[0]] = proba
            else:
                example.iloc[:, -1] = proba
            example.to_csv(out_path, index=False)
            print(f"[OK] Wrote predictions to {out_path} (matching example.csv)")
            return
        else:
            print("[WARN] example.csv length != test length; writing generic answers.csv")

    sub = pd.DataFrame({"target": proba})
    for cand in ("id", "Id", "ID"):
        if cand in test.columns:
            sub.insert(0, cand, test[cand].values)
            break
    sub.to_csv(out_path, index=False)
    print(f"[OK] Wrote predictions to {out_path}")


# ---------- Main ----------

def main():
    if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
        print("ERROR: put train.csv and test.csv in the current directory.", file=sys.stderr)
        sys.exit(1)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    if "target" not in train.columns:
        print("ERROR: 'target' column not found in train.csv", file=sys.stderr)
        sys.exit(1)

    y = train["target"].values.astype(int)
    X = train.drop(columns=["target"])
    X_test = test.copy()

    num_cols, cat_cols = detect_columns(X)
    print(f"[INFO] Detected {len(num_cols)} numeric and {len(cat_cols)} categorical columns.")

    # Preprocessors: разные для разных моделей (часто помогает)
    pre_std = preprocessor_std_ohe(num_cols, cat_cols)
    pre_ohe_no_scale = preprocessor_ohe_no_scale(num_cols, cat_cols)
    pre_qt_std = preprocessor_quantile_then_scale(num_cols, cat_cols, n_samples=len(X))

    # Base models (пара сильных конфигураций HistGB + GBC + LR)
    hgb1 = try_histgb(
        learning_rate=0.06, max_iter=800, max_depth=3,
        l2_regularization=1e-4, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
    )
    hgb2 = try_histgb(
        learning_rate=0.03, max_iter=1200, max_depth=None,
        l2_regularization=1e-4, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30
    )
    gbc = GradientBoostingClassifier(
        n_estimators=700, learning_rate=0.03, max_depth=3,
        subsample=0.9, random_state=RANDOM_STATE
    )
    lr = LogisticRegression(
        solver="lbfgs", penalty="l2", C=1.0, max_iter=2000, random_state=RANDOM_STATE
    )

    base_specs: List[Tuple[str, Pipeline]] = [
        ("HGB_depth3", Pipeline([("pre", pre_ohe_no_scale), ("clf", hgb1)])),
        ("HGB_deep",   Pipeline([("pre", pre_ohe_no_scale), ("clf", hgb2)])),
        ("GBC",        Pipeline([("pre", pre_ohe_no_scale), ("clf", gbc)])),
        ("LR_std",     Pipeline([("pre", pre_qt_std),       ("clf", lr)])),
    ]

    # OOF stacking
    oof_pred, test_pred, cv_scores, fitted_full = oof_stacking(
        X, y, X_test, base_specs, n_splits=5, random_state=RANDOM_STATE
    )

    # Meta-learner on OOF
    meta = LogisticRegression(
        solver="lbfgs", penalty="l2", C=1.0, max_iter=500, random_state=RANDOM_STATE
    )
    meta.fit(oof_pred, y)
    oof_meta = meta.predict_proba(oof_pred)[:, 1]
    cv_auc = roc_auc_score(y, oof_meta)
    print(f"[STACK] OOF meta AUC: {cv_auc:.5f}")

    # Final test predictions by stacking
    test_meta = meta.predict_proba(test_pred)[:, 1]

    # (опционально) лёгкая стабилизация крайних значений
    eps = 1e-6
    test_meta = np.clip(test_meta, eps, 1 - eps)

    write_submission(test_meta, X_test, out_path="answers.csv")


if __name__ == "__main__":
    main()