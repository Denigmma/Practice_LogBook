import argparse
import json
import math
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE = sqrt(mean((log1p(y_pred) - log1p(y_true))^2))"""
    # защита от отрицательных шумовых предсказаний
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def build_models(n_features: int):
    """Два базовых модели + веса ансамбля."""
    # Модель 1: Градиентный бустинг по исходным фичам
    gbr = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.9,
        random_state=42,
        loss="squared_error",
    )
    model_gbr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("gbr", gbr),
        ]
    )

    # Модель 2: Ридж на квадратичных полиномиальных признаках
    poly_block = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )

    return model_gbr, poly_block


def cv_report(model: Pipeline, X: pd.DataFrame, y_log: np.ndarray, name: str) -> float:
    """Оцениваем через RMSLE в «нативном» масштабе (экспонируя предсказания)."""
    # Кастомный scorer: модель обучается на log1p(y), но нам нужен RMSLE по expm1(pred)
    def _scorer(est, X_val, y_log_val):
        y_log_pred = est.predict(X_val)
        y_pred = np.expm1(y_log_pred)
        y_true = np.expm1(y_log_val)
        return -rmsle(y_true, y_pred)  # отрицательный, т.к. sklearn максимизирует

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_log, scoring=_scorer, cv=kf, n_jobs=None)
    mean_rmsle = -scores.mean()
    std_rmsle = scores.std()
    print(f"[CV] {name}: RMSLE = {mean_rmsle:.5f} ± {std_rmsle:.5f}")
    return mean_rmsle


def main():
    parser = argparse.ArgumentParser(description="Predict MSE for weight vectors.")
    parser.add_argument("--train", default="train_weights.csv")
    parser.add_argument("--test", default="test_weights.csv")
    parser.add_argument("--out", default="answers")
    parser.add_argument("--round_mse", type=int, default=3,
                        help="Округление MSE в JSON (число знаков после запятой). -1 = без округления")
    args = parser.parse_args()

    # --- Загрузка данных ---
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    feature_cols = [c for c in train.columns if c.startswith("W")]
    assert set(feature_cols) == set(test.columns), "test.csv должен иметь те же W0..W9"
    feature_cols = sorted(feature_cols, key=lambda x: int(x[1:]))  # гарантируем порядок W0..W9

    X = train[feature_cols].copy()
    y = train["MSE"].astype(float).values
    y = np.maximum(y, 0)  # на всякий
    y_log = np.log1p(y)

    # --- Модели ---
    m1, m2 = build_models(n_features=len(feature_cols))

    # Оборачиваем обучающие модели так, чтобы они предсказывали log1p(MSE)
    # (сколько-нибудь сложная трансформация цели в sklearn-пайплайне делается вручную)
    # 1) Обучаем
    m1.fit(X, y_log)
    m2.fit(X, y_log)

    # --- Кросс-валидационный отчёт (для самопроверки локально) ---
    # Можно закомментировать для ускорения
    try:
        _ = cv_report(m1, X, y_log, "GradientBoosting")
        _ = cv_report(m2, X, y_log, "Ridge(Poly2)")
    except Exception as e:
        print(f"[WARN] CV skipped: {e}")

    # --- Ансамбль ---
    # Подбираем веса простым правилом: чуть больше доверия бустингу
    w1, w2 = 0.6, 0.4

    # --- Обучение на всём трейне и предсказание ---
    m1.fit(X, y_log)
    m2.fit(X, y_log)

    X_test = test[feature_cols].copy()
    y_log_pred_1 = m1.predict(X_test)
    y_log_pred_2 = m2.predict(X_test)
    y_log_pred = w1 * y_log_pred_1 + w2 * y_log_pred_2

    y_pred = np.expm1(y_log_pred)          # возвращаемся к масштабу MSE
    y_pred = np.maximum(y_pred, 0.0)       # на всякий случай

    # --- Формирование JSON ответа ---
    out_records: List[dict] = []
    for i in range(len(test)):
        row = {col: (float(test.loc[i, col]) if np.issubdtype(type(test.loc[i, col]), np.number) else test.loc[i, col])
               for col in feature_cols}
        mse_val = float(y_pred[i])
        if args.round_mse >= 0:
            mse_val = float(np.round(mse_val, args.round_mse))
        row["MSE"] = mse_val
        out_records.append(row)

    # Имя файла строго 'answers' без пробелов
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(out_records)} predictions to '{out_path}'")


if __name__ == "__main__":
    main()
