"""
증강된 특징을 활용해 빠르게 회귀 모델을 비교/훈련하는 모듈.

해당 모듈은 함수를 통해 재사용할 수 있으며, 스크립트로 실행하면
자동으로 최신 모델을 `artifacts/models/best_improved_model.pkl`에 저장한다.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..config import EMBEDDINGS_PATH, MODEL_ARTIFACT_PATH, PROCESSED_DATA_PATH

warnings.filterwarnings("ignore")

try:  # Optional dependency
    import xgboost as xgb

    HAS_XGB = True
except ImportError:  # pragma: no cover - informative warning only
    HAS_XGB = False
    xgb = None  # type: ignore[assignment]


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def _load_dataset(data_path: Path, embeddings_path: Path) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load processed features and optional embeddings."""

    print(f"\n[1] Loading augmented data from {data_path.relative_to(PROCESSED_DATA_PATH.parents[1])}")
    df = pd.read_csv(data_path)
    print(f"    • Shape: {df.shape}")

    embeddings = None
    if embeddings_path.exists():
        print(f"\n[1-1] Loading embeddings from {embeddings_path.relative_to(embeddings_path.parents[2])}")
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        print(f"    • Embedding matrix: {len(embeddings)} × {len(embeddings[0])}")
    else:
        print("\n[1-1] Embeddings file not found → skipping")

    return df, embeddings


def train_realtime_model(
    data_path: Path = PROCESSED_DATA_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    artifact_path: Path = MODEL_ARTIFACT_PATH,
) -> dict:
    """
    Train multiple regressors on the augmented dataset and persist the best model.

    Returns:
        dict: Summary containing artifact path, metrics, and top models.
    """

    print("=" * 70)
    print("QUICK MODEL TRAINING - MULTI-MODEL COMPARISON")
    print("=" * 70)

    df, embeddings = _load_dataset(data_path, embeddings_path)

    X = df.drop(["queries", "MAL"], axis=1)
    y = df["MAL"]

    print(f"\nFeatures: {X.shape[1]} columns")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Encode categorical columns
    print("\n[2] Encoding categorical features...")
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"    • Categorical columns: {len(categorical_cols)}")

    label_encoders: dict[str, LabelEncoder] = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    print(f"    • Encoded shape: {X_encoded.shape}")
    print("\n[2-1] Embeddings omitted intentionally (small dataset → overfitting risk)")

    print("\n[3] Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_STATE
        )
    print(f"    • Train size: {X_train.shape[0]}")
    print(f"    • Test size : {X_test.shape[0]}")

    scaler = None

    print("\n[4] Baseline (mean predictor)...")
    y_pred_mean = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, y_pred_mean)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    baseline_r2 = r2_score(y_test, y_pred_mean)
    print(f"    • MAE : {baseline_mae:.4f}")
    print(f"    • RMSE: {baseline_rmse:.4f}")
    print(f"    • R²  : {baseline_r2:.4f}")

    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Ridge (alpha=0.5)": Ridge(alpha=0.5, random_state=RANDOM_STATE),
        "Ridge (alpha=2.0)": Ridge(alpha=2.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest (deep)": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "LightGBM (tuned)": lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    if HAS_XGB and xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        models["XGBoost (tuned)"] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print("\n[!] XGBoost 미설치 → 해당 모델은 스킵합니다.")

    print("\n[5] Training models...")
    print("=" * 70)

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

        print(f"    • MAE : {mae:.4f}")
        print(f"    • RMSE: {rmse:.4f}")
        print(f"    • R²  : {r2:.4f}")

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<22} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 70)
    print(f"{'Baseline (Mean)':<22} {baseline_mae:<12.4f} {baseline_rmse:<12.4f} {baseline_r2:<12.4f}")
    for model_name, metrics in results.items():
        print(f"{model_name:<22} {metrics['MAE']:<12.4f} {metrics['RMSE']:<12.4f} {metrics['R²']:<12.4f}")

    best_model_name = max(results.keys(), key=lambda k: results[k]["R²"])
    print(f"\n✓ Best model: {best_model_name}")
    print(f"    • R²  = {results[best_model_name]['R²']:.4f}")
    print(f"    • MAE = {results[best_model_name]['MAE']:.4f}")

    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL (Top 3 average)")
    print("=" * 70)

    top3_models = sorted(results.items(), key=lambda x: x[1]["R²"], reverse=True)[:3]
    ensemble_preds = np.zeros(len(y_test))
    for name, _ in top3_models:
        ensemble_preds += models[name].predict(X_test)
    ensemble_preds /= 3

    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    ensemble_r2 = r2_score(y_test, ensemble_preds)

    print(f"    • MAE : {ensemble_mae:.4f}")
    print(f"    • RMSE: {ensemble_rmse:.4f}")
    print(f"    • R²  : {ensemble_r2:.4f}")

    if ensemble_r2 > results[best_model_name]["R²"]:
        print("    → Ensemble 개선 확인! (best single 대비 향상)")
        best_model_name = "Ensemble"
        results["Ensemble"] = {"MAE": ensemble_mae, "RMSE": ensemble_rmse, "R²": ensemble_r2}
        best_model = None  # Ensemble 결과는 단일 모델 객체가 아님
    else:
        print("    → Ensemble이 추가 이득을 주지 못함")
        best_model = models[best_model_name]

    print("\nImprovement over baseline:")
    print(f"    • ΔR² : {results[best_model_name]['R²'] - baseline_r2:.4f}")
    print(f"    • MAE ↓: {(1 - results[best_model_name]['MAE'] / baseline_mae) * 100:.1f}%")

    feature_names_list = X_encoded.columns.tolist()

    if best_model is None:
        # Fall back to best single model for persistence
        best_model_name = max(
            (name for name in results if name != "Ensemble"),
            key=lambda k: results[k]["R²"],
        )
        best_model = models[best_model_name]

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "feature_names": feature_names_list,
                "label_encoders": label_encoders,
                "metrics": results[best_model_name],
                "uses_embeddings": False,
                "embedding_dim": 0 if embeddings is None else len(embeddings[0]),
            },
            f,
        )

    print(f"\n✓ Best model saved to {artifact_path.relative_to(artifact_path.parents[1])}")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    return {
        "artifact_path": artifact_path,
        "best_model_name": best_model_name,
        "metrics": results[best_model_name],
        "all_results": results,
    }


if __name__ == "__main__":
    train_realtime_model()