"""
Quick Model Training - Test 5 sklearn models on augmented data

Models:
1. Linear Regression (baseline)
2. Ridge Regression
3. Random Forest
4. Gradient Boosting (sklearn)
5. LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not installed, skipping XGBoost models")

import pickle
import warnings
warnings.filterwarnings('ignore')

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("QUICK MODEL TRAINING - 5 SKLEARN MODELS")
print("="*70)

# Load data
print("\n[1] Loading augmented data...")
df = pd.read_csv('augmented_data.csv')
print(f"Dataset shape: {df.shape}")

# Load embeddings
print("\n[1-1] Loading embeddings...")
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
print(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")

# Separate features and target
X = df.drop(['queries', 'MAL'], axis=1)
y = df['MAL']

print(f"Features: {X.shape[1]} columns")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# Encode categorical features
print("\n[2] Encoding categorical features...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {len(categorical_cols)}")

label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"Encoded shape: {X_encoded.shape}")

# Add embeddings as features (DISABLED - causes overfitting with small dataset)
# print("\n[2-1] Adding embeddings as features...")
# embeddings_df = pd.DataFrame(
#     embeddings,
#     columns=[f'emb_{i}' for i in range(len(embeddings[0]))]
# )
# X_encoded = pd.concat([X_encoded.reset_index(drop=True), embeddings_df], axis=1)
# print(f"Shape with embeddings: {X_encoded.shape}")

print(f"\n[2-1] NOT using embeddings (small dataset - would cause overfitting)")
print(f"Final shape: {X_encoded.shape}")

# Train/test split
print("\n[3] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# Scale features (DISABLED - tree-based models don't need it and it hurts performance)
# print("\n[3-1] Scaling features...")
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# print(f"Scaled features")

scaler = None  # No scaling

# Baseline (mean predictor)
print("\n[4] Baseline (mean predictor)...")
y_pred_mean = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, y_pred_mean)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
baseline_r2 = r2_score(y_test, y_pred_mean)
print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R²:   {baseline_r2:.4f}")

# Models to test
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'Ridge (alpha=0.5)': Ridge(alpha=0.5, random_state=RANDOM_STATE),
    'Ridge (alpha=2.0)': Ridge(alpha=2.0, random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Random Forest (deep)': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    ),
    'LightGBM (tuned)': lgb.LGBMRegressor(
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
        verbose=-1
    )
}

# Add XGBoost if available
if HAS_XGB:
    models['XGBoost'] = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    models['XGBoost (tuned)'] = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# Train and evaluate
print("\n[5] Training models...")
print("="*70)

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    }

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

# Summary
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
print("-"*70)
print(f"{'Baseline (Mean)':<20} {baseline_mae:<12.4f} {baseline_rmse:<12.4f} {baseline_r2:<12.4f}")
for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['MAE']:<12.4f} {metrics['RMSE']:<12.4f} {metrics['R²']:<12.4f}")

# Best model
best_model_name = max(results.keys(), key=lambda k: results[k]['R²'])
print(f"\n✓ Best model: {best_model_name}")
print(f"  R² = {results[best_model_name]['R²']:.4f}")
print(f"  MAE = {results[best_model_name]['MAE']:.4f} seconds")

# Try ensemble
print("\n" + "="*70)
print("ENSEMBLE MODEL")
print("="*70)
top3_models = sorted(results.items(), key=lambda x: x[1]['R²'], reverse=True)[:3]
print(f"Ensembling top 3 models:")
for name, metrics in top3_models:
    print(f"  - {name}: R²={metrics['R²']:.4f}")

# Ensemble prediction (average)
ensemble_preds = np.zeros(len(y_test))
for name, _ in top3_models:
    ensemble_preds += models[name].predict(X_test)
ensemble_preds /= 3

ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
ensemble_r2 = r2_score(y_test, ensemble_preds)

print(f"\nEnsemble Results:")
print(f"  MAE:  {ensemble_mae:.4f}")
print(f"  RMSE: {ensemble_rmse:.4f}")
print(f"  R²:   {ensemble_r2:.4f}")

if ensemble_r2 > results[best_model_name]['R²']:
    print(f"\n✓ Ensemble IMPROVED over best single model!")
    print(f"  R² improvement: {ensemble_r2 - results[best_model_name]['R²']:.4f}")
    best_model_name = 'Ensemble'
    results['Ensemble'] = {
        'MAE': ensemble_mae,
        'RMSE': ensemble_rmse,
        'R²': ensemble_r2
    }
else:
    print(f"\n✗ Ensemble did not improve (using best single model)")

# Improvement over baseline
print(f"\nImprovement over baseline:")
print(f"  R² improvement: {results[best_model_name]['R²'] - baseline_r2:.4f}")
print(f"  MAE reduction: {(1 - results[best_model_name]['MAE']/baseline_mae)*100:.1f}%")

# Save best model
best_model = models[best_model_name]

# Get feature names before scaling
feature_names_list = X_encoded.columns.tolist()

with open('best_improved_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names_list,
        'label_encoders': label_encoders,
        'metrics': results[best_model_name],
        'uses_embeddings': False,
        'embedding_dim': 0
    }, f)
print(f"\n✓ Best model saved to 'best_improved_model.pkl'")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)