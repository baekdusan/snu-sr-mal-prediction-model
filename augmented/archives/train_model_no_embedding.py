#!/usr/bin/env python3
"""
MAL Prediction Model Training WITHOUT Embeddings
Trains models using ONLY LLM-extracted features (no embeddings)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import os
import pickle
import json
import time
from datetime import datetime


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def encode_categorical_features(augmented_df):
    """
    Encode categorical features using One-Hot Encoding

    Args:
        augmented_df: DataFrame with augmented features

    Returns:
        DataFrame with one-hot encoded features and list of new column names
    """
    df = augmented_df.copy()

    # Automatically detect categorical columns (object/string type)
    # Exclude query text and target variable (MAL)
    exclude_columns = ['queries', 'MAL', 'query', 'mal']
    categorical_columns = [col for col in df.columns
                          if df[col].dtype == 'object' and col not in exclude_columns]

    if not categorical_columns:
        print("   No categorical features found")
        return df, []

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns, dtype=int)

    # Get list of new one-hot encoded columns
    new_columns = [col for col in df_encoded.columns if col not in df.columns]

    print(f"   One-hot encoded {len(categorical_columns)} categorical features:")
    for col in categorical_columns:
        one_hot_cols = [c for c in new_columns if c.startswith(col + '_')]
        print(f"   - {col} → {len(one_hot_cols)} columns: {one_hot_cols}")

    return df_encoded, new_columns


def main():
    """Main training pipeline with augmented features (NO EMBEDDINGS)"""

    # ========================================
    # 1. Load augmented features
    # ========================================
    print_section("STEP 1: Loading Augmented Features (NO EMBEDDINGS)")

    AUGMENTED_FILE = 'augmented_data.csv'

    if not os.path.exists(AUGMENTED_FILE):
        print(f"❌ Error: {AUGMENTED_FILE} not found!")
        print(f"   Please run 'augment_features.py' first to generate augmented features")
        return

    augmented_df = pd.read_csv(AUGMENTED_FILE)
    print(f"✅ Loaded {len(augmented_df)} samples with augmented features")
    print(f"   Columns: {list(augmented_df.columns)}")
    print(f"\n⚠️  NOTE: This experiment uses ONLY LLM-extracted features (NO embeddings)")


    # ========================================
    # 2. Identify and encode categorical features
    # ========================================
    print_section("STEP 2: Identifying and Encoding Features")

    # Identify feature columns (all columns except 'queries' and 'MAL')
    exclude_columns = ['queries', 'MAL', 'query', 'mal']  # Handle both upper/lower case
    all_feature_columns = [col for col in augmented_df.columns
                          if col not in exclude_columns]

    print(f"\n📊 Identified {len(all_feature_columns)} feature columns from CSV:")
    for col in all_feature_columns:
        print(f"   - {col}")

    # One-hot encode categorical features
    augmented_df_encoded, one_hot_columns = encode_categorical_features(augmented_df)

    # Build final feature column list
    # Include: numeric features + boolean features + one-hot encoded features
    feature_columns = []

    for col in all_feature_columns:
        if col in augmented_df_encoded.columns:
            # Convert boolean to int if needed
            if augmented_df_encoded[col].dtype == 'bool':
                augmented_df_encoded[col] = augmented_df_encoded[col].astype(int)
            feature_columns.append(col)

    # Add one-hot encoded columns
    feature_columns.extend(one_hot_columns)

    # Extract augmented features
    augmented_features = augmented_df_encoded[feature_columns].copy()

    print(f"\n✅ Final feature set: {len(feature_columns)} features (NO EMBEDDINGS)")
    print(f"   Original features: {len(all_feature_columns)}")
    print(f"   One-hot encoded features: {len(one_hot_columns)}")
    print(f"   Total: {len(feature_columns)}")


    # ========================================
    # 3. Normalize features
    # ========================================
    print_section("STEP 3: Normalizing Features")

    # Normalize augmented features
    scaler = StandardScaler()
    augmented_features_scaled = pd.DataFrame(
        scaler.fit_transform(augmented_features),
        columns=feature_columns
    )

    print(f"✅ Features normalized")
    print(f"   Total features: {len(augmented_features_scaled.columns)}")
    print(f"   Shape: {augmented_features_scaled.shape}")


    # ========================================
    # 4. Prepare data and split
    # ========================================
    print_section("STEP 4: Data Preparation and Splitting")

    # Features (X) and target (y)
    X = augmented_features_scaled.values

    # Find MAL column (handle both 'MAL' and 'mal')
    mal_column = 'MAL' if 'MAL' in augmented_df_encoded.columns else 'mal'
    y = augmented_df_encoded[mal_column].values

    # Train/test split (80:20, same random_state as baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✅ Data split completed")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   MAL range: {y.min():.2f} ~ {y.max():.2f}")


    # ========================================
    # 5. Train multiple models
    # ========================================
    print_section("STEP 5: Training Models (NO EMBEDDINGS)")

    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }

    # Store results
    results = {}

    for model_name, model in models.items():
        print(f"\n🔄 [{model_name}] Training...")
        start_time = time.time()

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        elapsed = time.time() - start_time

        # Store results
        results[model_name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_pred_test,
            'model': model
        }

        # Print results
        print(f"   ✅ Training completed ({elapsed:.3f}s)")
        print(f"      Train MSE: {train_mse:8.4f} | Test MSE: {test_mse:8.4f}")
        print(f"      Train MAE: {train_mae:8.4f} | Test MAE: {test_mae:8.4f}")
        print(f"      Train R²:  {train_r2:8.4f} | Test R²:  {test_r2:8.4f}")


    # ========================================
    # 6. Compare with baseline and augmented
    # ========================================
    print_section("STEP 6: Comparison with Baseline and Augmented")

    # Load baseline results
    BASELINE_FILE = 'baseline_results.json'
    AUGMENTED_FILE_JSON = 'augmented_results.json'

    comparison_data = []

    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        has_baseline = True
    else:
        has_baseline = False

    if os.path.exists(AUGMENTED_FILE_JSON):
        with open(AUGMENTED_FILE_JSON, 'r', encoding='utf-8') as f:
            augmented_results = json.load(f)
        has_augmented = True
    else:
        has_augmented = False

    if has_baseline or has_augmented:
        print("\n📊 Performance Comparison: Baseline vs Augmented vs No-Embedding\n")
        print(f"{'Model':<20} {'Baseline R²':>12} {'Augmented R²':>13} {'No-Embed R²':>13} {'vs Baseline':>12} {'vs Augmented':>13}")
        print("-" * 95)

        for model_name in results.keys():
            no_embed_r2 = results[model_name]['test_r2']

            baseline_r2 = baseline_results['models'][model_name]['test_r2'] if has_baseline else None
            augmented_r2 = augmented_results['models'][model_name]['test_r2'] if has_augmented else None

            baseline_str = f"{baseline_r2:12.4f}" if baseline_r2 is not None else "N/A".rjust(12)
            augmented_str = f"{augmented_r2:13.4f}" if augmented_r2 is not None else "N/A".rjust(13)

            vs_baseline = ""
            if baseline_r2 is not None:
                diff = no_embed_r2 - baseline_r2
                vs_baseline = f"{diff:+.4f}"

            vs_augmented = ""
            if augmented_r2 is not None:
                diff = no_embed_r2 - augmented_r2
                vs_augmented = f"{diff:+.4f}"

            print(f"{model_name:<20} {baseline_str} {augmented_str} {no_embed_r2:13.4f} {vs_baseline:>12} {vs_augmented:>13}")

    else:
        print("⚠️  No baseline or augmented results found for comparison.")


    # ========================================
    # 7. Model performance summary
    # ========================================
    print_section("STEP 7: Model Performance Summary (NO EMBEDDINGS)")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test MSE': [results[m]['test_mse'] for m in results.keys()],
        'Test MAE': [results[m]['test_mae'] for m in results.keys()],
        'Test R²': [results[m]['test_r2'] for m in results.keys()]
    })

    # Sort by Test MSE (ascending)
    comparison_df = comparison_df.sort_values('Test MSE')

    print("\n📊 Model Performance (Test Set):")
    print(comparison_df.to_string(index=False))

    # Best model
    best_model_name = comparison_df.iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   - Test MSE: {comparison_df.iloc[0]['Test MSE']:.4f}")
    print(f"   - Test MAE: {comparison_df.iloc[0]['Test MAE']:.4f}")
    print(f"   - Test R²:  {comparison_df.iloc[0]['Test R²']:.4f}")


    # ========================================
    # 8. Save best model and results
    # ========================================
    print_section("STEP 8: Saving Best Model and Results")

    MODEL_FILE = 'best_model_no_embedding.pkl'
    best_model = results[best_model_name]['model']

    # Save model, scaler, and feature info together
    model_package = {
        'model': best_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'one_hot_columns': one_hot_columns,
        'embedding_dimension': 0  # No embeddings
    }

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"✅ Best model package saved!")
    print(f"   Model: {best_model_name}")
    print(f"   File: {MODEL_FILE}")
    print(f"   Includes: model + scaler + feature info (NO EMBEDDINGS)")

    # Save no-embedding results to JSON
    no_embed_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'no_embedding_features_only',
        'dataset': {
            'total_samples': len(augmented_df_encoded),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': 'llm_extracted_only_no_embeddings',
            'embedding_dimension': 0,
            'augmented_features': len(feature_columns),
            'one_hot_encoded_features': len(one_hot_columns),
            'total_features': len(feature_columns),
            'mal_range': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            }
        },
        'augmented_feature_list': feature_columns,
        'models': {}
    }

    # Add all model results
    for model_name in results.keys():
        no_embed_results['models'][model_name] = {
            'train_mse': float(results[model_name]['train_mse']),
            'test_mse': float(results[model_name]['test_mse']),
            'train_mae': float(results[model_name]['train_mae']),
            'test_mae': float(results[model_name]['test_mae']),
            'train_r2': float(results[model_name]['train_r2']),
            'test_r2': float(results[model_name]['test_r2'])
        }

    # Mark best model
    no_embed_results['best_model'] = {
        'name': best_model_name,
        'test_mse': float(comparison_df.iloc[0]['Test MSE']),
        'test_mae': float(comparison_df.iloc[0]['Test MAE']),
        'test_r2': float(comparison_df.iloc[0]['Test R²'])
    }

    RESULTS_FILE = 'no_embedding_results.json'
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(no_embed_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 No-embedding results saved to: {RESULTS_FILE}")


    # ========================================
    # Summary
    # ========================================
    print_section("🎉 NO-EMBEDDING TRAINING COMPLETE")

    print(f"\n📝 Summary:")
    print(f"   - Dataset: {len(augmented_df_encoded)} queries")
    print(f"   - Embedding features: 0 (NONE)")
    print(f"   - Augmented features: {len(feature_columns)} (including {len(one_hot_columns)} one-hot encoded)")
    print(f"   - Total features: {len(feature_columns)}")
    print(f"   - Best model: {best_model_name}")
    print(f"   - Test R²: {comparison_df.iloc[0]['Test R²']:.4f}")
    print(f"   - Model saved to: {MODEL_FILE}")

    if has_baseline:
        baseline_best_r2 = baseline_results['best_model']['test_r2']
        no_embed_best_r2 = comparison_df.iloc[0]['Test R²']
        improvement = no_embed_best_r2 - baseline_best_r2

        print(f"\n📈 Comparison with baseline:")
        print(f"   - Baseline R²: {baseline_best_r2:.4f}")
        print(f"   - No-Embedding R²: {no_embed_best_r2:.4f}")
        print(f"   - Difference: {improvement:+.4f} ({improvement/baseline_best_r2*100:+.1f}%)")

    if has_augmented:
        augmented_best_r2 = augmented_results['best_model']['test_r2']
        no_embed_best_r2 = comparison_df.iloc[0]['Test R²']
        difference = no_embed_best_r2 - augmented_best_r2

        print(f"\n📉 Comparison with augmented (with embeddings):")
        print(f"   - Augmented R²: {augmented_best_r2:.4f}")
        print(f"   - No-Embedding R²: {no_embed_best_r2:.4f}")
        print(f"   - Difference: {difference:+.4f} ({difference/augmented_best_r2*100:+.1f}%)")

    print("=" * 80)


if __name__ == '__main__':
    main()
