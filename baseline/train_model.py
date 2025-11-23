#!/usr/bin/env python3
"""
MAL Prediction Model Training Script
Trains machine learning models to predict MAL (latency) from query embeddings
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import os
import pickle
import time


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    """Main training pipeline"""

    # ========================================
    # 1. Load data
    # ========================================
    print_section("STEP 1: Loading Data")

    data = pd.read_csv('rawdata.csv')
    print(f"✅ Loaded {len(data)} queries from rawdata.csv")
    print(f"   Columns: {list(data.columns)}")
    print(f"\nFirst 5 rows:")
    print(data.head())


    # ========================================
    # 2. Setup OpenAI client
    # ========================================
    print_section("STEP 2: OpenAI Client Setup")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Using placeholder key (update if needed)")
        api_key = "YOUR_API_KEY_HERE"

    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")


    # ========================================
    # 3. Generate/Load embeddings
    # ========================================
    print_section("STEP 3: Generating/Loading Embeddings")

    EMBEDDING_FILE = 'embeddings.pkl'

    def get_embedding(text, model="text-embedding-3-small"):
        """Generate text embedding using OpenAI API"""
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    # Check if embedding file exists
    if os.path.exists(EMBEDDING_FILE):
        print(f"📁 Found existing embedding file: {EMBEDDING_FILE}")
        print("   Loading embeddings... (saves API costs!)")

        start_time = time.time()
        with open(EMBEDDING_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        elapsed = time.time() - start_time

        print(f"✅ Embeddings loaded successfully ({elapsed:.3f}s)")
        print(f"   Embedding dimension: {len(embeddings[0])}")

    else:
        print("⚠️  Embedding file not found. Generating new embeddings...")
        print("   (This will call OpenAI API)")

        embeddings = []
        start_time = time.time()

        for idx, query in enumerate(data['queries']):
            if idx % 50 == 0:
                print(f"   Progress: {idx}/{len(data)} queries processed...")
            emb = get_embedding(query)
            embeddings.append(emb)

        elapsed = time.time() - start_time
        print(f"✅ Embeddings generated successfully ({elapsed:.3f}s)")
        print(f"   Embedding dimension: {len(embeddings[0])}")

        # Save embeddings
        with open(EMBEDDING_FILE, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"💾 Embeddings saved to '{EMBEDDING_FILE}'")
        print("   Next time, this file will be loaded instead of calling API!")

    # Convert to DataFrame
    embedding_df = pd.DataFrame(embeddings)
    print(f"\n📊 Embedding DataFrame shape: {embedding_df.shape}")


    # ========================================
    # 4. Prepare data and split
    # ========================================
    print_section("STEP 4: Data Preparation and Splitting")

    # Features (X) and target (y)
    X = embedding_df.values  # Embedding vectors as features
    y = data['MAL'].values   # MAL as target

    # Train/test split (80:20)
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
    print_section("STEP 5: Training Multiple ML Models")

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
    # 6. Compare model performance
    # ========================================
    print_section("STEP 6: Model Performance Comparison")

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
    # 7. Analyze Linear Regression formula
    # ========================================
    print_section("STEP 7: Linear Regression Model Analysis")

    linear_model = results['Linear Regression']['model']
    coefficients = linear_model.coef_
    intercept = linear_model.intercept_

    print(f"\n📐 Linear Formula:")
    print(f"   MAL = {intercept:.4f} + (w1 × emb1) + (w2 × emb2) + ... + (w{len(coefficients)} × emb{len(coefficients)})")

    print(f"\n📈 Coefficient Statistics:")
    print(f"   - Count: {len(coefficients)} (embedding dimensions)")
    print(f"   - Min: {coefficients.min():.6f}")
    print(f"   - Max: {coefficients.max():.6f}")
    print(f"   - Mean: {coefficients.mean():.6f}")
    print(f"   - Std: {coefficients.std():.6f}")

    # Top influential coefficients
    abs_coef = np.abs(coefficients)
    top_indices = np.argsort(abs_coef)[-10:][::-1]

    print(f"\n🔝 Top 10 Most Influential Coefficients:")
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i:2d}. Embedding dim [{idx:4d}]: {coefficients[idx]:10.6f}")


    # ========================================
    # 8. Analyze Random Forest
    # ========================================
    print_section("STEP 8: Random Forest Model Analysis")

    rf_model = results['Random Forest']['model']
    feature_importance = rf_model.feature_importances_

    print(f"\n🌲 Random Forest is an ensemble of decision trees")
    print(f"   Total trees: {rf_model.n_estimators}")

    print(f"\n📊 Feature Importance Statistics:")
    print(f"   - Min: {feature_importance.min():.6f}")
    print(f"   - Max: {feature_importance.max():.6f}")
    print(f"   - Mean: {feature_importance.mean():.6f}")
    print(f"   - Std: {feature_importance.std():.6f}")

    # Top important features
    top_feature_indices = np.argsort(feature_importance)[-10:][::-1]

    print(f"\n🔝 Top 10 Most Important Embedding Dimensions:")
    for i, idx in enumerate(top_feature_indices, 1):
        print(f"   {i:2d}. Embedding dim [{idx:4d}]: {feature_importance[idx]:.6f}")

    print(f"\n⚠️  Note: Random Forest is a non-linear model")
    print(f"   Cannot be expressed as a simple linear formula")
    print(f"   Uses {rf_model.n_estimators} decision trees to learn complex patterns")


    # ========================================
    # 9. Save best model and results
    # ========================================
    print_section("STEP 9: Saving Best Model and Results")

    MODEL_FILE = 'best_model.pkl'
    best_model = results[best_model_name]['model']

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"✅ Best model saved successfully!")
    print(f"   Model: {best_model_name}")
    print(f"   File: {MODEL_FILE}")
    print(f"   Test R²: {comparison_df.iloc[0]['Test R²']:.4f}")

    # Save baseline results to JSON
    import json
    from datetime import datetime

    baseline_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'baseline_embedding_only',
        'dataset': {
            'total_samples': len(data),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': 'embedding_only',
            'embedding_dimension': len(embeddings[0]),
            'mal_range': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            }
        },
        'models': {}
    }

    # Add all model results
    for model_name in results.keys():
        baseline_results['models'][model_name] = {
            'train_mse': float(results[model_name]['train_mse']),
            'test_mse': float(results[model_name]['test_mse']),
            'train_mae': float(results[model_name]['train_mae']),
            'test_mae': float(results[model_name]['test_mae']),
            'train_r2': float(results[model_name]['train_r2']),
            'test_r2': float(results[model_name]['test_r2'])
        }

    # Mark best model
    baseline_results['best_model'] = {
        'name': best_model_name,
        'test_mse': float(comparison_df.iloc[0]['Test MSE']),
        'test_mae': float(comparison_df.iloc[0]['Test MAE']),
        'test_r2': float(comparison_df.iloc[0]['Test R²'])
    }

    RESULTS_FILE = 'baseline_results.json'
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Baseline results saved to: {RESULTS_FILE}")
    print(f"   This will be used to compare with augmented features!")


    # ========================================
    # Summary
    # ========================================
    print_section("🎉 TRAINING COMPLETE")

    print(f"\n📝 Summary:")
    print(f"   - Dataset: {len(data)} queries")
    print(f"   - Embedding dimension: {len(embeddings[0])}")
    print(f"   - Models trained: {len(models)}")
    print(f"   - Best model: {best_model_name}")
    print(f"   - Test R²: {comparison_df.iloc[0]['Test R²']:.4f}")
    print(f"   - Model saved to: {MODEL_FILE}")
    print(f"\n✨ You can now use 'mal_prediction_model.py' to make predictions!")
    print("=" * 80)


if __name__ == '__main__':
    main()
