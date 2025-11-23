#!/usr/bin/env python3
"""
Analyze Feature Importance for Random Forest Model
Shows which features contribute most to MAL prediction
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    """Analyze feature importance"""

    # ========================================
    # 1. Load model
    # ========================================
    print_section("STEP 1: Loading Model")

    MODEL_FILE = 'best_model_augmented_v2_cleaned.pkl'

    try:
        with open(MODEL_FILE, 'rb') as f:
            model_package = pickle.load(f)

        model = model_package['model']
        feature_columns = model_package['feature_columns']
        embedding_dim = model_package['embedding_dimension']

        print(f"✅ Loaded model from {MODEL_FILE}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Total features: {len(feature_columns) + embedding_dim}")
        print(f"   - Embedding features: {embedding_dim}")
        print(f"   - Augmented features: {len(feature_columns)}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Check if it's a Random Forest model
    if not isinstance(model, RandomForestRegressor):
        print(f"❌ Model is {type(model).__name__}, not RandomForestRegressor")
        print("   Feature importance is only available for tree-based models")
        return

    # ========================================
    # 2. Extract feature importances
    # ========================================
    print_section("STEP 2: Extracting Feature Importances")

    # Get feature importances
    importances = model.feature_importances_

    # Create feature names (embeddings + augmented features)
    embedding_features = [f'emb_{i}' for i in range(embedding_dim)]
    all_features = embedding_features + feature_columns

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)

    print(f"✅ Extracted {len(importance_df)} feature importances")
    print(f"   Total importance sum: {importances.sum():.4f} (should be ~1.0)")

    # ========================================
    # 3. Analyze by feature type
    # ========================================
    print_section("STEP 3: Feature Type Analysis")

    # Separate embeddings vs augmented
    embedding_importance = importance_df[importance_df['feature'].str.startswith('emb_')]
    augmented_importance = importance_df[~importance_df['feature'].str.startswith('emb_')]

    embedding_total = embedding_importance['importance'].sum()
    augmented_total = augmented_importance['importance'].sum()

    print(f"\n📊 Importance by Feature Type:")
    print(f"   Embedding features:  {embedding_total:.4f} ({embedding_total*100:.1f}%)")
    print(f"   Augmented features:  {augmented_total:.4f} ({augmented_total*100:.1f}%)")

    # ========================================
    # 4. Top augmented features
    # ========================================
    print_section("STEP 4: Top Augmented Features")

    print(f"\n🏆 Top 30 Most Important Augmented Features:\n")
    print(f"{'Rank':>4}  {'Feature':<50}  {'Importance':>12}  {'Cumulative %':>13}")
    print("─" * 85)

    cumulative_importance = 0
    for idx, (_, row) in enumerate(augmented_importance.head(30).iterrows(), 1):
        cumulative_importance += row['importance']
        cumulative_pct = (cumulative_importance / augmented_total) * 100
        print(f"{idx:>4}  {row['feature']:<50}  {row['importance']:>12.6f}  {cumulative_pct:>12.1f}%")

    # ========================================
    # 5. Feature category analysis
    # ========================================
    print_section("STEP 5: Feature Category Analysis")

    # Group by category prefix
    category_importance = {}
    for _, row in augmented_importance.iterrows():
        feature = row['feature']
        importance = row['importance']

        # Extract prefix (category)
        prefix = feature.split('_')[0] if '_' in feature else 'OTHER'
        category_importance[prefix] = category_importance.get(prefix, 0) + importance

    # Sort by importance
    category_df = pd.DataFrame([
        {'category': cat, 'importance': imp}
        for cat, imp in category_importance.items()
    ]).sort_values('importance', ascending=False)

    print(f"\n📊 Importance by Feature Category:\n")
    print(f"{'Category':<15}  {'Importance':>12}  {'% of Augmented':>15}  {'% of Total':>12}")
    print("─" * 60)

    for _, row in category_df.iterrows():
        cat = row['category']
        imp = row['importance']
        pct_aug = (imp / augmented_total) * 100
        pct_total = (imp / 1.0) * 100
        print(f"{cat:<15}  {imp:>12.6f}  {pct_aug:>14.1f}%  {pct_total:>11.1f}%")

    # ========================================
    # 6. Top embedding dimensions
    # ========================================
    print_section("STEP 6: Top Embedding Dimensions")

    print(f"\n🧬 Top 20 Most Important Embedding Dimensions:\n")
    print(f"{'Rank':>4}  {'Dimension':<15}  {'Importance':>12}")
    print("─" * 40)

    for idx, (_, row) in enumerate(embedding_importance.head(20).iterrows(), 1):
        dim = row['feature'].replace('emb_', '')
        print(f"{idx:>4}  {dim:<15}  {row['importance']:>12.6f}")

    # ========================================
    # 7. Save results
    # ========================================
    print_section("STEP 7: Saving Results")

    # Save full importance list
    OUTPUT_FILE = 'feature_importance.csv'
    importance_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Full feature importance saved to: {OUTPUT_FILE}")

    # Save category summary
    CATEGORY_FILE = 'category_importance.csv'
    category_df.to_csv(CATEGORY_FILE, index=False)
    print(f"✅ Category importance saved to: {CATEGORY_FILE}")

    # ========================================
    # 8. Create visualization (optional)
    # ========================================
    print_section("STEP 8: Visualization")

    try:
        # Plot top 20 augmented features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Top augmented features
        top_augmented = augmented_importance.head(20)
        ax1.barh(range(len(top_augmented)), top_augmented['importance'])
        ax1.set_yticks(range(len(top_augmented)))
        ax1.set_yticklabels(top_augmented['feature'], fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance')
        ax1.set_title('Top 20 Augmented Features')
        ax1.grid(axis='x', alpha=0.3)

        # Category importance
        ax2.bar(range(len(category_df)), category_df['importance'])
        ax2.set_xticks(range(len(category_df)))
        ax2.set_xticklabels(category_df['category'], rotation=45, ha='right')
        ax2.set_ylabel('Importance')
        ax2.set_title('Feature Category Importance')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        PLOT_FILE = 'feature_importance.png'
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {PLOT_FILE}")

    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
        print("   (matplotlib may not be available in headless environment)")

    # ========================================
    # Summary
    # ========================================
    print_section("🎉 ANALYSIS COMPLETE")

    print(f"\n📝 Key Findings:")
    print(f"   - Total features: {len(all_features)}")
    print(f"   - Embedding contribution: {embedding_total*100:.1f}%")
    print(f"   - Augmented contribution: {augmented_total*100:.1f}%")
    print(f"   - Top augmented feature: {augmented_importance.iloc[0]['feature']}")
    print(f"     (importance: {augmented_importance.iloc[0]['importance']:.6f})")
    print(f"   - Top category: {category_df.iloc[0]['category']}")
    print(f"     (importance: {category_df.iloc[0]['importance']:.6f})")

    print(f"\n📁 Output Files:")
    print(f"   - {OUTPUT_FILE}: Full feature importance list")
    print(f"   - {CATEGORY_FILE}: Category-level importance")
    print(f"   - {PLOT_FILE}: Visualization (if available)")

    print("=" * 80)


if __name__ == '__main__':
    main()
