"""
Feature Selection Analysis for Model 1

This script analyzes all features used in Model 1 and performs:
1. Statistical significance testing (p-values)
2. Multicollinearity analysis (VIF)
3. Feature importance ranking
4. Comparison: Feature Selection vs PCA

Goal: Select optimal subset of features for improved model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE SELECTION ANALYSIS FOR MODEL 1")
print("=" * 80)

# Load data
print("\n1. Loading data...")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, '../data/final_dataset.csv'))
print(f"   Data shape: {data.shape}")

# Get all features (all except ID columns and target variable)
exclude_cols = ['participant_id', 'query_id', 'MAL', 'log_MAL']
all_features = [col for col in data.columns if col not in exclude_cols]
print(f"   Total features: {len(all_features)}")

# Convert categorical to numeric
for col in all_features:
    if data[col].dtype == 'object':
        data[col] = pd.Categorical(data[col]).codes

# Remove constant features (as in original model)
print("\n2. Removing constant features...")
X = data[all_features].copy()
constant_features = [col for col in all_features if X[col].nunique() == 1]
if constant_features:
    print(f"   Found {len(constant_features)} constant features:")
    for feat in constant_features:
        print(f"     - {feat}")
    all_features = [col for col in all_features if col not in constant_features]
    X = X[all_features]

# Remove highly correlated features (r > 0.95)
print("\n3. Removing highly correlated features (r > 0.95)...")
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
if high_corr_features:
    print(f"   Found {len(high_corr_features)} highly correlated features:")
    for feat in high_corr_features:
        correlated_with = upper_tri[feat][upper_tri[feat] > 0.95].index.tolist()
        print(f"     - {feat} (correlated with: {correlated_with})")
    all_features = [col for col in all_features if col not in high_corr_features]

print(f"   Remaining features after cleanup: {len(all_features)}")

# Load existing model coefficients
print("\n4. Loading existing Model 1 coefficients...")
coef_df = pd.read_csv(os.path.join(script_dir, '../outputs/lmm_model1_coefficients.csv'))
coef_df = coef_df[coef_df['feature'] != 'participant_id Var']
coef_df = coef_df[coef_df['feature'] != 'Intercept']
print(f"   Loaded {len(coef_df)} feature coefficients")

# Statistical significance analysis
print("\n5. Statistical Significance Analysis")
print("=" * 80)

sig_levels = [0.001, 0.01, 0.05, 0.10]
for level in sig_levels:
    n_sig = len(coef_df[coef_df['p_value'] < level])
    pct = n_sig / len(coef_df) * 100
    print(f"   p < {level:5.3f}: {n_sig:2d} features ({pct:5.1f}%)")

# Select significant features (p < 0.05)
significant_features = coef_df[coef_df['p_value'] < 0.05].copy()
significant_features = significant_features.sort_values('p_value')

print(f"\n6. Significant Features (p < 0.05): {len(significant_features)}")
print("=" * 80)
print(significant_features[['feature', 'estimate', 'p_value']].to_string(index=False))

# Save significant features
significant_feature_names = significant_features['feature'].tolist()

# Multicollinearity analysis (VIF)
print("\n7. Multicollinearity Analysis (VIF)")
print("=" * 80)
print("   Computing VIF for significant features...")

X_sig = data[significant_feature_names].copy()

# Calculate VIF
vif_data = pd.DataFrame()
vif_data['feature'] = significant_feature_names
vif_data['VIF'] = [
    variance_inflation_factor(X_sig.values, i)
    for i in range(len(significant_feature_names))
]
vif_data = vif_data.sort_values('VIF', ascending=False)

print(f"\n   VIF Analysis Results:")
print(vif_data.to_string(index=False))

high_vif = vif_data[vif_data['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n   ⚠️  {len(high_vif)} features with VIF > 10 (potential multicollinearity):")
    print(high_vif.to_string(index=False))
else:
    print(f"\n   ✓ No severe multicollinearity (all VIF < 10)")

# PCA Analysis
print("\n8. PCA Analysis")
print("=" * 80)

# Standardize features
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(data[all_features])

# Fit PCA
pca = PCA()
pca.fit(X_all_scaled)

# Cumulative explained variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)

# Find number of components for 90%, 95% variance
n_comp_90 = np.argmax(cumsum_var >= 0.90) + 1
n_comp_95 = np.argmax(cumsum_var >= 0.95) + 1

print(f"   Total features: {len(all_features)}")
print(f"   Components for 90% variance: {n_comp_90}")
print(f"   Components for 95% variance: {n_comp_95}")
print(f"\n   Explained variance by first 10 components:")
for i in range(min(10, len(pca.explained_variance_ratio_))):
    print(f"     PC{i+1}: {pca.explained_variance_ratio_[i]*100:5.2f}% "
          f"(cumulative: {cumsum_var[i]*100:5.2f}%)")

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, min(21, len(all_features)+1)),
            pca.explained_variance_ratio_[:20],
            alpha=0.7, color='steelblue')
axes[0].plot(range(1, min(21, len(all_features)+1)),
             pca.explained_variance_ratio_[:20],
             'ro-', linewidth=2)
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].set_title('Scree Plot (First 20 Components)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(all_features)+1), cumsum_var, 'b-', linewidth=2)
axes[1].axhline(y=0.90, color='r', linestyle='--', label='90% variance')
axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% variance')
axes[1].axvline(x=n_comp_90, color='r', linestyle=':', alpha=0.5)
axes[1].axvline(x=n_comp_95, color='g', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '../outputs/pca_analysis.png'), dpi=300, bbox_inches='tight')
print(f"\n   ✓ Saved: ../outputs/pca_analysis.png")

# Feature importance visualization
print("\n9. Feature Importance Visualization")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Top significant features
top_n = min(15, len(significant_features))
top_features = significant_features.head(top_n).copy()

# Bar plot of coefficients
axes[0].barh(range(len(top_features)), top_features['estimate'].values,
             color=['green' if x > 0 else 'red' for x in top_features['estimate'].values],
             alpha=0.7)
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels(top_features['feature'].values)
axes[0].set_xlabel('Coefficient Estimate', fontsize=12)
axes[0].set_title(f'Top {top_n} Most Significant Features (p < 0.05)',
                  fontsize=14, fontweight='bold')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3, axis='x')

# Add p-values as text
for i, (idx, row) in enumerate(top_features.iterrows()):
    axes[0].text(row['estimate'], i,
                f" p={row['p_value']:.4f}",
                va='center', fontsize=8)

# P-value distribution
axes[1].hist(coef_df['p_value'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[1].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05')
axes[1].axvline(x=0.01, color='orange', linestyle='--', linewidth=2, label='p = 0.01')
axes[1].set_xlabel('P-value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of P-values for All Features', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '../outputs/feature_importance.png'), dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: ../outputs/feature_importance.png")

# Comparison Summary
print("\n10. COMPARISON SUMMARY")
print("=" * 80)

comparison = pd.DataFrame({
    'Method': [
        'All Features (Original)',
        'Feature Selection (p < 0.05)',
        'Feature Selection (p < 0.01)',
        'PCA (90% variance)',
        'PCA (95% variance)'
    ],
    'N_Features': [
        len(all_features),
        len(significant_features[significant_features['p_value'] < 0.05]),
        len(significant_features[significant_features['p_value'] < 0.01]),
        n_comp_90,
        n_comp_95
    ],
    'Reduction': [
        '0%',
        f'{(1 - len(significant_features[significant_features["p_value"] < 0.05])/len(all_features))*100:.1f}%',
        f'{(1 - len(significant_features[significant_features["p_value"] < 0.01])/len(all_features))*100:.1f}%',
        f'{(1 - n_comp_90/len(all_features))*100:.1f}%',
        f'{(1 - n_comp_95/len(all_features))*100:.1f}%'
    ]
})

print(comparison.to_string(index=False))

# Recommendations
print("\n11. RECOMMENDATIONS")
print("=" * 80)

selected_p005 = significant_features[significant_features['p_value'] < 0.05]['feature'].tolist()
selected_p001 = significant_features[significant_features['p_value'] < 0.01]['feature'].tolist()

print(f"""
Recommendation for Feature Selection:

1. **Conservative (p < 0.01)**: {len(selected_p001)} features
   - Highest statistical confidence
   - Minimal multicollinearity risk
   - Good for interpretability

2. **Balanced (p < 0.05)**: {len(selected_p005)} features  ⭐ RECOMMENDED
   - Standard statistical threshold
   - Good balance of performance vs interpretability
   - Suitable for commercial use

3. **PCA ({n_comp_90} components for 90% variance)**:
   - Maximum dimensionality reduction
   - Orthogonal components (no multicollinearity)
   - Less interpretable (linear combinations)
   - Better for pure prediction

For commercial use with LLM feature extraction:
→ Use Feature Selection (p < 0.05) with {len(selected_p005)} features
  Benefits:
  - Each feature is interpretable
  - LLM can extract these directly from query text
  - Reduced API cost (fewer features to extract)
  - Improved extraction accuracy
""")

# Save selected features
print("\n12. Saving Selected Features")
print("=" * 80)

# Save p < 0.05 features
selected_features_df = significant_features[significant_features['p_value'] < 0.05].copy()
selected_features_df.to_csv(os.path.join(script_dir, '../outputs/selected_features_p005.csv'), index=False)
print(f"   ✓ Saved: ../outputs/selected_features_p005.csv ({len(selected_features_df)} features)")

# Save p < 0.01 features
selected_features_p001_df = significant_features[significant_features['p_value'] < 0.01].copy()
selected_features_p001_df.to_csv(os.path.join(script_dir, '../outputs/selected_features_p001.csv'), index=False)
print(f"   ✓ Saved: ../outputs/selected_features_p001.csv ({len(selected_features_p001_df)} features)")

# Save VIF analysis
vif_data.to_csv(os.path.join(script_dir, '../outputs/vif_analysis.csv'), index=False)
print(f"   ✓ Saved: ../outputs/vif_analysis.csv")

# Save PCA results
pca_summary = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained_Variance': pca.explained_variance_ratio_,
    'Cumulative_Variance': cumsum_var
})
pca_summary.to_csv(os.path.join(script_dir, '../outputs/pca_summary.csv'), index=False)
print(f"   ✓ Saved: ../outputs/pca_summary.csv")

print("\n" + "=" * 80)
print("✅ FEATURE SELECTION ANALYSIS COMPLETE!")
print("=" * 80)
print(f"""
Summary:
- Original features: {len(all_features)}
- Significant (p < 0.05): {len(selected_p005)}
- Significant (p < 0.01): {len(selected_p001)}
- PCA components (90% var): {n_comp_90}

Next steps:
1. Review selected features in outputs/selected_features_p005.csv
2. Retrain Model 1 with selected features
3. Update feature_specification.md
4. Create commercial predictor with simplified feature extraction
""")
