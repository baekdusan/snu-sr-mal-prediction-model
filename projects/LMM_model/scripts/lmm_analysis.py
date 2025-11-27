"""
Linear Mixed Model Analysis for MAL Prediction
Model: log(MAL_ij) = β0 + β'X_i + u_participant(j) + v_query(i) + ε_ij
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings('ignore', category=ConvergenceWarning)

print("=" * 80)
print("Linear Mixed Model Analysis for MAL Prediction")
print("=" * 80)

# 1. Load data
print("\n1. Loading data...")
data = pd.read_csv('final_dataset.csv')
print(f"   Data shape: {data.shape}")
print(f"   Participants: {data['participant_id'].nunique()}")
print(f"   Queries: {data['query_id'].nunique()}")

# 2. Prepare variables
print("\n2. Preparing variables...")
# Get feature columns
feature_cols = [col for col in data.columns if col.startswith('QL_')]
print(f"   Number of features: {len(feature_cols)}")

# Convert categorical features to numeric if needed
categorical_features = []
for col in feature_cols:
    if data[col].dtype == 'object':
        categorical_features.append(col)
        # Use one-hot encoding or label encoding
        data[col] = pd.Categorical(data[col]).codes

if categorical_features:
    print(f"   Converted {len(categorical_features)} categorical features to numeric")

# Check for multicollinearity using VIF
print("\n   Checking for multicollinearity...")
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
X = data[feature_cols].copy()
# Remove constant features
constant_features = []
for col in feature_cols:
    if X[col].nunique() == 1:
        constant_features.append(col)

if constant_features:
    print(f"   Removing {len(constant_features)} constant features")
    feature_cols = [col for col in feature_cols if col not in constant_features]
    X = X[feature_cols]

# Check for perfect correlation
print("   Checking for perfect correlations...")
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

if high_corr_features:
    print(f"   Removing {len(high_corr_features)} highly correlated features (r > 0.95)")
    feature_cols = [col for col in feature_cols if col not in high_corr_features]

print(f"   Final number of features: {len(feature_cols)}")

# 3. Model specification
print("\n3. Building Linear Mixed Model...")
print("   Model: log_MAL ~ features + (1|participant_id) + (1|query_id)")
print("   Fixed effects: β0 + β'X (intercept + all features)")
print("   Random effects: u_participant + v_query")

# Prepare data for modeling
# Note: statsmodels MixedLM supports only one grouping variable at a time
# We'll fit two models: one with participant RE, one with query RE
# Then we'll use a nested approach or pymer4 for crossed random effects

# First, let's try with participant as random effect
print("\n4. Fitting Model 1: Random intercept for participant...")
print("   (This may take a few minutes...)")

# Model 1: Participant random effect
model1 = MixedLM.from_formula(
    f"log_MAL ~ {' + '.join(feature_cols)}",
    groups="participant_id",
    data=data
)

result1 = model1.fit(method='powell', reml=True, maxiter=1000)

print("\n   ✓ Model 1 fitted successfully")
print(f"\n   Log-Likelihood: {result1.llf:.2f}")
print(f"   AIC: {result1.aic:.2f}")
print(f"   BIC: {result1.bic:.2f}")

# Model 2: Query random effect
print("\n5. Fitting Model 2: Random intercept for query...")

model2 = MixedLM.from_formula(
    f"log_MAL ~ {' + '.join(feature_cols)}",
    groups="query_id",
    data=data
)

result2 = model2.fit(method='powell', reml=True, maxiter=1000)

print("\n   ✓ Model 2 fitted successfully")
print(f"\n   Log-Likelihood: {result2.llf:.2f}")
print(f"   AIC: {result2.aic:.2f}")
print(f"   BIC: {result2.bic:.2f}")

# 6. Summary statistics
print("\n" + "=" * 80)
print("6. MODEL 1 SUMMARY (Random: Participant)")
print("=" * 80)
print(result1.summary())

print("\n" + "=" * 80)
print("7. MODEL 2 SUMMARY (Random: Query)")
print("=" * 80)
print(result2.summary())

# 7. Random effects variance components
print("\n" + "=" * 80)
print("8. VARIANCE COMPONENTS")
print("=" * 80)

print("\nModel 1 (Participant Random Effect):")
var_participant = result1.cov_re.iloc[0, 0]
var_residual1 = result1.scale
print(f"  Participant variance (u): {var_participant:.4f}")
print(f"  Residual variance (ε):    {var_residual1:.4f}")
print(f"  Total variance:           {var_participant + var_residual1:.4f}")
print(f"  ICC (Participant):        {var_participant / (var_participant + var_residual1):.4f}")

print("\nModel 2 (Query Random Effect):")
var_query = result2.cov_re.iloc[0, 0]
var_residual2 = result2.scale
print(f"  Query variance (v):       {var_query:.4f}")
print(f"  Residual variance (ε):    {var_residual2:.4f}")
print(f"  Total variance:           {var_query + var_residual2:.4f}")
print(f"  ICC (Query):              {var_query / (var_query + var_residual2):.4f}")

# 8. Fixed effects significance
print("\n" + "=" * 80)
print("9. SIGNIFICANT FIXED EFFECTS (p < 0.05)")
print("=" * 80)

# Model 1 significant effects
coef_df1 = pd.DataFrame({
    'feature': result1.params.index,
    'estimate': result1.params.values,
    'std_error': result1.bse.values,
    'p_value': result1.pvalues.values
})
coef_df1 = coef_df1.sort_values('p_value')

sig_features1 = coef_df1[coef_df1['p_value'] < 0.05]
print(f"\nModel 1: {len(sig_features1)} significant features")
print(sig_features1.head(20).to_string(index=False))

# Save coefficients
coef_df1.to_csv('lmm_model1_coefficients.csv', index=False)
print("\n   ✓ Saved: lmm_model1_coefficients.csv")

coef_df2 = pd.DataFrame({
    'feature': result2.params.index,
    'estimate': result2.params.values,
    'std_error': result2.bse.values,
    'p_value': result2.pvalues.values
})
coef_df2 = coef_df2.sort_values('p_value')
coef_df2.to_csv('lmm_model2_coefficients.csv', index=False)
print("   ✓ Saved: lmm_model2_coefficients.csv")

# 9. Diagnostic plots
print("\n10. Creating diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Linear Mixed Model Diagnostics', fontsize=16, fontweight='bold')

# Model 1 diagnostics
# Residuals vs Fitted
fitted1 = result1.fittedvalues
residuals1 = result1.resid

axes[0, 0].scatter(fitted1, residuals1, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Model 1: Residuals vs Fitted')
axes[0, 0].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals1, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Model 1: Q-Q Plot')
axes[0, 1].grid(True, alpha=0.3)

# Histogram of residuals
axes[0, 2].hist(residuals1, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Residuals')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Model 1: Residual Distribution')
axes[0, 2].grid(True, alpha=0.3)

# Model 2 diagnostics
fitted2 = result2.fittedvalues
residuals2 = result2.resid

axes[1, 0].scatter(fitted2, residuals2, alpha=0.5, color='green')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Model 2: Residuals vs Fitted')
axes[1, 0].grid(True, alpha=0.3)

stats.probplot(residuals2, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_color('green')
axes[1, 1].get_lines()[1].set_color('red')
axes[1, 1].set_title('Model 2: Q-Q Plot')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(residuals2, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 2].set_xlabel('Residuals')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Model 2: Residual Distribution')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lmm_diagnostics.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: lmm_diagnostics.png")

# 10. Random effects distribution
print("\n11. Plotting random effects distributions...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Participant random effects
participant_re = result1.random_effects
participant_re_values = [re[0] for re in participant_re.values()]

axes[0].hist(participant_re_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Random effect value')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Participant Random Effects (u)\nSD = {np.std(participant_re_values):.3f}')
axes[0].grid(True, alpha=0.3)

# Query random effects
query_re = result2.random_effects
query_re_values = [re[0] for re in query_re.values()]

axes[1].hist(query_re_values, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
axes[1].set_xlabel('Random effect value')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Query Random Effects (v)\nSD = {np.std(query_re_values):.3f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('random_effects.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: random_effects.png")

# 11. Save models
print("\n12. Saving models...")
import pickle

with open('lmm_model1.pkl', 'wb') as f:
    pickle.dump(result1, f)
print("   ✓ Saved: lmm_model1.pkl")

with open('lmm_model2.pkl', 'wb') as f:
    pickle.dump(result2, f)
print("   ✓ Saved: lmm_model2.pkl")

# 12. Model comparison
print("\n" + "=" * 80)
print("13. MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': ['Model 1 (Participant RE)', 'Model 2 (Query RE)'],
    'Log-Likelihood': [result1.llf, result2.llf],
    'AIC': [result1.aic, result2.aic],
    'BIC': [result1.bic, result2.bic],
    'Random Variance': [var_participant, var_query],
    'Residual Variance': [var_residual1, var_residual2],
    'ICC': [var_participant / (var_participant + var_residual1),
            var_query / (var_query + var_residual2)]
})

print(comparison.to_string(index=False))

print("\n" + "=" * 80)
print("✅ LINEAR MIXED MODEL ANALYSIS COMPLETE!")
print("=" * 80)
print("\nFiles created:")
print("  - lmm_model1.pkl (participant random effect model)")
print("  - lmm_model2.pkl (query random effect model)")
print("  - lmm_model1_coefficients.csv (fixed effects for model 1)")
print("  - lmm_model2_coefficients.csv (fixed effects for model 2)")
print("  - lmm_diagnostics.png (diagnostic plots)")
print("  - random_effects.png (random effects distributions)")
print("\nNote: For crossed random effects (both participant AND query),")
print("      consider using pymer4 or R's lme4 package.")
print("\nNext steps:")
print("  - Use these models for percentile predictions")
print("  - For new queries: extract features → predict log(MAL) → calculate percentiles")
