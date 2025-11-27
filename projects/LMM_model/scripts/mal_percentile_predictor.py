"""
MAL Percentile Predictor

This module provides a prediction API for estimating Maximum Acceptable Latency (MAL)
at different percentiles given a query and optionally a participant ID.

Input:
    - query: string (user query text)
    - percentile: float (e.g., 50, 90, 95)
    - participant_id: string (optional, for personalized prediction)

Output:
    - predicted_mal: float (MAL in seconds at given percentile)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')


class MALPercentilePredictor:
    """
    Predicts MAL at specified percentiles using Linear Mixed Model

    Handles both:
    - Warm-start: Known participant (personalized prediction)
    - Cold-start: Unknown participant (population-level prediction)
    """

    def __init__(self, model_path=None):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model (if None, will train on final_dataset.csv)
        """
        self.model = None
        self.feature_cols = None
        self.participant_random_effects = {}
        self.population_mean_random_effect = 0.0
        self.residual_std = None
        self.feature_extractors = {}

        if model_path:
            self.load_model(model_path)
        else:
            self.train()

    def train(self):
        """Train the LMM model on final_dataset.csv"""
        print("Training MAL Percentile Predictor...")

        # Load data
        data = pd.read_csv('final_dataset.csv')

        # Get feature columns
        self.feature_cols = [col for col in data.columns if col.startswith('QL_')]

        # Convert categorical to numeric
        for col in self.feature_cols:
            if data[col].dtype == 'object':
                data[col] = pd.Categorical(data[col]).codes

        # Remove constant and highly correlated features
        X = data[self.feature_cols].copy()
        constant_features = [col for col in self.feature_cols if X[col].nunique() == 1]
        self.feature_cols = [col for col in self.feature_cols if col not in constant_features]
        X = X[self.feature_cols]

        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        self.feature_cols = [col for col in self.feature_cols if col not in high_corr_features]

        print(f"  Using {len(self.feature_cols)} features")

        # Fit model
        print("  Fitting Linear Mixed Model...")
        model_formula = f"log_MAL ~ {' + '.join(self.feature_cols)}"
        model_fit = MixedLM.from_formula(model_formula, groups="participant_id", data=data)
        self.model = model_fit.fit(method='powell', reml=True, maxiter=1000)

        # Extract participant random effects
        for participant_id, re_values in self.model.random_effects.items():
            self.participant_random_effects[participant_id] = re_values.iloc[0]

        # Calculate population mean random effect (for cold-start)
        self.population_mean_random_effect = np.mean(list(self.participant_random_effects.values()))

        # Get residual standard deviation
        self.residual_std = np.sqrt(self.model.scale)

        print(f"  ✓ Model trained successfully")
        print(f"  ✓ {len(self.participant_random_effects)} participant random effects extracted")
        print(f"  ✓ Residual std: {self.residual_std:.4f}")

    def extract_features(self, query):
        """
        Extract features from query text using LLM-based feature extractor

        Args:
            query: Query text string

        Returns:
            dict: Feature name -> value mapping
        """
        try:
            from feature_extractor import QueryFeatureExtractor

            # Initialize extractor (will use ANTHROPIC_API_KEY from environment)
            if not hasattr(self, '_feature_extractor'):
                print("  Initializing LLM feature extractor...")
                self._feature_extractor = QueryFeatureExtractor()

            print(f"  Extracting features for: '{query}'...")
            features = self._feature_extractor.extract_features(query)
            print(f"  ✓ Extracted {len(features)} features")

            return features

        except ImportError:
            print(f"  ⚠️  Warning: feature_extractor module not found")
            print(f"     Install required: pip install anthropic")
            return None
        except Exception as e:
            print(f"  ❌ Feature extraction failed: {e}")
            print(f"     Falling back to manual feature input required")
            return None

    def predict_log_mal(self, features_dict, participant_id=None):
        """
        Predict log(MAL) for given features and participant

        Args:
            features_dict: Dictionary of feature values
            participant_id: Optional participant ID for personalized prediction

        Returns:
            tuple: (mean_log_mal, std_log_mal)
        """
        # Get fixed effects prediction
        fixed_pred = float(self.model.params['Intercept'])

        for feature, value in features_dict.items():
            if feature in self.model.params.index:
                # Ensure value is numeric
                try:
                    numeric_value = float(value)
                    fixed_pred += float(self.model.params[feature]) * numeric_value
                except (ValueError, TypeError):
                    # Skip non-numeric features
                    continue

        # Add random effect
        if participant_id and participant_id in self.participant_random_effects:
            # Warm-start: Use known participant's random effect
            random_effect = self.participant_random_effects[participant_id]
            scenario = "personalized"
        else:
            # Cold-start: Use population mean (conservative approach)
            random_effect = self.population_mean_random_effect
            scenario = "population-level"

        mean_log_mal = fixed_pred + random_effect
        std_log_mal = self.residual_std

        return mean_log_mal, std_log_mal, scenario

    def predict_mal_percentile(self, query=None, features=None, percentile=50, participant_id=None):
        """
        Predict MAL at specified percentile

        Args:
            query: Query text (if features not provided)
            features: Dict of feature values (if query not provided)
            percentile: Desired percentile (0-100)
            participant_id: Optional participant ID

        Returns:
            dict: {
                'mal_seconds': predicted MAL in seconds,
                'log_mal': log-scale prediction,
                'percentile': requested percentile,
                'scenario': 'personalized' or 'population-level',
                'confidence_interval': (lower, upper) 95% CI
            }
        """
        if features is None:
            if query is None:
                raise ValueError("Must provide either query or features")
            features = self.extract_features(query)
            if features is None:
                raise ValueError("Feature extraction failed. Please provide features directly.")

        # Get log(MAL) prediction
        mean_log_mal, std_log_mal, scenario = self.predict_log_mal(features, participant_id)

        # Calculate percentile
        z_score = stats.norm.ppf(percentile / 100)
        log_mal_percentile = mean_log_mal + z_score * std_log_mal

        # Transform back to original scale
        mal_seconds = np.exp(log_mal_percentile)

        # Calculate 95% confidence interval for the mean
        ci_lower = np.exp(mean_log_mal - 1.96 * std_log_mal)
        ci_upper = np.exp(mean_log_mal + 1.96 * std_log_mal)

        return {
            'mal_seconds': mal_seconds,
            'log_mal': log_mal_percentile,
            'percentile': percentile,
            'scenario': scenario,
            'mean_mal': np.exp(mean_log_mal),
            'confidence_interval_95': (ci_lower, ci_upper),
            'participant_id': participant_id if participant_id else 'unknown'
        }

    def predict_multiple_percentiles(self, query=None, features=None,
                                     percentiles=[10, 50, 90, 95], participant_id=None):
        """
        Predict MAL at multiple percentiles

        Args:
            query: Query text (if features not provided)
            features: Dict of feature values
            percentiles: List of percentiles to compute
            participant_id: Optional participant ID

        Returns:
            dict: Percentile -> MAL mapping with metadata
        """
        results = {}

        for p in percentiles:
            results[f'p{p}'] = self.predict_mal_percentile(
                query=query,
                features=features,
                percentile=p,
                participant_id=participant_id
            )

        return results


def demonstrate_predictor():
    """Demonstrate the predictor with example queries"""
    print("=" * 80)
    print("MAL PERCENTILE PREDICTOR DEMONSTRATION")
    print("=" * 80)

    # Initialize predictor
    predictor = MALPercentilePredictor()

    print("\n" + "=" * 80)
    print("DEMONSTRATION: Using actual query features from dataset")
    print("=" * 80)

    # Load actual data to get example features
    data = pd.read_csv('final_dataset.csv')

    # Example 1: Known participant (warm-start)
    print("\n" + "-" * 80)
    print("Example 1: KNOWN PARTICIPANT (Personalized Prediction)")
    print("-" * 80)

    example_row = data.iloc[0]
    participant_id = example_row['participant_id']
    features = {col: example_row[col] for col in predictor.feature_cols}

    print(f"\nParticipant: {participant_id}")
    print(f"Actual MAL: {example_row['MAL']:.2f} seconds")

    results = predictor.predict_multiple_percentiles(
        features=features,
        percentiles=[10, 50, 90, 95],
        participant_id=participant_id
    )

    print(f"\nPredicted MAL percentiles:")
    for percentile_key, result in results.items():
        print(f"  {percentile_key:4s}: {result['mal_seconds']:6.2f} seconds ({result['scenario']})")

    print(f"\nInterpretation:")
    print(f"  • 50% of the time, this user will accept latency up to {results['p50']['mal_seconds']:.1f}s")
    print(f"  • 90% of the time, this user will accept latency up to {results['p90']['mal_seconds']:.1f}s")
    print(f"  • To accommodate 95% of cases, plan for {results['p95']['mal_seconds']:.1f}s")

    # Example 2: Unknown participant (cold-start)
    print("\n" + "-" * 80)
    print("Example 2: UNKNOWN PARTICIPANT (Population-level Prediction)")
    print("-" * 80)

    example_row = data.iloc[100]
    features = {col: example_row[col] for col in predictor.feature_cols}

    print(f"\nNew user (no history)")
    print(f"Query features extracted")

    results_cold = predictor.predict_multiple_percentiles(
        features=features,
        percentiles=[10, 50, 90, 95],
        participant_id=None  # Cold-start
    )

    print(f"\nPredicted MAL percentiles (population average):")
    for percentile_key, result in results_cold.items():
        print(f"  {percentile_key:4s}: {result['mal_seconds']:6.2f} seconds ({result['scenario']})")

    print(f"\nInterpretation:")
    print(f"  • For a typical user, median MAL is {results_cold['p50']['mal_seconds']:.1f}s")
    print(f"  • To accommodate 90% of typical users, plan for {results_cold['p90']['mal_seconds']:.1f}s")
    print(f"  • Note: Once we have 5-10 observations for this user,")
    print(f"    we can switch to personalized predictions")

    # Example 3: Comparison across participants
    print("\n" + "-" * 80)
    print("Example 3: SAME QUERY, DIFFERENT PARTICIPANTS")
    print("-" * 80)

    # Use same features for 5 different participants
    same_query_features = {col: data.iloc[0][col] for col in predictor.feature_cols}

    print(f"\nSame query evaluated by 5 different participants:")
    print(f"{'Participant':<15} {'P50 (median)':<15} {'P90':<15} {'P95':<15}")
    print("-" * 60)

    for i in range(5):
        participant = data.iloc[i * 10]['participant_id']
        result = predictor.predict_multiple_percentiles(
            features=same_query_features,
            percentiles=[50, 90, 95],
            participant_id=participant
        )
        print(f"{participant:<15} {result['p50']['mal_seconds']:>8.2f}s      "
              f"{result['p90']['mal_seconds']:>8.2f}s      {result['p95']['mal_seconds']:>8.2f}s")

    print(f"\nInterpretation:")
    print(f"  • Same query, different patience levels!")
    print(f"  • Personalization is crucial for accurate predictions")

    # Save predictor for later use
    print("\n" + "=" * 80)
    print("Saving predictor for production use...")
    import pickle
    with open('mal_predictor.pkl', 'wb') as f:
        pickle.dump(predictor, f)
    print("  ✓ Saved: mal_predictor.pkl")

    # Save feature list for reference
    pd.DataFrame({'feature': predictor.feature_cols}).to_csv('required_features.csv', index=False)
    print("  ✓ Saved: required_features.csv")

    print("\n" + "=" * 80)
    print("USAGE EXAMPLE CODE")
    print("=" * 80)

    print("""
# Load predictor
import pickle
with open('mal_predictor.pkl', 'rb') as f:
    predictor = pickle.load(f)

# Scenario 1: Known user
features = extract_query_features(query_text)  # Your feature extraction
result = predictor.predict_mal_percentile(
    features=features,
    percentile=90,
    participant_id='P013'
)
print(f"90% accommodated MAL: {result['mal_seconds']:.1f} seconds")

# Scenario 2: New user (cold-start)
result = predictor.predict_mal_percentile(
    features=features,
    percentile=90,
    participant_id=None  # Unknown user
)
print(f"90% accommodated MAL (population): {result['mal_seconds']:.1f} seconds")

# Scenario 3: Multiple percentiles
results = predictor.predict_multiple_percentiles(
    features=features,
    percentiles=[50, 90, 95],
    participant_id='P013'
)
for p, r in results.items():
    print(f"{p}: {r['mal_seconds']:.1f}s")
    """)

    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_predictor()
