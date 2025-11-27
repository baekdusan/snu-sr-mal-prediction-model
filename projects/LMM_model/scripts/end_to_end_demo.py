"""
End-to-End MAL Prediction Demo

Demonstrates the complete pipeline:
1. Query text → LLM feature extraction → MAL prediction

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python end_to_end_demo.py
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

from feature_extractor import QueryFeatureExtractor
from mal_percentile_predictor import MALPercentilePredictor
import pickle


def demo_end_to_end():
    """Complete end-to-end demonstration"""
    print("=" * 80)
    print("END-TO-END MAL PREDICTION DEMONSTRATION")
    print("Query Text → Features → MAL Percentiles")
    print("=" * 80)

    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("\n❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please set it: export OPENAI_API_KEY=your_key_here")
        return

    # Load predictor
    print("\n1. Loading MAL predictor...")
    predictor_path = '../models/mal_predictor.pkl'
    if not os.path.exists(predictor_path):
        print(f"   ❌ Predictor not found at {predictor_path}")
        print("      Please run: python mal_percentile_predictor.py first")
        return

    with open(predictor_path, 'rb') as f:
        predictor = pickle.load(f)
    print("   ✓ Predictor loaded")

    # Initialize feature extractor
    print("\n2. Initializing LLM feature extractor...")
    try:
        extractor = QueryFeatureExtractor()
        print("   ✓ Feature extractor initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Test queries
    test_queries = [
        {
            'query': "지난주에 찍은 골프 스윙 영상 보여줘",
            'participant_id': 'P013',  # Known user
            'description': "Recall personal photo/video"
        },
        {
            'query': "오늘 날씨에 적합한 패션 스타일 추천해줘",
            'participant_id': None,  # New user (cold-start)
            'description': "Recommendation with context"
        },
        {
            'query': "이번 달에 총 얼마나 썼지?",
            'participant_id': 'P020',  # Known user
            'description': "Financial tracking/analytics"
        }
    ]

    print("\n" + "=" * 80)
    print("3. RUNNING END-TO-END PREDICTIONS")
    print("=" * 80)

    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        participant_id = test_case['participant_id']
        description = test_case['description']

        print(f"\n{'─' * 80}")
        print(f"Test Case {i}: {description}")
        print(f"{'─' * 80}")
        print(f"Query: \"{query}\"")
        print(f"Participant: {participant_id if participant_id else 'New User (Cold-start)'}")

        # Step 1: Extract features
        print(f"\n  [Step 1] Extracting features via LLM...")
        try:
            features = extractor.extract_features(query)
            print(f"  ✓ Extracted {len(features)} features")

            # Show some key features
            print(f"\n  Key extracted features:")
            print(f"    Task type: {features.get('QL_task_type')}")
            print(f"    Goal: {features.get('QL_goal_type')}")
            print(f"    Urgency: {features.get('QL_urgency_level')}")
            print(f"    Time ref: {features.get('QL_has_time_ref')}")
            print(f"    Personalization: {features.get('QL_personalization_depth')}")

        except Exception as e:
            print(f"  ❌ Feature extraction failed: {e}")
            continue

        # Step 2: Predict MAL percentiles
        print(f"\n  [Step 2] Predicting MAL percentiles...")
        try:
            results = predictor.predict_multiple_percentiles(
                features=features,
                percentiles=[10, 50, 90, 95],
                participant_id=participant_id
            )

            scenario = results['p50']['scenario']
            print(f"  ✓ Prediction complete ({scenario})")

            print(f"\n  Predicted MAL percentiles:")
            print(f"    10th: {results['p10']['mal_seconds']:6.1f}s  (optimistic)")
            print(f"    50th: {results['p50']['mal_seconds']:6.1f}s  (median)")
            print(f"    90th: {results['p90']['mal_seconds']:6.1f}s  (conservative) ⭐")
            print(f"    95th: {results['p95']['mal_seconds']:6.1f}s  (very conservative)")

            print(f"\n  Interpretation:")
            if scenario == 'personalized':
                print(f"    • This user typically waits up to {results['p50']['mal_seconds']:.0f}s (50% of time)")
                print(f"    • To accommodate 90% of this user's cases, plan for {results['p90']['mal_seconds']:.0f}s")
            else:
                print(f"    • For a typical new user, median MAL is {results['p50']['mal_seconds']:.0f}s")
                print(f"    • To accommodate 90% of users, plan for {results['p90']['mal_seconds']:.0f}s")
                print(f"    • Note: After 5-10 queries, we can personalize predictions")

        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            continue

    print("\n" + "=" * 80)
    print("4. USAGE IN PRODUCTION")
    print("=" * 80)

    print("""
# Example: Integrate into your application

from feature_extractor import QueryFeatureExtractor
from mal_percentile_predictor import MALPercentilePredictor
import pickle

# One-time setup
extractor = QueryFeatureExtractor()
with open('models/mal_predictor.pkl', 'rb') as f:
    predictor = pickle.load(f)

# For each user query
def get_mal_prediction(query_text, user_id=None):
    # Extract features from query
    features = extractor.extract_features(query_text)

    # Predict MAL at 90th percentile (conservative)
    result = predictor.predict_mal_percentile(
        features=features,
        percentile=90,
        participant_id=user_id
    )

    return result['mal_seconds']

# Use in UI/UX decisions
query = "오늘 날씨에 적합한 패션 스타일 추천해줘"
predicted_mal = get_mal_prediction(query, user_id='P013')

if predicted_mal > 30:
    show_progress_bar(f"Processing... (~{predicted_mal:.0f}s)")
else:
    show_spinner()
    """)

    print("\n" + "=" * 80)
    print("✅ END-TO-END DEMO COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Query text → LLM extracts 51 features automatically")
    print("  2. Features → LMM predicts personalized MAL percentiles")
    print("  3. Cold-start handled gracefully (population baseline)")
    print("  4. Ready for production deployment!")


if __name__ == "__main__":
    demo_end_to_end()
