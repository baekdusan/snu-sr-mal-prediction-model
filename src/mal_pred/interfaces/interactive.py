#!/usr/bin/env python3
"""
Interactive MAL Prediction

Run this script and enter your Korean queries to get MAL predictions in real-time.

Usage:
    python predict_interactive.py
"""

import sys
from ..predictor import MALPredictor


def print_header():
    """Print welcome header"""
    print("\n" + "="*80)
    print("🔮 MAL (Maximum Acceptable Latency) Prediction System")
    print("="*80)
    print("\nEnter Korean natural language queries to predict acceptable latency.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type 'verbose' to toggle detailed output.")
    print("Type 'explain' before a query to see detailed feature explanation.")
    print("="*80 + "\n")


def main():
    """Main interactive loop"""

    # Initialize predictor
    print("Initializing MAL predictor...")
    try:
        predictor = MALPredictor()
    except Exception as e:
        print(f"\n❌ Error initializing predictor: {e}")
        print("\nMake sure:")
        print("  1. OPENAI_API_KEY is set in environment")
        print("  2. artifacts/models/best_improved_model.pkl exists (run training first)")
        print("  3. docs/feature_specification.md exists")
        print("  4. data/intermediate/batch_responses/batch_1_response.md exists")
        return 1

    print("✓ Predictor ready!\n")

    # Print header
    print_header()

    # Interactive loop
    verbose = False
    explain_next = False

    while True:
        try:
            # Get user input
            user_input = input("🔍 Enter query: ").strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!\n")
                break

            # Check for verbose toggle
            if user_input.lower() == 'verbose':
                verbose = not verbose
                print(f"✓ Verbose mode: {'ON' if verbose else 'OFF'}\n")
                continue

            # Check for explain command
            if user_input.lower() == 'explain':
                explain_next = True
                print("✓ Next query will show detailed explanation\n")
                continue

            # If explain_next is set, remove 'explain' prefix if present
            if explain_next:
                if user_input.lower().startswith('explain '):
                    user_input = user_input[8:].strip()

            print("\n" + "-"*80)

            # Predict
            if explain_next:
                # Show detailed explanation
                predictor.explain_prediction(user_input)
                explain_next = False
            else:
                # Quick prediction
                try:
                    mal = predictor.predict(user_input, verbose=verbose)

                    # Display result
                    print(f"\n📊 Query: {user_input}")
                    print(f"⏱️  Predicted MAL: {mal:.2f} seconds")

                    # Add interpretation
                    if mal < 2:
                        print("   → Very fast response expected (instant)")
                    elif mal < 4:
                        print("   → Fast response expected (< 4s)")
                    elif mal < 8:
                        print("   → Moderate delay acceptable (4-8s)")
                    else:
                        print("   → Longer delay acceptable (> 8s)")

                except Exception as e:
                    print(f"\n❌ Error during prediction: {e}")

            print("-"*80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
