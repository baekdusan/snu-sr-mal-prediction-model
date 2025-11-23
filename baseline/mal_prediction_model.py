#!/usr/bin/env python3
"""
MAL Prediction CLI Tool
A CLI tool that predicts expected latency (MAL) for given queries
"""

import sys
import argparse
import pickle
import numpy as np
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor
import time
import os


class MALPredictor:
    """MAL Prediction Class"""

    def __init__(self, model_path='best_model.pkl', api_key=None):
        """
        Initialize predictor

        Args:
            model_path: Path to the saved model file
            api_key: OpenAI API key
        """
        self.model_path = model_path
        self.model = None
        self.client = None

        # Setup API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Use --api-key option or set OPENAI_API_KEY environment variable.")

    def load_model(self):
        """Load saved model"""
        print("━" * 60)
        print("🔄 Loading model...")

        start_time = time.time()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        elapsed = time.time() - start_time
        print(f"✅ Model loaded successfully ({elapsed:.3f}s)")
        print(f"   Model type: {type(self.model).__name__}")
        print("━" * 60)

    def get_embedding(self, text):
        """
        Convert text to embedding vector

        Args:
            text: Input text

        Returns:
            Embedding vector (numpy array)
        """
        print("\n🔄 Generating embedding...")
        start_time = time.time()

        # Initialize OpenAI client (only once)
        if self.client is None:
            self.client = OpenAI(api_key=self.api_key)

        # Generate embedding
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )

        embedding = response.data[0].embedding
        elapsed = time.time() - start_time

        print(f"✅ Embedding generated ({elapsed:.3f}s)")
        print(f"   Dimension: {len(embedding)}")

        return np.array(embedding).reshape(1, -1)

    def predict(self, query):
        """
        Predict MAL for given query

        Args:
            query: Input query string

        Returns:
            Predicted MAL value
        """
        print("\n" + "=" * 60)
        print(f"📝 Input Query: {query}")
        print("=" * 60)

        # 1. Generate embedding
        embedding = self.get_embedding(query)

        # 2. Model inference
        print("\n🔄 Running model inference...")
        start_time = time.time()

        predicted_mal = self.model.predict(embedding)[0]

        elapsed = time.time() - start_time
        print(f"✅ Inference completed ({elapsed:.3f}s)")

        # 3. Output results
        print("\n" + "=" * 60)
        print("📊 Prediction Result")
        print("=" * 60)
        print(f"🎯 Predicted Latency (MAL): {predicted_mal:.4f}ms")
        print("=" * 60)

        return predicted_mal


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Predicts expected latency (MAL) for given queries.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mal_prediction_model.py "Show me photos from last week"
  python mal_prediction_model.py "What's my schedule today" --model best_model.pkl
  python mal_prediction_model.py "How's the weather tomorrow?" --api-key YOUR_API_KEY
        """
    )

    parser.add_argument(
        'query',
        type=str,
        nargs='?',
        help='Query string to predict'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='best_model.pkl',
        help='Path to model file (default: best_model.pkl)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (can also use OPENAI_API_KEY environment variable)'
    )

    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Run in interactive mode'
    )

    args = parser.parse_args()

    try:
        # Initialize predictor
        predictor = MALPredictor(model_path=args.model, api_key=args.api_key)
        predictor.load_model()

        # Interactive mode
        if args.interactive or not args.query:
            print("\n💬 Interactive Mode (exit: type 'quit' or 'exit')\n")
            while True:
                try:
                    query = input("Query > ").strip()

                    if query.lower() in ['quit', 'exit', 'q']:
                        print("\n👋 Exiting program.")
                        break

                    if not query:
                        print("⚠️  Please enter a query.")
                        continue

                    predictor.predict(query)
                    print()

                except KeyboardInterrupt:
                    print("\n\n👋 Exiting program.")
                    break

        # Single query mode
        else:
            predictor.predict(args.query)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
