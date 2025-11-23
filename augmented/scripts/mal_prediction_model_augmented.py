#!/usr/bin/env python3
"""
MAL Prediction CLI Tool with Augmented Features
Predicts expected latency (MAL) using embeddings + LLM-extracted features
"""

import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from openai import OpenAI
import time
import os
import json


class MALPredictorAugmented:
    """MAL Prediction Class with Augmented Features"""

    def __init__(self, model_path='best_model_augmented.pkl',
                 augmented_csv='augmented_data.csv',
                 api_key=None):
        """
        Initialize predictor

        Args:
            model_path: Path to the saved augmented model file
            augmented_csv: Path to augmented data CSV (to get feature schema)
            api_key: OpenAI API key
        """
        self.model_path = model_path
        self.augmented_csv = augmented_csv
        self.model_package = None
        self.client = None
        self.feature_schema = None

        # Setup API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Use --api-key option or set OPENAI_API_KEY environment variable.")

    def load_model(self):
        """Load saved model and feature schema"""
        print("━" * 60)
        print("🔄 Loading augmented model...")

        start_time = time.time()

        # Load model package
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            self.model_package = pickle.load(f)

        # Load feature schema from CSV (CSV is pre-cleaned from leakage)
        if not os.path.exists(self.augmented_csv):
            raise FileNotFoundError(f"Augmented data CSV not found: {self.augmented_csv}")

        augmented_df = pd.read_csv(self.augmented_csv)

        # Extract feature schema (no need for leakage check - CSV is pre-cleaned)
        exclude_cols = ['queries', 'MAL', 'query', 'mal']
        self.feature_schema = [col for col in augmented_df.columns
                               if col not in exclude_cols]

        elapsed = time.time() - start_time
        print(f"✅ Model loaded successfully ({elapsed:.3f}s)")
        print(f"   Model type: {type(self.model_package['model']).__name__}")
        print(f"   Embedding dimension: {self.model_package['embedding_dimension']}")
        print(f"   Feature columns: {len(self.model_package['feature_columns'])}")
        print(f"   Feature schema from CSV: {self.feature_schema}")
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

        return np.array(embedding)

    def extract_features_with_llm(self, query):
        """
        Extract augmented features using LLM

        Args:
            query: Input query string

        Returns:
            Dictionary of feature values
        """
        print("\n🔄 Extracting features with LLM...")
        start_time = time.time()

        # Create prompt for feature extraction
        prompt = f"""Extract the following features from this query and return ONLY a JSON object.

Query: "{query}"

Features to extract (return as JSON):
{json.dumps(self.feature_schema, indent=2)}

Return ONLY valid JSON with these exact keys. For numeric features, use numbers. For categorical features, use strings.

Example format:
{{
  "QL_length": 42,
  "QL_word_count": 8,
  "QL_information_density": 0.75,
  "QL_task_type": "photo",
  "QL_urgency_level": 2,
  "QL_contextual_complexity": 4,
  "QL_consequence_severity": 2
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a feature extraction assistant. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            features = json.loads(content)

            elapsed = time.time() - start_time
            print(f"✅ Features extracted ({elapsed:.3f}s)")
            print(f"   Features: {list(features.keys())}")

            return features

        except Exception as e:
            print(f"⚠️  Error extracting features: {e}")
            raise

    def prepare_features(self, query_features):
        """
        Prepare feature vector with one-hot encoding

        Args:
            query_features: Dictionary of raw feature values

        Returns:
            Numpy array of processed features
        """
        print("\n🔄 Preparing features (one-hot encoding)...")

        # Create DataFrame with single row
        df = pd.DataFrame([query_features])

        # Identify categorical columns
        exclude_columns = ['queries', 'MAL', 'query', 'mal']
        categorical_columns = [col for col in df.columns
                              if df[col].dtype == 'object' and col not in exclude_columns]

        # Apply one-hot encoding
        if categorical_columns:
            df_encoded = pd.get_dummies(df, columns=categorical_columns,
                                       prefix=categorical_columns, dtype=int)
        else:
            df_encoded = df

        # Get the expected feature columns from model package
        expected_features = self.model_package['feature_columns']

        # Create feature vector with all expected features
        feature_vector = []
        for col in expected_features:
            if col in df_encoded.columns:
                feature_vector.append(df_encoded[col].values[0])
            else:
                # Missing one-hot column → set to 0
                feature_vector.append(0)

        print(f"✅ Features prepared")
        print(f"   Expected features: {len(expected_features)}")
        print(f"   Feature vector length: {len(feature_vector)}")

        return np.array(feature_vector)

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

        # 2. Extract features with LLM
        query_features = self.extract_features_with_llm(query)

        # 3. Prepare features (one-hot encoding)
        augmented_features = self.prepare_features(query_features)

        # 4. Combine embedding + augmented features
        print("\n🔄 Combining features...")
        combined_features = np.concatenate([embedding, augmented_features])

        # Apply scaler
        scaler = self.model_package['scaler']
        # Only scale augmented features (not embeddings)
        embedding_dim = self.model_package['embedding_dimension']
        scaled_augmented = scaler.transform(augmented_features.reshape(1, -1))
        combined_features_scaled = np.concatenate([
            embedding.reshape(1, -1),
            scaled_augmented
        ], axis=1)

        print(f"✅ Features combined")
        print(f"   Embedding: {embedding_dim} dims")
        print(f"   Augmented: {len(augmented_features)} dims")
        print(f"   Total: {combined_features_scaled.shape[1]} dims")

        # 5. Model inference
        print("\n🔄 Running model inference...")
        start_time = time.time()

        model = self.model_package['model']
        predicted_mal = model.predict(combined_features_scaled)[0]

        elapsed = time.time() - start_time
        print(f"✅ Inference completed ({elapsed:.3f}s)")

        # 6. Output results
        print("\n" + "=" * 60)
        print("📊 Prediction Result")
        print("=" * 60)
        print(f"🎯 Predicted Latency (MAL): {predicted_mal:.4f}ms")
        print("=" * 60)

        return predicted_mal


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Predicts expected latency (MAL) for given queries using augmented features.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mal_prediction_model_augmented.py "Show me photos from last week"
  python mal_prediction_model_augmented.py "What's my schedule today" --model best_model_augmented.pkl
  python mal_prediction_model_augmented.py "How's the weather tomorrow?" --api-key YOUR_API_KEY
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
        default='best_model_augmented.pkl',
        help='Path to augmented model file (default: best_model_augmented.pkl)'
    )

    parser.add_argument(
        '--csv',
        type=str,
        default='augmented_data.csv',
        help='Path to augmented data CSV (default: augmented_data.csv)'
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
        predictor = MALPredictorAugmented(
            model_path=args.model,
            augmented_csv=args.csv,
            api_key=args.api_key
        )
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
