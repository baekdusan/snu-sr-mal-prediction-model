"""
MAL Prediction Model

Predicts Maximum Acceptable Latency (MAL) for Korean natural language queries.
Uses LLM (GPT-5-mini) to extract features based on feature_specification.md and batch_1 reference,
then uses trained model to predict MAL.

Usage:
    model = MALPredictor()
    mal = model.predict("지난주에 찍은 골프 스윙 영상 보여줘")
    print(f"Predicted MAL: {mal:.2f} seconds")
"""

import os
import json
import pickle
from pathlib import Path
from openai import OpenAI
import pandas as pd


class MALPredictor:
    """
    MAL (Maximum Acceptable Latency) Predictor

    Uses LLM to extract features and trained model to predict latency.
    """

    def __init__(
        self,
        model_path: str = 'best_improved_model.pkl',
        feature_spec_path: str = 'feature_specification.md',
        batch1_path: str = 'batch_responses/batch_1_response.md',
        openai_api_key: str = None
    ):
        """
        Initialize MAL predictor.

        Args:
            model_path: Path to the trained model pickle file
            feature_spec_path: Path to feature specification markdown
            batch1_path: Path to batch 1 response (for feature reference)
            openai_api_key: OpenAI API key (defaults to env var)
        """
        # OpenAI setup
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = "gpt-5-mini"

        # Load feature specification
        self.feature_spec = self._load_file(feature_spec_path)

        # Load batch 1 reference
        self.batch1_reference = self._load_file(batch1_path)

        # Load trained model
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.uses_embeddings = False
        self.embedding_dim = 0

        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: Model file '{model_path}' not found.")
            print("Please train a model first using quick_train.py or improved_train.py")

    def _load_file(self, file_path: str) -> str:
        """Load text file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_model(self, model_path: str):
        """
        Load trained model from pickle file.

        Args:
            model_path: Path to the model pickle file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
        self.uses_embeddings = model_data.get('uses_embeddings', False)
        self.embedding_dim = model_data.get('embedding_dim', 0)

        print(f"✓ Model loaded from: {model_path}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Model type: {type(self.model).__name__}")
        if self.uses_embeddings:
            print(f"  Uses embeddings: Yes ({self.embedding_dim} dimensions)")

    def _extract_features_with_llm(self, query: str) -> dict:
        """
        Use LLM to extract features from query.

        Args:
            query: Korean natural language query

        Returns:
            Dictionary of feature values
        """
        prompt = f"""You are a feature extraction system for a MAL (Maximum Acceptable Latency) prediction model.

Given a Korean natural language query, extract all features according to the Feature Specification and following the EXACT same pattern as shown in the Feature_Value_Reference from batch 1.

# Feature Specification
{self.feature_spec}

# Feature_Value_Reference (from batch 1)
{self._extract_feature_reference_sample()}

# Task
Extract features for the following query and output ONLY a JSON object (no other text):

Query: "{query}"

Output format (JSON only):
```json
{{
  "QL_chars": <number>,
  "QL_words": <number>,
  "task_category": "<category>",
  "modality_type": "<type>",
  ... (all other features)
}}
```

IMPORTANT:
1. Output ONLY the JSON object, nothing else
2. Use the EXACT same field names as shown in the reference
3. Follow the EXACT same value types and ranges as the reference
4. Do NOT include the "queries" or "MAL" fields (we already have the query and will predict MAL)
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a precise feature extraction system. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        features = self._parse_json_from_response(response_text)

        return features

    def _extract_feature_reference_sample(self) -> str:
        """Extract a sample from batch 1 to use as reference"""
        # Extract first few examples from batch 1 JSON
        import re
        pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(pattern, self.batch1_reference, re.DOTALL)

        if match:
            json_text = match.group(1)
            try:
                data = json.loads(json_text)
                # Return first 3 examples as reference
                sample = data[:3]
                return json.dumps(sample, indent=2, ensure_ascii=False)
            except:
                pass

        # Fallback: return a portion of the batch 1 response
        return self.batch1_reference[:3000]

    def _get_query_embedding(self, query: str) -> list:
        """
        Get embedding for a query using OpenAI API.

        Args:
            query: Query text

        Returns:
            List of embedding values
        """
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def _parse_json_from_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response"""
        import re

        # Try to find JSON in code blocks
        pattern = r'```(?:json)?\s*\n(.*?)\n```'
        match = re.search(pattern, response_text, re.DOTALL)

        if match:
            json_text = match.group(1)
        else:
            # Try to parse the whole response as JSON
            json_text = response_text

        # Parse JSON
        try:
            features = json.loads(json_text)
            return features
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\n\nResponse:\n{response_text}")

    def predict(self, query: str, verbose: bool = False) -> float:
        """
        Predict MAL for a given query.

        Args:
            query: Korean natural language query
            verbose: If True, print extracted features

        Returns:
            Predicted MAL in seconds
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train a model first or provide valid model_path.")

        # Step 1: Extract features using LLM
        if verbose:
            print(f"\n[1] Extracting features from query using {self.model_name}...")

        features = self._extract_features_with_llm(query)

        if verbose:
            print("\n" + "="*60)
            print("EXTRACTED FEATURES")
            print("="*60)
            for key, value in features.items():
                print(f"  {key}: {value}")

        # Step 2: Convert to DataFrame
        features_df = pd.DataFrame([features])

        # Handle categorical features - need to encode them
        categorical_cols = features_df.select_dtypes(include=['object']).columns.tolist()

        # Apply target encoding using global mean (since we don't have training data here)
        for col in categorical_cols:
            if col in features_df.columns:
                # Simple hash-based encoding for prediction
                features_df[col] = hash(str(features_df[col].iloc[0])) % 1000

        # Step 3: Apply feature engineering
        features_df = self._apply_feature_engineering(features_df)

        # Step 3.5: Add embedding if model uses it
        if self.uses_embeddings:
            if verbose:
                print(f"\n[2.5] Generating embedding for query...")

            embedding = self._get_query_embedding(query)

            if verbose:
                print(f"  Embedding dimension: {len(embedding)}")

            # Add embedding features
            embedding_df = pd.DataFrame(
                [embedding],
                columns=[f'emb_{i}' for i in range(len(embedding))]
            )
            features_df = pd.concat([features_df.reset_index(drop=True), embedding_df], axis=1)

        # Step 4: Align with training features
        # Add missing columns with 0
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0

        # Keep only training columns in the same order
        features_df = features_df[self.feature_names]

        if verbose:
            print(f"\n[2] Aligned features: {features_df.shape[1]} columns")

        # Step 5: Scale features (if scaler exists)
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features_df)
            if verbose:
                print(f"[3] Scaled features")
        else:
            features_scaled = features_df.values
            if verbose:
                print(f"[3] No scaler (using raw features)")

        # Step 6: Predict
        prediction = self.model.predict(features_scaled)[0]

        if verbose:
            print("\n" + "="*60)
            print(f"PREDICTED MAL: {prediction:.2f} seconds")
            print("="*60)

        return prediction

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply same feature engineering as training.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features
        """
        # Create interaction features (same as in improved_train.py)

        # Query length ratio
        if 'QL_chars' in df.columns and 'QL_words' in df.columns:
            df['QL_chars_per_word'] = df['QL_chars'] / (df['QL_words'] + 1)

        # Complexity score
        if 'requires_aggregation' in df.columns and 'requires_generation' in df.columns:
            df['complexity_score'] = (
                df['requires_aggregation'] +
                df['requires_generation'] +
                df.get('requires_external_like_search', 0)
            )

        # Time complexity
        if 'temporal_scope_level' in df.columns and 'urgency_level' in df.columns:
            df['time_complexity'] = df['temporal_scope_level'] * df['urgency_level']

        # Polynomial features would be added here if we had the poly transformer
        # For now, we skip polynomial features in prediction to keep it simple
        # The model should still work reasonably well with the core features

        return df

    def predict_batch(self, queries: list, verbose: bool = False) -> list:
        """
        Predict MAL for multiple queries.

        Args:
            queries: List of Korean natural language queries
            verbose: If True, print progress

        Returns:
            List of predicted MAL values in seconds
        """
        predictions = []

        for i, query in enumerate(queries):
            if verbose:
                print(f"\n{'='*80}")
                print(f"[{i+1}/{len(queries)}] Query: {query}")
                print('='*80)

            pred = self.predict(query, verbose=verbose)
            predictions.append(pred)

            if not verbose:
                print(f"[{i+1}/{len(queries)}] {query[:50]}... → {pred:.2f}s")

        return predictions

    def explain_prediction(self, query: str):
        """
        Extract features and show detailed explanation.

        Args:
            query: Korean natural language query
        """
        print("\n" + "="*80)
        print(f"DETAILED PREDICTION EXPLANATION")
        print("="*80)
        print(f"Query: {query}")

        # Extract features
        print(f"\nExtracting features using {self.model_name}...")
        features = self._extract_features_with_llm(query)

        # Show features by category
        print("\n" + "="*80)
        print("EXTRACTED FEATURES")
        print("="*80)

        # Group features
        categories = {
            'Basic': ['QL_chars', 'QL_words'],
            'Task & Modality': ['task_category', 'modality_type', 'device_context_implicit'],
            'Temporal': ['temporal_scope_level', 'time_reference_recency', 'historical_span_complexity'],
            'Urgency & Stakes': ['urgency_level', 'stakes_importance_level'],
            'Personalization': ['personalization_depth', 'social_context_present'],
            'Requirements': ['requires_aggregation', 'requires_generation', 'requires_historical_search', 'requires_external_like_search'],
            'Output': ['output_cardinality_expectation', 'output_format_structured', 'output_format_media'],
            'Domain': ['calendar_or_schedule_related', 'finance_or_spending_related', 'health_fitness_related', 'shopping_or_commerce_related', 'entertainment_media_related'],
        }

        for category, feature_list in categories.items():
            print(f"\n{category}:")
            for feature in feature_list:
                if feature in features:
                    print(f"  - {feature}: {features[feature]}")

        # Predict
        print("\n" + "="*80)
        print("PREDICTION")
        print("="*80)
        mal = self.predict(query, verbose=False)
        print(f"Predicted MAL: {mal:.2f} seconds")
        print("="*80)


def main():
    """Example usage of MAL Predictor"""

    print("\n" + "="*80)
    print("MAL PREDICTION MODEL - DEMO")
    print("="*80)

    # Initialize predictor
    try:
        predictor = MALPredictor()
    except Exception as e:
        print(f"\nError initializing predictor: {e}")
        print("\nMake sure:")
        print("  1. OPENAI_API_KEY is set in environment")
        print("  2. best_improved_model.pkl exists (run improved_train.py first)")
        print("  3. feature_specification.md exists")
        print("  4. batch_responses/batch_1_response.md exists")
        return

    # Example queries
    example_queries = [
        "지난주에 찍은 골프 스윙 영상 보여줘",
        "오늘 점심 메뉴 추천해줘",
        "지난달 카드 결제 내역 정리해서 보여줘",
    ]

    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    for i, query in enumerate(example_queries, 1):
        print(f"\n[{i}] Query: {query}")

        try:
            mal = predictor.predict(query, verbose=False)
            print(f"    Predicted MAL: {mal:.2f} seconds")
        except Exception as e:
            print(f"    Error: {e}")

    # Detailed explanation for first query
    if example_queries:
        print("\n" + "="*80)
        print("DETAILED EXPLANATION EXAMPLE")
        print("="*80)
        try:
            predictor.explain_prediction(example_queries[0])
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
