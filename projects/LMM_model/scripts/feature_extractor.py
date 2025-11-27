"""
Feature Extractor for MAL Prediction

Uses OpenAI GPT-5-mini to extract features from Korean query text based on
feature_specification.md guidelines.

This module replaces the manual feature extraction by automatically generating
all 51 features from a query string.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class QueryFeatureExtractor:
    """
    Extracts query features using OpenAI GPT-5-mini based on feature_specification.md
    """

    def __init__(self, api_key: str = None):
        """
        Initialize feature extractor

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=self.api_key)

        # Load feature specification from consolidated docs
        self.feature_spec_path = PROJECT_ROOT / "docs" / "feature_specification.md"
        self.batch_example_path = (
            PROJECT_ROOT / "data" / "intermediate" / "batch_responses" / "batch_1_response.md"
        )

        # Read specification
        with open(self.feature_spec_path, "r", encoding="utf-8") as f:
            self.feature_spec = f.read()

        # Read example
        with open(self.batch_example_path, "r", encoding="utf-8") as f:
            self.batch_example = f.read()

    def extract_features(self, query: str) -> Dict[str, Any]:
        """
        Extract all features from a query string using Claude

        Args:
            query: Korean query text

        Returns:
            Dictionary of feature_name -> value
        """
        # Create prompt for GPT
        prompt = self._create_extraction_prompt(query)

        # Call OpenAI Responses API with gpt-5-mini
        response = self.client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a feature extraction expert for a Maximum Acceptable Latency (MAL) prediction model.",
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=2048,
        )

        # Parse response
        response_text = self._collect_text_output(response)

        # Extract JSON from response
        features = self._parse_features_from_response(response_text)

        return features

    def _collect_text_output(self, response) -> str:
        """Collect text content from Responses API output."""
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        if not getattr(response, "output", None):
            return ""

        chunks = []
        for item in response.output:
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    chunks.append(content.text)
        return "".join(chunks)

    def _create_extraction_prompt(self, query: str) -> str:
        """Create prompt for GPT to extract features"""

        prompt = f"""Your task is to extract ALL features from the given Korean query based on the feature specification document below.

# Feature Specification
{self.feature_spec}

# Example Output Format
Here are examples from previous extractions showing the exact JSON structure and feature values:

{self.batch_example[:3000]}

# Your Task
Extract ALL features for the following query:

**Query:** "{query}"

**Instructions:**
1. Follow the feature definitions EXACTLY as specified in the feature specification
2. For binary features: return 0 or 1
3. For ordinal features: return the appropriate integer level (0, 1, 2, 3, etc.)
4. For categorical features: return the exact category string as shown in examples
5. For numeric features: calculate the exact value
6. Include ALL features shown in the example (51+ features)
7. Match the exact JSON structure from the examples

**Output Format:**
Return ONLY a valid JSON object with ALL features. Do NOT include any markdown code blocks, explanations, or additional text.
Just return the raw JSON object starting with {{ and ending with }}.

Now extract features for: "{query}"
"""
        return prompt

    def _parse_features_from_response(self, response_text: str) -> Dict[str, Any]:
        """Parse features from Claude's response"""
        # Try to find JSON in response
        try:
            # First try direct parse
            features = json.loads(response_text)
            return features
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                features = json.loads(json_str)
                return features
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                features = json.loads(json_str)
                return features
            else:
                # Try to find any JSON object
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    features = json.loads(json_match.group())
                    return features
                else:
                    raise ValueError(f"Could not parse JSON from response: {response_text[:200]}...")

    def extract_batch(self, queries: list) -> list:
        """
        Extract features for multiple queries

        Args:
            queries: List of query strings

        Returns:
            List of feature dictionaries
        """
        results = []
        for i, query in enumerate(queries):
            print(f"Extracting features for query {i+1}/{len(queries)}: {query[:50]}...")
            try:
                features = self.extract_features(query)
                results.append({
                    'query': query,
                    'features': features,
                    'success': True
                })
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'query': query,
                    'features': None,
                    'success': False,
                    'error': str(e)
                })

        return results


def demo_feature_extraction():
    """Demonstrate feature extraction"""
    print("=" * 80)
    print("MAL Feature Extractor - Demo")
    print("=" * 80)

    # Initialize extractor
    print("\nInitializing feature extractor...")
    extractor = QueryFeatureExtractor()
    print("✓ Extractor initialized")

    # Test queries
    test_queries = [
        "지난주에 찍은 골프 스윙 영상 보여줘",
        "오늘 날씨에 적합한 패션 스타일 추천해줘",
        "이번 달에 총 얼마나 썼지?",
        "소연이가 추천했던 영화 콘텐츠 제목이 뭐였지?",
        "현재 위치에서 걸어서 갈 수 있는 맛집 추천해줘"
    ]

    print(f"\n{'='*80}")
    print("Extracting features for test queries...")
    print(f"{'='*80}\n")

    results = extractor.extract_batch(test_queries)

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}: {result['query']}")
        print(f"{'─'*80}")

        if result['success']:
            features = result['features']
            print(f"✓ Feature extraction successful")
            print(f"\nKey features:")
            print(f"  Task type: {features.get('QL_task_type')}")
            print(f"  Goal: {features.get('QL_goal_type')}")
            print(f"  Urgency: {features.get('QL_urgency_level')}")
            print(f"  Personalization: {features.get('QL_personalization_depth')}")
            print(f"  Time ref: {features.get('QL_has_time_ref')}")
            print(f"  Location coupled: {features.get('QL_location_coupled')}")
            print(f"  Recommender task: {features.get('QL_recommender_task')}")

            print(f"\nAll {len(features)} features extracted:")
            for k, v in sorted(features.items())[:10]:  # Show first 10
                print(f"  {k}: {v}")
            print(f"  ... ({len(features) - 10} more features)")
        else:
            print(f"❌ Error: {result.get('error')}")

    # Save results
    import json
    output_path = '../outputs/feature_extraction_demo.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print("✅ Demo complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    demo_feature_extraction()
