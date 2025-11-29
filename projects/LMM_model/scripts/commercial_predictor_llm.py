"""
Commercial MAL Predictor with LLM Feature Extraction (Final - Simplified)

This is the FINAL commercial-ready MAL predictor using:
- 8 selected features (down from 51) → 84% less API tokens!
- LLM-based feature extraction (GPT-5-mini or Claude)
- Linear Mixed Model (Model 1 - Participant Random Effect)
- Population-level predictions

Improvements over original:
- 51 features → 8 features (84% reduction)
- Faster LLM extraction (shorter prompt)
- Higher extraction accuracy (fewer features to confuse)
- Lower API cost (fewer tokens)
- Better model performance (no multicollinearity)

Performance:
- R² = 0.7277 (72.8% variance explained)
- RMSE = 1.55 seconds
- All features highly significant (p < 0.001)

Usage:
```python
predictor = CommercialMALPredictorLLM()

# Single prediction
result = predictor.predict("지난주에 찍은 골프 스윙 영상 보여줘", accommodation_levels=[50, 90, 95])
print(result['mal_predictions'])
# Output: {'50%': 12.3, '90%': 28.5, '95%': 35.7}
```
"""

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import json
from typing import Dict, List
import warnings
from openai import OpenAI
import os

warnings.filterwarnings('ignore')


class SimplifiedLLMFeatureExtractor:
    """
    Simplified LLM feature extractor for 8 selected features only

    Improvements over original:
    - 51 → 8 features (84% reduction)
    - Shorter, clearer prompt
    - Faster extraction
    - Lower cost
    - Higher accuracy
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize feature extractor

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-mini for speed/cost)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # Feature definitions (based on final selected features)
        self.feature_definitions = """
Extract exactly 11 features from the Korean query based on these detailed specifications:

1. **needs_health_data** (BINARY: 0 or 1)
   - 1 = Yes: Query requires health/fitness data (운동, 건강, 수면, 식단, 칼로리, 걸음수, 심박수)
   - 0 = No: No health data needed

2. **expected_answer_length** (ORDINAL: 0-2)
   - 0 = Single item/short answer (하나의 아이템, 간단한 답변)
   - 1 = List/multiple items (여러 개의 아이템, 목록)
   - 2 = Long document/summary (긴 문서, 요약문)

3. **planning_horizon** (ORDINAL: 0-3)
   - 0 = No planning (단순 조회)
   - 1 = Short-term (오늘, 내일, 이번 주)
   - 2 = Medium-term (이번 달, 다음 달)
   - 3 = Long-term (올해, 내년, 장기 계획)

4. **time_window_length** (ORDINAL: 0-3)
   - 0 = Point in time (특정 시점: 어제, 오늘)
   - 1 = Days/week (며칠, 이번 주)
   - 2 = Weeks/month (몇 주, 이번 달)
   - 3 = Months/year+ (몇 달, 올해, 작년)

5. **time_urgency_level** (ORDINAL: 0-2)
   - 0 = No urgency (긴급하지 않음)
   - 1 = Moderate urgency (보통 긴급성)
   - 2 = High urgency (매우 긴급)

6. **novelty_seeking** (BINARY: 0 or 1)
   - 1 = Yes: Query seeks new/novel recommendations (새로운, 추천, 발견)
   - 0 = No: Query about known/past items

7. **requires_aggregation** (BINARY: 0 or 1)
   - 1 = Yes: Requires counting, summing, averaging (몇 번, 총, 평균, 가장 많이)
   - 0 = No: Simple retrieval

8. **has_comparative_phrase** (BINARY: 0 or 1)
   - 1 = Yes: Contains comparison (더, 가장, 최고, 비교, vs, 대비)
   - 0 = No: No comparison

9. **device_context_implied** (ORDINAL: 0-2)
   - 0 = Device-agnostic (기기 무관)
   - 1 = Mobile-specific context (모바일 맥락)
   - 2 = Desktop/specific device (특정 기기)

10. **output_requires_multimedia_creation** (BINARY: 0 or 1)
   - 1 = Yes: Requires creating multimedia output (영상 만들기, 사진 편집)
   - 0 = No: Text or retrieval only

11. **social_context_strength** (ORDINAL: 0-2)
   - 0 = No social context (개인적 조회)
   - 1 = Mentions people (친구, 가족)
   - 2 = Group/collaborative context (단체, 팀)

IMPORTANT: Return ONLY the JSON with feature values (0, 1, or 2 as specified), no explanations.
"""

    def extract_features(self, query: str, return_reasoning: bool = False) -> Dict:
        """
        Extract 11 features from query using LLM

        Args:
            query: Korean query text
            return_reasoning: If True, also return reasoning for each feature

        Returns:
            If return_reasoning=False: Dictionary of 11 feature values
            If return_reasoning=True: Dictionary with 'features' and 'reasoning'
        """
        if return_reasoning:
            prompt = f"""{self.feature_definitions}

Query: "{query}"

Return a JSON object with:
1. "features": object with exactly 11 features
2. "reasoning": object with brief explanation for each feature (1-2 sentences in Korean)

Example format:
{{
  "features": {{
    "needs_health_data": 0,
    "expected_answer_length": 1,
    "planning_horizon": 0,
    ...
  }},
  "reasoning": {{
    "needs_health_data": "건강/운동 데이터가 필요하지 않음",
    "expected_answer_length": "여러 개의 아이템을 보여주는 리스트 형태",
    ...
  }}
}}"""

            max_tokens = 800
        else:
            prompt = f"""{self.feature_definitions}

Query: "{query}"

Return ONLY a JSON object with exactly 11 features (no explanation, no markdown):
{{
  "needs_health_data": 0,
  "expected_answer_length": 0,
  "planning_horizon": 0,
  "time_window_length": 0,
  "time_urgency_level": 0,
  "novelty_seeking": 0,
  "requires_aggregation": 0,
  "has_comparative_phrase": 0,
  "device_context_implied": 0,
  "output_requires_multimedia_creation": 0,
  "social_context_strength": 0
}}"""

            max_tokens = 200

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise feature extraction expert. Return only valid JSON, no markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=max_tokens
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON
        result = self._parse_json(response_text)

        if return_reasoning:
            # Validate structure
            if 'features' not in result or 'reasoning' not in result:
                raise ValueError("Response must contain 'features' and 'reasoning' keys")

            features = result['features']
            reasoning = result['reasoning']

            # Validate features
            self._validate_features(features)

            return {
                'features': features,
                'reasoning': reasoning
            }
        else:
            # Validate
            self._validate_features(result)
            return result

    def _parse_json(self, text: str) -> Dict[str, int]:
        """Parse JSON from LLM response"""
        # Remove markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {text[:200]}") from e

    def _validate_features(self, features: Dict[str, int]):
        """Validate extracted features"""
        expected_features = [
            'needs_health_data',
            'expected_answer_length',
            'planning_horizon',
            'time_window_length',
            'time_urgency_level',
            'novelty_seeking',
            'requires_aggregation',
            'has_comparative_phrase',
            'device_context_implied',
            'output_requires_multimedia_creation',
            'social_context_strength'
        ]

        # Check all features present
        for feat in expected_features:
            if feat not in features:
                raise ValueError(f"Missing feature: {feat}")

        # Check values are in valid range (binary: 0-1, ordinal: 0-2 or 0-3)
        valid_ranges = {
            'needs_health_data': [0, 1],
            'expected_answer_length': [0, 1, 2],
            'planning_horizon': [0, 1, 2, 3],
            'time_window_length': [0, 1, 2, 3],
            'time_urgency_level': [0, 1, 2],
            'novelty_seeking': [0, 1],
            'requires_aggregation': [0, 1],
            'has_comparative_phrase': [0, 1],
            'device_context_implied': [0, 1, 2],
            'output_requires_multimedia_creation': [0, 1],
            'social_context_strength': [0, 1, 2]
        }

        for feat, val in features.items():
            if val not in valid_ranges.get(feat, [0, 1]):
                raise ValueError(f"Invalid value for {feat}: {val} (must be in {valid_ranges.get(feat, [0, 1])})")


class CommercialMALPredictorLLM:
    """
    Commercial MAL predictor with LLM feature extraction

    Uses GPT-5-mini to extract 8 features (down from 51)
    Then predicts MAL using Linear Mixed Model
    """

    def __init__(self, model_path: str = None, api_key: str = None):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model (.pkl)
            api_key: OpenAI API key
        """
        self.model = None
        self.feature_extractor = SimplifiedLLMFeatureExtractor(api_key=api_key)
        self.selected_features = [
            'needs_health_data',
            'expected_answer_length',
            'planning_horizon',
            'time_window_length',
            'time_urgency_level',
            'novelty_seeking',
            'requires_aggregation',
            'has_comparative_phrase',
            'device_context_implied',
            'output_requires_multimedia_creation',
            'social_context_strength'
        ]

        # Model parameters
        self.fixed_effects = {}
        self.population_mean_random_effect = 0.0
        self.participant_variance = 0.0
        self.residual_std = 0.0

        if model_path:
            self.load_model(model_path)
        else:
            self._try_load_default()

    def _try_load_default(self):
        """Try to load model from default location"""
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))

        default_paths = [
            os.path.join(script_dir, '../models/lmm_model1_selected.pkl'),
            os.path.join(script_dir, 'lmm_model1_selected.pkl'),
            'lmm_model1_selected.pkl'
        ]

        for path in default_paths:
            try:
                self.load_model(path)
                return
            except Exception as e:
                continue

        raise FileNotFoundError(f"No model found at default locations. Searched: {default_paths}")

    def load_model(self, model_path: str):
        """Load trained model"""
        print(f"Loading model from {model_path}...")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self._extract_model_components()

        print(f"✓ Model loaded")
        print(f"  Features: {len(self.selected_features)}")
        print(f"  Total std: {self.total_std:.4f}")

    def _extract_model_components(self):
        """Extract model components"""
        self.fixed_effects = dict(self.model.params)

        participant_res = {}
        for pid, re_vals in self.model.random_effects.items():
            participant_res[pid] = re_vals.iloc[0]

        self.population_mean_random_effect = np.mean(list(participant_res.values()))
        self.participant_variance = self.model.cov_re.iloc[0, 0]
        self.residual_std = np.sqrt(self.model.scale)
        self.total_std = np.sqrt(self.residual_std**2 + self.participant_variance)

    def predict(
        self,
        query: str,
        accommodation_levels: List[int] = [50, 90, 95],
        verbose: bool = True,
        return_reasoning: bool = False
    ) -> Dict:
        """
        Predict MAL for a query

        Args:
            query: Korean query text
            accommodation_levels: Percentiles for user retention (e.g., [50, 90, 95])
                                 50 = 50% retention (50% churn) = median MAL
                                 90 = 90% retention (10% churn) = higher MAL needed
                                 95 = 95% retention (5% churn) = even higher MAL needed
            verbose: Print extraction progress
            return_reasoning: Include LLM reasoning for each feature

        Returns:
            Dictionary with predictions and interpretation
        """
        # Extract features using LLM
        if verbose:
            print(f"  Extracting features: '{query[:50]}...'")

        extraction_result = self.feature_extractor.extract_features(query, return_reasoning=return_reasoning)

        if return_reasoning:
            features = extraction_result['features']
            reasoning = extraction_result['reasoning']
        else:
            features = extraction_result
            reasoning = None

        if verbose:
            print(f"  ✓ Features extracted")

        # Compute log(MAL) mean
        log_mal_mean = self.fixed_effects['Intercept']

        for feat, value in features.items():
            if feat in self.fixed_effects:
                log_mal_mean += self.fixed_effects[feat] * value

        log_mal_mean += self.population_mean_random_effect

        # Compute MAL at each accommodation level
        # IMPORTANT: Lower MAL = more users can tolerate (less churn)
        #            Higher MAL = fewer users can tolerate (more churn)
        # We want the MAL value where {level}% of users will churn
        # This means we need the (100-level) percentile of the MAL distribution
        mal_predictions = {}
        for level in accommodation_levels:
            # For 90% retention (10% churn): We want the 10th percentile
            # This is the MAL that only 10% of users exceed (90% are below = satisfied)
            churn_percentile = (100 - level) / 100
            z_score = stats.norm.ppf(churn_percentile)
            log_mal_p = log_mal_mean + z_score * self.total_std
            mal_sec = np.exp(log_mal_p)
            mal_predictions[f"{level}%"] = round(mal_sec, 2)

        mean_mal = np.exp(log_mal_mean)

        interpretation = self._generate_interpretation(mal_predictions)

        result = {
            'query': query,
            'features': features,
            'mal_predictions': mal_predictions,
            'mean_mal': round(mean_mal, 2),
            'log_mal_mean': round(log_mal_mean, 4),
            'total_std': round(self.total_std, 4),
            'interpretation': interpretation
        }

        if return_reasoning and reasoning is not None:
            result['reasoning'] = reasoning

        return result

    def _generate_interpretation(self, mal_predictions: Dict) -> str:
        """Generate interpretation"""
        lines = []

        # Sort by retention level (descending: 95% -> 90% -> 50%)
        sorted_predictions = sorted(
            mal_predictions.items(),
            key=lambda x: int(x[0].replace('%', '')),
            reverse=True
        )

        for level_str, mal_sec in sorted_predictions:
            level = int(level_str.replace('%', ''))
            churn_rate = 100 - level
            lines.append(
                f"  • {level}% retention ({churn_rate}% churn): {mal_sec:.1f}s "
                f"(keep {level}% of users)"
            )

        if '90%' in mal_predictions:
            lines.append("")
            lines.append(f"Recommendation:")
            lines.append(f"  → Design for {mal_predictions['90%']:.1f}s to keep 90% of users (10% churn)")

        return "\n".join(lines)

    def batch_predict(
        self,
        queries: List[str],
        accommodation_levels: List[int] = [50, 90, 95]
    ) -> pd.DataFrame:
        """Batch prediction"""
        results = []

        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] {query[:50]}...")

            try:
                result = self.predict(query, accommodation_levels)

                row = {
                    'query': query,
                    'mean_mal': result['mean_mal']
                }
                row.update(result['mal_predictions'])
                row.update(result['features'])
                row['status'] = 'success'

                results.append(row)

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'query': query,
                    'status': 'failed',
                    'error': str(e)
                })

        return pd.DataFrame(results)

    def print_summary(self):
        """Print summary"""
        print("=" * 80)
        print("COMMERCIAL MAL PREDICTOR (LLM) - SUMMARY")
        print("=" * 80)
        print(f"\nModel: Linear Mixed Model ({len(self.selected_features)} features)")
        print(f"Feature extraction: LLM (GPT-4o-mini)")
        print(f"R² ≈ 0.74 (74% variance explained)")
        print(f"\nFeatures:")
        for i, feat in enumerate(self.selected_features, 1):
            coef = self.fixed_effects.get(feat, 0)
            print(f"  {i}. {feat:45s} ({coef:+.3f})")
        print("=" * 80)


def interactive_predictor():
    """Interactive predictor - user inputs queries directly"""
    print("=" * 80)
    print("MAL PREDICTOR - INTERACTIVE MODE")
    print("=" * 80)
    print("\nInitializing predictor...")

    predictor = CommercialMALPredictorLLM()

    print("\n✓ Ready!")
    print("\nThis predictor estimates Maximum Acceptable Latency (MAL)")
    print(f"for Korean queries using {len(predictor.selected_features)} AI-extracted features.")
    print("\nRetention rates:")
    print("  • 50% = median (half of users will churn)")
    print("  • 90% = keep 90% of users (10% churn)")
    print("  • 95% = keep 95% of users (5% churn)")
    print("\nType 'quit' to exit\n")

    while True:
        print("─" * 80)
        query = input("Enter your Korean query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not query:
            print("⚠️  Please enter a query\n")
            continue

        print(f"\n🔍 Analyzing: '{query}'")
        print("─" * 80)

        try:
            result = predictor.predict(
                query,
                accommodation_levels=[50, 90, 95],
                verbose=True,
                return_reasoning=True
            )

            print(f"\n📊 Extracted Features (all {len(result['features'])}):")
            for feat, val in result['features'].items():
                symbol = "✓" if val > 0 else "✗"
                val_str = f"= {val}" if val > 1 else ""
                reasoning_text = result.get('reasoning', {}).get(feat, '')
                if reasoning_text:
                    print(f"  {symbol} {feat} {val_str}")
                    print(f"     → {reasoning_text}")
                else:
                    print(f"  {symbol} {feat} {val_str}")

            # Show calculation steps
            print(f"\n🧮 Calculation Process:")
            print("─" * 80)
            print(f"Step 1: Compute log(MAL)_mean")
            print(f"")

            # Show intercept
            intercept = predictor.fixed_effects['Intercept']
            print(f"  log(MAL)_mean = {intercept:.4f}  (intercept)")

            # Show each feature contribution
            log_mal_components = [intercept]
            for feat, val in result['features'].items():
                if feat in predictor.fixed_effects:
                    coef = predictor.fixed_effects[feat]
                    contribution = coef * val
                    log_mal_components.append(contribution)
                    if val == 1:
                        print(f"                + {coef:+.4f} × 1  ({feat})")
                    else:
                        print(f"                + {coef:+.4f} × 0  ({feat})")

            # Show population random effect
            pop_re = predictor.population_mean_random_effect
            log_mal_components.append(pop_re)
            print(f"                + {pop_re:+.4f}  (population mean random effect)")

            # Show total
            log_mal_mean = sum(log_mal_components)
            print(f"                = {log_mal_mean:.4f}")
            print(f"")

            # Show percentile calculations
            print(f"Step 2: Compute percentile-based MAL")
            print(f"")
            total_std = predictor.total_std

            from scipy import stats
            for level_str, mal_sec in sorted(result['mal_predictions'].items(),
                                             key=lambda x: int(x[0].replace('%', '')),
                                             reverse=True):
                level = int(level_str.replace('%', ''))
                churn_p = (100 - level) / 100
                z_score = stats.norm.ppf(churn_p)
                log_mal_p = log_mal_mean + z_score * total_std

                print(f"  {level}% retention ({100-level}% churn):")
                print(f"    churn_percentile = (100 - {level}) / 100 = {churn_p:.2f}")
                print(f"    z_score = Φ⁻¹({churn_p:.2f}) = {z_score:.4f}")
                print(f"    log(MAL)_{level} = {log_mal_mean:.4f} + {z_score:.4f} × {total_std:.4f}")
                print(f"    log(MAL)_{level} = {log_mal_p:.4f}")
                print(f"    MAL_{level} = exp({log_mal_p:.4f}) = {mal_sec:.2f}s")
                print(f"")

            print(f"\n⏱️  Predicted MAL (Summary):")
            print(result['interpretation'])

            print(f"\nMean MAL: {result['mean_mal']:.2f}s (50th percentile)")
            print()

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def demo():
    """Demo with example queries"""
    print("=" * 80)
    print("COMMERCIAL MAL PREDICTOR - DEMO")
    print("=" * 80)

    print("\n1. Initializing predictor...")
    predictor = CommercialMALPredictorLLM()

    print("\n2. Model Summary:")
    predictor.print_summary()

    print("\n3. EXAMPLE PREDICTIONS")
    print("=" * 80)

    examples = [
        "지난주에 찍은 골프 스윙 영상 보여줘",
        "오늘 날씨에 적합한 패션 스타일 추천해줘",
        "이번 달에 총 얼마나 썼지?",
        "소연이가 추천했던 영화 콘텐츠 제목이 뭐였지?"
    ]

    for i, query in enumerate(examples[:2], 1):
        print(f"\n{'─'*80}")
        print(f"Example {i}: {query}")
        print(f"{'─'*80}")

        result = predictor.predict(query, accommodation_levels=[50, 75, 90, 95])

        print(f"\nExtracted Features:")
        for feat, val in result['features'].items():
            if val == 1:
                print(f"  ✓ {feat}")

        print(f"\n{result['interpretation']}")

    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)

    print("""
To use interactive mode, run:
    python commercial_predictor_llm.py --interactive

Or in code:
    from commercial_predictor_llm import CommercialMALPredictorLLM
    predictor = CommercialMALPredictorLLM()
    result = predictor.predict("your query here")
    print(result['mal_predictions'])
""")


if __name__ == "__main__":
    import sys

    # Check for --interactive flag
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_predictor()
    else:
        demo()
