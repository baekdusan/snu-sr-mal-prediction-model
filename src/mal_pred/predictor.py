"""
실시간 쿼리 예측을 위한 MALPredictor 클래스.

- LLM을 이용해 새로운 쿼리를 feature space로 사상
- `artifacts/models/best_improved_model.pkl`에 저장된 모델로 MAL 예측
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from .config import (
    BATCH_RESPONSES_DIR,
    DATA_GEN_PROMPT_PATH,
    FEATURE_SPEC_PATH,
    MODEL_ARTIFACT_PATH,
)


class MALPredictor:
    """LLM feature generator + scikit-learn 회귀 모델을 묶은 래퍼."""

    def __init__(
        self,
        model_path: Path | None = None,
        feature_spec_path: Path = FEATURE_SPEC_PATH,
        feature_reference_path: Path | None = None,
        data_prompt_path: Path = DATA_GEN_PROMPT_PATH,
    ) -> None:
        self.model_path = model_path or MODEL_ARTIFACT_PATH
        self.feature_spec_path = feature_spec_path
        self.feature_reference_path = feature_reference_path or (
            BATCH_RESPONSES_DIR / "batch_1_response.md"
        )
        self.data_prompt_path = data_prompt_path

        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []
        self._label_encoders: dict[str, Any] = {}
        self._client: OpenAI | None = None

        self._feature_spec = self._safe_read(self.feature_spec_path)
        self._feature_reference = self._safe_read(self.feature_reference_path)
        self._prompt_template = self._safe_read(self.data_prompt_path)

        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _safe_read(self, path: Path | None) -> str:
        if path and path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _load_model(self) -> None:
        import pickle

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run the training script first."
            )

        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)

        self._model = bundle["model"]
        self._scaler = bundle.get("scaler")
        self._feature_names = bundle.get("feature_names", [])
        self._label_encoders = bundle.get("label_encoders", {})
        self._metrics = bundle.get("metrics", {})

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------
    def _call_llm(self, query: str) -> dict:
        system_prompt = self._prompt_template
        csv_text = 'queries\n"' + query.replace('"', "'") + '"'

        user_message = f"""You are now in **prediction mode**.
You must annotate exactly ONE query using the existing Feature Specification and Feature_Value_Reference.
Return ONLY a JSON array with a single object containing `queries` + all feature columns. Do NOT include MAL.

Feature Specification:
{self._feature_spec}

Feature_Value_Reference (from batch 1):
{self._feature_reference}

Dataset:
```csv
{csv_text}
```
"""

        response = self._ensure_client().chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        return self._extract_json(response.choices[0].message.content)[0]

    def _extract_json(self, content: str) -> list[dict]:
        pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError("LLM output에 JSON 블록이 없습니다.")
        return json.loads(match.group(1))

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_feature_frame(self, feature_dict: dict) -> pd.DataFrame:
        row = {col: feature_dict.get(col) for col in self._feature_names}
        df = pd.DataFrame([row])

        for col, encoder in self._label_encoders.items():
            if col not in df.columns:
                df[col] = encoder.transform([encoder.classes_[0]])[0]
                continue

            value = str(df.at[0, col])
            try:
                df.at[0, col] = encoder.transform([value])[0]
            except ValueError:
                fallback = encoder.classes_[0]
                df.at[0, col] = encoder.transform([fallback])[0]

        for col in df.columns:
            if col in self._label_encoders:
                continue
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                df[col] = 0

        df = df[self._feature_names]

        if self._scaler is not None:
            df = pd.DataFrame(self._scaler.transform(df), columns=self._feature_names)

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, query: str, verbose: bool = False) -> float:
        features = self._call_llm(query)
        df = self._build_feature_frame(features)
        mal_seconds = float(self._model.predict(df)[0])

        if verbose:
            self._print_feature_table(features, mal_seconds)

        return mal_seconds

    def predict_with_features(self, features: dict, verbose: bool = False) -> float:
        df = self._build_feature_frame(features)
        mal_seconds = float(self._model.predict(df)[0])
        if verbose:
            self._print_feature_table(features, mal_seconds)
        return mal_seconds

    def explain_prediction(self, query: str) -> float:
        features = self._call_llm(query)
        df = self._build_feature_frame(features)
        mal_seconds = float(self._model.predict(df)[0])
        self._print_feature_table(features, mal_seconds)
        return mal_seconds

    def _print_feature_table(self, features: dict, mal: float) -> None:
        print("\nFeature snapshot")
        for key in sorted(features.keys()):
            print(f"  - {key}: {features[key]}")
        print(f"\nPredicted MAL: {mal:.2f}s")
