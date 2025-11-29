"""
공통 경로 및 상수 정의.

모듈 어디에서든 동일한 기준 경로를 사용하도록 Path 객체를 한곳에서 관리한다.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Data layout
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "rawdata.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "augmented_data.csv"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
BATCH_RESPONSES_DIR = INTERMEDIATE_DIR / "batch_responses"
ARCHIVES_DIR = INTERMEDIATE_DIR / "archives"

# Artifact paths
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "models" / "best_improved_model.pkl"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings" / "embeddings.pkl"

# Documentation assets
DOCS_DIR = PROJECT_ROOT / "docs"
PROMPTS_DIR = DOCS_DIR / "prompts"
FEATURE_SPEC_PATH = DOCS_DIR / "feature_specification.md"
FLOWCHART_PATH = DOCS_DIR / "flowchart.md"

# Prompt files
FEATURE_DESIGN_PROMPT_PATH = PROMPTS_DIR / "feature_design_prompt.md"
DATA_GEN_PROMPT_PATH = PROMPTS_DIR / "data_generation_prompt.md"
DATA_AUGMENT_PROMPT_PATH = PROMPTS_DIR / "augment_prompt.md"

# Default models
MODEL_HIGH = "gpt-5.1"
MODEL_LOW = "gpt-5-mini"




