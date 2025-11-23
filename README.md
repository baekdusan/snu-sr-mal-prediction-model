# SNU SR MAL Prediction Model

Maximum Acceptable Latency (MAL) prediction model for personal search queries using augmented features and machine learning.

## Overview

This project predicts the Maximum Acceptable Latency (MAL) for personal search queries by extracting rich linguistic, semantic, and task-related features from query text. The model uses a two-stage feature augmentation approach powered by GPT-5.1.

## Project Structure

```
mal-prediction-model/
├── baseline/                    # Baseline model implementation
│   ├── mal_prediction_model.py  # Baseline model using embeddings only
│   ├── train_model.py          # Training script
│   ├── rawdata.csv             # Raw query data
│   └── baseline_results.json   # Baseline performance results
│
├── augmented/                   # Augmented feature approach
│   ├── scripts/
│   │   ├── augment_features_v3.py          # Feature extraction script (2-stage)
│   │   ├── mal_prediction_model_augmented.py
│   │   └── train_model_augmented.py
│   ├── prompts/
│   │   └── augment_prompt_v2.md           # System prompt for feature extraction
│   ├── data/
│   │   ├── rawdata.csv
│   │   └── augmented_data_v3.csv          # Generated augmented features
│   ├── results/
│   │   ├── feature_schema_v3.json         # Feature schema definition
│   │   └── batch_responses_v3.json
│   └── archives/                          # Previous versions and experiments
│
└── README.md
```

## Features

### Augmented Feature Categories (30+ features)

The model extracts features across 8 categories:

- **TXT_*** (9 features): Text statistics (char count, word count, punctuation, etc.)
- **LING_*** (8 features): Linguistic features (query type, tense, negation, clause complexity)
- **SEM_*** (13 features): Semantic features (domain, entities, specificity, ambiguity)
- **TASK_*** (9 features): Task complexity (operations, filtering, modality types)
- **TEMP_*** (5 features): Temporal features (reference type, recency, time range)
- **INTENT_*** (5 features): User intent (urgency, priority, emotional tone)
- **INFO_*** (4 features): Information density (constraints, precision/recall requirements)
- **COMP_*** (4 features): Computational demand (search space, index complexity)

## Installation

```bash
# Clone the repository
git clone https://github.com/baekdusan/snu-sr-mal-prediction-model.git
cd snu-sr-mal-prediction-model

# Install dependencies
pip install pandas numpy scikit-learn openai
```

## Usage

### 1. Feature Augmentation

Extract rich features from raw queries using GPT-5.1:

```bash
cd augmented/scripts
export OPENAI_API_KEY='your-api-key'
python augment_features_v3.py
```

**Two-Stage Processing:**
- **Stage 1**: Analyzes sample queries and defines a consistent feature schema (30+ features)
- **Stage 2**: Applies the schema to all queries in batches of 50 with validation and retry logic

### 2. Model Training

Train the augmented model:

```bash
python train_model_augmented.py
```

### 3. Baseline Comparison

Run the baseline model (embeddings only):

```bash
cd ../../baseline
python train_model.py
```

## Model Performance

### Baseline (Embeddings Only)
- Best Model: Linear Regression
- Test R²: 0.536
- Test MAE: 9.42

### Augmented Features
- **58 features** extracted
- Combines embeddings (1536D) + LLM-extracted features (58)
- Improved interpretability through human-understandable features

## Key Components

### Feature Extraction (`augment_features_v3.py`)

- **GPT-5.1** for schema definition (accuracy)
- **GPT-5.1-mini** for batch processing (cost-effective)
- Batch size: 50 queries
- Max retries: 5 per batch
- Validation: Ensures schema alignment and data quality

### Feature Schema

Defined using Human Factors & Ergonomics principles:
- Cognitive load indicators
- Task complexity metrics
- User intent analysis
- Temporal context

## Technical Details

### Validation & Retry Logic

Each batch is validated for:
1. Row count match
2. All feature columns present
3. No extra columns
4. Column order consistency
5. Missing values < 50% per column
6. Query text matches original

Failed batches are retried up to 5 times automatically.

### Cost Optimization

- Schema definition: `gpt-5.1` (once)
- Batch processing: `gpt-5.1-mini` (repeated)
- Reduces cost while maintaining quality

## Citation

```
@misc{mal-prediction-2024,
  title={MAL Prediction Model for Personal Search Queries},
  author={Seoul National University SR Lab},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
