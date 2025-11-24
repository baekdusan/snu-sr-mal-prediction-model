# MAL Prediction Model

Maximum Acceptable Latency (MAL) prediction model for Korean natural language queries using LLM-augmented features and machine learning.

## Overview

This project predicts the Maximum Acceptable Latency (MAL) for natural language queries in Korean. The model uses theory-grounded features extracted from query text and augmented using GPT-5.1/GPT-5-mini to train regression models that predict how long users are willing to wait for a response.

## Project Structure

```
mal-prediction-model/
├── rawdata.csv                      # Original dataset (256 queries)
├── augmented_data.csv               # Feature-augmented dataset
├── augment_pipeline.py              # Data augmentation pipeline
├── improved_train.py                # Advanced model training script
├── quick_train.py                   # Quick baseline training
├── feature_design_prompt.md         # LLM prompt for feature design
├── data_generation_prompt.md        # LLM prompt for data generation
├── feature_specification.md         # Designed feature specifications
├── batch_responses/                 # LLM batch generation responses
│   ├── batch_1_response.md
│   ├── batch_2_response.md
│   └── ...
├── embeddings.pkl                   # Pre-computed embeddings
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Features

The model uses 60+ theory-grounded features including:

### Core Features
- **Query Length**: Character count, word count
- **Task Category**: Retrieve, summarize, generate, compare, recommend, etc.
- **Modality Type**: Text, photo, video, audio, calendar, finance, health, etc.
- **Temporal Scope**: Momentary, day/week, month, multi-month/year
- **Urgency Level**: Low (retrospective), medium (planning), high (now/today)
- **Personalization Depth**: Generic, light personal context, deep personal data mining
- **Task Requirements**: Aggregation, generation, historical search, external search
- **Output Expectations**: Cardinality (single/few/many items)
- **Stakes Importance**: Low (entertainment), medium (shopping), high (finance/ID)
- **Context**: Social context, device context, location context

### Domain-Specific Features
- Calendar/schedule related
- Finance/spending related
- Shopping/commerce related
- Entertainment/media related
- Health/fitness related
- Multi-source integration requirements

### Advanced Features
- Media transformation requirements
- Comparison tasks
- List ordering requirements
- Named person count
- Goal/target presence
- Emotional/preference inference

See [feature_specification.md](feature_specification.md) for complete feature definitions and theoretical rationale.

## Pipeline

### 1. Feature Design (GPT-5.1)
Analyzes all 256 queries to design theory-grounded, interpretable features for MAL prediction.

```bash
# Step 1 is included in augment_pipeline.py
```

### 2. Data Augmentation (GPT-5.1 + GPT-5-mini)
Batch processing with structure validation:
- Batch 1 (GPT-5.1): Rows 1-32 + Feature Value Reference
- Batches 2-8 (GPT-5-mini): Rows 33-256 using reference from Batch 1

```bash
python augment_pipeline.py
```

### 3. Model Training
Train multiple regression models with advanced feature engineering and hyperparameter tuning.

```bash
# Quick baseline training
python quick_train.py

# Advanced training with hyperparameter tuning
python improved_train.py
```

## Installation

### Requirements
- Python 3.8+
- OpenAI API key (for data augmentation)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd mal-prediction-model

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (for augmentation pipeline)
export OPENAI_API_KEY="your-api-key-here"
```

### Dependencies
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
openai>=1.0.0  # For augmentation pipeline
```

## Usage

### Quick Start

```bash
# Run the full pipeline (if starting from scratch)
python augment_pipeline.py
python improved_train.py
```

### Data Augmentation

The augmentation pipeline (`augment_pipeline.py`) supports resume mode:

```python
# Resume from existing progress (default)
python augment_pipeline.py

# Start from scratch
# Edit augment_pipeline.py: main(resume=False)
```

The pipeline automatically:
- Skips completed steps
- Validates JSON structure across batches
- Ensures field consistency
- Saves checkpoints for each batch

### Model Training

The improved training script (`improved_train.py`) includes:

**Feature Engineering:**
- Target encoding for categorical features
- Polynomial features for key numerical variables
- Interaction features (chars per word, complexity score, time complexity)
- Standard scaling

**Models Trained:**
- Ridge Regression (tuned)
- Lasso Regression (tuned)
- ElasticNet (tuned)
- Random Forest (tuned)
- LightGBM (tuned)
- XGBoost (tuned, optional)
- CatBoost (tuned, optional)
- Neural Network MLP (tuned)

**Hyperparameter Tuning:**
- RandomizedSearchCV with cross-validation
- Custom parameter grids for each model
- Optimized for R² score

## Results

The best model is automatically saved to `best_improved_model.pkl` with:
- Trained model
- Fitted scaler
- Feature names
- Performance metrics (MAE, RMSE, R²)

Example output:
```
MODEL COMPARISON (SORTED BY R²)
Model                          MAE          RMSE         R²
--------------------------------------------------------------------------------
Baseline (Mean)                X.XXXX       X.XXXX       X.XXXX
LightGBM (Tuned)              X.XXXX       X.XXXX       X.XXXX
XGBoost (Tuned)               X.XXXX       X.XXXX       X.XXXX
Random Forest (Tuned)         X.XXXX       X.XXXX       X.XXXX
...
```

## Data Format

### Input (rawdata.csv)
```csv
queries,MAL
"오늘 점심 메뉴 추천해줘",2.5
"지난달 카드 결제 내역 정리해서 보여줘",8.0
...
```

### Output (augmented_data.csv)
```csv
queries,MAL,QL_chars,QL_words,task_category,modality_type,...
"오늘 점심 메뉴 추천해줘",2.5,15,4,recommend_content,text_note,...
...
```

## Technical Details

### Augmentation Pipeline
- Uses JSON output format with structure validation
- Validates field consistency across batches
- No hardcoded schema - validates based on Batch 1's structure
- Automatic resume capability with checkpoint tracking

### Feature Engineering
- Target encoding to prevent data leakage
- Polynomial features (degree=2, interaction_only) for key variables
- Domain-specific interaction features
- Standard scaling for all features

### Model Selection
- Comprehensive model comparison
- Automated hyperparameter tuning
- Cross-validation for robust evaluation
- Best model selected by R² score

## Theoretical Foundation

Features are designed based on:
- **Task Complexity Theory**: Aggregation, generation, multi-source integration
- **Cognitive Load Theory**: Query length, output cardinality, specificity
- **Urgency and Temporal Theory**: Time scope, urgency level, recency
- **Personalization Theory**: Depth of personalization, social context
- **Stakes Theory**: Financial, identity, schedule importance
- **Modality Theory**: Text, media, calendar, finance, health domains

## Troubleshooting

### Common Issues

**OpenAI API Errors:**
- Ensure `OPENAI_API_KEY` is set correctly
- Check API rate limits
- Verify model availability (gpt-5.1, gpt-5-mini)

**Field Mismatch Errors:**
- The pipeline validates structure consistency automatically
- If errors occur, check `ERROR_LOG.md` for details
- Delete problematic batch files in `batch_responses/` and rerun

**Missing Dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**XGBoost/CatBoost Installation Issues:**
```bash
# macOS
brew install libomp
pip install xgboost catboost

# Linux
pip install xgboost catboost

# Windows
pip install xgboost catboost
```

## Performance Optimization

- Use `quick_train.py` for fast prototyping
- Adjust `n_iter` in RandomizedSearchCV to balance speed vs. accuracy
- Set `n_jobs=-1` to use all CPU cores
- Use `model="haiku"` for faster, cheaper augmentation (requires code modification)

## Contributing

To extend the project:

1. Add new features in `feature_specification.md`
2. Update prompts in `feature_design_prompt.md` and `data_generation_prompt.md`
3. Modify `augment_pipeline.py` to regenerate augmented data
4. Add new models in `improved_train.py`

## License

[Specify your license here]

## Contact

[Specify contact information here]

## References

- Feature design based on cognitive psychology and HCI latency research
- LLM augmentation using OpenAI GPT-5.1 and GPT-5-mini
- Machine learning models from scikit-learn, XGBoost, LightGBM, CatBoost

## Acknowledgments

- OpenAI for GPT-5.1 and GPT-5-mini models
- scikit-learn, XGBoost, LightGBM, CatBoost development teams
