# MAL Prediction Model

Maximum Acceptable Latency (MAL) prediction for Korean natural-language queries using LLM-augmented features.

## 1. Architecture at a Glance

- **Realtime Regressor (this repo root)**  
  - LLM-based feature augmentation (`docs/prompts/*`)  
  - Classical ML regression ensemble trained on `data/processed/augmented_data.csv`  
  - Interactive CLI predictor backed by `artifacts/models/best_improved_model.pkl`

- **Percentile LMM module (`projects/LMM_model/`)**  
  - Linear mixed-model workflow for participant-level percentile estimation  
  - Detailed documentation lives inside `projects/LMM_model/README.md`

## 2. Directory Layout

```
mal-prediction-model/
├── data/
│   ├── raw/rawdata.csv
│   ├── processed/augmented_data.csv
│   └── intermediate/
│       ├── batch_responses/        # GPT outputs per batch
│       └── archives/               # Historical backups
├── artifacts/
│   ├── models/best_improved_model.pkl
│   └── embeddings/embeddings.pkl
├── docs/
│   ├── feature_specification.md
│   ├── flowchart.md
│   ├── prompts/ (LLM instructions)
│   └── reports/ (error logs & update notes)
├── src/mal_pred/
│   ├── pipelines/augment.py        # Core augmentation pipeline
│   ├── training/realtime.py        # Multi-model trainer
│   ├── interfaces/interactive.py   # Interactive CLI logic
│   └── predictor.py                # MALPredictor class
├── scripts/
│   ├── augment.py                  # User-facing CLI wrapper
│   ├── train_realtime.py
│   └── predict_cli.py
├── projects/LMM_model/             # Percentile modeling workspace
├── requirements.txt
└── README.md
```

## 3. Getting Started

```bash
git clone <repo-url>
cd mal-prediction-model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # required for augmentation & inference
```

> The helper scripts automatically prepend `src/` to `PYTHONPATH`, so you can run them from the project root without extra setup.

## 4. End-to-End Workflow

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python scripts/augment.py` | Resumes the GPT-based feature augmentation pipeline. Add `--fresh` to discard partial progress. |
| 2 | `python scripts/train_realtime.py` | Compares multiple regressors (Ridge, tree ensembles, LightGBM/XGBoost) and persists the best model to `artifacts/models/`. |
| 3 | `python scripts/predict_cli.py` | Launches the interactive shell. Type `verbose` to toggle feature dumps or `explain` to force a one-off explanation. |

### Under the Hood

- `src/mal_pred/pipelines/augment.py` handles feature design, batch processing, schema validation, and MAL merge.
- `src/mal_pred/training/realtime.py` label-encodes categorical fields, evaluates >8 candidate models, and stores the top performer + metadata.
- `src/mal_pred/predictor.py` loads the saved artifact, calls GPT-5-mini with the feature spec/value reference, and feeds the final feature row to the regressor.

## 5. Data & Artifacts

| Location | Contents |
|----------|----------|
| `data/raw/rawdata.csv` | 256 labeled Korean queries (ground-truth MAL) |
| `data/processed/augmented_data.csv` | Feature-augmented dataset (queries + MAL + 60+ engineered fields) |
| `data/intermediate/batch_responses/` | Raw GPT outputs per batch (JSON + feature value reference) |
| `artifacts/models/best_improved_model.pkl` | Pickled estimator, encoders, metrics (refreshed via `train_realtime.py`) |
| `artifacts/embeddings/embeddings.pkl` | Optional semantic embeddings (currently unused to avoid overfitting) |

## 6. Documentation Hub

- `docs/feature_specification.md` — full feature dictionary + theory
- `docs/flowchart.md` — visual pipeline summary
- `docs/prompts/*.md` — reusable instructions for GPT agents
- `docs/reports/summary_updates.md` — chronological change log
- `projects/LMM_model/README.md` — percentile modeling deep-dive (mixed effects, diagnostics, API usage)

## 7. Troubleshooting Checklist

| Issue | Quick Fix |
|-------|-----------|
| Missing `OPENAI_API_KEY` | `export OPENAI_API_KEY=...` before running scripts. |
| `batch_x_response.md` not found | Re-run `python scripts/augment.py --fresh` to regenerate. |
| `best_improved_model.pkl` missing/outdated | Execute `python scripts/train_realtime.py`. |
| Predictor throws encoding errors | Ensure `docs/feature_specification.md` and `data/intermediate/batch_responses/batch_1_response.md` exist—they inform the LLM prompt. |
| Import errors when calling modules manually | Use the provided `scripts/*.py` wrappers or set `PYTHONPATH=$PWD/src`. |

## 8. Extending the Repo

1. Update `docs/feature_specification.md` + prompts → rerun augmentation.
2. Fork `src/mal_pred/training/realtime.py` to plug in new model families.
3. Wrap `mal_pred.predictor.MALPredictor` inside your own REST/gRPC service for production.
4. Explore `projects/LMM_model/` for percentile APIs, diagnostics, and future roadmap.

## 9. References & Credits

- Feature design grounded in cognitive load, acceptable-waiting-time, and task complexity literature.
- LLM augmentation powered by OpenAI GPT-5.1 / GPT-5-mini.
- Regressors built with scikit-learn, LightGBM, XGBoost (optional).
- Mixed-model pipeline built with statsmodels (`projects/LMM_model`).

즐거운 실험 되세요! 🚀
