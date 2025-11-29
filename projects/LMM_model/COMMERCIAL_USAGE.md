# Commercial MAL Predictor - Usage Guide

**Production-ready MAL prediction for any Korean query**

---

## 🎯 What This Does

Predicts **Maximum Acceptable Latency (MAL)** for Korean queries at different user retention levels.

**Key Concept - Retention vs Churn**:
- **50% retention** (50% churn) = Median MAL → half of users will leave
- **90% retention** (10% churn) = Higher MAL → keep 90% of users, only 10% leave
- **95% retention** (5% churn) = Even higher MAL → keep 95% of users, only 5% leave

**The higher the retention rate, the higher the MAL you need to design for.**

---

## 📦 Installation

```bash
pip install openai pandas numpy scipy statsmodels
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## 🚀 Quick Start

### **Method 1: Interactive Mode** (직접 입력)

```bash
cd projects/LMM_model/scripts
python commercial_predictor_llm.py --interactive
```

Then type your queries:
```
Enter your Korean query: 지난주에 찍은 골프 스윙 영상 보여줘

🔍 Analyzing: '지난주에 찍은 골프 스윙 영상 보여줘'
────────────────────────────────────────────────────────────────────────────────
  Extracting features: '지난주에 찍은 골프 스윙 영상 보여줘...'
  ✓ Features extracted

📊 Extracted Features:
  ✓ QL_media_domain

⏱️  Predicted MAL:
  • 50% retention (50% churn): 12.3s (keep 50% of users)
  • 90% retention (10% churn): 28.5s (keep 90% of users)
  • 95% retention (5% churn): 35.7s (keep 95% of users)

  Recommendation:
    → Design for 28.5s to keep 90% of users (10% churn)

Mean MAL: 12.3s (50th percentile)
```

### **Method 2: Python API**

```python
from commercial_predictor_llm import CommercialMALPredictorLLM

# Initialize (one time)
predictor = CommercialMALPredictorLLM()

# Predict for a query
query = "오늘 날씨에 적합한 패션 스타일 추천해줘"
result = predictor.predict(query, accommodation_levels=[50, 90, 95])

print(result['mal_predictions'])
# Output: {'50%': 10.5, '90%': 24.3, '95%': 30.5}

print(result['interpretation'])
# Output:
#   • 50% retention (50% churn): 10.5s (keep 50% of users)
#   • 90% retention (10% churn): 24.3s (keep 90% of users)
#   • 95% retention (5% churn): 30.5s (keep 95% of users)
#
#   Recommendation:
#     → Design for 24.3s to keep 90% of users (10% churn)
```

### **Method 3: Batch Processing**

```python
queries = [
    "지난주에 찍은 골프 스윙 영상 보여줘",
    "오늘 날씨에 적합한 패션 스타일 추천해줘",
    "이번 달에 총 얼마나 썼지?",
    "소연이가 추천했던 영화 콘텐츠 제목이 뭐였지?"
]

df_results = predictor.batch_predict(queries, accommodation_levels=[50, 90, 95])
df_results.to_csv('mal_predictions.csv', index=False)
```

---

## 📊 Understanding the Output

### **Retention Rate Interpretation**

| Retention | Churn | MAL Example | Meaning |
|-----------|-------|-------------|---------|
| **50%** | 50% | 12.3s | Half of users will leave if latency > 12.3s |
| **90%** | 10% | 28.5s | Only 10% will leave if latency ≤ 28.5s |
| **95%** | 5% | 35.7s | Only 5% will leave if latency ≤ 35.7s |

**Design Rule**:
- For **critical features**: Design for 95% retention (low churn tolerance)
- For **standard features**: Design for 90% retention
- For **experimental features**: Design for 50-75% retention

### **Feature Extraction**

The predictor automatically extracts 8 features using GPT-4o-mini:

1. `QL_requires_structured_output` - Needs list/table/formatted output
2. `QL_long_horizon_planning` - Long-term analysis (yearly, semi-annual)
3. `QL_calendar_schedule_domain` - Calendar, meetings, appointments
4. `QL_social_context` - Involves people, friends, groups
5. `QL_weather_coupled` - Weather-related
6. `QL_media_domain` - Music, movies, videos, podcasts
7. `QL_question_formality` - Polite question form vs imperative
8. `QL_recall_specific_entity` - "Where/when/what was it" queries

---

## 💡 Example Use Cases

### **Use Case 1: Mobile App Design**

```python
# Test critical user flow
query = "현재 위치에서 걸어서 갈 수 있는 맛집 추천해줘"
result = predictor.predict(query, accommodation_levels=[90, 95])

print(f"Design latency target: {result['mal_predictions']['90%']}s")
# Design for 90% retention to minimize churn
```

### **Use Case 2: A/B Testing**

```python
# Test different query phrasings
queries = [
    "음악 추천해줘",  # Short imperative
    "음악 추천해줄래?",  # Polite question
    "나한테 맞는 음악 리스트 만들어줘"  # Personalized + structured
]

for query in queries:
    result = predictor.predict(query, verbose=False)
    print(f"{query:40s} → 90% retention: {result['mal_predictions']['90%']}s")
```

### **Use Case 3: SLA Definition**

```python
# Define SLAs based on query type
query_types = {
    "urgent": "지금 여기 할인되는 카드 뭐야?",
    "standard": "이번 달에 총 얼마나 썼지?",
    "analytical": "올해 상반기에 가장 많이 산 카테고리 뭐야?"
}

for qtype, query in query_types.items():
    result = predictor.predict(query, verbose=False)
    print(f"{qtype:12s} SLA: {result['mal_predictions']['90%']:.1f}s (90% retention)")
```

---

## 🔧 Advanced Usage

### **Custom Retention Levels**

```python
result = predictor.predict(
    "쿼리",
    accommodation_levels=[10, 25, 50, 75, 90, 95, 99]
)
```

### **Silent Mode (no prints)**

```python
result = predictor.predict(query, verbose=False)
```

### **Access Raw Features**

```python
result = predictor.predict(query)

# See which features were detected
for feat, val in result['features'].items():
    if val == 1:
        print(f"✓ {feat}")
```

---

## ⚡ Performance

- **Feature extraction**: ~0.5-1s per query (GPT-4o-mini)
- **Model prediction**: <1ms
- **Total**: ~0.5-1s per query
- **Accuracy**: R² = 0.73 (73% variance explained)
- **Cost**: ~$0.0001 per query (GPT-4o-mini pricing)

---

## 🎓 Technical Details

### **Model**
- **Type**: Linear Mixed Model (LMM)
- **Features**: 8 (down from 51)
- **Random Effect**: Participant (accounts for individual differences)
- **Prediction**: Population-level (works for all users)

### **Assumptions**
- Log-normal distribution of MAL
- 70.5% of variance is due to individual differences
- 29.5% is due to query characteristics

### **Validation**
- Training data: 2,560 observations (80 participants × 256 queries)
- R² = 0.7277
- RMSE = 1.55 seconds
- All features: p < 0.001 (highly significant)
- No multicollinearity (VIF < 10)

---

## ❓ FAQ

**Q: What if my query is in English?**
A: The model is trained on Korean queries only. English queries may not work well.

**Q: Can I predict for a specific user?**
A: Currently, this is population-level prediction (average across all users). Individual prediction would require user-specific data.

**Q: What if I don't have OpenAI API key?**
A: You need an API key for LLM feature extraction. Alternative: Use rule-based extraction (see `commercial_predictor_final.py`).

**Q: How accurate is this?**
A: R² = 0.73 means the model explains 73% of variance. The remaining 27% is due to individual differences and noise.

**Q: Should I design for 90% or 95% retention?**
A:
- **90% (10% churn)**: Standard for most features
- **95% (5% churn)**: Critical features, low churn tolerance
- **50% (50% churn)**: Exploratory features, high churn acceptable

---

## 📝 Files

- `commercial_predictor_llm.py`: Main predictor (LLM-based)
- `models/lmm_model1_selected.pkl`: Trained model (8 features)
- `docs/feature_specification_selected.md`: Feature definitions

---

## 🚀 Production Deployment

```python
# Example: Flask API
from flask import Flask, request, jsonify
from commercial_predictor_llm import CommercialMALPredictorLLM

app = Flask(__name__)
predictor = CommercialMALPredictorLLM()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query = data.get('query')
    retention_levels = data.get('retention_levels', [50, 90, 95])

    result = predictor.predict(query, accommodation_levels=retention_levels, verbose=False)

    return jsonify({
        'query': query,
        'predictions': result['mal_predictions'],
        'mean_mal': result['mean_mal'],
        'interpretation': result['interpretation']
    })

if __name__ == '__main__':
    app.run(port=5000)
```

---

## 📞 Support

For issues or questions, refer to:
- Feature definitions: [feature_specification_selected.md](../docs/feature_specification_selected.md)
- Model training: `retrain_model1_selected.py`
- Feature analysis: `feature_selection_analysis.py`
