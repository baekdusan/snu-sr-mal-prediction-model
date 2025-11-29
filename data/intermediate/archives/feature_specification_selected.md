# Selected Features for MAL Prediction (Final - 8 Features)

**Last Updated**: 2025-11-28
**Model**: Linear Mixed Model (Model 1 - Participant Random Effect)
**Performance**: R² = 0.7277, RMSE = 1.55s

---

## Overview

This document specifies the **final 8 features** selected for the MAL prediction model after:
1. Statistical significance testing (p < 0.05)
2. Multicollinearity removal (VIF < 10)
3. Feature reduction (from 51 → 8 features, 84% reduction)

These features represent the optimal balance between:
- Statistical significance
- Interpretability
- Ease of extraction (LLM or rule-based)
- Low multicollinearity
- Commercial viability

---

## Final Feature Set (8 Features)

| **Feature Name** | **Type** | **Coefficient** | **P-value** | **Effect Direction** |
|---|---|---|---|---|
| `QL_requires_structured_output` | binary | +0.478 | 7.78e-37 | **Increases MAL** |
| `QL_long_horizon_planning` | binary | +0.174 | 4.21e-14 | **Increases MAL** |
| `QL_calendar_schedule_domain` | binary | -0.305 | 2.08e-11 | **Decreases MAL** |
| `QL_social_context` | binary | +0.153 | 1.61e-09 | **Increases MAL** |
| `QL_weather_coupled` | binary | -0.462 | 5.97e-06 | **Decreases MAL** |
| `QL_media_domain` | binary | -0.097 | 2.34e-05 | **Decreases MAL** |
| `QL_question_formality` | ordinal | -0.097 | 2.74e-05 | **Decreases MAL** |
| `QL_recall_specific_entity` | binary | -0.122 | 7.00e-05 | **Decreases MAL** |

---

## Feature Definitions

### 1. **QL_requires_structured_output** (BINARY)
**Definition**: Query explicitly requires structured output format
**Values**:
- `1` = Yes (structured output required)
- `0` = No (simple text output)

**Examples**:
- `1`: "가계부로 작성해줘", "쇼핑리스트 만들어줘", "리스트 만들어줘"
- `0`: "어디였지?", "언제였지?", "추천해줘"

**Rationale**: Structured outputs require additional formatting and organization, making users more tolerant of processing time.

**Extraction Logic**:
```python
if any(keyword in query for keyword in ["가계부", "리스트", "목록", "정리해줘", "작성해줘"]):
    return 1
else:
    return 0
```

---

### 2. **QL_long_horizon_planning** (BINARY)
**Definition**: Query involves long-term patterns or annual/half-year summaries
**Values**:
- `1` = Yes (long-term analysis)
- `0` = No (short-term or no time window)

**Examples**:
- `1`: "올해 상반기에 몇 번이나 샀지?", "지난해에 약속이 몇 번…", "올해 가장 많이…"
- `0`: "어제", "오늘", "이번 주", "지난주"

**Rationale**: Long-horizon analysis feels like "reporting/analytics", making users more tolerant of waiting time.

**Extraction Logic**:
```python
if any(keyword in query for keyword in ["올해", "지난해", "상반기", "하반기", "작년", "연간"]):
    return 1
else:
    return 0
```

---

### 3. **QL_calendar_schedule_domain** (BINARY)
**Definition**: Query is about calendar, schedule, appointments, meetings
**Values**:
- `1` = Yes (calendar/schedule related)
- `0` = No

**Examples**:
- `1`: "일정", "약속", "회의", "미팅", "세미나", "생일", "기념일"
- `0`: "사진", "음악", "쇼핑", "운동"

**Rationale**: Calendar queries are often time-critical (before a meeting), reducing acceptable latency.

**Extraction Logic**:
```python
if any(keyword in query for keyword in ["일정", "약속", "회의", "미팅", "세미나", "생일", "기념일", "스케줄"]):
    return 1
else:
    return 0
```

---

### 4. **QL_social_context** (BINARY)
**Definition**: Query involves named people, relationships, or group interactions
**Values**:
- `1` = Yes (social context present)
- `0` = No (individual query)

**Examples**:
- `1`: "소연이", "지민이", "친구들", "가족", "팀원들", "동료들", "단체 메신저", "재희랑"
- `0`: "내", "나", "나의", (no specific person mentioned)

**Rationale**: Socially contextual tasks may feel more complex (analyzing group chat, planning with friends), increasing MAL tolerance.

**Extraction Logic**:
```python
social_keywords = ["친구", "가족", "팀원", "동료", "단체", "그룹", "함께"]
# Also check for proper nouns (Korean names ending in 이/가/은/는/와/과)
if any(keyword in query for keyword in social_keywords) or has_proper_names(query):
    return 1
else:
    return 0
```

---

### 5. **QL_weather_coupled** (BINARY)
**Definition**: Query mentions weather as context
**Values**:
- `1` = Yes (weather-coupled)
- `0` = No

**Examples**:
- `1`: "오늘같은 날씨에 하면 좋을 운동", "오늘 날씨에 적합한 패션 스타일", "날씨가 좋은 날 갈 만한"
- `0`: (no weather mention)

**Rationale**: Weather-coupled tasks are highly context-dependent and often immediate (today/now), reducing acceptable latency.

**Extraction Logic**:
```python
if "날씨" in query:
    return 1
else:
    return 0
```

---

### 6. **QL_media_domain** (BINARY)
**Definition**: Query is about media content (music, videos, movies, TV, podcasts)
**Values**:
- `1` = Yes (media-related)
- `0` = No

**Examples**:
- `1`: "음악", "노래", "영화", "드라마", "유튜브", "영상", "팟캐스트", "라디오", "OTT"
- `0`: "사진" (photo/video is separate), "쇼핑", "운동"

**Rationale**: Media queries often expect quick recall ("어제 들은 노래 제목"), reducing MAL.

**Extraction Logic**:
```python
media_keywords = ["음악", "노래", "영화", "드라마", "유튜브", "영상", "팟캐스트", "라디오", "OTT", "넷플릭스"]
if any(keyword in query for keyword in media_keywords):
    return 1
else:
    return 0
```

---

### 7. **QL_question_formality** (ORDINAL)
**Definition**: Level of question formality
**Values**:
- `0` = Imperative without question ending ("…해줘")
- `1` = Polite question ("…해줄래?", "…뭐였지?", "…어디였지?")

**Examples**:
- `0`: "추천해줘", "보여줘", "만들어줘"
- `1`: "추천해줄래?", "뭐였지?", "어디였지?", "언제였지?"

**Rationale**: Polite/indirect forms correlate with exploratory or less time-pressured tasks, potentially increasing MAL.

**Extraction Logic**:
```python
if any(ending in query for ending in ["줄래", "였지", "었지", "나요", "까요"]):
    return 1
else:
    return 0
```

---

### 8. **QL_recall_specific_entity** (BINARY)
**Definition**: Asking "어디였지/언제였지/무엇이었지/제목이 뭐였지" about a specific past entity
**Values**:
- `1` = Yes (specific recall query)
- `0` = No

**Examples**:
- `1`: "어디였지?", "언제였지?", "제목이 뭐였지?", "이름이 뭐였지?"
- `0`: "추천해줘", "보여줘", "요약해줘"

**Rationale**: These feel like simple memory lookups, so users expect near-instant answers and lower MAL.

**Extraction Logic**:
```python
if any(pattern in query for pattern in ["었지", "였지", "제목이 뭐", "이름이 뭐", "어디였", "언제였"]):
    return 1
else:
    return 0
```

---

## Removed Features (Why They Were Excluded)

### **High Multicollinearity (VIF > 10)**
1. `QL_language_complexity_proxy` (VIF = 42.6) - Correlated with `QL_word_len`
2. `QL_output_size_expectation` (VIF = 33.3) - Correlated with `QL_requires_structured_output`
3. `QL_word_len` (VIF = 24.1) - Correlated with `QL_language_complexity_proxy`

### **Not Statistically Significant (p > 0.05)**
- 38 features were not significant (p > 0.05)
- Including: `QL_has_time_ref`, `QL_personalization_depth`, `QL_data_modalities`, etc.

---

## Feature Extraction Guidelines

### **For LLM-based Extraction**
Use this simplified prompt with GPT/Claude:

```
Extract the following 8 features from the Korean query:

1. QL_requires_structured_output (0/1): Does the query require structured output like lists, tables, or formatted documents?
2. QL_long_horizon_planning (0/1): Does the query involve long-term analysis (yearly, semi-annual)?
3. QL_calendar_schedule_domain (0/1): Is the query about calendar, schedule, meetings, appointments?
4. QL_social_context (0/1): Does the query involve specific people, friends, family, or groups?
5. QL_weather_coupled (0/1): Does the query mention weather?
6. QL_media_domain (0/1): Is the query about music, movies, videos, podcasts, or media content?
7. QL_question_formality (0/1): Is the query polite/question form (줄래, 었지, 였지) vs imperative (해줘)?
8. QL_recall_specific_entity (0/1): Is the query asking "where/when/what was it" about a specific past item?

Query: "{query}"

Return JSON:
{
  "QL_requires_structured_output": 0,
  "QL_long_horizon_planning": 0,
  "QL_calendar_schedule_domain": 0,
  "QL_social_context": 0,
  "QL_weather_coupled": 0,
  "QL_media_domain": 0,
  "QL_question_formality": 0,
  "QL_recall_specific_entity": 0
}
```

### **For Rule-based Extraction** (Faster, Cheaper)
Use keyword matching (see extraction logic above for each feature).

---

## Model Prediction Formula

```python
log_MAL = Intercept +
          0.478 × QL_requires_structured_output +
          0.174 × QL_long_horizon_planning +
         -0.305 × QL_calendar_schedule_domain +
          0.153 × QL_social_context +
         -0.462 × QL_weather_coupled +
         -0.097 × QL_media_domain +
         -0.097 × QL_question_formality +
         -0.122 × QL_recall_specific_entity +
          population_mean_random_effect

# For percentile p (e.g., 50%, 90%, 95%):
z_score = norm.ppf(p / 100)
log_MAL_percentile = log_MAL + z_score × total_std

# Transform back to seconds:
MAL_seconds = exp(log_MAL_percentile)
```

---

## Commercial Use Example

```python
# Query: "지난주에 찍은 골프 스윙 영상 보여줘"

features = {
    'QL_requires_structured_output': 0,  # No list/structure
    'QL_long_horizon_planning': 0,       # "지난주" is short-term
    'QL_calendar_schedule_domain': 0,    # Not calendar-related
    'QL_social_context': 0,              # No people mentioned
    'QL_weather_coupled': 0,             # No weather
    'QL_media_domain': 1,                # "영상" = video/media
    'QL_question_formality': 0,          # "보여줘" = imperative
    'QL_recall_specific_entity': 0       # Not asking "어디였지"
}

# Predict:
# 50% accommodation: 12.3 seconds
# 90% accommodation: 28.5 seconds
# 95% accommodation: 35.7 seconds
```

---

## Validation & Performance

- **Training data**: 2,560 observations (80 participants × 256 queries)
- **R² = 0.7277**: Model explains 72.8% of variance
- **RMSE = 1.55 seconds**: Average prediction error
- **ICC = 0.705**: 70.5% of variance due to individual differences
- **All features**: p < 0.001 (highly significant)
- **All VIF < 10**: No multicollinearity

---

## Changelog

**2025-11-28**: Initial selected feature set (8 features)
- Reduced from 51 → 8 features (84% reduction)
- Removed multicollinear features (VIF > 10)
- Removed non-significant features (p > 0.05)
- Model performance: R² = 0.7277

---

## References

- Original feature specification: `feature_specification.md`
- Model training: `retrain_model1_selected.py`
- Feature selection analysis: `feature_selection_analysis.py`
