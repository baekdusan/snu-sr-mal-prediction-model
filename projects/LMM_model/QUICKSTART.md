# MAL Predictor - Quick Start Guide

**쿼리 텍스트만으로 MAL 예측 - OpenAI GPT-4o-mini 사용!** ⚡

---

## 🎉 자동 Feature 추출

쿼리 텍스트만 입력하면 LLM(GPT-4o-mini)이 자동으로 51개 features를 추출합니다!

```python
query = "지난주에 찍은 골프 스윙 영상 보여줘"
features = extractor.extract_features(query)  # ✨ LLM이 자동 추출!
```

---

## 🚀 빠른 시작

### 0. 사전 요구사항

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY=your_api_key_here

# 필요한 패키지 설치
pip install openai
```

### 1. End-to-End 예측 (가장 간단!)

```python
import sys
sys.path.append('scripts')

from feature_extractor import QueryFeatureExtractor
from mal_percentile_predictor import MALPercentilePredictor
import pickle

# 모델 로드
with open('models/mal_predictor.pkl', 'rb') as f:
    predictor = pickle.load(f)

# Feature extractor 초기화
extractor = QueryFeatureExtractor()

# 쿼리 텍스트로 바로 예측!
query = "오늘 날씨에 적합한 패션 스타일 추천해줘"

# Step 1: Features 자동 추출
features = extractor.extract_features(query)

# Step 2: MAL 예측
result = predictor.predict_mal_percentile(
    features=features,
    percentile=90,
    participant_id='P013'  # or None for cold-start
)

print(f"90% 수용 MAL: {result['mal_seconds']:.1f}초")
```

### 2. 전체 데모 실행

```bash
cd LMM_model/scripts
export OPENAI_API_KEY=your_key
python end_to_end_demo.py
```

---

## 📖 상세 사용법

### Feature Extractor 사용

```python
from feature_extractor import QueryFeatureExtractor

extractor = QueryFeatureExtractor()

# 단일 쿼리
features = extractor.extract_features("이번 달에 총 얼마나 썼지?")

print(f"Extracted {len(features)} features:")
print(f"  Task type: {features['QL_task_type']}")
print(f"  Goal: {features['QL_goal_type']}")
print(f"  Financial domain: {features['QL_financial_domain']}")
# ... 51 features total

# 배치 처리
queries = [
    "지난주에 찍은 사진 보여줘",
    "오늘 날씨 추천",
    "소연이가 말했던 영화 제목"
]

results = extractor.extract_batch(queries)
for r in results:
    if r['success']:
        print(f"✓ {r['query']}: {len(r['features'])} features")
    else:
        print(f"✗ {r['query']}: {r['error']}")
```

### 통합 예측 파이프라인

```python
def predict_mal_from_text(query_text, user_id=None, percentile=90):
    """
    쿼리 텍스트로부터 MAL 예측

    Args:
        query_text: 사용자 쿼리 (한국어)
        user_id: 사용자 ID (optional)
        percentile: 원하는 percentile (default: 90)

    Returns:
        float: 예측된 MAL (초)
    """
    # Feature 추출
    features = extractor.extract_features(query_text)

    # MAL 예측
    result = predictor.predict_mal_percentile(
        features=features,
        percentile=percentile,
        participant_id=user_id
    )

    return result['mal_seconds']


# 사용 예시
mal_90 = predict_mal_from_text(
    "현재 위치에서 걸어서 갈 수 있는 맛집 추천해줘",
    user_id='P013',
    percentile=90
)

if mal_90 > 30:
    show_progress_bar()
else:
    show_spinner()
```

---

## 💡 실전 활용 예시

### 1. 적응형 UI/UX

```python
# 쿼리 받기
query = user_input()

# Feature 추출
features = extractor.extract_features(query)

# 90th percentile 예측
mal_90 = predictor.predict_mal_percentile(
    features=features,
    percentile=90,
    participant_id=current_user_id
)['mal_seconds']

# UI 결정
if mal_90 > 30:
    show_progress_bar(f"처리 중... (~{mal_90:.0f}초)")
elif mal_90 > 10:
    show_spinner("잠시만 기다려주세요...")
else:
    show_instant_response()
```

### 2. 쿼리 복잡도 안내

```python
features = extractor.extract_features(query)

if features['QL_language_complexity_proxy'] > 0.8:
    warn_user("복잡한 질문입니다. 간단히 다시 물어보시겠어요?")

if features['QL_requires_cross_modal']:
    inform_user("여러 데이터를 종합 분석 중입니다. 잠시만 기다려주세요.")
```

### 3. 다중 Percentile 제공

```python
features = extractor.extract_features(query)

results = predictor.predict_multiple_percentiles(
    features=features,
    percentiles=[50, 90, 95],
    participant_id=user_id
)

show_to_user(f"""
예상 대기 시간:
  • 보통: {results['p50']['mal_seconds']:.0f}초
  • 안전하게: {results['p90']['mal_seconds']:.0f}초
  • 최대: {results['p95']['mal_seconds']:.0f}초
""")
```

---

## 🔍 Feature 추출 상세

### LLM 기반 자동 추출

Feature extractor는 OpenAI GPT-4o-mini를 사용하여 `feature_specification.md`에 정의된 51개 features를 자동으로 추출합니다.

**장점:**
- ✅ 수동 입력 불필요
- ✅ 일관된 추출 품질
- ✅ 빠른 응답 (~1-2초)
- ✅ 저렴한 비용 (~$0.0001/query)

**전체 Feature 목록:**
```python
# outputs/required_features.csv 참조
import pandas as pd
features_list = pd.read_csv('outputs/required_features.csv')
print(features_list)
```

---

## ⚠️ 주의사항

### 1. API 키 필수

```bash
# 환경변수 설정
export OPENAI_API_KEY=sk-...

# 또는 Python에서
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
```

### 2. API 비용

- **GPT-4o-mini**: ~$0.0001/query (매우 저렴!)
- **Anthropic Claude**: ~$0.003/query (30배 비쌈)
- 배치 처리 시에도 부담 없음

### 3. 성능

- Feature 추출: ~1-2초/쿼리 (GPT-4o-mini)
- MAL 예측: ~0.01초
- **병목: Feature 추출** → 캐싱 권장

---

## 🆘 문제 해결

### Q1: "OPENAI_API_KEY not found" 에러
**A**: API 키를 환경변수로 설정하세요
```bash
export OPENAI_API_KEY=your_key_here
```

### Q2: Feature 추출 실패 시
**A**: 수동으로 features 제공 가능
```python
# Fallback: 수동 features
manual_features = {
    'QL_task_type': 'retrieve_item',
    'QL_goal_type': 'remember/recall',
    # ... provide manually
}

result = predictor.predict_mal_percentile(
    features=manual_features,
    percentile=90
)
```

### Q3: 캐싱 구현
**A**: 같은 쿼리 재사용
```python
# 캐싱 예시
feature_cache = {}

def get_features_cached(query):
    if query not in feature_cache:
        feature_cache[query] = extractor.extract_features(query)
    return feature_cache[query]
```

---

## 📊 성능 비교

### GPT-4o-mini vs Claude

| Metric | GPT-4o-mini | Claude Sonnet |
|--------|-------------|---------------|
| **비용/query** | ~$0.0001 ✅ | ~$0.003 |
| **속도** | ~1-2초 ✅ | ~2-3초 |
| **품질** | 매우 우수 ✅ | 매우 우수 |
| **안정성** | 높음 ✅ | 높음 |

**결론**: GPT-4o-mini가 비용 효율적이고 빠름!

---

## 📚 더 알아보기

- 전체 문서: [README.md](README.md)
- Feature 정의: `../feature_specification.md`
- 전체 데모: `python scripts/end_to_end_demo.py`

---

## 🎯 TL;DR

```python
# 3줄로 끝내는 MAL 예측
from feature_extractor import QueryFeatureExtractor
from mal_percentile_predictor import MALPercentilePredictor

extractor = QueryFeatureExtractor()  # OPENAI_API_KEY 필요
predictor = MALPercentilePredictor()

# 쿼리 → 예측
query = "오늘 날씨에 맞는 옷 추천해줘"
features = extractor.extract_features(query)  # GPT-4o-mini 자동 추출
result = predictor.predict_mal_percentile(features, percentile=90)
print(f"90% MAL: {result['mal_seconds']:.1f}초")
```

**끝!** 🎉

---

**Made with ❤️ using OpenAI GPT-4o-mini + Linear Mixed Models**
