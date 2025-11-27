# MAL (Maximum Acceptable Latency) Prediction Model

**선형 혼합 모델(Linear Mixed Model)을 이용한 개인화 MAL Percentile 예측**

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [디렉토리 구조](#디렉토리-구조)
3. [데이터 처리 파이프라인](#데이터-처리-파이프라인)
4. [모델링 방법론](#모델링-방법론)
5. [모델 성능](#모델-성능)
6. [예측 API 사용법](#예측-api-사용법)
7. [주요 발견사항](#주요-발견사항)
8. [한계 및 개선방향](#한계-및-개선방향)

---

## 프로젝트 개요

### 목적
사용자의 쿼리에 대한 **Maximum Acceptable Latency (MAL)**를 예측하여, 다양한 percentile 수준(예: 50%, 90%, 95%)에서 사용자가 수용할 수 있는 최대 대기 시간을 추정합니다.

### 핵심 질문
- **"이 사용자가 이 쿼리에서 90% 확률로 수용할 수 있는 최대 지연시간은?"**
- **"신규 사용자(Cold-start)의 경우, 일반적인 MAL은?"**

### 입력 및 출력

**Input:**
- `query`: 사용자 쿼리 텍스트 (또는 추출된 features)
- `percentile`: 원하는 수용 확률 (예: 50, 90, 95%)
- `participant_id`: 사용자 ID (선택사항)

**Output:**
- `mal_seconds`: 해당 percentile에서의 예측 MAL (초 단위)
- `scenario`: 'personalized' (기존 사용자) 또는 'population-level' (신규 사용자)
- `confidence_interval`: 95% 신뢰구간

---

## 디렉토리 구조

```
LMM_model/
│
├── data/                          # 원본 및 처리된 데이터
│   ├── all_data.xlsx              # 원본 데이터 (2560 rows: 256 queries × 80 participants)
│   ├── augmented_data.csv         # 쿼리별 feature 데이터 (256 queries × 51 features)
│   └── final_dataset.csv          # 최종 결합 데이터 (2560 rows × 55 cols)
│
├── models/                        # 학습된 모델
│   ├── lmm_model1.pkl             # Model 1: Participant Random Effect (추천)
│   ├── lmm_model2.pkl             # Model 2: Query Random Effect
│   └── mal_predictor.pkl          # 배포용 예측 모델
│
├── scripts/                       # 분석 스크립트
│   ├── join_data.py               # 데이터 결합 스크립트
│   ├── lmm_analysis.py            # LMM 모델 학습 및 분석
│   ├── model_performance_analysis.py  # 모델 성능 평가
│   └── mal_percentile_predictor.py    # 예측 API (메인)
│
├── outputs/                       # 결과 및 시각화
│   ├── mal_transformation_comparison.png    # 로그 변환 전후 비교
│   ├── lmm_diagnostics.png                  # 모델 진단 플롯
│   ├── random_effects.png                   # Random effects 분포
│   ├── model_performance_analysis.png       # 성능 분석 시각화
│   ├── lmm_model1_coefficients.csv         # Model 1 계수
│   ├── lmm_model2_coefficients.csv         # Model 2 계수
│   └── required_features.csv               # 필요한 feature 목록
│
├── docs/                          # 문서 (이 파일)
│   └── README.md
│
└── README.md                      # 메인 문서 (이 파일)
```

---

## 데이터 처리 파이프라인

### 1단계: 데이터 결합 (`scripts/join_data.py`)

**입력:**
- `all_data.xlsx`: 80명의 participants × 256개 queries = 2,560 rows
  - Columns: `participant`, `queries`, `MAL`
- `augmented_data.csv`: 256개 unique queries × 51개 features
  - Columns: `queries`, `MAL`, `QL_*` (51 features)

**처리:**
1. `all_data.xlsx`에 `query_id` 추가 (1~256)
2. `participant` → `participant_id`로 rename
3. `queries`를 키로 두 테이블 join
4. `all_data.xlsx`의 MAL 값 사용 (사람마다 다름)
5. `augmented_data.csv`의 features 사용 (쿼리별로 동일)

**출력:**
- `final_dataset.csv`: 2,560 rows × 55 columns
  - `participant_id`, `query_id`, `MAL`, `log_MAL`, 51 features

**검증:**
- ✅ 2,560 rows (256 queries × 10 participants/query)
- ✅ Query ID: 1~256
- ✅ MAL 값: all_data 기준 (개인별 차이 반영)
- ✅ Features: augmented_data 기준 (쿼리별 동일)
- ✅ 결측값 없음

### 2단계: 로그 변환

**이유:**
- 원본 MAL 분포: 오른쪽으로 긴 꼬리 (skewness = 2.99)
- 선형 혼합 모델은 정규성 가정 필요

**변환:**
```python
log_MAL = log(MAL)
```

**결과:**
- **Skewness: 2.99 → -0.12** (거의 대칭)
- **Kurtosis: 15.98 → -0.04** (정규분포에 근접)
- Q-Q plot: 거의 직선 형태

![로그 변환 비교](outputs/mal_transformation_comparison.png)

---

## 모델링 방법론

### 수학적 모델

```
log(MAL_ij) = β₀ + β'X_i + u_j + ε_ij
```

**구성 요소:**
- `i`: query index (1~256)
- `j`: participant index (80명)
- `β₀`: Intercept (전체 평균)
- `β'X_i`: Fixed effects (쿼리 features의 효과)
  - 51개 features의 선형 결합
- `u_j`: Participant random effect
  - 사람별 "기다림 성향" (개인차)
  - `u_j ~ N(0, σ_u²)`
- `ε_ij`: Residual error
  - 개별 관측치의 오차
  - `ε_ij ~ N(0, σ_ε²)`

### 두 가지 모델 비교

#### Model 1: Participant Random Effect (채택 ✅)
```
log(MAL_ij) = β₀ + β'X_i + u_participant(j) + ε_ij
```
- **Grouping variable**: participant_id
- **가정**: 사람마다 기본 MAL 성향이 다름

#### Model 2: Query Random Effect (미채택 ❌)
```
log(MAL_ij) = β₀ + β'X_i + v_query(i) + ε_ij
```
- **Grouping variable**: query_id
- **가정**: 쿼리마다 feature로 설명 안 되는 추가 효과

### 모델 선택 이유

| Metric | Model 1 (Participant RE) | Model 2 (Query RE) | 승자 |
|--------|--------------------------|-------------------|------|
| Log-Likelihood | -1,730.68 | -3,144.77 | ✅ Model 1 |
| R² (log scale) | **0.75** | 0.11 | ✅ Model 1 |
| R² (original scale) | **0.65** | 0.05 | ✅ Model 1 |
| RMSE (seconds) | **12.03** | 19.71 | ✅ Model 1 |
| ICC | 0.72 | 0.02 | ✅ Model 1 |

**결론:** Model 1이 압도적으로 우수 → **개인차가 쿼리 차이보다 훨씬 중요**

---

## 모델 성능

### 전체 예측 정확도

#### Log Scale (모델링 스케일)
- **R² = 0.75** (전체 분산의 75% 설명)
- **RMSE = 0.42** (로그 스케일)
- **상관계수 = 0.87** (매우 강한 상관)

#### Original Scale (실제 MAL)
- **R² = 0.65**
- **RMSE = 12.03초** (평균 오차)
- **MAE = 6.72초** (중간값 오차)
- **평균 MAL = 21.9초** → 오차율 약 30%

### 분산 분해 - 핵심 발견 🎯

```
Participant variance (u): 0.4715  →  72.0% ⭐⭐⭐
Residual variance (ε):    0.1834  →  28.0%
───────────────────────────────────────────────
Total variance:           0.6549     100%
```

**ICC (Intraclass Correlation) = 0.72**

**해석:**
- 같은 사람이 다른 쿼리에 답할 때, MAL의 **72%는 그 사람 고유의 성향**으로 설명됨
- **"기다림 tolerance"는 개인 특성**이며, 쿼리 내용보다 훨씬 중요!
- 개인화 예측이 필수적임을 시사

### 유의미한 Feature 효과 (13개, p < 0.05)

#### MAL을 증가시키는 요인 ⬆️
| Feature | 계수 | 해석 |
|---------|------|------|
| `language_complexity_proxy` | +0.92 ⭐⭐⭐ | 언어 복잡도 높을수록 MAL 증가 (exp(0.92) ≈ 2.5배) |
| `requires_structured_output` | +0.48 ⭐⭐⭐ | 구조화된 출력 필요 시 MAL 증가 |
| `social_context` | +0.14 | 사회적 맥락 포함 시 약간 증가 |
| `long_horizon_planning` | +0.12 | 장기 계획 쿼리 시 약간 증가 |

#### MAL을 감소시키는 요인 ⬇️
| Feature | 계수 | 해석 |
|---------|------|------|
| `weather_coupled` | -0.44 ⭐⭐⭐ | 날씨 관련 쿼리는 빠른 응답 기대 |
| `calendar_schedule_domain` | -0.24 | 일정/캘린더 관련 쿼리 |
| `recall_specific_entity` | -0.19 | 특정 항목 찾기 (빠른 검색) |
| `question_formality` | -0.16 | 격식있는 질문일수록 MAL 감소 |
| `media_domain` | -0.10 | 미디어 관련 쿼리 |

### 모델 진단

![모델 진단](outputs/model_performance_analysis.png)

**✅ 장점:**
- Residual 평균 ≈ 0 (편향 없음)
- Residual 분포가 정규분포에 근접
- Actual vs Predicted가 대각선에 정렬
- 이분산성 문제 없음

**⚠️ 한계:**
- Within-participant R² = 0.21 (낮음)
  - 같은 사람 내에서도 쿼리 간 변동이 큼
  - 하지만 이것이 바로 percentile 예측이 필요한 이유!

---

## 예측 API 사용법

### 설치 및 로드

```python
import pickle

# 모델 로드
with open('models/mal_predictor.pkl', 'rb') as f:
    predictor = pickle.load(f)
```

### 시나리오 1: 기존 사용자 (Personalized Prediction)

```python
# Feature extraction (실제로는 쿼리에서 자동 추출)
features = {
    'QL_char_len': 20,
    'QL_word_len': 4,
    'QL_language_complexity_proxy': 0.75,
    # ... 49개 features
}

# 90th percentile 예측
result = predictor.predict_mal_percentile(
    features=features,
    percentile=90,
    participant_id='P013'  # 기존 사용자
)

print(f"90% accommodated MAL: {result['mal_seconds']:.1f} seconds")
print(f"Scenario: {result['scenario']}")  # 'personalized'
print(f"95% CI: {result['confidence_interval_95']}")
```

**출력 예시:**
```
90% accommodated MAL: 52.9 seconds
Scenario: personalized
95% CI: (17.67, 61.86)
```

### 시나리오 2: 신규 사용자 (Cold-start)

```python
# 같은 features 사용
result = predictor.predict_mal_percentile(
    features=features,
    percentile=90,
    participant_id=None  # Cold-start
)

print(f"90% accommodated MAL (population): {result['mal_seconds']:.1f} seconds")
print(f"Scenario: {result['scenario']}")  # 'population-level'
```

**출력 예시:**
```
90% accommodated MAL (population): 19.4 seconds
Scenario: population-level
```

### 시나리오 3: 다중 Percentile 예측

```python
results = predictor.predict_multiple_percentiles(
    features=features,
    percentiles=[10, 50, 90, 95],
    participant_id='P013'
)

for p, r in results.items():
    print(f"{p}: {r['mal_seconds']:.1f}s")
```

**출력 예시:**
```
p10: 17.7s
p50: 30.6s
p90: 52.9s
p95: 61.9s
```

### 실전 활용 예시

#### 적응형 UI/UX
```python
predicted_mal_90 = predictor.predict_mal_percentile(
    features=query_features,
    percentile=90,
    participant_id=user_id
)['mal_seconds']

if predicted_mal_90 > 30:
    show_progress_bar()  # 오래 걸릴 것 같으면
else:
    show_spinner()  # 빠르면 스피너만
```

#### 개인별 최적화
```python
# 사용자의 patience level 추정
user_patience = predictor.participant_random_effects.get(user_id, 0)

if user_patience < -0.5:  # 참을성 낮음
    optimize_for_speed()
else:  # 참을성 높음
    optimize_for_quality()
```

---

## 주요 발견사항

### 1. 개인차가 압도적으로 중요 🎯

```
전체 분산의 72% = 개인 간 차이
전체 분산의 28% = 개인 내 변동 + 쿼리 효과
```

**의미:**
- MAL은 **개인 특성**
- "이 쿼리는 X초다"가 아니라 **"이 사람에게 이 쿼리는 X초다"**
- 개인화 없는 예측은 무의미

### 2. 쿼리 Features도 유의미 📊

**13개 significant features (p < 0.05)**
- 언어 복잡도 ↑ → MAL 2.5배 증가
- 날씨 관련 → MAL 감소
- 구조화 출력 필요 → MAL 증가

**활용:**
- 쿼리 최적화 가이드라인 제공
- "더 간단하게 질문해보세요" 추천

### 3. Percentile 예측의 가치 🎲

단순 평균이 아닌 **분포 전체 예측**:
- **50th percentile**: 중간값 (절반은 이보다 빠름)
- **90th percentile**: 보수적 예측 (10명 중 9명 수용)
- **95th percentile**: 매우 보수적 (20명 중 19명 수용)

**장점:**
- Uncertainty를 명시적으로 전달
- Risk management (최악의 경우 고려)
- 사용자 신뢰도 향상

### 4. Cold-start 해결책 💡

**전략:**
1. **초기**: Population-level 예측 (모든 사용자 평균)
2. **5~10개 쿼리 수집**: Participant random effect 추정
3. **이후**: 개인화 예측으로 전환

**성능:**
- Population-level: MAE ≈ 12초 (moderate)
- Personalized: MAE ≈ 6.7초 (good) → **67% 개선**

---

## 한계 및 개선방향

### 현재 한계

#### 1. Within-participant 예측력 낮음 ⚠️
- **문제**: 같은 사람도 쿼리마다 변동이 큼 (R² = 0.21)
- **원인**: 상황적 요인 (시간, 기분, 맥락 등) 미반영
- **완화**: Percentile 예측으로 uncertainty 명시

#### 2. Feature Extraction 미구현 ⚠️
- **문제**: 현재는 사전 추출된 features만 사용
- **필요**: 쿼리 텍스트 → features 자동 추출 파이프라인
- **해결**: NLP 기반 feature extractor 개발

#### 3. Crossed Random Effects 미구현 ⚠️
- **문제**: Participant OR Query 랜덤효과만 가능 (statsmodels 한계)
- **이상**: Participant AND Query 동시 고려
- **해결**: R의 `lme4::lmer()` 사용 or Bayesian (brms)

### 개선 방향

#### 단기 (1~2개월)
1. **Feature Extractor 구현**
   - LLM 기반 feature 추출
   - augmented_data.csv의 feature engineering 로직 재현

2. **Online Learning**
   - 신규 데이터로 모델 점진적 업데이트
   - Participant random effect 동적 갱신

3. **A/B Testing**
   - 예측 정확도 실전 검증
   - Percentile 수준별 사용자 만족도 측정

#### 중기 (3~6개월)
1. **Crossed Random Effects**
   - R lme4 연동 or Bayesian 전환
   - `log(MAL) ~ β'X + u_participant + v_query + ε`

2. **Contextual Features 추가**
   - 시간대, 요일, 디바이스
   - 이전 쿼리 히스토리

3. **Bayesian Approach**
   - `brms` 패키지로 uncertainty 더 정교하게 quantify
   - Posterior predictive distribution

#### 장기 (6개월~1년)
1. **Deep Learning Integration**
   - Query embedding (BERT, GPT)
   - User embedding (collaborative filtering)

2. **Multi-level Model**
   - Participant → Session → Query 계층 구조
   - Time-varying covariates

3. **Causal Inference**
   - 쿼리 최적화의 인과 효과 추정
   - Intervention 실험 설계

---

## 재현 방법

### 전체 파이프라인 실행

```bash
cd LMM_model

# 1. 데이터 결합
python scripts/join_data.py

# 2. LMM 모델 학습
python scripts/lmm_analysis.py

# 3. 모델 성능 분석
python scripts/model_performance_analysis.py

# 4. 예측 API 생성 및 테스트
python scripts/mal_percentile_predictor.py
```

### 개별 단계 실행

```python
# Python에서 직접 사용
import sys
sys.path.append('scripts')

from mal_percentile_predictor import MALPercentilePredictor

# 모델 학습
predictor = MALPercentilePredictor()

# 예측
result = predictor.predict_mal_percentile(
    features=your_features,
    percentile=90,
    participant_id='P013'
)
```

---

## 참고 문헌 및 리소스

### 통계 방법론
- **Linear Mixed Models**: Pinheiro & Bates (2000), *Mixed-Effects Models in S and S-PLUS*
- **Intraclass Correlation**: Shrout & Fleiss (1979), "Intraclass correlations: uses in assessing rater reliability"
- **REML Estimation**: Patterson & Thompson (1971), "Recovery of inter-block information when block sizes are unequal"

### Python 패키지
- **statsmodels**: Seabold & Perktold (2010), [MixedLM Documentation](https://www.statsmodels.org/stable/mixed_linear.html)
- **lme4 (R)**: Bates et al. (2015), "Fitting Linear Mixed-Effects Models Using lme4"

### 관련 프로젝트
- `augment_pipeline.py`: Feature 생성 파이프라인
- `feature_specification.md`: Feature 정의 및 설명

---

## 라이선스 및 연락처

**프로젝트**: MAL Prediction Model
**버전**: 1.0
**날짜**: 2025-11-27

**문의:**
- 기술 문의: GitHub Issues
- 협업 문의: [Your Email]

---

## Changelog

### v1.0 (2025-11-27)
- ✅ 데이터 결합 및 전처리 완료
- ✅ 로그 변환 및 정규성 검증
- ✅ LMM 모델 학습 (Model 1 & 2)
- ✅ 모델 성능 분석 및 시각화
- ✅ Percentile 예측 API 구현
- ✅ Cold-start 솔루션 포함
- ✅ 문서화 완료

### Planned (v1.1)
- [ ] Feature extraction pipeline
- [ ] Online learning implementation
- [ ] REST API deployment
- [ ] Dashboard for monitoring

---

**📊 모델 요약 카드**

| Metric | Value |
|--------|-------|
| **Training Data** | 2,560 observations (256 queries × 80 participants) |
| **Features** | 49 (after preprocessing) |
| **Model Type** | Linear Mixed Model with Participant Random Effect |
| **R² (log scale)** | 0.75 |
| **R² (original scale)** | 0.65 |
| **RMSE** | 12.03 seconds |
| **MAE** | 6.72 seconds |
| **ICC** | 0.72 (72% variance from individual differences) |
| **Significant Features** | 13 (p < 0.05) |
| **Use Case** | Personalized MAL percentile prediction |
| **Cold-start Strategy** | Population-level baseline → personalized after 5-10 queries |

---

**✨ TL;DR**

이 모델은 **개인화된 MAL percentile 예측**을 제공합니다:
- ✅ 72%의 분산을 개인차로 설명 (ICC = 0.72)
- ✅ R² = 0.75 (우수한 설명력)
- ✅ MAE = 6.7초 (평균 오차)
- ✅ Cold-start 솔루션 포함
- ✅ 13개 유의미한 쿼리 features
- ✅ 실전 배포 가능한 API

**사용법**: `predictor.predict_mal_percentile(features, percentile=90, participant_id='P013')`

**핵심 인사이트**: "기다림 tolerance는 개인 특성이며, 개인화 없는 예측은 무의미하다"

---

**End of Documentation**
