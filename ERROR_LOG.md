# MAL Prediction Model - Error Log & Solutions

프로젝트 진행 중 발생한 오류와 해결 방법을 정리한 문서

---

## 1. OpenAI API - Temperature Parameter Error

### 오류 내용
```
openai.BadRequestError: Error code: 400 - {'error': {'message': "Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported.", 'type': 'invalid_request_error', 'param': 'temperature', 'code': 'unsupported_value'}}
```

### 발생 위치
`augment_pipeline.py` - `call_openai()` 함수

### 원인
- `gpt-5-mini` 모델은 temperature 파라미터를 지원하지 않음
- 모든 모델에 동일하게 `temperature=0.7`을 적용하려고 시도

### 해결 방법
```python
def call_openai(system_prompt: str, user_message: str, model: str) -> str:
    kwargs = {
        "model": model,
        "messages": [...]
    }

    # gpt-5-mini는 temperature 미지원
    if model != MODEL_LOW:  # MODEL_LOW = "gpt-5-mini"
        kwargs["temperature"] = 0.7

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
```

### 교훈
- API 호출 시 모델별 지원 파라미터를 확인해야 함
- 조건부로 파라미터를 추가하는 방식으로 유연성 확보

---

## 2. CSV Parsing Error - Field Count Mismatch

### 오류 내용
```
pandas.errors.ParserError: Error tokenizing data. C error: Expected 43 fields in line 36, saw 44
```

### 발생 위치
- `augmented_data.csv` 파일 읽기 시
- 총 81개 행에서 field count 불일치 발생

### 원인
LLM(GPT-5.1, GPT-5-mini)이 CSV를 생성할 때 일관성 문제:
1. **Extra fields**: 36-65행, 156-225행 등 → 44-45개 필드 (1-2개 초과)
2. **Missing fields**: 110-127행, 246-253행 등 → 42개 필드 (1개 부족)

**근본 원인**:
- LLM이 배치별로 CSV를 생성하면서 일부 feature 컬럼을 누락하거나 중복 생성
- Feature Value Reference를 제공했음에도 불구하고 일관성 유지 실패
- 특히 categorical 값에 쉼표(,)가 포함되거나 quoting이 잘못된 경우 발생

### 해결 방법

#### Step 1: CSV 재병합
```python
# remerge_csv.py
# 배치 응답 파일에서 직접 CSV 블록 추출하여 재병합
pattern = r'```csv\n(.*?)\n```'
match = re.search(pattern, content, re.DOTALL)
```

#### Step 2: Field Count 수정
```python
# clean_csv_data.py
if len(row) < len(header):
    # Missing fields - 기본값으로 padding
    fixed_row = row + ['0'] * (len(header) - len(row))
elif len(row) > len(header):
    # Extra fields - truncate (heuristic)
    fixed_row = row[:len(header)]
```

### 결과
- 원본: 256행 → 파싱 실패 81행 → 유효 데이터 175행
- 수정 후: 256행 모두 복구 → `augmented_data_final.csv`

### 교훈 및 개선 방안

#### 1. **LLM CSV 생성의 한계**
- LLM은 구조화된 데이터(CSV) 생성에 취약함
- 특히 배치 처리 시 일관성 유지 어려움

#### 2. **근본적 해결책**
```python
# 향후 개선 방향
# 1. LLM이 JSON 형식으로 출력하도록 변경
output_format = "json"  # CSV 대신 JSON 사용

# 2. Pydantic 등으로 schema validation
from pydantic import BaseModel
class QueryFeatures(BaseModel):
    queries: str
    MAL: float
    QL_char: int
    # ... 모든 필드 정의

# 3. 생성 후 즉시 validation
for batch_response in responses:
    try:
        QueryFeatures(**batch_response)
    except ValidationError as e:
        # 재생성 또는 수정
```

#### 3. **중간 검증 추가**
```python
# augment_pipeline.py 개선안
def validate_csv_response(response: str, expected_fields: int) -> bool:
    csv_content = extract_csv_from_response(response)
    reader = csv.reader(csv_content.split('\n'))

    for i, row in enumerate(reader):
        if len(row) != expected_fields:
            print(f"⚠ Warning: Row {i} has {len(row)} fields, expected {expected_fields}")
            return False
    return True

# 배치 생성 직후 검증
if not validate_csv_response(response, 43):
    # 재시도 또는 에러 처리
```

---

## 3. Resume 기능 필요성

### 배경
- 파이프라인 실행 중 에러 발생 시 처음부터 재실행해야 함
- API 비용이 발생하므로 비효율적

### 해결 방법
```python
def check_existing_files() -> dict:
    """이미 생성된 파일 확인"""
    status = {
        'feature_spec': os.path.exists('feature_specification.md'),
        'batch_1': os.path.exists('batch_1_response.md'),
        'batches': {i: os.path.exists(f'batch_{i}_response.md') for i in range(2, 9)}
    }
    return status

def main(resume: bool = True):
    status = check_existing_files() if resume else {...}

    # 이미 존재하는 파일은 skip
    if status['feature_spec']:
        feature_spec = load_existing_feature_spec()
    else:
        feature_spec = step1_feature_design(all_rows)
```

### 효과
- 중간에 실패해도 이미 생성된 배치는 재사용
- API 비용 절감

---

## 4. Data Quality Issues

### 발견된 문제들

#### A. Placeholder 값 ('none')
- `embedding_cluster_id`: 57.4% (147/256)
- `media_type`: 39.1% (100/256)
- `embedding_complexity_score`: 25% (64/256)

**원인**: LLM이 해당 feature가 적용되지 않는 쿼리에 대해 'none' 할당

**처리 방법**:
- 'none'을 유효한 카테고리로 유지 (의미 있음)
- Tree-based 모델은 자동으로 처리 가능
- One-hot encoding 시 'none'도 하나의 컬럼으로 생성

#### B. Feature 중복 (73/256 rows)
- 동일한 feature 값을 가지지만 다른 MAL 값

**예시**:
```
"지난달에 운동 몇 번이나 했지?" → MAL: 12.72
"지난달에 걷기 몇 번이나 했지?" → MAL: 20.82
"지난달에 요가 몇 번이나 했지?" → MAL: 15.47
```

**판단**: 정상적인 데이터 변동성
- 같은 쿼리 패턴도 사용자마다 다른 기대치를 가질 수 있음
- 모델이 학습해야 할 노이즈/불확실성

---

## 5. 향후 예방 체크리스트

### Pipeline 실행 전
- [ ] API key 환경변수 설정 확인
- [ ] 모델별 지원 파라미터 확인
- [ ] 출력 형식(CSV/JSON) 결정
- [ ] Validation schema 정의

### Pipeline 실행 중
- [ ] 각 배치 생성 후 즉시 field count 검증
- [ ] 오류 발생 시 자동 재시도 로직
- [ ] 중간 파일 저장 (resume 가능하도록)

### Pipeline 실행 후
- [ ] 전체 데이터 파싱 테스트
- [ ] Missing value 확인
- [ ] Duplicate 확인
- [ ] Feature distribution 확인

---

## 6. 코드 개선 제안 (미래)

### A. 출력 형식 변경: CSV → JSON
```python
# data_generation_prompt.md 수정
OUTPUT FORMAT:
Return a JSON array where each element represents one query:
[
  {
    "queries": "...",
    "MAL": 12.34,
    "QL_char": 15,
    ...
  },
  ...
]
```

**장점**:
- Parsing 오류 없음
- Schema validation 가능
- 중첩 구조 지원

### B. Streaming validation
```python
def generate_with_validation(prompt, model):
    max_retries = 3
    for attempt in range(max_retries):
        response = call_openai(prompt, model)

        try:
            data = json.loads(response)
            # Validate schema
            for item in data:
                QueryFeatures(**item)
            return data
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    ```

### C. Feature 일관성 강화
```python
# Feature Specification에 validation rules 추가
{
  "feature_name": "QL_char",
  "type": "numeric",
  "validation": {
    "min": 0,
    "max": 1000,
    "required": True
  }
}
```

---

## Summary

| 오류 | 원인 | 해결 | 재발 방지 |
|------|------|------|-----------|
| Temperature error | 모델별 파라미터 차이 | 조건부 파라미터 추가 | API 문서 사전 확인 |
| CSV parsing error | LLM 일관성 부족 | 수동 수정 + validation | JSON 출력 + schema validation |
| Resume 기능 부재 | 처음부터 재실행 필요 | 체크포인트 시스템 | 기본 기능으로 포함 |
| Placeholder 값 | LLM 판단 | 유효한 카테고리로 유지 | 명시적 가이드라인 |
| Feature 중복 | 자연스러운 변동성 | 정상으로 간주 | N/A |

---

**작성일**: 2025-11-23
**버전**: 1.0
**다음 업데이트**: 모델 학습 후 추가 이슈 발생 시