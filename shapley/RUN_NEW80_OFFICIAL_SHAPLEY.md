# 새 계정 80편: 공식 자료 통제와 사건별 exact Shapley

이 문서는 `new80_bundle`에서 요약·사건·속성·semantic-v2 판정 입력 생성이 끝난 뒤 실행한다.
개발 200편의 파일럿 결과를 복사해 재사용하지 않고 새 80편에서 입력을 새로 만든다.

## 동결할 모델 역할

- Gemma 4: 전체 영상 1회 요약과 시간 구간별 VLM 요약
- API 사건 추출 모델: 기존 200편과 동일한 모델·프롬프트로 사건 그래프 생성
- Qwen 3.5-27B: actor / target / purpose / authenticity 의미 속성 추출
- Qwen3-8B-AWQ: 아래 K0/K1/KSHAM 및 Shapley coalition의 최종 판정기

Qwen 3.5-27B와 Qwen3-8B는 역할이 다르다. 새 80편의 의미 속성 모델이나 프롬프트가 기존
200편과 다르면 두 집합을 합친 성능표를 만들지 말고 별도 실험으로 보고한다.

## 연구 질문

1. 올바르게 대응된 공식 자료가 단순히 공식 문구를 추가한 효과보다 더 유용한가?
2. 어떤 시간 정박 사건이 Qwen3-8B의 점수에 실제로 영향을 주었는가?

## 필수 입력 점검

```bash
cd ~/new80/new80_bundle
test -f data/summaries_temporal.json
test -f llm_graph/graph_events_temporal.json
test -f experiments/event_attributes.json
test -f experiments/semantic_event_attributes.json
test -f experiments/ksc2026_inputs_semantic_v2_new80.json
```

## A. 공식 자료가 실제 대응 때문에 도움이 되는가

### A1. 영상 사건과 공식 패턴의 label-free 대응

```bash
python experiments/match_official_patterns.py \
  --semantic experiments/semantic_event_attributes.json \
  --ids-from experiments/ksc2026_inputs_semantic_v2_new80.json \
  --out results/official_pattern_matches_new80.json
```

### A2. 공식 요소가 하나 이상 관찰된 영상만 선택

선택 과정은 정답 라벨을 사용하지 않는다. 라벨은 선택이 끝난 뒤 평가용으로만 붙는다.

```bash
python experiments/prepare_official_records.py \
  --matches results/official_pattern_matches_new80.json \
  --inputs experiments/ksc2026_inputs_semantic_v2_new80.json \
  --min-supported 1 \
  --out results/official_records_supported_new80.json
```

### A3. K0/K1/KSHAM 입력 생성

- K0: 시간 정박 영상 사건만
- K1: K0 + 실제로 대응된 공식 요소와 출처
- KSHAM: K0 + 같은 개수·형식의 무관한 공식 요소

주 비교는 `K1 vs KSHAM`이다. `K1 vs K0`만 비교하면 공식 문구를 추가한 프라이밍과 올바른
대응 효과를 구분할 수 없다.

```bash
python experiments/build_knowledge_shapley_inputs.py \
  --pilot results/official_records_supported_new80.json \
  --original experiments/ksc2026_inputs_semantic_v2_new80.json \
  --patterns data/reference_fraud_kg/patterns_seed.json \
  --sources data/reference_fraud_kg/sources.json \
  --control-out results/knowledge_controls_supported_new80.json \
  --shapley-out results/unused_supported_coalitions.json
```

### A4. Qwen3-8B 고정 판정

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/eval_ksc_local.py \
  --model models/Qwen3-8B-AWQ \
  --input results/knowledge_controls_supported_new80.json \
  --conditions K0 K1 KSHAM \
  --batch-size 4 --awq-backend torch_awq \
  --output results/eval_knowledge_controls_supported_new80_qwen3_8b.json
```

결과의 `K1_vs_KSHAM` AUC/AP 차이와 McNemar 검정을 우선 본다. 표본이 작으면 성능 확정 주장보다
각 영상의 `K1-KSHAM` 점수 변화와 오탐 사례를 보고한다.

## B. 사건별 exact Shapley

### B1. 같은 Qwen3-8B로 M2 점수 생성

이미 Qwen3-8B M2 결과가 있으면 이 단계는 생략한다.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/eval_ksc_local.py \
  --model models/Qwen3-8B-AWQ \
  --input experiments/ksc2026_inputs_semantic_v2_new80.json \
  --conditions M2 \
  --batch-size 4 --awq-backend torch_awq \
  --output results/eval_new80_qwen3_8b_m2.json
```

### B2. 감사 대상과 모든 사건 coalition 생성

감사 대상은 M2의 모든 오분류와 `0.1 <= score <= 0.9` 비포화 사례의 합집합이다. 이 선택은
설명 분석용이며 독립 분류 성능 표본으로 해석하지 않는다. 사건 player 선택에는 라벨을 사용하지 않는다.

```bash
python experiments/build_shapley_31_inputs.py \
  --eval results/eval_new80_qwen3_8b_m2.json \
  --inputs experiments/ksc2026_inputs_semantic_v2_new80.json \
  --original experiments/ksc2026_inputs_semantic_v2_new80.json \
  --matches results/official_pattern_matches_new80.json \
  --semantic experiments/semantic_event_attributes.json \
  --patterns data/reference_fraud_kg/patterns_seed.json \
  --sources data/reference_fraud_kg/sources.json \
  --out results/shapley_coalitions_new80.json
```

### B3. coalition 점수와 Shapley 분석

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/eval_ksc_local.py \
  --model models/Qwen3-8B-AWQ \
  --input results/shapley_coalitions_new80.json \
  --conditions SHAP \
  --batch-size 4 --awq-backend torch_awq \
  --output results/eval_shapley_new80_qwen3_8b.json

python experiments/analyze_shapley_31.py \
  --eval results/eval_shapley_new80_qwen3_8b.json \
  --input results/shapley_coalitions_new80.json \
  --out results/analysis_shapley_new80.json
```

주 지표는 efficiency error, top-event comprehensiveness, sufficiency, 상위 양의 사건 제거 후 오탐
교정률, 반증 사건의 음의 Shapley 비율이다. Shapley는 사건의 사실성을 증명하지 않으며 모델의 입력
민감도만 측정한다.

## C. 돌려보낼 파일

```bash
tar -czf new80_official_shapley_outputs.tar.gz \
  results/official_pattern_matches_new80.json \
  results/official_records_supported_new80.json \
  results/knowledge_controls_supported_new80.json \
  results/eval_knowledge_controls_supported_new80_qwen3_8b.json \
  results/eval_new80_qwen3_8b_m2.json \
  results/shapley_coalitions_new80.json \
  results/eval_shapley_new80_qwen3_8b.json \
  results/analysis_shapley_new80.json
```

`unused_supported_coalitions.json`은 전달하지 않아도 된다. 모델, 영상, API 키도 포함하지 않는다.
