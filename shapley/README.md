# new80 사건별 exact Shapley

`new80_official_shapley_addon`의 실험 코드를 원본 그대로 보관한다. 이 별도 실험에서는 Qwen3-8B-AWQ가 시간 구간별 요약과 사건 텍스트를 입력받는다. 같은 영상에서 사건 묶음을 넣고 뺀 모든 조합을 채점해 사건별 평균 한계 기여도를 계산한다. 여기서 계산하는 값은 입력 구성에 따른 **Qwen3-8B 점수의 변화**다. 발표용 영상 10편의 분석은 [TEMPO-10 Shapley 결과](tempo10/README.md)에 있다. 기존 TEMPO 점수의 집계 기여도와 Qwen3-8B 관찰 입력 기여도를 각각 제공한다.

| 단계 | 코드 |
|---|---|
| 기준 점수에서 설명 검토 사례를 골라 사건 조합 구성 | [build_shapley_31_inputs.py](experiments/build_shapley_31_inputs.py) |
| 공식 자료 통제군과 사건 텍스트 구성 | [build_knowledge_shapley_inputs.py](experiments/build_knowledge_shapley_inputs.py) |
| Qwen3-8B로 사건 조합별 점수 계산 | [eval_ksc_local.py](experiments/eval_ksc_local.py) |
| 정확 Shapley 값과 사건 제거 전후 지표 계산 | [analyze_shapley_31.py](experiments/analyze_shapley_31.py) |

## 실행

전체 추론에는 별도로 준비된 `new80_bundle`의 요약, 사건 그래프, 의미 속성, M2 입력과 Qwen3-8B-AWQ 모델이 필요하다. `build_ksc_inputs.py`, `eval_abc.py` 등 기존 `new80_bundle/experiments/` 모듈도 사용한다. [원본 실행 가이드](RUN_NEW80_OFFICIAL_SHAPLEY.md)의 `experiments/`와 `data/reference_fraud_kg/` 경로는 기존 `new80_bundle` 작업본을 기준으로 한다. 이 디렉터리의 동명 파일을 그 작업본에 배치한 뒤 가이드를 따른다.

이미 사건 조합과 점수 JSON이 있다면, 표준 라이브러리만으로 분석 코드를 직접 실행할 수 있다.

```bash
python shapley/experiments/analyze_shapley_31.py \
  --input /path/to/shapley_coalitions_new80.json \
  --eval /path/to/eval_shapley_new80_qwen3_8b.json \
  --out /path/to/analysis_shapley_new80.json
```

재현에 필요한 영상 원본, 판정 입력 및 결과, 모델 파일과 인증 정보는 별도로 관리한다. `SHA256SUMS`는 이 폴더의 추가 README를 제외한 원본 애드온 파일을 검증한다.
