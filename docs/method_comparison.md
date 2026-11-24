# Qwen·InternVL 방법론별 성능

기존 실행에서 저장된 test 수치를 백본별로 정리했다. 소수 네 자리로 반올림했으며, 전체 정밀도와 원본 파일명·SHA-256·결과 키는 [CSV](../run/method_comparisons.csv)와 [JSON](../run/method_comparisons.json)에 있다. AUC는 ROC-AUC이고 AP는 원본 `map` 필드다.

## Qwen2.5-VL-72B-AWQ

| 방법 | AUC | AP | F1@0.50 | Acc@0.50 |
|---|---:|---:|---:|---:|
| Vanilla | 0.3230 | 0.2835 | 0.4476 | 0.3417 |
| Vanilla + CoT | 0.4693 | 0.3505 | 0.4286 | 0.4667 |
| Baseline EM | 0.5979 | 0.4037 | 0.4176 | 0.5583 |
| Baseline EM ensemble | 0.6030 | 0.4029 | 0.5789 | 0.6000 |
| Temporal EM | 0.6745 | 0.4972 | 0.4884 | 0.6333 |
| Tool EM | 0.6926 | 0.4984 | 0.5806 | 0.6750 |
| 기존 TEMPO / CYREN | 0.6929 | 0.5423 | 0.5895 | 0.6750 |
| APO¹ | 0.4511 | 0.3266 | 0.4885 | 0.4417 |
| ERM¹ | 0.4479 | 0.3292 | 0.4655 | 0.4833 |
| TEMPO + Guard-only (12라운드) | 0.5235 | 0.3675 | 0.5128 | 0.5289 |
| TEMPO + Guard + Reflect (12라운드) | 0.8607 | 0.7773 | 0.7391 | 0.8017 |

¹ APO·ERM의 영상 점수는 프레임 평균 대신 사기 판정 프레임 비율이다. 보관 구현 기준으로 test 판정에는 해당 비율에 0.50을 적용했다. 과거 Qwen 요약 결과에는 실제 채점 영상 ID가 보존되지 않아 최신 121편의 test와 일대일 대조할 수 없다.

## InternVL2.5-78B-AWQ

| 방법 | AUC | AP | F1@0.50 | Acc@0.50 |
|---|---:|---:|---:|---:|
| Vanilla | 0.5000 | 0.3969 | 0.4483 | 0.4711 |
| Vanilla + CoT | 0.6627 | 0.5198 | 0.5556 | 0.4050 |
| Baseline EM | 0.6222 | 0.5387 | 0.5138 | 0.5620 |
| Baseline EM ensemble | 0.5620 | 0.4325 | 0.5000 | 0.5537 |
| Temporal EM | 0.5911 | 0.4396 | 0.5045 | 0.5455 |
| Tool EM | 0.5515 | 0.4335 | 0.4833 | 0.4876 |
| 기존 TEMPO / CYREN single | 0.6480 | 0.5190 | 0.5149 | 0.5950 |
| 기존 TEMPO / CYREN top-3 | 0.6874 | 0.5468 | 0.5652 | 0.6694 |
| TEMPO + Guard + Reflect (12라운드) | 0.7329 | 0.6537 | 0.5455 | 0.7107 |

과거 InternVL 요약 결과에도 채점 영상 ID가 없다. 확인 가능한 InternVL Guard-only, APO, ERM의 test 결과는 이 표에 없다. Qwen EvoPrompt의 보관 JSON에는 validation 점수만 있다. Fine-tuning 비교는 이 결과 묶음의 범위에 들어 있지 않다.

## 동일 test 여부와 별도 실행

- Qwen의 12라운드 Guard-only와 Guard + Reflect는 train/dev 영상 멤버십, test 영상 ID·라벨, test의 3,320개 영상·시점 쌍이 일치했다. 이 두 실행에서 test AP는 각각 **0.3675와 0.7773**이었다. Guard + Reflect 실행에서 Reflect 후보 채택은 0회였다. 한 번씩 수행한 12라운드 실행의 성능 차이로 Reflect의 독립 효과를 추정할 수 없다.
- 두 백본의 Guard + Reflect 결과는 각각 test 121편이지만 공통 영상은 **21편**이다. 공통 영상의 기준 라벨은 모두 일치했다. 위 표는 각 백본의 저장 결과를 보여주므로 두 백본의 AUC·AP 차이를 동일 test 기반 순위로 해석할 수 없다.
- Dev에서 선택한 임계값으로 본 test F1/Acc는 Qwen Guard-only `0.33 → 0.5693 / 0.5124`, Qwen Guard + Reflect `0.28 → 0.7273 / 0.7273`, InternVL Guard + Reflect `0.35 → 0.6078 / 0.6694`다. 발표 영상 10개는 계속 Qwen Guard + Reflect의 `0.28` 결과를 사용한다.

### Qwen 결정론적 3라운드 pilot

| 방법 | Test AUC | Test AP |
|---|---:|---:|
| Guard-only | 0.5739 | 0.3976 |
| Guard + Reflect | 0.6110 | 0.4131 |
| Guard + Reflect 반복 | 0.6116 | 0.4224 |
| Shadow-Reflect | — | — |

Shadow-Reflect의 보관 JSON에는 test 프레임 예측이 0건이고 평가 지표가 `null`이다. 표의 빈 칸은 계산 가능한 test 성능이 없다는 뜻이다.
