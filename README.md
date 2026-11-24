# TEMPO 발표 자료

2026년 10월에 저장된 Qwen TEMPO/CYREN 실행 `vdc_reflect_full_qwen_seed42`의 분류 및 프롬프트 선택 기록과 발표용 시간별 화면 발췌다. 발표용 영상 10개는 **이 실행의 test 121개에서 뽑은 동일 영상**이다. 사후 근거 추출은 별도 서버에서 수행했다.

| 자료 | 파일 |
|---|---|
| 파이프라인, 모델, 프롬프트 및 가드 | [실행 설명](docs/tempo_run.md), [프롬프트](run/prompts.json), [보관 코드](code_snapshot/) |
| 실제 프롬프트 채택 기록 | [24회 선택](run/guard_updates.csv), [실행 로그 발췌 및 줄 번호](run/guard_log_extract.json) |
| 전체 평가 | [dev/test 지표](run/metrics.json), [영상별 판정](run/video_predictions.csv), [프레임별 점수](run/frame_scores.csv) |
| Qwen·InternVL 방법론별 성능 | [발표용 비교표 및 평가 조건](docs/method_comparison.md), [CSV](run/method_comparisons.csv), [원본 출처 및 SHA-256](run/method_comparisons.json) |
| 발표용 10개 | [영상 ID 및 207개 프레임 점수](run/presentation_cases.json) |
| 출처 및 무결성 | [원본 파일명, SHA-256, 실행 설정](run/provenance.json) |
| 사기·정상 각 1편의 1초 간격 캡처 | [타임라인](timeline/README.md) |
| 근거 그래프 검토 | [Cypher](graph/review.cypher), [스키마와 산출물 위치](graph/README.md) |

10개 사례는 TP 3편, TN 3편, FP 2편, FN 2편이다. 저장된 평균 점수 기준으로 TP·FP의 높은 점수 사례와 TN의 낮은 점수 사례를 골랐고, test의 FN 2편을 모두 포함했다. 따라서 전체 성능은 test 121개의 지표로 보고한다. `normal_`과 `abnormal_`은 파일명 접두사이며 평가 라벨은 결과 JSON의 `label` 필드다.

다른 서버에서 생산한 `temporal_observations.json`과 `tempo10_evidence_graph.json`은 후속 커밋 대상으로 남겨두었다. 이 저장소에는 발표용 화면 발췌, 저장 예측과 검토 쿼리가 있다.
