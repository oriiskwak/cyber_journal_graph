# 발표용 TEMPO 근거 그래프

이 폴더의 두 JSON은 기존 TEMPO 10개 사례의 사후 관찰 결과를 공개 발표용으로 마스킹한 사본이다.

| 파일 | 내용 |
|---|---|
| [temporal_observations.json](temporal_observations.json) | 영상 10개의 207개 시점에 대한 사건 요약, 화면 문구, 객체, 엔티티 등 기존 관찰 결과 |
| [tempo10_evidence_graph.json](tempo10_evidence_graph.json) | 기존 영상별 판정·프레임 점수와 사후 관찰을 같은 시점에 연결한 Neo4j 적재용 JSON |
| [review.cypher](review.cypher) | 공개 사본을 적재한 환경에서 사례, 근거, 그래프 건수를 조회하는 예시 |

화면 문구, 엔티티 이름, 관찰 요약·설명에서 연락처와 개인 이름·계정 표기를 가렸다. 객체·엔티티 ID와 모든 관계 ID는 표시 문구를 포함하지 않는 값으로 바꿨으며, 관계의 양쪽 노드 참조도 함께 갱신했다. 발표 사례와 연결하기 위해 기존 영상 ID는 유지했다.

[발표 사례 데이터](../run/presentation_cases.json)와 대조해 영상 ID 10개, 관찰 시점과 프레임 점수 207개, 임계값 `0.28`이 일치하는 것을 확인했다. 마스킹한 그래프는 **노드 592개**, **관계 1,141개**이고, `Event`·`TempoFrameScore` 각 207개와 `ALIGNS_WITH` 207개를 포함한다. 관계 양쪽 참조와 Neo4j 로더 dry-run을 점검했다.

`TempoFrameScore`는 저장된 72B 분류기 출력이다. `Event`, `Evidence`, `Entity`, `VisualEvidence`, `Tactic`은 7B 모델의 사후 관찰이다. `ALIGNS_WITH`는 같은 영상의 같은 시점을 연결하므로 근거 검토에 사용할 수 있다. 발표용으로 직접 검토한 1초 단위 예시는 [타임라인](../timeline/README.md)을 참고한다.

공개 사본은 일부 ID가 원본과 다르므로, 실제로 적재할 때는 원본 그래프가 이미 있는 데이터베이스와 분리된 환경을 사용한다.
