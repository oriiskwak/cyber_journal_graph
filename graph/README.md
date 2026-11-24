# Neo4j 근거 그래프 검토

별도 서버에서 생성한 `temporal_observations.json`과 `tempo10_evidence_graph.json`을 후속 커밋에서 이 디렉터리에 둘 수 있다. 발표용으로 내보낼 때는 화면 문구뿐 아니라 엔티티 이름, 관찰 설명, 노드 ID와 관계 ID에 들어간 연락처·계정·민감 정보도 가려야 한다. 현재 저장소에는 검토용 Cypher와 저장 예측이 담겨 있다.

다른 서버의 검증 보고 기준으로 10개 영상에 Event 및 TempoFrameScore 각 207개와 ALIGNS_WITH 207개가 생성됐다. 전체 592개 노드와 1,141개 관계가 보고됐다. 이 수치는 다른 서버의 적재 전 점검이며, Aura에서의 연결이나 실제 적재 여부는 여기서 확인하지 않았다.

Neo4j Aura의 Query에 [review.cypher](review.cypher)의 쿼리를 한 블록씩 실행한다. `TempoFrameScore`의 점수는 원래 72B 분류기의 출력이다. `Event`, `Evidence`, `Entity`, `VisualEvidence`, `Tactic`은 7B 근거 모델의 사후 관찰이다. 발표용 객체 범주는 별도 검토 자료에서 확인한다. [타임라인](../timeline/README.md)
