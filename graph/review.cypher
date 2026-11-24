// 사례별 원래 TEMPO 판정, 기준 라벨, 영상 점수
MATCH (v:Video)-[:HAS_TEMPO_PREDICTION]->(p:TempoPrediction),
      (v)-[:HAS_HUMAN_REVIEW]->(h:HumanReview)
WHERE v.id STARTS WITH 'tempo10:'
RETURN v.video_id AS video_id, h.case_type AS case_type, h.human_label AS reference_label,
       p.predicted_label AS prediction, p.scam_score AS TEMPO_score, p.threshold AS threshold
ORDER BY case_type, video_id;

// 사기·정상 발표 영상의 저장 점수와 시점별 사후 관찰
// 원문 OCR이나 엔티티 값의 표시 전에 마스킹된 산출물을 사용한다.
MATCH (v:Video)-[:HAS_TEMPO_PREDICTION]->(:TempoPrediction)-[:HAS_FRAME_SCORE]->(f:TempoFrameScore)
WHERE v.id STARTS WITH 'tempo10:'
  AND v.video_id IN ['abnormal_BU2FfjTqGY8', 'normal_qsHWozaXnnw']
MATCH (f)-[a:ALIGNS_WITH]->(e:Event)
OPTIONAL MATCH (txt:Evidence)-[:EVIDENCE_FOR]->(e)
OPTIONAL MATCH (e)-[:SHOWS]->(obj:VisualEvidence)
RETURN v.video_id AS video_id, f.second AS TEMPO_second, f.segment AS TEMPO_segment,
       f.scam_probability AS TEMPO_score, e.second AS observation_second,
       e.event_type AS observation_event_type, count(DISTINCT txt) AS screen_text_items,
       count(DISTINCT obj) AS visual_objects, a.distance_seconds AS alignment_gap_seconds
ORDER BY video_id, TEMPO_second;

// 그래프 뷰. 발표 화면에는 텍스트 속성이 가려진 인스턴스를 사용한다.
MATCH path=(v:Video)-[:HAS_TEMPO_PREDICTION]->(:TempoPrediction)
           -[:HAS_FRAME_SCORE]->(:TempoFrameScore)-[:ALIGNS_WITH]->(:Event)
WHERE v.id = 'tempo10:video:abnormal_BU2FfjTqGY8'
RETURN path;

// 적재 건수 검증
MATCH (n)
WHERE n.id STARTS WITH 'tempo10:'
RETURN labels(n) AS labels, count(*) AS total
ORDER BY total DESC;

MATCH ()-[r]->()
WHERE r.id STARTS WITH 'tempo10:'
RETURN type(r) AS relationship, count(*) AS total
ORDER BY total DESC;
