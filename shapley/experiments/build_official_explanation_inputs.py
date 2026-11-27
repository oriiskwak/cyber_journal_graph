"""Build E0--E3 inputs for the 10-video official-pattern explanation pilot.

E0: segment summary
E1: E0 + timestamped video events and counterevidence
E2: E1 + official pattern/source mappings
E3: E2 after removing one central video event, its overlapping summary lines,
    and every official mapping supported only by that event
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = HERE / "official_pattern_pilot_10_dev.json"
ORIGINAL = HERE / "ksc2026_inputs_semantic_v2_dev.json"
PATTERNS = ROOT / "data/reference_fraud_kg/patterns_seed.json"
SOURCES = ROOT / "data/reference_fraud_kg/sources.json"
OUT = HERE / "official_explanation_inputs_10_dev.json"
TIME_RE = re.compile(r"^(\d+(?:\.\d+)?)초\s*\|")


def support_count(match: dict) -> int:
    return sum(e["required"] and e["status"] == "support" for e in match["elements"])


def choose_match(record: dict) -> dict:
    return sorted(record["matches"], key=lambda m: (
        -int(m["candidate"]),
        -(support_count(m) / m["required_total"]),
        -support_count(m), m["pattern_id"]))[0]


def representative_events(match: dict, max_per_element: int = 2) -> tuple[dict, dict]:
    """Return deduplicated events and element-to-event-index links."""
    events, links = {}, {}
    for element in match["elements"]:
        candidates = sorted(element["video_evidence"], key=lambda x: (
            x["t_start"], x["t_end"], x["event_index"]))
        if len(candidates) > max_per_element:
            candidates = [candidates[0], candidates[-1]]
        links[element["element_id"]] = []
        for event in candidates:
            idx = event["event_index"]
            events[idx] = event
            links[element["element_id"]].append(idx)
    return events, links


def evidence_text(event: dict) -> str:
    quotes = " / ".join(event["evidence"]) or event["summary"]
    fields = [f"{event['t_start']:g}~{event['t_end']:g}초", f"사건={event['event_type']}",
              f"원문={quotes}"]
    for key in ("channel", "claim", "context", "purpose", "authenticity"):
        value = event.get(key)
        if value not in (None, "none", "not_applicable"):
            fields.append(f"{key}={value}")
    return " | ".join(fields)


def selected_counterevidence(record: dict, limit: int = 3) -> list[dict]:
    seen, output = set(), []
    for event in sorted(record["counterevidence"], key=lambda x: (x["t_start"], x["event_index"])):
        if event["event_index"] in seen:
            continue
        seen.add(event["event_index"]); output.append(event)
        if len(output) == limit:
            break
    return output


def video_block(events: dict, counters: list[dict]) -> str:
    lines = ["[영상에서 추출한 시간 정박 사건]"]
    if events:
        for n, (_, event) in enumerate(sorted(events.items()), 1):
            lines.append(f"V{n}: {evidence_text(event)}")
    else:
        lines.append("공식 패턴 요소를 직접 지지하는 사건을 찾지 못함")
    lines.append("[영상 내 반증·정상 맥락 후보]")
    if counters:
        for n, event in enumerate(counters, 1):
            lines.append(f"C{n}: reasons={','.join(event['reasons'])} | {evidence_text(event)}")
    else:
        lines.append("추출된 반증 후보 없음")
    return "\n".join(lines)


def official_block(match: dict, links: dict, events: dict, pattern: dict,
                   source_by_id: dict) -> str:
    names = {idx: f"V{n}" for n, idx in enumerate(sorted(events), 1)}
    sources = [source_by_id[s] for s in pattern["source_ids"]]
    lines = ["[공식 수법 지식과 영상 사건의 대응]",
             f"공식 패턴={pattern['name']} ({pattern['pattern_id']})",
             "주의=부분 일치는 사기 확정이 아니며, 보이지 않는 요소는 미확인임"]
    for source in sources:
        lines.append(f"출처={source['organization']} | {source['title']} | "
                     f"{source['published']} | {source['url']}")
    pattern_elements = {e["id"]: e for e in pattern["elements"]}
    for element in match["elements"]:
        idxs = [idx for idx in links[element["element_id"]] if idx in events]
        status = "support" if idxs else "unconfirmed"
        refs = ",".join(names[i] for i in idxs) if idxs else "없음"
        locator = pattern_elements[element["element_id"]]["locator"]
        required = "필수" if element["required"] else "선택"
        lines.append(f"{element['element_id']}({required}) {element['action']} | "
                     f"상태={status} | 영상근거={refs} | 공식위치={locator}")
    return "\n".join(lines)


def remove_overlapping_summary(m2: str, event: dict) -> str:
    output = []
    removed = 0
    for line in m2.splitlines():
        match = TIME_RE.match(line)
        if match and event["t_start"] <= float(match.group(1)) <= event["t_end"]:
            removed += 1
            continue
        output.append(line)
    return "\n".join(output), removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, default=PILOT)
    ap.add_argument("--original", type=Path, default=ORIGINAL)
    ap.add_argument("--patterns", type=Path, default=PATTERNS)
    ap.add_argument("--sources", type=Path, default=SOURCES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))["records"]
    original = {r["id"]: r for r in json.loads(args.original.read_text(encoding="utf-8"))["records"]}
    patterns = {p["pattern_id"]: p for p in json.loads(args.patterns.read_text(encoding="utf-8"))["patterns"]}
    sources = {s["source_id"]: s for s in json.loads(args.sources.read_text(encoding="utf-8"))["sources"]}
    output = []
    for record in pilot:
        match = choose_match(record)
        pattern = patterns[match["pattern_id"]]
        events, links = representative_events(match)
        counters = selected_counterevidence(record)
        m2 = original[record["id"]]["inputs"]["M2"]
        e1_block = video_block(events, counters)
        e2_block = official_block(match, links, events, pattern, sources)

        # Centrality is the number of official elements supported by an event.
        frequencies = Counter(idx for idxs in links.values() for idx in idxs)
        drop_idx = sorted(events, key=lambda idx: (-frequencies[idx], events[idx]["t_start"], idx))[0]
        drop_event = events[drop_idx]
        remaining_events = {idx: e for idx, e in events.items() if idx != drop_idx}
        remaining_counters = [e for e in counters if e["event_index"] != drop_idx]
        e3_summary, removed_lines = remove_overlapping_summary(m2, drop_event)
        e3_video = video_block(remaining_events, remaining_counters)
        e3_official = official_block(match, links, remaining_events, pattern, sources)

        output.append({"id": record["id"], "source": record["source"],
                       "gold_dev_only": record["gold_dev_only"],
                       "inputs": {"E0": m2, "E1": m2 + "\n\n" + e1_block,
                                  "E2": m2 + "\n\n" + e1_block + "\n\n" + e2_block,
                                  "E3": e3_summary + "\n\n" + e3_video + "\n\n" + e3_official},
                       "explanation_meta": {"target_pattern": match["pattern_id"],
                           "candidate": match["candidate"], "drop_event_index": drop_idx,
                           "drop_interval": [drop_event["t_start"], drop_event["t_end"]],
                           "drop_event_type": drop_event["event_type"],
                           "removed_summary_lines": removed_lines}})

    payload = {"dataset_role": "development_only", "n": len(output),
               "conditions": {"E0": "segment summary", "E1": "E0 + timestamped video evidence",
                              "E2": "E1 + official pattern/source mapping",
                              "E3": "E2 with central video evidence and overlapping summary removed"},
               "records": output}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for condition in payload["conditions"]:
        sizes = [len(r["inputs"][condition]) for r in output]
        print(f"{condition}: median_chars={statistics.median(sizes):.0f} max_chars={max(sizes)}")
    for r in output:
        m = r["explanation_meta"]
        print(f"{r['id']} pattern={m['target_pattern']} drop={m['drop_event_type']} "
              f"{m['drop_interval']} lines={m['removed_summary_lines']}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
