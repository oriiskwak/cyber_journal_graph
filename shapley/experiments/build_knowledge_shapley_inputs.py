"""Build knowledge controls and exact-Shapley coalition inputs for 10 videos.

Knowledge controls
  K0: timestamped video evidence only
  K1: K0 + source-backed mappings for actually supported elements only
  KSHAM: K0 + the same number of unrelated official elements

Shapley players are evidence bundles.  Each bundle contains the overlapping
segment-summary lines, one timestamped event, and only mappings supported by
that event.  All player-overlapping lines are removed from the coalition
baseline before subsets are added back, avoiding duplicate attribution.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import build_official_explanation_inputs as expl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = HERE / "official_pattern_pilot_10_dev.json"
ORIGINAL = HERE / "ksc2026_inputs_semantic_v2_dev.json"
PATTERNS = ROOT / "data/reference_fraud_kg/patterns_seed.json"
SOURCES = ROOT / "data/reference_fraud_kg/sources.json"
CONTROL_OUT = HERE / "knowledge_controls_10_dev.json"
SHAPLEY_OUT = HERE / "shapley_coalitions_10_dev.json"
TIME_RE = re.compile(r"^(\d+(?:\.\d+)?)초\s*\|")


def matched_only_block(match: dict, links: dict, events: dict,
                       pattern: dict, sources: dict) -> str:
    names = {idx: f"V{n}" for n, idx in enumerate(sorted(events), 1)}
    element_by_id = {e["id"]: e for e in pattern["elements"]}
    supported = []
    for element in match["elements"]:
        idxs = [idx for idx in links[element["element_id"]] if idx in events]
        if idxs:
            supported.append((element, idxs))
    lines = ["[공식 자료에서 실제 영상 근거와 대응된 요소만 표시]"]
    for source_id in pattern["source_ids"]:
        s = sources[source_id]
        lines.append(f"출처={s['organization']} | {s['title']} | {s['published']} | {s['url']}")
    for element, idxs in supported:
        loc = element_by_id[element["element_id"]]["locator"]
        lines.append(f"공식요소={element['action']} | 영상근거={','.join(names[i] for i in idxs)} | "
                     f"공식위치={loc}")
    if not supported:
        lines.append("실제 대응 요소 없음")
    return "\n".join(lines)


def sham_block(match: dict, events: dict, pattern_order: list[dict], sources: dict) -> str:
    supported_n = sum(e["status"] == "support" for e in match["elements"])
    target_pos = next(i for i, p in enumerate(pattern_order) if p["pattern_id"] == match["pattern_id"])
    sham = pattern_order[(target_pos + 1) % len(pattern_order)]
    names = [f"V{n}" for n in range(1, len(events) + 1)] or ["V1"]
    # Keep the surface header identical to K1 so the model cannot identify the
    # control from an explicit "sham" label.  Only the official elements differ.
    lines = ["[공식 자료에서 실제 영상 근거와 대응된 요소만 표시]"]
    for source_id in sham["source_ids"]:
        s = sources[source_id]
        lines.append(f"출처={s['organization']} | {s['title']} | {s['published']} | {s['url']}")
    for n, element in enumerate(sham["elements"][:supported_n]):
        lines.append(f"공식요소={element['action']} | 영상근거={names[n % len(names)]} | "
                     f"공식위치={element['locator']}")
    if supported_n == 0:
        lines.append("실제 대응 요소 없음")
    return "\n".join(lines)


def overlapping_lines(m2: str, event: dict) -> list[str]:
    result = []
    for line in m2.splitlines():
        match = TIME_RE.match(line)
        if match and event["t_start"] <= float(match.group(1)) <= event["t_end"]:
            result.append(line)
    return result


def baseline_without_players(m2: str, players: list[dict]) -> str:
    result = []
    for line in m2.splitlines():
        match = TIME_RE.match(line)
        if match:
            t = float(match.group(1))
            if any(p["event"]["t_start"] <= t <= p["event"]["t_end"] for p in players):
                continue
        result.append(line)
    return "\n".join(result)


def player_bundle(player: dict) -> str:
    lines = [f"[근거 {player['player_id']}]", *player["summary_lines"],
             "사건원문=" + expl.evidence_text(player["event"])]
    for action in player["official_actions"]:
        lines.append(f"공식자료대응={action['action']} | 출처={action['source']} | 위치={action['locator']}")
    for reason in player["counter_reasons"]:
        lines.append(f"반증맥락={reason}")
    return "\n".join(lines)


def build_players(m2: str, match: dict, links: dict, support_events: dict,
                  counters: list[dict], pattern: dict, sources: dict) -> list[dict]:
    element_meta = {e["id"]: e for e in pattern["elements"]}
    event_actions: dict[int, list[dict]] = {}
    for element in match["elements"]:
        for idx in links[element["element_id"]]:
            if idx not in support_events:
                continue
            source_id = pattern["source_ids"][0]
            event_actions.setdefault(idx, []).append({
                "action": element["action"], "locator": element_meta[element["element_id"]]["locator"],
                "source": f"{sources[source_id]['organization']}·{sources[source_id]['title']}"})
    counter_by_idx = {e["event_index"]: e for e in counters}
    candidate_events = dict(support_events)
    for e in counters:
        candidate_events[e["event_index"]] = e
    # Prefer events supporting multiple official elements, then counterevidence,
    # then chronological order.  Maximum four keeps exact enumeration at 16 calls/video.
    ranked = sorted(candidate_events, key=lambda idx: (
        -len(event_actions.get(idx, [])), -int(idx in counter_by_idx),
        candidate_events[idx]["t_start"], idx))[:4]
    players = []
    for n, idx in enumerate(ranked, 1):
        event = candidate_events[idx]
        players.append({"player_id": f"P{n}", "event_index": idx, "event": event,
                        "summary_lines": overlapping_lines(m2, event),
                        "official_actions": event_actions.get(idx, []),
                        "counter_reasons": counter_by_idx.get(idx, {}).get("reasons", [])})
    return players


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, default=PILOT)
    ap.add_argument("--original", type=Path, default=ORIGINAL)
    ap.add_argument("--patterns", type=Path, default=PATTERNS)
    ap.add_argument("--sources", type=Path, default=SOURCES)
    ap.add_argument("--control-out", type=Path, default=CONTROL_OUT)
    ap.add_argument("--shapley-out", type=Path, default=SHAPLEY_OUT)
    args = ap.parse_args()

    pilot_payload = json.loads(args.pilot.read_text(encoding="utf-8"))
    pilot = pilot_payload["records"]
    dataset_role = pilot_payload.get("dataset_role", "selected_evidence_analysis")
    original = {r["id"]: r for r in json.loads(args.original.read_text(encoding="utf-8"))["records"]}
    pattern_order = json.loads(args.patterns.read_text(encoding="utf-8"))["patterns"]
    patterns = {p["pattern_id"]: p for p in pattern_order}
    sources = {s["source_id"]: s for s in json.loads(args.sources.read_text(encoding="utf-8"))["sources"]}
    control_records, coalition_records, manifest = [], [], []
    for record in pilot:
        match = expl.choose_match(record)
        pattern = patterns[match["pattern_id"]]
        events, links = expl.representative_events(match)
        counters = expl.selected_counterevidence(record)
        m2 = original[record["id"]]["inputs"]["M2"]
        video = expl.video_block(events, counters)
        k0 = m2 + "\n\n" + video
        k1 = k0 + "\n\n" + matched_only_block(match, links, events, pattern, sources)
        ksham = k0 + "\n\n" + sham_block(match, events, pattern_order, sources)
        control_records.append({"id": record["id"], "source": record["source"],
            "gold_dev_only": record["gold_dev_only"],
            "inputs": {"K0": k0, "K1": k1, "KSHAM": ksham}})

        players = build_players(m2, match, links, events, counters, pattern, sources)
        baseline = baseline_without_players(m2, players)
        for mask in range(1 << len(players)):
            chosen = [p for i, p in enumerate(players) if mask & (1 << i)]
            evidence = baseline
            if chosen:
                evidence += "\n\n" + "\n\n".join(player_bundle(p) for p in chosen)
            coalition_records.append({"id": f"{record['id']}__s{mask:02x}",
                "source": record["source"], "gold_dev_only": record["gold_dev_only"],
                "inputs": {"SHAP": evidence}, "coalition_meta": {
                    "video_id": record["id"], "mask": mask, "n_players": len(players)}})
        manifest.append({"video_id": record["id"], "gold_dev_only": record["gold_dev_only"],
                         "target_pattern": match["pattern_id"], "players": players,
                         "empty_id": f"{record['id']}__s00",
                         "full_id": f"{record['id']}__s{(1 << len(players)) - 1:02x}"})

    args.control_out.write_text(json.dumps({"dataset_role": dataset_role,
        "conditions": {"K0": "video evidence only", "K1": "matched official elements only",
                       "KSHAM": "unrelated official elements, format control"},
        "records": control_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    args.shapley_out.write_text(json.dumps({"dataset_role": dataset_role,
        "condition": "SHAP", "manifest": manifest, "records": coalition_records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"controls={len(control_records)} saved={args.control_out}")
    print(f"coalitions={len(coalition_records)} saved={args.shapley_out}")
    for condition in ("K0", "K1", "KSHAM"):
        z = [len(r["inputs"][condition]) for r in control_records]
        print(f"{condition}: median_chars={statistics.median(z):.0f} max_chars={max(z)}")
    for m in manifest:
        print(f"{m['video_id']} players={len(m['players'])} subsets={1 << len(m['players'])}")


if __name__ == "__main__":
    main()
