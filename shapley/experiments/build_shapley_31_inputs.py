"""Build exact-Shapley coalitions for all M2 errors plus non-saturated cases.

Selection (development-only analysis):
  - every M2 error at threshold 0.5
  - every video with 0.1 <= M2 score <= 0.9

Player construction is label-free.  Up to four timestamped event bundles are
chosen from official matches, counterevidence, and the frozen graph selector.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import build_ksc_inputs as base
import build_knowledge_shapley_inputs as ks
import build_official_explanation_inputs as expl
import eval_abc as ab
import match_official_patterns as matcher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL = HERE / "eval_relation_controls_qwen3_8b_awq.json"
INPUTS = HERE / "ksc2026_inputs_relation_controls_dev.json"
ORIGINAL = HERE / "ksc2026_inputs_semantic_v2_dev.json"
MATCHES = HERE / "official_pattern_matches_dev.json"
SEMANTIC = HERE / "semantic_event_attributes.json"
PATTERNS = ROOT / "data/reference_fraud_kg/patterns_seed.json"
SOURCES = ROOT / "data/reference_fraud_kg/sources.json"
OUT = HERE / "shapley_coalitions_31_dev.json"
TIME_RE = re.compile(r"^(\d+(?:\.\d+)?)초\s*\|")


def official_actions(record: dict, patterns: dict, sources: dict) -> dict[int, list[dict]]:
    output: dict[int, list[dict]] = defaultdict(list)
    seen = set()
    for match in record["matches"]:
        pattern = patterns[match["pattern_id"]]
        meta = {e["id"]: e for e in pattern["elements"]}
        source_id = pattern["source_ids"][0]
        source = sources[source_id]
        for element in match["elements"]:
            if element["status"] != "support":
                continue
            for evidence in element["video_evidence"]:
                idx = evidence["event_index"]
                key = (idx, element["action"], source_id)
                if key in seen:
                    continue
                seen.add(key)
                output[idx].append({"action": element["action"],
                    "locator": meta[element["element_id"]]["locator"],
                    "source": f"{source['organization']}·{source['title']}"})
    return output


def choose_indices(events: list[base.Event], official: dict, counters: dict) -> list[int]:
    selected, _ = base.select(events)
    candidates = set(selected) | set(official) | set(counters)
    if not candidates:
        candidates = set(range(min(4, len(events))))

    def rank(idx: int):
        event = events[idx]
        risk = max(base.cue_score(event), base.action_score(event), 0.0)
        return (-len(official.get(idx, [])), -risk, event.t_start, idx)

    chosen = []
    # Preserve one explicit counterexample when present.
    if counters:
        chosen.append(sorted(counters, key=lambda i: (
            -len(counters[i].get("reasons", [])), events[i].t_start, i))[0])
    for idx in sorted(candidates, key=rank):
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) == 4:
            break
    return sorted(chosen, key=lambda i: (events[i].t_start, i))


def fallback_summary_players(m2: str) -> list[dict]:
    timed = []
    for line in m2.splitlines():
        match = TIME_RE.match(line)
        if match:
            timed.append((float(match.group(1)), line))
    if not timed:
        raise ValueError("no graph events and no timestamped summary lines")
    k = min(4, len(timed))
    positions = [round(i * (len(timed)-1) / (k-1)) for i in range(k)] if k > 1 else [0]
    players = []
    for n, pos in enumerate(dict.fromkeys(positions), 1):
        timestamp, line = timed[pos]
        event = {"event_index": -n, "event_id": f"summary-{n}",
            "event_type": "SEGMENT_OBSERVATION", "t_start": timestamp, "t_end": timestamp,
            "summary": line, "evidence": [line], "channel": "none", "claim": "none",
            "context": "none", "purpose": "none", "authenticity": "not_applicable"}
        players.append({"player_id": f"P{n}", "event_index": -n, "event": event,
            "summary_lines": [line], "official_actions": [], "counter_reasons": [],
            "fallback": "no_graph_event"})
    return players


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=EVAL)
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--original", type=Path, default=ORIGINAL)
    ap.add_argument("--matches", type=Path, default=MATCHES)
    ap.add_argument("--semantic", type=Path, default=SEMANTIC)
    ap.add_argument("--patterns", type=Path, default=PATTERNS)
    ap.add_argument("--sources", type=Path, default=SOURCES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    evaluation = json.loads(args.eval.read_text(encoding="utf-8"))["results"]
    input_payload = json.loads(args.inputs.read_text(encoding="utf-8"))
    base_inputs = input_payload["records"]
    labels = {r["id"]: r["gold_dev_only"] for r in base_inputs}
    original = {r["id"]: r for r in json.loads(args.original.read_text(encoding="utf-8"))["records"]}
    matched = {r["id"]: r for r in json.loads(args.matches.read_text(encoding="utf-8"))["records"]}
    semantic = json.loads(args.semantic.read_text(encoding="utf-8"))["records"]
    patterns = {p["pattern_id"]: p for p in json.loads(args.patterns.read_text(encoding="utf-8"))["patterns"]}
    sources = {s["source_id"]: s for s in json.loads(args.sources.read_text(encoding="utf-8"))["sources"]}

    chosen = []
    for vid, result in evaluation.items():
        score = result["M2"]["fraud_score"]
        gold = labels[vid] == "사기"
        error = (score >= 0.5) != gold
        boundary = 0.1 <= score <= 0.9
        if error or boundary:
            outcome = "TP" if gold and score >= 0.5 else "FN" if gold else "FP" if score >= 0.5 else "TN"
            chosen.append((vid, score, error, boundary, outcome))
    chosen.sort(key=lambda x: x[0])
    graph_events = base.load_events([x[0] for x in chosen])
    coalition_records, manifest = [], []
    for vid, score, error, boundary, outcome in chosen:
        rows, sem = graph_events[vid], semantic.get(vid, [])
        if len(rows) != len(sem):
            raise ValueError(f"semantic/event mismatch {vid}")
        official = official_actions(matched[vid], patterns, sources)
        counter_list = matcher.counterevidence(rows, sem)
        counters = {e["event_index"]: e for e in counter_list}
        m2 = original[vid]["inputs"]["M2"]
        if rows:
            indices = choose_indices(rows, official, counters)
            players = []
            for n, idx in enumerate(indices, 1):
                event = matcher.event_record(idx, rows[idx], sem[idx])
                players.append({"player_id": f"P{n}", "event_index": idx, "event": event,
                    "summary_lines": ks.overlapping_lines(m2, event),
                    "official_actions": official.get(idx, []),
                    "counter_reasons": counters.get(idx, {}).get("reasons", [])})
        else:
            players = fallback_summary_players(m2)
        if not players:
            raise ValueError(f"no Shapley player for {vid}")
        baseline = ks.baseline_without_players(m2, players)
        for mask in range(1 << len(players)):
            subset = [p for i, p in enumerate(players) if mask & (1 << i)]
            evidence = baseline
            if subset:
                evidence += "\n\n" + "\n\n".join(ks.player_bundle(p) for p in subset)
            coalition_records.append({"id": f"{vid}__s{mask:02x}",
                "source": original[vid]["source"], "gold_dev_only": labels[vid],
                "inputs": {"SHAP": evidence}, "coalition_meta": {
                    "video_id": vid, "mask": mask, "n_players": len(players)}})
        manifest.append({"video_id": vid, "gold_dev_only": labels[vid],
            "m2_score": score, "m2_outcome": outcome, "is_error": error,
            "is_boundary": boundary, "players": players,
            "empty_id": f"{vid}__s00",
            "full_id": f"{vid}__s{(1 << len(players)) - 1:02x}"})

    payload = {"dataset_role": input_payload.get("dataset_role", "selected_evidence_audit"),
        "selection": "union of all M2 errors and 0.1<=M2 score<=0.9",
        "condition": "SHAP", "manifest": manifest, "records": coalition_records}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"videos={len(manifest)} coalitions={len(coalition_records)}")
    print("boundary=" + str(sum(m["is_boundary"] for m in manifest)) +
          " errors=" + str(sum(m["is_error"] for m in manifest)))
    for outcome in ("TP", "TN", "FP", "FN"):
        print(f"{outcome}={sum(m['m2_outcome']==outcome for m in manifest)}")
    print("with_official=" + str(sum(any(p["official_actions"] for p in m["players"]) for m in manifest)))
    print("with_counter=" + str(sum(any(p["counter_reasons"] for p in m["players"]) for m in manifest)))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
