"""Match extracted video events to the small official fraud-pattern graph.

This is a label-free candidate matcher.  It does not call an LLM and does not
declare a video fraudulent.  Every match retains its timestamp and source text;
missing elements remain `unconfirmed` rather than becoming negative evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_ksc_inputs as base  # noqa: E402
import eval_abc as ab  # noqa: E402

DEFAULT_PATTERNS = ROOT / "data/reference_fraud_kg/patterns_seed.json"
DEFAULT_SEMANTIC = HERE / "semantic_event_attributes.json"
DEFAULT_OUT = HERE / "official_pattern_matches_dev.json"


def text_of(e: base.Event) -> str:
    return " ".join((e.summary, *e.evidence, *e.entities)).lower()


def rule(pattern: str, element: str, e: base.Event, s: dict) -> bool:
    t = text_of(e)
    private_lead = e.channel == "external_private" and s["purpose"] == "investment_lead"
    rules = {
        ("P01_IMPERSONATION_APP_DEPOSIT", "E1"): e.typ == "IMPERSONATE" and s["authenticity"] in {"alleged", "unknown"},
        ("P01_IMPERSONATION_APP_DEPOSIT", "E2"): e.typ == "PROMISE_HIGH_RETURN" and e.claim == "guaranteed",
        ("P01_IMPERSONATION_APP_DEPOSIT", "E3"): e.typ in {"INDUCE_CONTACT", "INDUCE_LINK"} and private_lead,
        ("P01_IMPERSONATION_APP_DEPOSIT", "E4"): e.typ == "INDUCE_APP_INSTALL" and s["purpose"] in {"investment_lead", "payment"},
        ("P01_IMPERSONATION_APP_DEPOSIT", "E5"): e.typ == "REQUEST_TRANSFER" or s["purpose"] == "payment",
        ("P02_FREE_CONTENT_TO_FAKE_APP", "E1"): e.typ == "OFFER_FREE" and any(k in t for k in ("책", "정보", "리딩", "종목")),
        ("P02_FREE_CONTENT_TO_FAKE_APP", "E2"): e.typ in {"INDUCE_CONTACT", "INDUCE_LINK"} and private_lead,
        ("P02_FREE_CONTENT_TO_FAKE_APP", "E3"): e.typ in {"IMPERSONATE", "CITE_AUTHORITY"} and s["authenticity"] in {"alleged", "unknown"},
        ("P02_FREE_CONTENT_TO_FAKE_APP", "E4"): e.typ == "INDUCE_APP_INSTALL" and s["purpose"] in {"investment_lead", "payment"},
        ("P03_FAKE_TRADING_SYSTEM", "E1"): e.typ == "PROMISE_HIGH_RETURN" and e.claim == "guaranteed",
        ("P03_FAKE_TRADING_SYSTEM", "E2"): e.typ == "SHOW_RETURN_PROOF" and any(k in t for k in ("수익", "거래", "계좌", "hts", "mts")),
        ("P03_FAKE_TRADING_SYSTEM", "E3"): e.typ == "INDUCE_APP_INSTALL" and s["authenticity"] != "official",
        ("P03_FAKE_TRADING_SYSTEM", "E4"): e.typ == "REQUEST_TRANSFER" or s["purpose"] == "payment",
        ("P03_FAKE_TRADING_SYSTEM", "E5"): any(k in t for k in ("출금", "잠적", "인출")) and e.typ in {"CLAIM_SYSTEM_ERROR", "REQUEST_TRANSFER"},
        ("P04_LOSS_RECOVERY_IMPERSONATION", "E1"): e.typ == "IMPERSONATE" and s["authenticity"] in {"alleged", "unknown"},
        ("P04_LOSS_RECOVERY_IMPERSONATION", "E2"): e.typ == "CITE_AUTHORITY" and any(k in t for k in ("공문", "금융위원", "금융감독", "경찰", "법원")),
        ("P04_LOSS_RECOVERY_IMPERSONATION", "E3"): e.typ == "PROMISE_REFUND" or ("손실" in t and "보상" in t),
        ("P04_LOSS_RECOVERY_IMPERSONATION", "E4"): e.typ in {"REQUEST_TRANSFER", "INDUCE_APP_INSTALL", "INDUCE_CONTACT"} and s["purpose"] in {"investment_lead", "payment"},
        # E1 and E4 describe hidden/off-video trading behavior and normally remain unconfirmed.
        ("P05_UNDISCLOSED_FRONT_RUNNING", "E1"): "선매수" in t,
        ("P05_UNDISCLOSED_FRONT_RUNNING", "E2"): any(k in t for k in ("이해관계 미표시", "보유 사실 미표시")),
        ("P05_UNDISCLOSED_FRONT_RUNNING", "E3"): e.typ == "RECOMMEND_ASSET" and any(k in t for k in ("급등", "무조건", "바닥", "테마", "수혜주")),
        ("P05_UNDISCLOSED_FRONT_RUNNING", "E4"): any(k in t for k in ("선행매매", "매수세", "물량받이")),
    }
    return bool(rules.get((pattern, element), False))


def event_record(index: int, e: base.Event, s: dict) -> dict:
    return {"event_index": index, "event_id": e.eid, "event_type": e.typ,
            "t_start": e.t_start, "t_end": e.t_end, "summary": e.summary,
            "evidence": list(e.evidence), "channel": e.channel, "claim": e.claim,
            "context": e.context, "purpose": s["purpose"],
            "authenticity": s["authenticity"]}


def counterevidence(events: list[base.Event], sem: list[dict]) -> list[dict]:
    found = []
    for i, (e, s) in enumerate(zip(events, sem)):
        t = text_of(e)
        reasons = []
        if e.context == "warning" or s["purpose"] == "warning":
            reasons.append("fraud_warning_context")
        if s["purpose"] == "education" or e.typ == "EXPLAIN_KNOWLEDGE":
            reasons.append("education_context")
        if s["purpose"] in {"platform_engagement", "audience_participation"}:
            reasons.append("nonfinancial_participation")
        if s["authenticity"] == "official":
            reasons.append("official_source_or_product")
        if any(k in t for k in ("예금자보호", "원금 손실", "투자설명서", "위험 고지", "투자 적기가 아닐")):
            reasons.append("risk_disclosure_or_caution")
        if reasons:
            found.append({"reasons": sorted(set(reasons)), **event_record(i, e, s)})
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", type=Path, default=DEFAULT_PATTERNS)
    ap.add_argument("--semantic", type=Path, default=DEFAULT_SEMANTIC)
    ap.add_argument("--ids-from", type=Path,
                    help="optional input JSON; restrict matching to its record IDs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    ids, _, temporal, _ = ab.load_all()
    if args.ids_from:
        wanted = [r["id"] for r in json.loads(
            args.ids_from.read_text(encoding="utf-8"))["records"]]
        available = set(ids)
        missing = [vid for vid in wanted if vid not in available]
        if missing:
            raise ValueError(f"ids absent from pipeline outputs: {missing[:5]}")
        ids = wanted
    if args.limit:
        ids = ids[:args.limit]
    events = base.load_events(ids)
    semantic = json.loads(args.semantic.read_text(encoding="utf-8"))["records"]
    patterns = json.loads(args.patterns.read_text(encoding="utf-8"))["patterns"]
    records = []
    for vid in ids:
        rows, sem = events[vid], semantic.get(vid, [])
        if len(rows) != len(sem):
            raise ValueError(f"semantic/event mismatch {vid}: {len(sem)}/{len(rows)}")
        matches = []
        for p in patterns:
            elements = []
            for element in p["elements"]:
                candidates = [event_record(i, e, sem[i]) for i, e in enumerate(rows)
                              if rule(p["pattern_id"], element["id"], e, sem[i])]
                elements.append({"element_id": element["id"], "action": element["action"],
                                 "required": element["required"],
                                 "status": "support" if candidates else "unconfirmed",
                                 "video_evidence": candidates})
            required = [x for x in elements if x["required"]]
            support_required = sum(x["status"] == "support" for x in required)
            matches.append({"pattern_id": p["pattern_id"], "name": p["name"],
                            "source_ids": p["source_ids"], "elements": elements,
                            "required_supported": support_required,
                            "required_total": len(required),
                            "candidate": support_required >= 2})
        records.append({"id": vid, "source": temporal[vid]["source"],
                        "matches": matches, "counterevidence": counterevidence(rows, sem)})

    payload = {"scope": "official-pattern candidate matching",
               "warning": "candidate matches are not fraud verdicts",
               "patterns_file": str(args.patterns.resolve()), "records": records}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates = Counter()
    for r in records:
        for m in r["matches"]:
            if m["candidate"]:
                candidates[m["pattern_id"]] += 1
    print(f"videos={len(records)}")
    print("candidate_videos=" + str(sum(any(m['candidate'] for m in r['matches']) for r in records)))
    for pid, count in candidates.items():
        print(f"{pid}={count}")
    print("with_counterevidence=" + str(sum(bool(r["counterevidence"]) for r in records)))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
