"""Aggregate exact Shapley faithfulness and error-audit results for 31 videos."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def shapley(scores: dict[int, float], n: int) -> list[float]:
    values = []
    for i in range(n):
        value = 0.0
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            k = mask.bit_count()
            w = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
            value += w * (scores[mask | (1 << i)] - scores[mask])
        values.append(value)
    return values


def mean(values):
    return statistics.mean(values) if values else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=HERE / "eval_shapley_31_qwen3_8b_awq.json")
    ap.add_argument("--input", type=Path, default=HERE / "shapley_coalitions_31_dev.json")
    ap.add_argument("--out", type=Path, default=HERE / "analysis_shapley_31.json")
    args = ap.parse_args()
    result = json.loads(args.eval.read_text(encoding="utf-8"))
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    if not result.get("complete"):
        raise ValueError("evaluation is incomplete")
    rows, all_players = [], []
    for item in dataset["manifest"]:
        vid, players = item["video_id"], item["players"]
        n = len(players)
        scores = {mask: result["results"][f"{vid}__s{mask:02x}"]["SHAP"]["fraud_score"]
                  for mask in range(1 << n)}
        phi = shapley(scores, n)
        positive = max(range(n), key=lambda i: phi[i])
        negative = min(range(n), key=lambda i: phi[i])
        full_mask = (1 << n)-1
        p_rows = []
        for i, (p, value) in enumerate(zip(players, phi)):
            row = {"video_id": vid, "player_id": p["player_id"],
                "event_type": p["event"]["event_type"],
                "interval": [p["event"]["t_start"], p["event"]["t_end"]],
                "has_official": bool(p["official_actions"]),
                "official_actions": [x["action"] for x in p["official_actions"]],
                "has_counter": bool(p["counter_reasons"]),
                "counter_reasons": p["counter_reasons"], "shapley": value}
            p_rows.append(row); all_players.append(row)
        row = {"video_id": vid, "gold": item["gold_dev_only"],
            "m2_score": item["m2_score"], "m2_outcome": item["m2_outcome"],
            "is_error": item["is_error"], "is_boundary": item["is_boundary"],
            "empty_score": scores[0], "full_score": scores[full_mask],
            "full_minus_empty": scores[full_mask]-scores[0],
            "top_positive": p_rows[positive], "top_negative": p_rows[negative],
            "comprehensiveness": scores[full_mask]-scores[full_mask & ~(1<<positive)],
            "sufficiency": scores[1<<positive]-scores[0],
            "positive_removal_flips_to_normal": scores[full_mask]>=.5 and scores[full_mask & ~(1<<positive)]<.5,
            "negative_removal_increase": scores[full_mask & ~(1<<negative)]-scores[full_mask],
            "efficiency_error": sum(phi)-(scores[full_mask]-scores[0]), "players": p_rows}
        rows.append(row)

    boundary = [r for r in rows if r["is_boundary"]]
    errors = [r for r in rows if r["is_error"]]
    counters = [p for p in all_players if p["has_counter"]]
    official = [p for p in all_players if p["has_official"]]
    nonofficial = [p for p in all_players if not p["has_official"]]
    fp_top = Counter(r["top_positive"]["event_type"] for r in errors if r["m2_outcome"]=="FP")
    fn_top = Counter(r["top_positive"]["event_type"] for r in errors if r["m2_outcome"]=="FN")
    aggregate = {
        "n_videos": len(rows), "n_boundary": len(boundary), "n_errors": len(errors),
        "max_abs_efficiency_error": max(abs(r["efficiency_error"]) for r in rows),
        "boundary_mean_comprehensiveness": mean([r["comprehensiveness"] for r in boundary]),
        "boundary_mean_sufficiency": mean([r["sufficiency"] for r in boundary]),
        "boundary_positive_removal_flip_rate": mean([r["positive_removal_flips_to_normal"] for r in boundary]),
        "counter_negative_fraction": mean([p["shapley"] < 0 for p in counters]),
        "counter_mean_shapley": mean([p["shapley"] for p in counters]),
        "official_mean_abs_shapley": mean([abs(p["shapley"]) for p in official]),
        "nonofficial_mean_abs_shapley": mean([abs(p["shapley"]) for p in nonofficial]),
        "fp_top_event_types": dict(fp_top), "fn_top_event_types": dict(fn_top)}
    args.out.write_text(json.dumps({"aggregate": aggregate, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print("\nPer-video top evidence")
    for r in rows:
        p=r["top_positive"]
        print(f"{r['video_id']} {r['m2_outcome']} score={r['m2_score']:.4f} "
              f"top={p['event_type']} phi={p['shapley']:+.4f} "
              f"comp={r['comprehensiveness']:+.4f} suff={r['sufficiency']:+.4f}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
