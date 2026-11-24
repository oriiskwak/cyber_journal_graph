#!/usr/bin/env python3
"""Export saved Qwen and InternVL metrics with an audit of test coverage."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


DESTINATION = Path(__file__).resolve().parents[1] / "run"
QWEN = "Qwen2.5-VL-72B-AWQ"
INTERN = "InternVL2.5-78B-AWQ"
Q = "results/qwen/"
I = "results_intern/"
FULL_QWEN = Q + "vdc_reflect_full_qwen_seed42.json"
GUARD_QWEN = Q + "vdc_guard_only_full_qwen_seed42.json"
FULL_INTERN = I + "vdc_reflect_full_seed42.json"
RUNS = (
    (QWEN, "Vanilla", Q + "baseline_result.json", "test", "historical"),
    (QWEN, "Vanilla + CoT", "results/vanilla_cot_result.json", "test", "historical"),
    (QWEN, "Baseline EM", Q + "baseline_em_result.json", "best_test", "historical"),
    (QWEN, "Baseline EM (ensemble)", Q + "baseline_em_ensemble_result.json", "best_test", "historical"),
    (QWEN, "Temporal EM", "results/temporal_em_result.json", "best_test", "historical"),
    (QWEN, "Tool EM", "results/tool_em_result.json", "best_test", "historical"),
    (QWEN, "Original TEMPO / CYREN", Q + "tempo_cov_total_covfix_result.json", "best_test", "historical"),
    (QWEN, "APO", "results/apo_result.json", "best_test", "historical_ratio_score"),
    (QWEN, "ERM", "results/erm_result.json", "best_test", "historical_ratio_score"),
    (QWEN, "TEMPO + Guard-only", GUARD_QWEN, "evaluation.test_at_0.5", "full"),
    (QWEN, "TEMPO + Guard + Reflect", FULL_QWEN, "evaluation.test_at_0.5", "full"),
    (INTERN, "Vanilla", I + "vanilla_intern_result.json", "test", "historical"),
    (INTERN, "Vanilla + CoT", I + "vanilla_cot_intern_result.json", "test", "historical"),
    (INTERN, "Baseline EM", I + "baseline_em_intern_result.json", "best_test", "historical"),
    (INTERN, "Baseline EM (ensemble)", I + "baseline_em_intern_ensemble_result.json", "best_test", "historical"),
    (INTERN, "Temporal EM", I + "temporal_em_intern_result.json", "best_test", "historical"),
    (INTERN, "Tool EM", I + "tool_em_intern_result.json", "best_test", "historical"),
    (INTERN, "Original TEMPO / CYREN (single)", I + "tempo_cov_intern_evolve_result.json", "single_test", "historical"),
    (INTERN, "Original TEMPO / CYREN (top-3)", I + "tempo_cov_intern_evolve_result.json", "best_test", "historical"),
    (INTERN, "TEMPO + Guard + Reflect", FULL_INTERN, "evaluation.test_at_0.5", "full"),
    (QWEN, "Deterministic Guard-only", Q + "vdc_deterministic_guard_pilot3_full_qwen_seed42.json", "evaluation.test_at_0.5", "pilot"),
    (QWEN, "Deterministic Guard + Reflect", Q + "vdc_deterministic_reflect_pilot3_full_qwen_seed42.json", "evaluation.test_at_0.5", "pilot"),
    (QWEN, "Deterministic Guard + Reflect (repeat)", Q + "vdc_deterministic_reflect_repro2_pilot3_full_qwen_seed42.json", "evaluation.test_at_0.5", "pilot"),
    (QWEN, "Shadow-Reflect", Q + "vdc_deterministic_shadow_reflect_pilot3_full_qwen_seed42.json", "evaluation.test_at_0.5", "pilot"),
)


def read_run(root, filename):
    data = (root / filename).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def test_labels(data):
    return {frame["video"]: frame["label"] for frame in data["evaluation"]["test_frame_predictions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    summaries = []
    rows = []
    for backbone, method, filename, key, protocol in RUNS:
        data, digest = read_run(args.source_root, filename)
        evaluation = data.get("evaluation") or {}
        metrics = evaluation["test_at_0.5"] if key.startswith("evaluation.") else data[key]
        frames = evaluation.get("test_frame_predictions")
        row = {
            "backbone": backbone, "method": method, "protocol": protocol,
            "rounds": data.get("rounds_configured"),
            "test_videos_with_predictions": len(test_labels(data)) if frames is not None else None,
            "test_frames_with_predictions": len(frames) if frames is not None else None,
            "score_threshold": 0.5,
            **{metric: metrics.get(metric) for metric in ("auc", "map", "f1", "acc")},
            "dev_selected_threshold": evaluation.get("threshold_from_dev"),
            "f1_at_dev_threshold": evaluation.get("test_at_dev_threshold", {}).get("f1"),
            "acc_at_dev_threshold": evaluation.get("test_at_dev_threshold", {}).get("acc"),
            "source_file": filename, "source_key": key, "source_sha256": digest,
        }
        rows.append(row)
        updates = data.get("reflect_records", [])
        summaries.append({**row, "stored_test_metrics": metrics,
                          "stored_dev_metrics": evaluation.get("dev"),
                          "stored_test_metrics_at_dev_threshold": evaluation.get("test_at_dev_threshold"),
                          "stored_split_metadata": data.get("split"),
                          "guard_trigger_count": sum(bool(x.get("reflect_triggered")) for x in updates) if updates else None,
                          "prompt_choices": Counter(x.get("selected", "skipped") for x in updates) if updates else None})

    qwen, guard, intern = [read_run(args.source_root, path)[0] for path in (FULL_QWEN, GUARD_QWEN, FULL_INTERN)]
    qwen_ids, guard_ids, intern_ids = map(test_labels, (qwen, guard, intern))
    common = qwen_ids.keys() & intern_ids.keys()
    audit = {
        "qwen_full_runs_have_identical_test_ids_and_labels": qwen_ids == guard_ids,
        "qwen_full_runs_have_identical_test_video_second_pairs": {
            (frame["video"], frame["second"]) for frame in qwen["evaluation"]["test_frame_predictions"]
        } == {
            (frame["video"], frame["second"]) for frame in guard["evaluation"]["test_frame_predictions"]
        },
        "qwen_full_runs_have_identical_train_and_dev_membership": all(
            set(qwen[key]) == set(guard[key]) for key in ("train_videos", "dev_videos")
        ),
        "qwen_test_video_count": len(qwen_ids), "internvl_test_video_count": len(intern_ids),
        "qwen_internvl_test_video_overlap": len(common),
        "qwen_internvl_label_mismatches_in_overlap": sum(qwen_ids[key] != intern_ids[key] for key in common),
    }
    evoprompt, evoprompt_digest = read_run(args.source_root, "results/evoprompt_result.json")
    missing = {"EvoPrompt": {"source_file": "results/evoprompt_result.json", "source_sha256": evoprompt_digest,
                             "stored_split_metadata": evoprompt.get("split"), "validation_score": evoprompt.get("val_score"),
                             "test_metrics_available": False}}

    with (DESTINATION / "method_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (DESTINATION / "method_comparisons.json").write_text(
        json.dumps({"evaluation_set_audit": audit, "runs": summaries, "missing_test_metrics": missing},
                   ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
