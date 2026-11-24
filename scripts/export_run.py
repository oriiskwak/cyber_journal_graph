#!/usr/bin/env python3
"""Export an allowlisted presentation view of a saved TEMPO run."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = "vdc_reflect_full_qwen_seed42"


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    original = json.loads(args.result.read_text(encoding="utf-8"))
    prior_cases = json.loads(args.cases.read_text(encoding="utf-8"))["videos"]
    evaluation = original["evaluation"]
    threshold = evaluation["threshold_from_dev"]
    cases = {row["video_id"]: row for row in prior_cases}

    out = ROOT / "run"
    out.mkdir(exist_ok=True)
    frames = []
    videos = []
    test_by_id = {}
    for split in ("dev", "test"):
        grouped = defaultdict(list)
        for row in evaluation[f"{split}_frame_predictions"]:
            grouped[row["video"]].append(row)
        for video_id, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: row["second"])
            score = statistics.mean(row["pred"] for row in rows)
            label = rows[0]["label"]
            assert len({row["label"] for row in rows}) == 1
            label_at_threshold = "abnormal" if score >= threshold else "normal"
            case = {("abnormal", "abnormal"): "TP", ("normal", "normal"): "TN",
                    ("normal", "abnormal"): "FP", ("abnormal", "normal"): "FN"}[label, label_at_threshold]
            demo = split == "test" and video_id in cases
            videos.append({"run_id": RUN, "split": split, "video_id": video_id,
                           "reference_label": label, "mean_frame_score": score,
                           "threshold": threshold, "predicted_label": label_at_threshold,
                           "case_type": case, "frames": len(rows), "presentation_case": demo})
            for row in rows:
                frames.append({"run_id": RUN, "split": split, "video_id": video_id,
                               "second": row["second"], "segment": row["segment"],
                               "model_score": row["pred"], "reference_frame_score": row["gt"],
                               "frame_kl": row["kl"]})
            if demo:
                old = cases[video_id]
                assert old["video_scam_score_mean"] == score and old["case_type"] == case
                assert old["threshold"] == threshold
                assert sorted((row["second"], row["segment"], row["pred"]) for row in rows) == sorted(
                    (row["second"], row["segment"], row["scam_probability"]) for row in old["frames"]
                )
                test_by_id[video_id] = {"video_id": video_id, "reference_label": label,
                                        "predicted_label": label_at_threshold, "case_type": case,
                                        "mean_frame_score": score, "threshold": threshold,
                                        "frames": [{"second": row["second"], "segment": row["segment"],
                                                    "model_score": row["pred"]} for row in rows]}

    assert len(test_by_id) == len(cases) == 10
    assert sum(len(case["frames"]) for case in test_by_id.values()) == 207
    save_csv(out / "frame_scores.csv", frames)
    save_csv(out / "video_predictions.csv", videos)
    save_json(out / "presentation_cases.json", {"run_id": RUN, "split": "test", "threshold": threshold,
                                                "videos": list(test_by_id.values())})
    metrics = {key: evaluation[key] for key in ("threshold_from_dev", "dev", "test_at_0.5", "test_at_dev_threshold")}
    metrics["test_confusion_at_dev_threshold"] = Counter(row["case_type"] for row in videos if row["split"] == "test")
    save_json(out / "metrics.json", metrics)
    save_json(out / "prompts.json", {"initial": "Predict the probability (0.0-1.0) that this video contains a cyber scam.",
                                     "best_early": original["best_early"], "best_late": original["best_late"],
                                     "score_suffix": "\nReturn ONLY a single float 0.0–1.0."})

    keys = ("segment", "reflect_triggered", "forced_for_smoke", "current_map", "new_map", "reflect_map", "selected")
    updates = [{"step": index, **{key: row[key] for key in keys}} for index, row in enumerate(original["reflect_records"], 1)]
    assert all(row["reflect_triggered"] == (row["new_map"] < row["current_map"] - original["reflect_eps"]) for row in updates)
    for row in updates:
        choices = {"current": row["current_map"], "new": row["new_map"]}
        if row["reflect_map"] is not None:
            choices["reflect"] = row["reflect_map"]
        assert row["selected"] == max(choices, key=choices.get)
    save_csv(out / "guard_updates.csv", updates)

    log_examples = []
    for number, line in enumerate(args.log.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith('[VDC Reflect] {'):
            entry = json.loads(line.split(" ", 2)[2])
            fields = {key: entry[key] for key in keys}
            log_examples.append({"source_line": number, "event": "[VDC Reflect]", **fields})
    assert len(log_examples) == len(updates)
    assert all(all(update[key] == log[key] for key in keys) for update, log in zip(updates, log_examples))
    save_json(out / "guard_log_extract.json", log_examples)
    save_json(out / "provenance.json", {
        "run_id": RUN, "source_result": args.result.name, "source_result_sha256": sha256(args.result),
        "source_log": args.log.name, "source_log_sha256": sha256(args.log),
        "reference_commit": "70e57b38e8b65cd39b11c1ffef335e13fb0fd1dd",
        "commit_status": "post-run archive containing byte-identical result, not execution-time SHA",
        "settings": {"seed": original["seed"], "rounds": original["rounds_configured"],
                     "guard_epsilon": original["reflect_eps"], "force_reflect": original["force_reflect"],
                     "threshold_from_dev": threshold, "source_model": "Qwen2.5-VL-72B-Instruct-AWQ",
                     "quantization": "awq_marlin", "dtype": "float16", "tensor_parallel_size": 2,
                     "max_model_len": 8192, "max_num_seqs": 4, "gpu_memory_utilization": 0.85,
                     "image_limit_per_prompt": 1},
        "train_video_count": len(original["train_videos"]),
        "frame_counts": Counter(frame["split"] for frame in frames),
        "video_counts": Counter(video["split"] for video in videos),
        "elapsed_seconds": original["elapsed_seconds"],
    })


if __name__ == "__main__":
    main()
