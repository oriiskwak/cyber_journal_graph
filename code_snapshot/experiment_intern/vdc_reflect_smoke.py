"""One-round, seed-42 smoke test for degradation-aware reflection.

This intentionally uses four balanced train videos and four balanced dev videos.
It validates the full Early/Late E-step, prompt update, reflection, and scoring
path without touching the historical TEMPO result or checkpoint files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("INTERN_QUANTIZATION", "awq_marlin")
os.environ.setdefault("INTERN_PATCH_AWQ_MARLIN", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# This host's shared HF_HOME points at /opt/huggingface_cache.  Its model
# files are readable, but its token file is not.  Offline loading needs no
# token, so keep token discovery in the user's writable cache instead.
os.environ["HF_TOKEN_PATH"] = str(
    Path.home() / ".cache" / "huggingface" / "vdc_no_token"
)

SEED = 42
FULL_RUN = os.environ.get("VDC_FULL_RUN", "0") == "1"
DETERMINISTIC = os.environ.get("VDC_DETERMINISTIC", "0") == "1"
BACKEND = os.environ.get("VDC_BACKEND", "intern").lower()
if BACKEND not in {"intern", "qwen"}:
    raise ValueError(f"unsupported VDC_BACKEND={BACKEND!r}")
REFLECT_EPS = float(os.environ.get("VDC_REFLECT_EPS", "0.01"))
ENABLE_REFLECT = os.environ.get("VDC_ENABLE_REFLECT", "1") == "1"
ALLOW_REFLECT_SELECTION = os.environ.get(
    "VDC_ALLOW_REFLECT_SELECTION", "1"
) == "1"
FORCE_REFLECT = os.environ.get(
    "VDC_FORCE_REFLECT", "0" if FULL_RUN else "1"
) == "1"
if FORCE_REFLECT and not ENABLE_REFLECT:
    raise ValueError("VDC_FORCE_REFLECT=1 requires VDC_ENABLE_REFLECT=1")
DEFAULT_METHOD_TAG = "vdc_reflect" if ENABLE_REFLECT else "vdc_guard_only"
METHOD_TAG = os.environ.get("VDC_METHOD_TAG", DEFAULT_METHOD_TAG)
RUN_TAG = f"{METHOD_TAG}_{'full' if FULL_RUN else 'smoke'}_{BACKEND}_seed42"
OUTPUT_DIR = (ROOT / "results_intern" if BACKEND == "intern"
              else ROOT / "results" / "qwen")
OUTPUT = OUTPUT_DIR / f"{RUN_TAG}.json"


def balanced_subset(dataset: list[dict], per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    normal = [v for v in dataset if v["label"] == "normal"]
    abnormal = [v for v in dataset if v["label"] == "abnormal"]
    rng.shuffle(normal)
    rng.shuffle(abnormal)
    return normal[:per_label] + abnormal[:per_label]


def main() -> None:
    started = time.perf_counter()
    random.seed(SEED)

    import numpy as np
    if BACKEND == "intern":
        import experiment_intern.tempo_cov as tempo
    else:
        import experiment2.tempo_cov as tempo
    from em_core_tool2 import (
        _text_generate,
        crossover_prompts,
        generate_prompt_candidates,
    )

    np.random.seed(SEED)
    tempo.SEED = SEED
    tempo.MAX_ROUNDS = int(os.environ.get(
        "VDC_MAX_ROUNDS", "12" if FULL_RUN else "1"
    ))
    tempo.MINIBATCH = 32 if FULL_RUN else 4
    tempo.DEV_SUBSAMPLE = 50 if FULL_RUN else 4
    tempo.PHASE1_MAX_ROUNDS = 6 if FULL_RUN else 99
    tempo.CHECKPOINT_PATH = OUTPUT_DIR / f"{RUN_TAG}_checkpoint.json"
    records: list[dict] = []

    def reflect_prompt(seg_name: str, current_prompt: str, current_map: float,
                       new_prompt: str, new_map: float, feedback: str,
                       seed: int | None = None) -> tuple[str, str]:
        request = f"""You are diagnosing a failed cyber-scam video prompt update.
Segment: {seg_name}
Previous prompt (mAP={current_map:.6f}):
{current_prompt}

Candidate prompt (mAP={new_map:.6f}):
{new_prompt}

Feedback that produced the candidate:
{feedback}

Explain why the candidate may have degraded, then write a corrected visual-only
policy under 200 words. The policy must require ONLY one float from 0.0 to 1.0.
Do not mention external tools.

<diagnosis>[brief diagnosis]</diagnosis>
<policy>[corrected prompt]</policy>"""
        raw = _text_generate(request, temperature=0.3, max_tokens=500, seed=seed)
        diagnosis_match = re.search(r"<diagnosis>(.*?)</diagnosis>", raw, re.DOTALL)
        policy_match = re.search(r"<policy>(.*?)</policy>", raw, re.DOTALL)
        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else raw[:500]
        policy = policy_match.group(1).strip() if policy_match else current_prompt
        return policy, diagnosis

    def score_same_subset(seg_name: str, prompt: str, fixed_prompt: str,
                          dev_sub: list[dict], this_fb_mem, other_fb_mem):
        if seg_name == "early":
            result = tempo.score_dataset(dev_sub, prompt, fixed_prompt,
                                          this_fb_mem, other_fb_mem)
        else:
            result = tempo.score_dataset(dev_sub, fixed_prompt, prompt,
                                          other_fb_mem, this_fb_mem)
        metrics = tempo.compute_metrics(result)
        return metrics.get("map") or 0.0, metrics

    step_index = 0

    def m_step_reflect(seg_name, current_prompt, fixed_other_prompt, feedback,
                       pm, pos_examples, devset, this_fb_mem, other_fb_mem,
                       n_candidates=3):
        nonlocal step_index
        step_index += 1
        if not feedback:
            if not FORCE_REFLECT:
                records.append({"segment": seg_name, "skipped": "no_feedback"})
                return current_prompt, pm, 0.0, {}
            feedback = (
                "Smoke-test fallback: preserve specificity while balancing false "
                "positive and false negative visual cues."
            )

        candidates = generate_prompt_candidates(
            current_prompt, feedback, pos_examples, n=n_candidates
        )
        new_prompt = crossover_prompts(candidates, pm.get_top_k())
        dev_seed = SEED * 100000 + step_index * 10 + 4
        dev_rng = random.Random(dev_seed) if DETERMINISTIC else random
        dev_sub = dev_rng.sample(devset, min(tempo.DEV_SUBSAMPLE, len(devset)))
        current_map, current_metrics = score_same_subset(
            seg_name, current_prompt, fixed_other_prompt, dev_sub,
            this_fb_mem, other_fb_mem,
        )
        new_map, new_metrics = score_same_subset(
            seg_name, new_prompt, fixed_other_prompt, dev_sub,
            this_fb_mem, other_fb_mem,
        )
        triggered = ENABLE_REFLECT and (
            FORCE_REFLECT or new_map < current_map - REFLECT_EPS
        )
        reflected_prompt = None
        diagnosis = ""
        reflected_map = None
        reflected_metrics = None
        if triggered:
            reflect_seed = SEED * 100000 + step_index * 10 + 3
            reflected_prompt, diagnosis = reflect_prompt(
                seg_name, current_prompt, current_map, new_prompt, new_map, feedback,
                seed=reflect_seed if DETERMINISTIC else None,
            )
            reflected_map, reflected_metrics = score_same_subset(
                seg_name, reflected_prompt, fixed_other_prompt, dev_sub,
                this_fb_mem, other_fb_mem,
            )

        choices = [(current_map, current_prompt, current_metrics, "current"),
                   (new_map, new_prompt, new_metrics, "new")]
        if reflected_map is not None and ALLOW_REFLECT_SELECTION:
            choices.append((reflected_map, reflected_prompt, reflected_metrics, "reflect"))
        selected_map, selected_prompt, selected_metrics, selected = max(
            choices, key=lambda item: item[0]
        )
        record = {
            "segment": seg_name,
            "reflect_triggered": triggered,
            "allow_reflect_selection": ALLOW_REFLECT_SELECTION,
            "forced_for_smoke": FORCE_REFLECT,
            "current_map": current_map,
            "new_map": new_map,
            "reflect_map": reflected_map,
            "selected": selected,
            "diagnosis": diagnosis,
            "step_index": step_index,
            "dev_subset_seed": dev_seed if DETERMINISTIC else None,
            "dev_subset_videos": [v.get("video_name") for v in dev_sub],
            "current_prompt": current_prompt,
            "new_prompt": new_prompt,
            "reflected_prompt": reflected_prompt,
        }
        records.append(record)
        print("[VDC Update] " + json.dumps(record, ensure_ascii=False), flush=True)
        return selected_prompt, pm, selected_map, selected_metrics

    tempo._m_step_seg_cov = m_step_reflect
    if not FULL_RUN:
        tempo.CHECKPOINT_PATH.unlink(missing_ok=True)
    all_data = tempo.load_gt()
    train, dev, test = tempo.split_data(all_data, 0.6, 0.2, SEED)
    train_run = train if FULL_RUN else balanced_subset(train, per_label=2, seed=SEED)
    dev_run = dev if FULL_RUN else balanced_subset(dev, per_label=2, seed=SEED + 1)

    (best_early, best_late, _early_fb, _late_fb,
     _early_pm, _late_pm, _cache, log) = tempo.run_tempo_cov_total2b(
        train_run, dev_run
    )
    evaluation = None
    if FULL_RUN:
        print("[VDC Update] final dev evaluation", flush=True)
        dev_results = tempo.score_dataset(
            dev, best_early, best_late, _early_fb, _late_fb
        )
        threshold = tempo.find_best_threshold(dev_results)
        print("[VDC Update] final test evaluation", flush=True)
        test_results = tempo.score_dataset(
            test, best_early, best_late, _early_fb, _late_fb
        )
        evaluation = {
            "threshold_from_dev": threshold,
            "dev": tempo.compute_metrics(dev_results),
            "test_at_0.5": tempo.compute_metrics(test_results),
            "test_at_dev_threshold": tempo.compute_metrics(test_results, threshold),
            "dev_frame_predictions": dev_results,
            "test_frame_predictions": test_results,
        }
    payload = {
        "status": "ok",
        "purpose": "full_seed42_experiment" if FULL_RUN else "mechanics_and_runtime_smoke_only",
        "backend": BACKEND,
        "method_tag": METHOD_TAG,
        "deterministic": DETERMINISTIC,
        "seed": SEED,
        "rounds_configured": tempo.MAX_ROUNDS,
        "train_videos": [v["video_name"] for v in train_run],
        "dev_videos": [v["video_name"] for v in dev_run],
        "force_reflect": FORCE_REFLECT,
        "enable_reflect": ENABLE_REFLECT,
        "allow_reflect_selection": ALLOW_REFLECT_SELECTION,
        "reflect_eps": REFLECT_EPS,
        "reflect_records": records,
        "best_early": best_early,
        "best_late": best_late,
        "optimization_log": log,
        "evaluation": evaluation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[VDC Update] saved: {OUTPUT}")


if __name__ == "__main__":
    main()
