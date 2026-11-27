"""Join label-free official matches with frozen evaluation labels.

Selection uses only the number of supported official elements. Gold labels are
copied after selection solely for downstream evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def supported_count(record: dict) -> int:
    return sum(e.get("status") == "support"
               for m in record.get("matches", []) for e in m.get("elements", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True,
                    help="frozen semantic-v2 input JSON containing id, source and gold label")
    ap.add_argument("--min-supported", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    matches = json.loads(args.matches.read_text(encoding="utf-8"))["records"]
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))["records"]
    by_id = {r["id"]: r for r in inputs}
    output = []
    for record in matches:
        n = supported_count(record)
        if n < args.min_supported:
            continue
        original = by_id.get(record["id"])
        if original is None:
            raise ValueError(f"match id absent from frozen inputs: {record['id']}")
        output.append({**record, "gold_dev_only": original["gold_dev_only"],
                       "source": original.get("source", record.get("source")),
                       "supported_elements": n})

    args.out.write_text(json.dumps({
        "dataset_role": "independent_test_evidence_analysis",
        "selection": f"label-free: at least {args.min_supported} supported official element(s)",
        "records": output,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"input_videos={len(matches)} selected={len(output)} min_supported={args.min_supported}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
