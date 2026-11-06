import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_episodes(input_path, output):
    groups = defaultdict(list)
    unavailable = defaultdict(int)
    for number, line in enumerate(Path(input_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        benchmark = row.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError(f"Episode row {number} has no benchmark")
        if row.get("status") == "unavailable":
            unavailable[benchmark] += 1
            continue
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Episode row {number} has an invalid normalized score")
        groups[benchmark].append({"score": score, "tokens": int(row.get("tokens", 0)),
                                  "store_size": int(row.get("store_size", 0))})
    if not groups and not unavailable:
        raise ValueError("No episode records are available")
    summary = {}
    for benchmark in sorted(set(groups) | set(unavailable)):
        rows = groups[benchmark]
        scores = [row["score"] for row in rows]
        summary[benchmark] = {
            "scored": len(rows), "unavailable": unavailable[benchmark],
            "mean_normalized_score": statistics.fmean(scores) if scores else None,
            "sample_std_normalized_score": statistics.stdev(scores) if len(scores) > 1 else 0.0 if scores else None,
            "mean_tokens": statistics.fmean(row["tokens"] for row in rows) if rows else None,
            "mean_store_size": statistics.fmean(row["store_size"] for row in rows) if rows else None,
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "episode_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
    return summary
