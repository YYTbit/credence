import json
import math
import statistics
from collections import defaultdict
from html import escape
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
    if not groups:
        raise ValueError("No scored episode records are available")
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
    _plot(summary, output / "episode_scores.svg")
    return summary


def _plot(summary, path):
    names = [name for name, values in summary.items() if values["scored"]]
    width, height, left, right, top, bottom = 880, 410, 100, 35, 56, 110
    chart_width, chart_height = width - left - right, height - top - bottom
    slot = chart_width / max(1, len(names))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="440" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#202936">Recorded normalized episode scores</text>']
    for tick in range(6):
        value = tick / 5
        y = top + chart_height * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dee8"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475467">{value:.1f}</text>')
    for index, name in enumerate(names):
        row = summary[name]
        x = left + index * slot + slot * 0.22
        bar_width = slot * 0.56
        mean = row["mean_normalized_score"]
        bar_height = chart_height * mean
        y = top + chart_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="#3976a8"/>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#202936">{mean:.2f}</text>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{top+chart_height+24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#202936">{escape(name)}</text>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{top+chart_height+43}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#667085">n={row["scored"]}</text>')
    parts.append(f'<text x="25" y="{top + chart_height/2}" transform="rotate(-90 25 {top + chart_height/2})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#202936">Mean score</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
