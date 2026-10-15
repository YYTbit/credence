# Input preparation

Install the released environments for API-Bank, WebShop, ScienceWorld, ToolBench, and AppWorld. Preserve their task identifiers, splits, action interfaces, step limits, and official graders. Training uses the first four environments. AppWorld Test-Normal measures transfer.

## Task records

Provide one JSON object per line in `data/tasks.jsonl`.

```json
{"benchmark":"scienceworld","task_id":"task-0","split":"train","text":"Task description","parameters":{}}
```

Benchmark identifiers are `scienceworld`, `webshop`, `api_bank`, `toolbench`, and `appworld`. Splits are `train`, `test`, and `test_normal`. The environment receives the complete task record. Its parameters carry the native environment identifiers and options.

## Environment interface

Provide a JSON manifest mapping benchmark identifiers to environment factories.

```json
{"scienceworld":{"factory":"credence.benchmarks:ScienceWorldEnvironment","kwargs":{}}}
```

The supplied [manifest](../configs/environments.json) selects the five native wrappers. Install the released SDKs and data in their documented locations, then set `sdk_path`, `tool_root_dir`, and environment options. `sdk_python` selects an interpreter with the corresponding SDK and this package installed. ToolBench uses ToolEval and environment-supplied API credentials.

| Benchmark | Task parameters | Workspace interface |
| --- | --- | --- |
| ScienceWorld | `task_name`, `variation`, optional `simplification` | `benchmark.step(action)` |
| WebShop | released `goal_id` | `benchmark.step(action)` |
| API-Bank | `apis`, released `ground_truth` API records | `benchmark.call(name, **parameters)` |
| ToolBench | released `query_json`, `tool_descriptions` | `benchmark.call(name, **parameters)` |
| AppWorld | released `task_id` | `benchmark.execute(code)` |

Native wrappers retain environment state in separate processes and expose `benchmark.py` within each workspace. Official graders provide normalized scores. Custom factories can implement `start(task, workspace, seed)` with session methods `step(action)`, `grade(answer)`, `close()`, and `initial_observation`.

`credence.environments:CommandEnvironment` accepts `command` and `timeout` keyword arguments. Its process reads JSON requests from stdin and writes one JSON response per line to stdout.

| Request | Parameters | Response |
| --- | --- | --- |
| `reset` | `task`, `workspace`, `seed` | `observation` |
| `step` | `action` | `observation` |
| `grade` | `answer` | `score` |

`WorkspaceEnvironment` executes local software fixtures with `files` and `expected` fields.

## Warm-start collection

Warm-start collection runs the base model on the four training distributions in round-robin order. It excludes non-training records, records each complete scored trajectory, and generates writer, builder, and curator targets from the recorded trajectory and touched assets. Target outputs pass the same format, Python compilation, and curation-operation checks used at runtime. A target that fails validation is recorded in the manifest count and omitted from supervised training.

```bash
python -m credence collect-warm-start --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --output data/warm_start
python -m credence sft --config configs/qwen3_8b.yaml \
  --examples data/warm_start/head_examples.jsonl --output checkpoints/warm_start
```

The default collection contains 1,000 trajectories, evenly interleaved across ScienceWorld, WebShop, API-Bank, and ToolBench. `--trajectories` sets a smaller collection size for preparation checks. The output contains `trajectories.jsonl`, `head_examples.jsonl`, and `manifest.json`, which records task identifiers, seeds, target counts, model source and revision when exposed by the model configuration, split filtering, and file hashes. The base model generates both the trajectories and head targets with its adapters disabled. The paper specifies the base-model source and training split but does not state a decoding temperature; collection follows the runtime's greedy rollout policy and records that choice.

## Execution

Validate inputs before loading a backbone.

```bash
python -m credence validate --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json
python -m credence train --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --resume results/credence/checkpoint --output results/continued
python -m credence run --checkpoint results/credence/checkpoint \
  --tasks data/evaluation.jsonl --environments configs/environments.json \
  --output results/evaluation --seed 149
```

The output directory contains execution records and resumable checkpoints, including pending candidate groups. Every fifth stream position supplies calibration data and is excluded from policy and predictor updates. Evaluation clones the trained benchmark store for each task. Unavailable grader results are recorded without numeric scores. Model access credentials are supplied through the environment.

Summaries and an SVG score plot are generated from scored episode records only.

```bash
python -m credence artifacts --episodes results/evaluation/episodes.jsonl \
  --output results/evaluation/artifacts
```
