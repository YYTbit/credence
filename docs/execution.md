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

## Warm-start records

Use recorded completions for all three heads. Each JSON line has `head`, `prompt`, and `response`. Heads are `writer`, `builder`, and `curator`. Responses follow the paper templates in the executable prompt module. The warm start establishes the reference adapters.

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
