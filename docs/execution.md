# Input preparation

Install the released environments for API-Bank, WebShop, ScienceWorld, ToolBench, and AppWorld. Preserve their task identifiers, splits, action interfaces, step limits, and official graders. Training uses the first four environments. AppWorld Test-Normal measures transfer.

| Environment | Official installation and data |
| --- | --- |
| API-Bank | [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank) |
| WebShop | [WebShop](https://github.com/princeton-nlp/WebShop/tree/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd) |
| ScienceWorld | [ScienceWorld](https://github.com/allenai/ScienceWorld/tree/e8216d6044e8e39be9fcb185e3b2dfb602584b52) |
| ToolBench | [ToolBench and ToolEval](https://github.com/OpenBMB/ToolBench/tree/d56fdd89faf8c91fa135090b212bb9057ee5cfc2) |
| AppWorld | [AppWorld](https://github.com/StonyBrookNLP/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5) |

Install each SDK and its released data using its own instructions. Separate SDK interpreters can be selected with `sdk_python` in `configs/environments.json`; install this package in those interpreters as well. The model environment uses the versions in `pyproject.toml`.

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

## Warm-start collection

Warm-start collection runs the base model on the four training distributions in round-robin order. It excludes non-training records, records each complete scored trajectory, and generates writer, builder, and curator targets from the recorded trajectory and touched assets. Target outputs pass the same format, Python compilation, and curation-operation checks used at runtime. A target that fails validation is recorded in the manifest count and omitted from supervised training.

```bash
python -m credence collect-warm-start --config configs/${MODEL}.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --output data/warm_start/${MODEL}
python -m credence sft --config configs/${MODEL}.yaml \
  --examples data/warm_start/${MODEL}/head_examples.jsonl \
  --output checkpoints/warm_start/${MODEL}
```

Set `MODEL=qwen3_8b` or `MODEL=llama3_1_8b`, matching the README. The default collection contains 1,000 trajectories, evenly interleaved across ScienceWorld, WebShop, API-Bank, and ToolBench. Each completed trajectory and its accepted targets are committed together. Repeating the same command continues from the committed cursor. The output contains `trajectories.jsonl`, `head_examples.jsonl`, and `manifest.json`, with task identifiers, generation seeds, target counts, model revision, and file hashes. The base model generates the trajectories and head targets with its adapters disabled.

## Resuming training

Resume SFT with the same model, examples, epochs, and learning rate:

```bash
python -m credence sft --config configs/${MODEL}.yaml \
  --examples data/warm_start/${MODEL}/head_examples.jsonl \
  --resume checkpoints/warm_start/${MODEL} --output checkpoints/warm_start/${MODEL}
```

Resume joint training from its committed optimizer batch:

```bash
python -m credence train --config configs/${MODEL}.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --resume results/${MODEL}/train/checkpoint --output results/${MODEL}/train
```

Checkpoints contain the native adapters, tokenizer, generation configuration, projection and outcome model. Training state includes optimizer moments, RNG state, stream positions, version ledgers, pending candidate groups, calibration records, and policy-update counts. Resume retains the original task and environment inputs. Every fifth stream position supplies calibration data and is excluded from policy and predictor updates.

## Evaluation

```bash
python -m credence run --config configs/${MODEL}.yaml \
  --checkpoint results/${MODEL}/train/checkpoint \
  --tasks data/evaluation.jsonl --environments configs/environments.json \
  --output results/${MODEL}/evaluation
```

Evaluation isolates task stores and commits episode records incrementally. Repeat the command to resume the same evaluation. The completed collection produces `episode_summary.json` from the recorded scores, including scored-task and unavailable-task counts. Model access and ToolEval credentials are supplied through the environment.
