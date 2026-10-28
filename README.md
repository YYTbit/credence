# CREDENCE: Co-Evolving Memory, Tools, and Curation for Long-Horizon Agents via Counterfactual Credit

Long-horizon language agents accumulate memories and reusable tools to improve
across tasks. Existing approaches use separate or episode-level feedback, yet
asset creation and curation are coupled, and individual contributions can emerge
much later. To address these challenges, we propose CREDENCE, which jointly
trains a memory writer, tool builder, and curator. We therefore supervise joint
learning through counterfactual removal effects that quantify asset utility.
Specifically, an outcome model predicts task scores after asset removal, and the
difference from observed scores estimates each asset’s contribution. Accumulated
credit returns delayed utility to creators, while a calibrated margin guides
curation. During joint training, matched continuations preserve candidate
comparisons until these rewards mature, allowing all three policies to adapt as
stores evolve. To characterize when accumulated credit supports curation, we
derive a conditional sign certificate under bounded estimation error. We
evaluate CREDENCE on five public benchmarks covering tool calling, web shopping,
interactive science, tool use, and API programming. Across Qwen3-8B and
Llama-3.1-8B-Instruct, it improves average performance by 7.5 and 7.1 points
over the strongest baseline. Ablations support the complementary roles of joint
training and removal-based credit, while calibration shows 99.2% sign agreement
on acted decisions. Code is available at
<https://anonymous.4open.science/r/credence-7446/>.

![Comparison of store evolution strategies](assets/overview.png)

## Installation

Use Python 3.10 or later and a PyTorch installation appropriate for the
execution hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Data and environments

Install the released environments and prepare task records using their native
identifiers, splits, and graders.

| Environment | Role |
| --- | --- |
| API-Bank | Training and tool-calling evaluation |
| WebShop | Training and shopping evaluation |
| ScienceWorld | Training and interactive-science evaluation |
| ToolBench | Training and tool-use evaluation |
| AppWorld | Test-Normal transfer evaluation |

The [execution guide](docs/execution.md) specifies JSONL task fields and native
SDK options. [configs/environments.json](configs/environments.json) selects the
five supplied wrappers. Set SDK locations and task parameters before running
collection.

Validate task and environment inputs without loading the backbone.

```bash
python -m credence validate --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json
```

## Warm start

Collect base-model rollouts from the four training distributions. The default
collection contains 1,000 trajectories and produces validated writer, builder,
and curator targets.

```bash
python -m credence collect-warm-start --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --output data/warm_start

python -m credence sft --config configs/qwen3_8b.yaml \
  --examples data/warm_start/head_examples.jsonl \
  --output checkpoints/warm_start
```

Collection records task identities, seeds, model provenance, and target counts.
Supervised training initializes the three policy adapters from these examples.

## Joint training

Train the memory writer, tool builder, curator, encoder, and outcome model with
matched task streams and delayed credit.

```bash
python -m credence train --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --warm-start checkpoints/warm_start --output results/credence
```

Every fifth stream position supplies calibration observations and is excluded
from policy and predictor updates. Pending candidate groups retain behavior
probabilities and their original policy versions until rewards mature.

Resume the saved optimizer, stores, candidate groups, calibration, and stream
positions.

```bash
python -m credence train --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --resume results/credence/checkpoint --output results/continued
```

## Evaluation

Run native evaluation tasks using the trained checkpoint and benchmark store.

```bash
python -m credence run --checkpoint results/credence/checkpoint \
  --tasks data/evaluation.jsonl --environments configs/environments.json \
  --output results/evaluation --seed 149
```

Each evaluation task receives a cloned store. The supplied wrappers retain
official task scores; unavailable grader results are recorded separately.
`python -m credence artifacts --help` describes aggregation of completed episode
records.

## Configuration

Choose [Qwen3-8B](configs/qwen3_8b.yaml) or
[Llama-3.1-8B-Instruct](configs/llama3_1_8b.yaml) and set the model source,
device, and seed for the run.

| Setting | Value |
| --- | --- |
| Policy adapters | LoRA rank 16, scale 32 |
| Candidate group size | 8 |
| Creator-credit horizon | 20 episodes |
| Maximum policy lag | 32 updates |
| Selected context assets | 24 |
| Store capacity | 1,000 assets |
| Episode token budget | 28,000 |

The [execution guide](docs/execution.md) also specifies warm-start provenance,
calibration positions, and native task parameters.

## Software checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

CPU checks cover delayed credit, store operations, masked policy likelihoods,
joint gradients, and checkpoint continuation.

## License

[MIT License](LICENSE).
