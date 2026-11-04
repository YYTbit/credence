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
on acted decisions.

![Comparison of store evolution strategies](assets/overview.png)

## Installation

Use Python 3.10 or later and install PyTorch for your system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Data preparation

Install the benchmark environments and prepare `data/tasks.jsonl` following the
[data guide](docs/execution.md). Set the environment paths in
[configs/environments.json](configs/environments.json) and choose a model profile
from [configs/](configs).

## Training

Collect warm-start examples and train the initial policy adapters:

```bash
python -m credence collect-warm-start --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --output data/warm_start

python -m credence sft --config configs/qwen3_8b.yaml \
  --examples data/warm_start/head_examples.jsonl \
  --output checkpoints/warm_start
```

Use the policy adapters trained above to initialize joint training:

```bash
python -m credence train --config configs/qwen3_8b.yaml \
  --tasks data/tasks.jsonl --environments configs/environments.json \
  --warm-start checkpoints/warm_start --output results/credence
```

## Evaluation

Evaluate the trained checkpoint on the benchmark tasks:

```bash
python -m credence run --checkpoint results/credence/checkpoint \
  --tasks data/evaluation.jsonl --environments configs/environments.json \
  --output results/evaluation
```

## Tests

```bash
python -m pytest -q
```

## License

[MIT License](LICENSE).
