from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self, hidden, dimension):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dimension))

    def forward(self, states):
        return F.normalize(self.layers(states.float()), dim=-1)


class OutcomeModel(nn.Module):
    def __init__(self, dimension=6174, hidden=512):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, summaries):
        return self.layers(summaries.float()).squeeze(-1)


class LanguageModels:
    heads = ("writer", "builder", "curator")

    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device = next(model.parameters()).device
        self.projection = Projection(model.config.hidden_size, config.embedding_dim).to(self.device)
        self.outcome = OutcomeModel(config.summary_dim, config.predictor_hidden).to(self.device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.eval()

    @classmethod
    def load(cls, config):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        base = AutoModelForCausalLM.from_pretrained(config.backbone, torch_dtype=getattr(torch, config.dtype))
        adapter = LoraConfig(r=config.adapter_rank, lora_alpha=config.adapter_alpha,
                             lora_dropout=0, target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(base, adapter, adapter_name="writer")
        for head in ("builder", "curator"):
            model.add_adapter(head, adapter)
        encoder = LoraConfig(r=config.adapter_rank, lora_alpha=config.adapter_alpha,
                             lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model.add_adapter("encoder", encoder)
        model.to(config.device)
        instance = cls(model, tokenizer, config)
        instance.refresh_references()
        return instance

    def select(self, head):
        self.model.set_adapter(head)
        for name, parameter in self.model.named_parameters():
            if "_ref." in name:
                parameter.requires_grad_(False)

    def refresh_references(self):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        for head in self.heads:
            reference = head + "_ref"
            if reference not in self.model.peft_config:
                self.model.add_adapter(reference, self.model.peft_config[head])
            state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(
                self.model, adapter_name=head).items()}
            set_peft_model_state_dict(self.model, state, adapter_name=reference)
        self.select("writer")

    def parameters(self):
        adapters = [parameter for name, parameter in self.model.named_parameters()
                    if "lora_" in name and "_ref." not in name]
        return adapters + list(self.projection.parameters()) + list(self.outcome.parameters())

    def prompt_ids(self, prompt):
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                     tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False)
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text, limit):
        return self.tokenizer.decode(self.tokenizer.encode(text, add_special_tokens=False)[:limit],
                                     skip_special_tokens=True)

    def encode(self, texts, grad=False):
        if not texts:
            return torch.zeros((0, self.config.embedding_dim), device=self.device)
        self.select("encoder")
        self.model.train(grad)
        values = []
        with nullcontext() if grad else torch.no_grad():
            for start in range(0, len(texts), self.config.encoder_batch):
                tokens = self.tokenizer(texts[start:start + self.config.encoder_batch], return_tensors="pt",
                                        padding=True, truncation=True, max_length=self.config.encoder_tokens).to(self.device)
                states = self.model.get_base_model().model(**tokens, use_cache=False).last_hidden_state
                positions = tokens["attention_mask"].sum(dim=1) - 1
                pooled = states[torch.arange(len(states), device=self.device), positions]
                values.append(self.projection(pooled))
            return torch.cat(values)

    def _grammar(self, schema):
        if schema is None or not self.config.grammar_mask:
            return None
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        return build_transformers_prefix_allowed_tokens_fn(self.tokenizer, JsonSchemaParser(schema))

    def _masked_logits(self, logits, prefix, grammar):
        if grammar is None:
            return logits.float()
        allowed = grammar(0, prefix)
        mask = torch.full_like(logits, -torch.inf, dtype=torch.float32)
        mask[allowed] = logits[allowed].float()
        return mask

    def generate(self, head, prompt, max_tokens, seed=0, schema=None, sample=True):
        from .groups import Sample
        ids = self.prompt_ids(prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        grammar = self._grammar(schema)
        self.model.eval()
        output, probabilities = [], []
        context = self.model.disable_adapter() if head == "agent" else nullcontext()
        if head != "agent":
            self.select(head)
        with context, torch.no_grad():
            inputs = torch.tensor([ids], device=self.device)
            cache = None
            for _ in range(max_tokens):
                result = self.model(input_ids=inputs, past_key_values=cache, use_cache=True)
                prefix = torch.tensor(ids + output, device=self.device)
                logits = self._masked_logits(result.logits[0, -1], prefix, grammar)
                logprobs = logits.log_softmax(dim=-1)
                token = int(torch.multinomial(logprobs.exp(), 1, generator=generator)) if sample else int(logprobs.argmax())
                output.append(token)
                probabilities.append(logprobs[token].cpu())
                cache = result.past_key_values
                inputs = torch.tensor([[token]], device=self.device)
                if token == self.tokenizer.eos_token_id:
                    break
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        return Sample(prompt, ids, output, text, torch.stack(probabilities), schema)

    def logprobs(self, head, sample, grad=True):
        self.select(head)
        self.model.train(grad)
        ids = torch.tensor([sample.prompt_ids + sample.output_ids], device=self.device)
        grammar = self._grammar(sample.schema)
        with nullcontext() if grad else torch.no_grad():
            result = self.model(input_ids=ids, use_cache=False)
            start = len(sample.prompt_ids) - 1
            values = []
            for i, token in enumerate(sample.output_ids):
                prefix = ids[0, :len(sample.prompt_ids) + i]
                logits = self._masked_logits(result.logits[0, start + i], prefix, grammar)
                values.append(logits.log_softmax(dim=-1)[token])
            return torch.stack(values)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "adapters", selected_adapters=list(self.heads) + ["encoder"]
                                   + [head + "_ref" for head in self.heads])
        self.tokenizer.save_pretrained(path / "tokenizer")
        torch.save({"projection": self.projection.state_dict(), "outcome": self.outcome.state_dict()},
                   path / "auxiliary.pt")
        self.config.save(path / "config.yaml")

    def restore(self, path):
        from peft import load_peft_weights, set_peft_model_state_dict
        path = Path(path)
        for head in (*self.heads, "encoder", *(head + "_ref" for head in self.heads)):
            state = load_peft_weights(str(path / "adapters" / head), device=str(self.device))
            set_peft_model_state_dict(self.model, state, adapter_name=head)
        state = torch.load(path / "auxiliary.pt", map_location=self.device, weights_only=True)
        self.projection.load_state_dict(state["projection"])
        self.outcome.load_state_dict(state["outcome"])
