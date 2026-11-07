from contextlib import nullcontext
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoding import GenerationPolicy, StructuredOutput, final_answer, reasoning_prefix


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
        self.embedding_cache = OrderedDict()
        self._grammar_tokens = None
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
    def load(cls, config, checkpoint=None):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
        from .checkpoint import compatible_checkpoint
        if checkpoint is not None:
            checkpoint = compatible_checkpoint(config, checkpoint)
        set_seed(config.seed)
        base = AutoModelForCausalLM.from_pretrained(config.backbone, revision=config.revision,
                                                   torch_dtype=getattr(torch, config.dtype))
        config.revision = getattr(base.config, "_commit_hash", None) or config.revision
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer" if checkpoint is not None else config.backbone,
                                                  **({} if checkpoint is not None else {"revision": config.revision}))
        if checkpoint is not None and (checkpoint / "generation_config.json").is_file():
            base.generation_config = GenerationConfig.from_pretrained(checkpoint)
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
        if checkpoint is not None:
            instance.restore(checkpoint)
        return instance

    def select(self, head):
        self.model.set_adapter(head)
        for name, parameter in self.model.named_parameters():
            if "_ref." in name:
                parameter.requires_grad_(False)

    def refresh_references(self):
        # Snapshot each head so its KL reference stays fixed while that policy trains.
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
                                                     **self.config.chat_template_kwargs)
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def response_ids(self, prompt, response):
        if reasoning_prefix(self.tokenizer, prompt, self.config.chat_template_kwargs):
            if response.startswith("<think>"):
                response = response[len("<think>"):]
            elif not response.startswith("</think>"):
                response = "</think>\n\n" + response
        return self.tokenizer.encode(response, add_special_tokens=False) + [self.tokenizer.eos_token_id]

    def truncate(self, text, limit):
        return self.tokenizer.decode(self.tokenizer.encode(text, add_special_tokens=False)[:limit],
                                     skip_special_tokens=True)

    def invalidate_embeddings(self):
        self.embedding_cache.clear()

    def encode(self, texts, grad=False):
        if grad or not texts:
            return self._encode(texts, grad)
        missing = list(dict.fromkeys(text for text in texts if text not in self.embedding_cache))
        if missing:
            values = self._encode(missing, False).detach().cpu()
            self.embedding_cache.update(zip(missing, values))
        result = torch.stack([self.embedding_cache[text] for text in texts]).to(self.device)
        for text in texts:
            self.embedding_cache.move_to_end(text)
        while len(self.embedding_cache) > max(2048, 4 * self.config.store_capacity):
            self.embedding_cache.popitem(last=False)
        return result

    def _encode(self, texts, grad=False):
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

    def _grammar(self, schema, prompt="", prefix_open=None):
        if schema is None or not self.config.grammar_mask:
            return None
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        if self._grammar_tokens is None:
            self._grammar_tokens = build_token_enforcer_tokenizer_data(self.tokenizer)
        opened = (reasoning_prefix(self.tokenizer, prompt, self.config.chat_template_kwargs)
                  if prefix_open is None else prefix_open)
        return StructuredOutput(self.tokenizer, self._grammar_tokens, schema, opened)

    def generate(self, head, prompt, max_tokens, seed=0, schema=None, sample=None):
        from .groups import Sample
        if max_tokens < 1:
            raise ValueError("Generation requires a positive response budget")
        ids = self.prompt_ids(prompt)
        if not ids:
            raise ValueError("Generation requires a nonempty prompt")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        opened = reasoning_prefix(self.tokenizer, prompt, self.config.chat_template_kwargs)
        grammar = self._grammar(schema, prompt, opened)
        policy = GenerationPolicy(self.model.get_base_model(), self.tokenizer, self.config, head,
                                  len(ids), max_tokens, grammar, self.device, sample)
        self.model.eval()
        output, probabilities = [], []
        finish_reason = "length"
        context = self.model.disable_adapter() if head == "agent" else nullcontext()
        if head != "agent":
            self.select(head)
        with context, torch.no_grad():
            inputs = torch.tensor([ids], device=self.device)
            # Reuse the KV cache and feed only the newest token after the prompt pass.
            cache = None
            for step in range(max_tokens):
                result = self.model(input_ids=inputs, past_key_values=cache, use_cache=policy.settings.use_cache)
                history = torch.tensor([ids + output], device=self.device)
                logits = policy.scores(history, result.logits[0, -1])
                token, probability = policy.draw(logits, generator)
                output.append(token)
                probabilities.append(probability.cpu())
                cache = result.past_key_values
                inputs = torch.tensor([[token]] if cache is not None else [ids + output], device=self.device)
                if policy.stopping(torch.tensor([ids + output], device=self.device), None).item():
                    finish_reason = "length" if step == max_tokens - 1 and token not in policy.eos else "stop"
                    break
        text, complete = final_answer(self.tokenizer, output, opened, policy.eos)
        return Sample(prompt, ids, output, text, torch.stack(probabilities), schema, seed,
                      policy.state, complete, finish_reason, opened)

    def logprobs(self, head, sample, grad=True):
        self.select(head)
        self.model.train(grad)
        ids = torch.tensor([sample.prompt_ids + sample.output_ids], device=self.device)
        grammar = self._grammar(sample.schema, sample.prompt, getattr(sample, "reasoning_prefix_open", None))
        recorded = getattr(sample, "generation", None)
        policy = (GenerationPolicy(self.model.get_base_model(), self.tokenizer, self.config, head,
                                   len(sample.prompt_ids), recorded["max_new_tokens"], grammar,
                                   self.device, recorded=recorded) if recorded is not None else None)
        with nullcontext() if grad else torch.no_grad():
            result = self.model(input_ids=ids, use_cache=False)
            start = len(sample.prompt_ids) - 1
            # Apply the generation-time grammar at each prefix when scoring output tokens.
            values = []
            for i, token in enumerate(sample.output_ids):
                prefix = ids[:, :len(sample.prompt_ids) + i]
                logits = result.logits[0, start + i].float()
                if policy is not None:
                    logits = policy.scores(prefix, logits)
                elif grammar is not None:
                    from .decoding import SchemaLogitsProcessor
                    logits = SchemaLogitsProcessor(grammar)(prefix, logits.unsqueeze(0))[0]
                values.append(logits.log_softmax(dim=-1)[token])
            return torch.stack(values)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "adapters", selected_adapters=list(self.heads) + ["encoder"]
                                   + [head + "_ref" for head in self.heads])
        self.tokenizer.save_pretrained(path / "tokenizer")
        self.model.generation_config.save_pretrained(path)
        torch.save({"projection": self.projection.state_dict(), "outcome": self.outcome.state_dict()},
                   path / "auxiliary.pt")
        self.config.save(path / "config.yaml")

    def restore(self, path):
        from peft import load_peft_weights, set_peft_model_state_dict
        from transformers import AutoTokenizer, GenerationConfig
        from .checkpoint import compatible_checkpoint
        path = compatible_checkpoint(self.config, path)
        for head in (*self.heads, "encoder", *(head + "_ref" for head in self.heads)):
            state = load_peft_weights(str(path / "adapters" / head), device=str(self.device))
            set_peft_model_state_dict(self.model, state, adapter_name=head)
        state = torch.load(path / "auxiliary.pt", map_location=self.device, weights_only=True)
        self.projection.load_state_dict(state["projection"])
        self.outcome.load_state_dict(state["outcome"])
        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        self.tokenizer.padding_side = "right"
        self._grammar_tokens = None
        if (path / "generation_config.json").is_file():
            self.model.generation_config = GenerationConfig.from_pretrained(path)
        self.invalidate_embeddings()
