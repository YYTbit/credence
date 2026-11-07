from copy import deepcopy

import torch


DECODING_PROTOCOL = "native-structured-v1"


def reasoning_prefix(tokenizer, prompt, template_kwargs):
    vocabulary = tokenizer.get_vocab()
    if not tokenizer.chat_template or not {"<think>", "</think>"}.issubset(vocabulary):
        return False
    messages = [{"role": "user", "content": prompt}]
    before = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                          **template_kwargs)
    after = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **template_kwargs)
    suffix = after[len(before):] if after.startswith(before) else ""
    return suffix.rfind("<think>") > suffix.rfind("</think>")


def reasoning_started(tokenizer, response):
    begin = tokenizer.get_vocab().get("<think>")
    if begin is None or begin not in response:
        return False
    return not tokenizer.decode(response[:response.index(begin)], skip_special_tokens=False).strip()


def final_answer(tokenizer, response, prefix_open, eos=()):
    if prefix_open or reasoning_started(tokenizer, response):
        end = tokenizer.get_vocab().get("</think>")
        if end is None or end not in response:
            return "", False
        response = response[response.index(end) + 1:]
    while response and response[-1] in eos:
        response = response[:-1]
    return tokenizer.decode(response, skip_special_tokens=False), True


class StructuredOutput:
    def __init__(self, tokenizer, tokenizer_data, schema, prefix_open):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        self.direct = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(schema))
        self.returned = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(schema))
        self.tokenizer = tokenizer
        self.prefix_open = prefix_open
        self.prefix_length = None
        self.begin = tokenizer.get_vocab().get("<think>")
        self.end = tokenizer.get_vocab().get("</think>")

    def __call__(self, batch, tokens):
        if self.prefix_length is None:
            self.prefix_length = len(tokens)
        response = tokens[self.prefix_length:].tolist()
        if self.prefix_open or reasoning_started(self.tokenizer, response):
            if self.end is not None and self.end in response:
                return self.returned(batch, tokens)
            return None
        allowed = self.direct(batch, tokens)
        if self.begin is not None and self.end is not None and not self.tokenizer.decode(response, skip_special_tokens=False).strip():
            allowed = [*allowed, self.begin]
        return allowed


class SchemaLogitsProcessor:
    def __init__(self, grammar):
        from transformers import PrefixConstrainedLogitsProcessor
        self.grammar = grammar
        self.allowed = None
        self.native = PrefixConstrainedLogitsProcessor(self.allowed_tokens, num_beams=1)

    def allowed_tokens(self, batch, tokens):
        return self.allowed

    def __call__(self, input_ids, scores):
        self.allowed = self.grammar(0, input_ids[0])
        return scores if self.allowed is None else self.native(input_ids, scores)


class GenerationPolicy:
    def __init__(self, backbone, tokenizer, config, head, prefix_length, max_tokens,
                 grammar, device, sample=None, recorded=None):
        from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
        native = backbone.generation_config
        if recorded is not None:
            settings = GenerationConfig.from_dict(deepcopy(recorded))
        elif head == "agent":
            settings = deepcopy(native)
            unused = settings.update(**config.generation)
            if unused:
                raise ValueError("Unknown native generation settings")
        else:
            # Full support keeps GRPO behavior, current, and reference likelihoods comparable.
            settings = GenerationConfig(do_sample=True, temperature=float(config.policy_temperature), top_k=0,
                                        eos_token_id=native.eos_token_id, bos_token_id=native.bos_token_id,
                                        pad_token_id=native.pad_token_id)
        if sample is not None:
            settings.do_sample = sample
        if settings.num_beams != 1 or settings.num_return_sequences != 1 or settings.penalty_alpha:
            raise ValueError("Store continuations require single-sequence sampling or greedy decoding")
        if settings.eos_token_id is None:
            settings.eos_token_id = tokenizer.eos_token_id
        if settings.pad_token_id is None:
            settings.pad_token_id = tokenizer.pad_token_id
        settings.max_new_tokens = max_tokens
        settings.max_length = prefix_length + max_tokens
        settings.validate()
        self.state = settings.to_dict()
        backbone._prepare_special_tokens(settings, device=device)
        self.eos = set(settings._eos_token_tensor.tolist()) if settings._eos_token_tensor is not None else set()
        processors = LogitsProcessorList([SchemaLogitsProcessor(grammar)]) if grammar is not None else LogitsProcessorList()
        self.processors = backbone._get_logits_processor(
            settings, input_ids_seq_length=prefix_length, encoder_input_ids=None,
            prefix_allowed_tokens_fn=None, logits_processor=processors, device=device)
        self.stopping = backbone._get_stopping_criteria(settings, stopping_criteria=StoppingCriteriaList(),
                                                       tokenizer=tokenizer)
        self.settings = settings

    def scores(self, history, logits):
        return self.processors(history, logits.float().unsqueeze(0))[0]

    def draw(self, scores, generator):
        if not torch.isfinite(scores).any():
            raise RuntimeError("Generation has no admissible vocabulary token")
        logprobs = scores.log_softmax(-1)
        if self.settings.do_sample:
            token = int(torch.multinomial(logprobs.exp(), 1, generator=generator))
            return token, logprobs[token]
        return int(scores.argmax()), scores.new_zeros(())
