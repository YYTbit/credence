import pytest
import torch


@pytest.fixture
def language_models():
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from credence.config import Config
    from credence.models import LanguageModels
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 \n",
              '{"title":"memory","text":"use the tool"}',
              '{"name":"count","code":"def count(items): return len(items)"}',
              '{"asset_id":"memory-00000001","action":"keep","target":null}']
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=160,
                                  special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                                        bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    config = Config(embedding_dim=4, context_budget=3, predictor_hidden=8,
                    adapter_rank=2, adapter_alpha=4, group_size=2, warmup_episodes=0)
    torch.manual_seed(17)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16,
                                       intermediate_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                       max_position_embeddings=4096, attention_dropout=0))
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora, adapter_name="writer")
    for name in ("builder", "curator", "encoder"):
        model.add_adapter(name, lora)
    instance = LanguageModels(model, tokenizer, config)
    instance.refresh_references()
    return instance
