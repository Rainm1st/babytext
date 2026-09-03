from transformers import PretrainedConfig


class GPTBertConfig(PretrainedConfig):
    model_type = "gpt_bert_mntp"

    def __init__(
        self,
        vocab_size=16000,
        hidden_size=720,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
        position_bucket_size=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        unk_token_id=3,
        mask_token_id=4,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.position_bucket_size = position_bucket_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id
        self.tie_word_embeddings = True
        self.architectures = ["GPTBertForMaskedLM"]
        self.auto_map = {
            "AutoConfig": "configuration_gpt_bert.GPTBertConfig",
            "AutoModel": "modeling_gpt_bert.GPTBertModel",
            "AutoModelForMaskedLM": "modeling_gpt_bert.GPTBertForMaskedLM",
            "AutoModelForCausalLM": "modeling_gpt_bert.GPTBertForCausalLM",
        }
