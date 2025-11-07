from __future__ import annotations
import math
from dataclasses import dataclass

from typing import Tuple
import sys, argparse
from tinygrad import Tensor, nn, UOp, TinyJit, getenv


class SimpleTokenizer:
  def __init__(self, normal_tokens: dict[str, int], special_tokens: dict[str, int]):
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    def ucat_range(pre: str):
      return "".join(re.escape(chr(cp)) for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)).startswith(pre))

    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
      + f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    )
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}

  @staticmethod
  def from_gguf_kv(kv: dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    # if kv["tokenizer.ggml.pre"] not in ("llama3", "llama-v3", "llama-bpe"):
    # raise ValueError(f"Invalid tokenizer preset '{kv['tokenizer.ggml.pre']}'")
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = helpers.partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens))

  def _encode_word(self, word: bytes) -> list[int]:
    if (early_token := self._normal_tokens.get(word)) is not None:
      return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j] + parts[j + 1], sys.maxsize), j) for j in range(len(parts) - 1)])[1]
      if i == -1:
        break
      parts[i : i + 2] = [parts[i] + parts[i + 1]]
    try:
      return [self._normal_tokens[p] for p in parts]
    except KeyError:
      raise RuntimeError("token not found")

  def _encode_sentence(self, chunk: str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]

  def encode(self, text: str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos : match.start(0)]) + [self._special_tokens[text[match.start(0) : match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids: list[int]) -> str:
    return b"".join(self._tok2bytes[tid] for tid in ids).decode()

  def role(self, role: str):
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")


@dataclass
class ModelConfig:
  num_hidden_layers: int = 36  # block_count
  num_experts: int = 28  # exper_count
  experts_per_token: int = 4  # expert_used_count
  vocab_size: int = 201088  # ??
  hidden_size: int = 2880  #
  intermediate_size: int = 2880
  swiglu_limit: float = 7.0
  head_dim: int = 64
  num_attention_heads: int = 64  # head_count
  num_key_value_heads: int = 8  # head_count_kv
  sliding_window: int = 128  # sliding_window
  initial_context_length: int = 4096  # scaling.original_context_length
  rope_theta: float = 150000.0
  rope_scaling_factor: float = 32.0
  rope_ntk_alpha: float = 1.0
  rope_ntk_beta: float = 32.0


def build_config_from_kv(kv: dict) -> ModelConfig:
  """Convert GGUF KV pairs to ModelConfig.

  Mapping from GGUF keys to ModelConfig fields:
  - gpt-oss.block_count -> num_hidden_layers
  - gpt-oss.expert_count -> num_experts
  - gpt-oss.expert_used_count -> experts_per_token
  - gpt-oss.embedding_length -> hidden_size
  - gpt-oss.feed_forward_length -> intermediate_size
  - gpt-oss.attention.head_count -> num_attention_heads
  - gpt-oss.attention.head_count_kv -> num_key_value_heads
  - gpt-oss.attention.key_length -> head_dim
  - gpt-oss.attention.sliding_window -> sliding_window
  - gpt-oss.rope.freq_base -> rope_theta
  - gpt-oss.rope.scaling.factor -> rope_scaling_factor
  - gpt-oss.context_length -> initial_context_length
  """
  return ModelConfig(
    num_hidden_layers=kv.get("gpt-oss.block_count", 36),
    num_experts=kv.get("gpt-oss.expert_count", 28),
    experts_per_token=kv.get("gpt-oss.expert_used_count", 4),
    hidden_size=kv.get("gpt-oss.embedding_length", 2880),
    intermediate_size=kv.get("gpt-oss.feed_forward_length", 2880),
    num_attention_heads=kv.get("gpt-oss.attention.head_count", 64),
    num_key_value_heads=kv.get("gpt-oss.attention.head_count_kv", 8),
    head_dim=kv.get("gpt-oss.attention.key_length", 64),
    sliding_window=kv.get("gpt-oss.attention.sliding_window", 128),
    rope_theta=kv.get("gpt-oss.rope.freq_base", 150000.0),
    rope_scaling_factor=kv.get("gpt-oss.rope.scaling.factor", 32.0),
    initial_context_length=kv.get("gpt-oss.context_length", 4096),
  )


def _apply_rotary_emb(
  x: Tensor,
  cos: Tensor,
  sin: Tensor,
) -> Tensor:
  cos = cos.unsqueeze(-2)
  sin = sin.unsqueeze(-2)
  x1, x2 = x.chunk(2, dim=-1)
  o1 = x1 * cos - x2 * sin
  o2 = x2 * cos + x1 * sin
  return o1.cat(o2, dim=-1)


class RotaryEmbedding:
  def __init__(
    self,
    head_dim: int,
    base: int,
    initial_context_length: int = 4096,
    scaling_factor: float = 1.0,
    ntk_alpha: float = 1.0,
    ntk_beta: float = 32.0,
  ) -> None:
    super().__init__()
    self.head_dim = head_dim
    self.base = base
    self.initial_context_length = initial_context_length
    self.scaling_factor = scaling_factor
    self.ntk_alpha = ntk_alpha
    self.ntk_beta = ntk_beta

  def _compute_concentration_and_inv_freq(self) -> Tuple[float, Tensor]:
    """See YaRN paper: https://arxiv.org/abs/2309.00071"""
    freq = self.base ** (Tensor.arange(0, self.head_dim, 2) / self.head_dim)
    if self.scaling_factor > 1.0:
      concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration

      d_half = self.head_dim / 2
      # NTK by parts
      low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
      high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
      assert 0 < low < high < d_half - 1

      interpolation = 1.0 / (self.scaling_factor * freq)
      extrapolation = 1.0 / freq

      ramp = (Tensor.arange(d_half, device=freq.device) - low) / (high - low)
      mask = 1 - ramp.clamp(0, 1)

      inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
      concentration = 1.0
      inv_freq = 1.0 / freq

    return concentration, inv_freq

  def _compute_cos_sin(self, num_tokens: int):
    concentration, inv_freq = self._compute_concentration_and_inv_freq()
    t = Tensor.arange(num_tokens)
    freqs = Tensor.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin

  def __call__(
    self,
    query: Tensor,
    key: Tensor,
  ) -> tuple[Tensor, Tensor]:
    num_tokens = query.shape[0]
    cos, sin = self._compute_cos_sin(int(num_tokens))

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_dim)
    query = _apply_rotary_emb(query, cos, sin)
    query = query.reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_dim)
    key = _apply_rotary_emb(key, cos, sin)
    key = key.reshape(key_shape)
    return query, key


def sdpa(Q: Tensor, K: Tensor, V: Tensor, S: Tensor, sm_scale, sliding_window=0):
  # sliding_window == 0 means no sliding window
  n_tokens, n_heads, q_mult, d_head = Q.shape
  assert K.shape == (n_tokens, n_heads, d_head)
  assert V.shape == (n_tokens, n_heads, d_head)
  K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
  V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
  S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)

  new_full = Tensor.full((n_tokens, n_tokens), -float("inf"))
  mask = new_full.triu(diagonal=1)
  if sliding_window > 0:
    new_full_two = Tensor.full((n_tokens, n_tokens), -float("inf"))
    mask += new_full_two.tril(diagonal=-sliding_window)
  QK = Tensor.einsum("qhmd,khmd->hmqk", Q, K)
  QK = QK * sm_scale
  QK += mask[None, None, :, :]
  QK = QK.cat(S, dim=-1)
  W = QK.softmax(axis=-1)
  W = W[..., :-1]
  attn = Tensor.einsum("hmqk,khmd->qhmd", W, V)
  return attn.reshape(n_tokens, -1)


class AttentionBlock:
  def __init__(
    self,
    config: ModelConfig,
    layer_idx: int = 0,
  ):
    super().__init__()
    self.head_dim = config.head_dim
    self.num_attention_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    # Only apply sliding window to every other layer
    self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
    self.sinks = Tensor.zeros(config.num_attention_heads)
    self.norm = nn.RMSNorm(config.hidden_size)
    qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
    self.qkv = nn.Linear(config.hidden_size, qkv_dim)
    self.out = nn.Linear(
      config.head_dim * config.num_attention_heads,
      config.hidden_size,
    )
    self.sm_scale = 1 / math.sqrt(config.head_dim)
    self.rope = RotaryEmbedding(
      config.head_dim,
      int(config.rope_theta),
      initial_context_length=config.initial_context_length,
      scaling_factor=config.rope_scaling_factor,
      ntk_alpha=config.rope_ntk_alpha,
      ntk_beta=config.rope_ntk_beta,
    )

  def __call__(self, x: Tensor) -> Tensor:
    t = self.norm(x)
    qkv = self.qkv(t)
    q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
    k = qkv[
      :,
      self.num_attention_heads * self.head_dim : (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
    ].contiguous()
    v = qkv[
      :,
      (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
      * self.head_dim,
    ].contiguous()

    q = q.view(
      -1,
      self.num_key_value_heads,
      self.num_attention_heads // self.num_key_value_heads,
      self.head_dim,
    )
    k = k.view(-1, self.num_key_value_heads, self.head_dim)
    v = v.view(-1, self.num_key_value_heads, self.head_dim)
    q, k = self.rope(q, k)
    t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
    t = self.out(t)
    t = x + t
    return t


def swiglu(x: Tensor, alpha: float = 1.702, limit: float = 7.0):
  x_glu, x_linear = x[..., ::2], x[..., 1::2]
  # Clamp the input values
  x_glu = x_glu.clamp(None, limit)
  x_linear = x_linear.clamp(-limit, limit)
  out_glu = x_glu * (alpha * x_glu).sigmoid()
  # Note we add an extra bias of 1 to the linear layer
  return out_glu * (x_linear + 1)


class MLPBlock:
  def __init__(
    self,
    config: ModelConfig,
  ):
    super().__init__()
    self.num_experts = config.num_experts
    self.experts_per_token = config.experts_per_token
    self.swiglu_limit = config.swiglu_limit
    self.norm = nn.RMSNorm(config.hidden_size)
    self.gate = nn.Linear(config.hidden_size, config.num_experts)

    self.mlp1_weight = Tensor.zeros(
      config.num_experts,
      config.intermediate_size * 2,
      config.hidden_size,
    )
    self.mlp1_bias = Tensor.zeros(config.num_experts, config.intermediate_size * 2)

    self.mlp2_weight = Tensor.zeros(
      config.num_experts,
      config.hidden_size,
      config.intermediate_size,
    )
    self.mlp2_bias = Tensor.zeros(
      config.num_experts,
      config.hidden_size,
    )

  def __call__(self, x: Tensor) -> Tensor:
    t = self.norm(x)
    g = self.gate(t)
    expert_values, expert_indices = g.topk(k=self.experts_per_token, dim=-1)
    expert_weights = expert_values.softmax(1)

    # MLP #1
    mlp1_weight = self.mlp1_weight[expert_indices, ...]
    mlp1_bias = self.mlp1_bias[expert_indices, ...]
    t = Tensor.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    t = swiglu(t, limit=self.swiglu_limit)

    # MLP #2
    mlp2_weight = self.mlp2_weight[expert_indices, ...]
    mlp2_bias = self.mlp2_bias[expert_indices, ...]
    t = Tensor.einsum("beck,bek->bec", mlp2_weight, t)
    t += mlp2_bias

    # Weighted sum of experts
    t = Tensor.einsum("bec,be->bc", t, expert_weights)

    return x + t


class TransformerBlock:
  def __init__(
    self,
    config: ModelConfig,
    layer_idx: int,
  ):
    super().__init__()
    self.layer_idx = layer_idx
    self.attn = AttentionBlock(config, layer_idx)
    self.mlp = MLPBlock(config)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.attn(x)
    x = self.mlp(x)
    return x


class Transformer:
  def __init__(
    self,
    config: ModelConfig,
  ):
    super().__init__()
    self.token_embd = nn.Embedding(config.vocab_size, config.hidden_size)
    self.blk = [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    self.norm = nn.RMSNorm(config.hidden_size)
    self.unembedding = nn.Linear(
      config.hidden_size,
      config.vocab_size,
      bias=False,
    )

    self.forward_jit = TinyJit(self.forward)
    self.max_context = 2048

  def forward(self, x: Tensor) -> Tensor:
    x = self.embedding(x)
    for block in self.block:
      x = block(x)
    x = self.norm(x)
    x = self.unembedding(x)
    return x

  def generate(self, tokens: list[int], start_pos=0):
    v_start_pos = UOp.variable("start_pos", 1, self.max_context - 1)
    start_pos = 0
    t = Tensor([tokens[start_pos:]], dtype="int32")
    self.forward_jit.reset()  # TODO: why is this required? root cause the issue and make it not be needed
    while len(tokens) < self.max_context:
      t = self(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

  def __call__(self, tokens: Tensor, start_pos: int | UOp = 0) -> Tensor:
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens)

  # @staticmethod
  # def from_checkpoint(path: str, device: str | torch.device = "cuda") -> "Transformer":
  #   if not isinstance(device, torch.device):
  #     device = torch.device(device)

  #   config_path = os.path.join(path, "config.json")
  #   with open(config_path, "r") as f:
  #     json_config = json.load(f)
  #     config = ModelConfig(**json_config)

  #   model = Transformer(
  #     config=config,
  #   )
  #   model.eval()

  #   # Load weights
  #   my_rank = dist.get_rank() if dist.is_initialized() else 0
  #   world_size = dist.get_world_size() if dist.is_initialized() else 1
  #   per_rank_intermediate_size = config.intermediate_size // world_size

  #   checkpoint = Checkpoint(path, device)

  #   for name, param in model.named_parameters():
  #     loaded_tensor = checkpoint.get(name)

  #     # Note: it would be more efficient to do sharding before upcasting from MXFP4,
  #     # but for simplicity we do it after.
  #     if "mlp1" in name:  # both weight and bias
  #       loaded_tensor = loaded_tensor[
  #         :,
  #         my_rank * 2 * per_rank_intermediate_size : (my_rank + 1) * 2 * per_rank_intermediate_size,
  #         ...,
  #       ]
  #     elif "mlp2_weight" in name:  # only weight
  #       loaded_tensor = loaded_tensor[
  #         ...,
  #         my_rank * per_rank_intermediate_size : (my_rank + 1) * per_rank_intermediate_size,
  #       ]
  #     try:
  #       param.data.copy_(loaded_tensor)
  #     except:
  #       print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
  #       raise

  #   return model


# class TokenGenerator:
#   def __init__(self, checkpoint: str, device: torch.device):
#     self.device = device
#     self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

#   def generate(self, prompt_tokens: list[int], stop_tokens: list[int], temperature: float = 1.0, max_tokens: int = 0, return_logprobs: bool = False):
#     tokens = list(prompt_tokens)
#     num_generated_tokens = 0
#     while max_tokens == 0 or num_generated_tokens < max_tokens:
#       logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]
#       if temperature == 0.0:
#         predicted_token = torch.argmax(logits, dim=-1).item()
#       else:
#         probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
#         predicted_token = torch.multinomial(probs, num_samples=1).item()
#       tokens.append(predicted_token)
#       num_generated_tokens += 1

#       if return_logprobs:
#         logprobs = torch.log_softmax(logits, dim=-1)
#         selected_logprobs = logprobs[predicted_token].item()
#         yield predicted_token, selected_logprobs
#       else:
#         yield predicted_token

#       if predicted_token in stop_tokens:
#         break


models = {
  #  "20B": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "20B": "https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q6_K.gguf",
}


def rename_state_dict_keys(state_dict: dict, kv: dict) -> dict:
  for i in range(0, kv['gpt-oss.block_count']):
    # attention
    state_dict[f'blk.{i}.attn.sinks'] = state_dict.pop(f'blk.{i}.attn_sinks.weight')
    state_dict[f'blk.{i}.attn.norm.weight'] = state_dict.pop(f'blk.{i}.attn_norm.weight')
    state_dict[f'blk.{i}.attn.qkv.weight'] = state_dict.pop(f'blk.{i}.attn_qkv.weight')

  return state_dict


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", choices=list(models.keys()), default=list(models.keys())[0], help="Model size")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  args = parser.parse_args()

  # load the model
  #
  kv, state_dict = nn.state.gguf_load(Tensor.from_url(models[args.size]).to(None))

  model_config = build_config_from_kv(kv)

  model = Transformer(model_config)

  nn.state.load_state_dict(model, rename_state_dict_keys(state_dict, kv))

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int = kv["tokenizer.ggml.bos_token_id"]
  eos_id: int = kv["tokenizer.ggml.eos_token_id"]

  ids: list[int] = [bos_id]
  while 1:
    start_pos = len(ids) - 1
    try:
      ids += tok.role("user") + tok.encode(input(">>> ")) + [eos_id] + tok.role("assistant")
    except EOFError:
      break
    for next_id in model.generate(ids, start_pos):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id:
        break
