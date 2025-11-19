from __future__ import annotations
import math
from dataclasses import dataclass

from pathlib import Path
from typing import Tuple
import sys


from tinygrad import Tensor, nn, UOp, TinyJit,  dtypes
from tinygrad.apps.debug import log_tensor
from tinygrad.dtype import DType


from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


@dataclass
class ModelConfig:
  num_hidden_layers: int = 24  # block_count
  num_experts: int = 32  # exper_count
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
    # qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
    # self.qkv = nn.Linear(config.hidden_size, qkv_dim)
    self.attn_q = nn.Linear(config.hidden_size, config.head_dim * config.num_attention_heads, bias=True)
    self.attn_k = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=True)
    self.attn_v = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=True)
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
    # qkv = self.qkv(t)
    # q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
    # k = qkv[
    #   :,
    #   self.num_attention_heads * self.head_dim : (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
    # ].contiguous()
    # v = qkv[
    #   :,
    #   (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
    #   * self.head_dim,
    # ].contiguous()

    q = self.attn_q(t)
    k = self.attn_k(t)
    v = self.attn_v(t)
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

    # the model weights loaded from GGUF are already split into ffn_up and ffn_gate experts
    self.gate_up_proj = Tensor.zeros(
       config.num_experts,
       config.intermediate_size * 2,
       90, 16
    )
    self.gate_up_proj_bias = Tensor.zeros(config.num_experts, config.intermediate_size * 2)
    self.gate_up_proj_scales = Tensor.zeros(config.num_experts, config.intermediate_size * 2, 90)

    self.down_proj = Tensor.zeros(
       config.num_experts,
       config.intermediate_size,
       90,16
    )
    self.down_proj_bias = Tensor.zeros(config.num_experts, config.intermediate_size)
    self.down_proj_scales = Tensor.zeros(config.num_experts, config.intermediate_size, 90)

    #self.mlp1_weights = Tensor.zeros(config.num_experts, config.intermediate_size * 2, config.intermediate_size)
    #self.mlp2_weights = Tensor.zeros(config.num_experts, config.intermediate_size, config.intermediate_size)

  def build_layers(self):
    self.mlp1_weights = _get_mxfp4_tensor(self.gate_up_proj, self.gate_up_proj_scales)
    self.mlp2_weights = _get_mxfp4_tensor(self.down_proj, self.down_proj_scales)


  def __call__(self, x: Tensor) -> Tensor:
    t = self.norm(x)
    g = self.gate(t)
    expert_values, expert_indices = g.topk(k=self.experts_per_token, dim=-1)
    expert_weights = expert_values.softmax(1)

    # MLP #1
    mlp1_weight_exp = self.mlp1_weights[expert_indices, ...]
    mlp1_bias_exp = self.gate_up_proj_bias[expert_indices, ...]
    t = Tensor.einsum("beck,bk->bec", mlp1_weight_exp.squeeze(0), t.squeeze(0))
    t += mlp1_bias_exp.squeeze(0)
    t = swiglu(t, limit=self.swiglu_limit)


    # MLP #2
    mlp2_weight_sliced = self.mlp2_weights[expert_indices, ...]
    mlp2_bias = self.down_proj_bias[expert_indices, ...]
    t = Tensor.einsum("beck,bek->bec", mlp2_weight_sliced.squeeze(0), t.squeeze(0))
    t += mlp2_bias.squeeze(0)

    # Weighted sum of experts
    t = Tensor.einsum("bec,be->bc", t, expert_weights.squeeze(0))

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
    self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    self.block = [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    self.norm = nn.RMSNorm(config.hidden_size)
    self.unembedding = nn.Linear(
      config.hidden_size,
      config.vocab_size,
      bias=False,
    )

    self.forward_jit = TinyJit(self.forward)
    self.max_context = 2048

  def build_layers(self):
    for block in self.block:
      block.mlp.build_layers()

  def forward(self, x: Tensor) -> Tensor:
    x = self.embedding(x)
    for block in self.block:
      x = block(x)
    x = self.norm(x)
    x = self.unembedding(x)
    out = x[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)
    return out

  def generate(self, tokens: list[int], start_pos=0):
    start_pos = 0
    t = Tensor([tokens[start_pos:]], dtype="int32")
    # self.forward_jit.reset()  # TODO: why is this required? root cause the issue and make it not be needed
    while len(tokens) < self.max_context:
      out = self(t, start_pos)
      next_id = int(out.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

  def __call__(self, tokens: Tensor, start_pos: int | UOp = 0) -> Tensor:
    return self.forward_jit(tokens)

models = {
  "20B": "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf"
}


def rename_state_dict_keys(state_dict: dict, layers: int) -> dict:
  # norm.weight
  state_dict["norm.weight"] = state_dict.pop("model.norm.weight")
  state_dict["unembedding.weight"] = state_dict.pop("lm_head.weight")
  state_dict["embedding.weight"] = state_dict.pop("model.embed_tokens.weight")

  for i in range(0, layers):
    # attention
    state_dict[f"block.{i}.attn.sinks"] = state_dict.pop(f"model.layers.{i}.self_attn.sinks")
    state_dict[f"block.{i}.attn.norm.weight"] = state_dict.pop(f"model.layers.{i}.input_layernorm.weight")

    # blk.0.attn.attn_q.weight
    state_dict[f"block.{i}.attn.attn_q.weight"] = state_dict.pop(f"model.layers.{i}.self_attn.q_proj.weight")
    state_dict[f"block.{i}.attn.attn_q.bias"] = state_dict.pop(f"model.layers.{i}.self_attn.q_proj.bias")
    state_dict[f"block.{i}.attn.attn_k.weight"] = state_dict.pop(f"model.layers.{i}.self_attn.k_proj.weight")
    state_dict[f"block.{i}.attn.attn_k.bias"] = state_dict.pop(f"model.layers.{i}.self_attn.k_proj.bias")
    state_dict[f"block.{i}.attn.attn_v.weight"] = state_dict.pop(f"model.layers.{i}.self_attn.v_proj.weight")
    state_dict[f"block.{i}.attn.attn_v.bias"] = state_dict.pop(f"model.layers.{i}.self_attn.v_proj.bias")
    # model.layers.0.atself_attn.t.weight
    state_dict[f"block.{i}.attn.out.weight"] = state_dict.pop(f"model.layers.{i}.self_attn.o_proj.weight")
    state_dict[f"block.{i}.attn.out.bias"] = state_dict.pop(f"model.layers.{i}.self_attn.o_proj.bias")

    # mlp
    # model.layers.0.mlp.norm.weight
    state_dict[f"block.{i}.mlp.norm.weight"] = state_dict.pop(f"model.layers.{i}.post_attention_layernorm.weight")
    # model.layers.0.mlp.gate.weight
    state_dict[f"block.{i}.mlp.gate.weight"] = state_dict.pop(f"model.layers.{i}.mlp.router.weight")
    state_dict[f"block.{i}.mlp.gate.bias"] = state_dict.pop(f"model.layers.{i}.mlp.router.bias")

    # model.layers.0.mlp.mlp1_weight
    # state_dict[f'block.{i}.mlp.mlp1_weight'] = state_dict.pop(f'model.layers.{i}.ffn_gate_exps.weight')
    # state_dict[f'block.{i}.mlp.mlp1_bias'] = state_dict.pop(f'model.layers.{i}.ffn_gate_exps.bias')

    # ffn_up_exps
    state_dict[f"block.{i}.mlp.gate_up_proj"] = state_dict.pop(f"model.layers.{i}.mlp.experts.gate_up_proj_blocks")
    state_dict[f"block.{i}.mlp.gate_up_proj_bias"] = state_dict.pop(f"model.layers.{i}.mlp.experts.gate_up_proj_bias")
    state_dict[f"block.{i}.mlp.gate_up_proj_scales"] = state_dict.pop(f"model.layers.{i}.mlp.experts.gate_up_proj_scales")

    # model.layers.0.mlp.mlp2_weight
    state_dict[f"block.{i}.mlp.down_proj"] = state_dict.pop(f"model.layers.{i}.mlp.experts.down_proj_blocks")
    state_dict[f"block.{i}.mlp.down_proj_bias"] = state_dict.pop(f"model.layers.{i}.mlp.experts.down_proj_bias")
    state_dict[f"block.{i}.mlp.down_proj_scales"] = state_dict.pop(f"model.layers.{i}.mlp.experts.down_proj_scales")

  return state_dict

def main():

  path = Path('/Users/calebouellette/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')
  state_dict = nn.state.safe_load(path.joinpath('model-00000-of-00002.safetensors'))
  state_dict = state_dict | nn.state.safe_load(path.joinpath('model-00001-of-00002.safetensors'))
  state_dict = state_dict | nn.state.safe_load(path.joinpath('model-00002-of-00002.safetensors'))

  model_config = ModelConfig() # TODO Load
  model = Transformer(model_config)

  nn.state.load_state_dict(model, rename_state_dict_keys(state_dict, model_config.num_hidden_layers))
  model.build_layers()

  encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

  system_message_content = (
      SystemContent.new()
      .with_reasoning_effort(REASONING_EFFORT['low'])
  )

  system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
  messages = [system_message]
  conversation = Conversation.from_messages(messages)
  tokens = encoding.render_conversation(conversation)

  # load generation_config.json from path

  while 1:
    start_pos = len(tokens) - 1
    for next_id in model.generate(tokens, start_pos):
      sys.stdout.write(encoding.decode([next_id]))
      sys.stdout.flush()
      if next_id == 123: # TODO fix
        break


FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def _get_mxfp4_tensor(
    blocks: Tensor,
    scales: Tensor,
    *,
    dtype: DType = dtypes.bfloat16,
    rows_per_chunk: int = 16384 * 512,
) -> Tensor:


    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    lut = Tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total   = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = Tensor.empty(rows_total, B * 2, dtype=dtype)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).cast(dtypes.long)
        idx_hi = (blk >> 4).cast(dtypes.long)

        sub = out[r0:r1].contiguous()
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        # torch.ldexp(sub, exp, out=sub)
        exp = Tensor([2.0]).pow(exp.unsqueeze(-1))
        sub = sub * exp # seems like we should be assigning to out
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
