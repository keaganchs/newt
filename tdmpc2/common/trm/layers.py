from typing import Tuple
from copy import deepcopy

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"
	

class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation.
	"""

	def __init__(self, *args, act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act

	def forward(self, x):
		x = super().forward(x)
		return self.act(self.ln(x))

	def __repr__(self):
		if isinstance(self.act, nn.Sequential):
			act = '[' + ', '.join([m.__class__.__name__ for m in self.act]) + ']'
		else:
			act = self.act.__class__.__name__
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}, "\
			f"act={act})"


def mlp(in_dim, mlp_dims, out_dim, act=None):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1]))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def policy(in_dim, mlp_dims, out_dim, act=None):
	"""
	Policy network for TD-MPC2.
	Vanilla MLP with ReLU activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(nn.Linear(dims[i], dims[i+1]))
		mlp.append(nn.ReLU())
	mlp.append(nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


class QEnsemble(nn.Module):
	"""
	Vectorized ensemble of Q-networks. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		in_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
		mlp_dims = 2*[cfg.mlp_dim]
		out_dim = max(cfg.num_bins, 1)
		self._Qs = nn.ModuleList([mlp(in_dim, mlp_dims, out_dim) for _ in range(cfg.num_q)])
		if cfg.compile:
			if cfg.rank == 0:
				print('Compiling QEnsemble forward...')
			self._forward = torch.compile(self._forward_impl, mode='reduce-overhead')
		else:
			self._forward = self._forward_impl
	
	def _forward_impl(self, x):
		outs = [q(x) for q in self._Qs]
		return torch.stack(outs, dim=0)

	def forward(self, x):
		return self._forward(x)


class QOnlineTargetEnsemble(nn.Module):
	"""
	Online and target Q-ensembles for TD-MPC2. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.online = QEnsemble(cfg)
		self.target = deepcopy(self.online)
		self.tau = cfg.tau
		self.target.train(False)
		self.track_grad(False, network='target')

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		self.online.train(mode)
		self.target.train(False)
		return self
	
	def track_grad(self, mode=True, network='online'):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		"""
		assert network in {'online', 'target'}
		module = self.online if network == 'online' else self.target
		for p in module.parameters():
			p.requires_grad_(mode)

	@torch.no_grad()
	def hard_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.copy_(op.data)

	@torch.no_grad()
	def soft_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.lerp_(op.data, self.tau)

	def forward(self, x, target=False):
		if target:
			return self.target(x)
		else:
			return self.online(x)
		

def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	if cfg.obs == 'state':
		out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	elif cfg.obs == 'rgb':
		out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim + cfg.obs_shape['rgb'][0], max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	else:
		raise NotImplementedError(f"Unexpected observation type: {cfg.obs}")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Attempts to automatically convert a model checkpoint (e.g. add/remove DDP 'module.' prefixes).
	"""
	encoder_key = 'module._encoder.state.0.weight'
	if encoder_key in source_state_dict and encoder_key not in target_state_dict:
		# Remove 'module.' prefix from all keys in source_state_dict
		source_state_dict = {k[len('module.'):]: v for k, v in source_state_dict.items()}
	if encoder_key in target_state_dict and encoder_key not in source_state_dict:
		# Add 'module.' prefix to all keys in source_state_dict
		source_state_dict = {'module.' + k: v for k, v in source_state_dict.items()}

	for key in ['_encoder.state.0.weight', 'module._encoder.state.0.weight']:
		if key in target_state_dict and key in source_state_dict and \
				target_state_dict[key].shape != source_state_dict[key].shape:
			# possible rgb input in target but not in source, we should pad
			print('Warning: unexpected shape mismatch in encoder weights, attempting to pad source weights...')
			pad = target_state_dict[key].shape[1] - source_state_dict[key].shape[1]
			assert pad > 0, 'pad should be positive'
			pad_tensor = torch.zeros(source_state_dict[key].shape[0], pad, device=source_state_dict[key].device)
			source_state_dict[key] = torch.cat([source_state_dict[key], pad_tensor], dim=1)

	if '_action_masks' in target_state_dict and '_action_masks' in source_state_dict and \
			source_state_dict['_action_masks'].shape != target_state_dict['_action_masks'].shape:
		# repeat first dimension to match
		source_state_dict['_action_masks'] = source_state_dict['_action_masks'].repeat(
			target_state_dict['_action_masks'].shape[0] // source_state_dict['_action_masks'].shape[0], 1)
		if '_task_emb.weight' in source_state_dict:
			source_state_dict['_task_emb.weight'] = source_state_dict['_task_emb.weight'].repeat(
				target_state_dict['_action_masks'].shape[0] // source_state_dict['_task_emb.weight'].shape[0], 1)
		
	if '_task_emb.weight' in source_state_dict and not '_task_emb.weight' in target_state_dict:
		# delete task embedding from source state dict
		source_state_dict.pop('_task_emb.weight', None)

	return source_state_dict


def print_mismatched_tensors(target_state_dict, source_state_dict):
	target_keys = set(target_state_dict.keys())
	source_keys = set(source_state_dict.keys())

	# Keys in source but not in target
	for key in source_keys - target_keys:
		print(f"[Extra in source] {key}: shape={tuple(source_state_dict[key].shape)}")

	# Keys in target but not in source
	for key in target_keys - source_keys:
		print(f"[Missing in source] {key}: expected shape={tuple(target_state_dict[key].shape)}")

	# Keys present in both but with shape mismatch
	for key in target_keys & source_keys:
		try:
			t_shape = tuple(target_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in target_state_dict] {key}: {e}")
			continue
		try:
			s_shape = tuple(source_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in source_state_dict] {key}: {e}")
			continue
		if t_shape != s_shape:
			print(f"[Shape mismatch] {key}: target={t_shape}, source={s_shape}")


"""
Begin TRM Layers, copied from the original implementation
"""

CosSin = Tuple[torch.Tensor, torch.Tensor]


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)
	

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Real Weights
        # Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        # Local embedding IDs, not persistent
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)
            
        # Training mode, fill puzzle embedding from weights
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


"""
End TRM Layers
"""

