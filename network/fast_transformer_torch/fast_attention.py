import numpy as np
import torch
import torch.nn as nn
from rotary_embedding_torch import apply_rotary_emb
from einops import reduce

class FastAttention(nn.Module):
    """Fast Attention Module of FastTransformer architecture
       https://arxiv.org/abs/2108.09084
    """
    def __init__(self, dim, mask, heads=8, dim_head=64, max_seq_len=None, pos_emb=None):
        super(FastAttention, self).__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.mask = mask

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if pos_emb is None and max_seq_len is None:
            raise Exception(
                "If you are using Rotary positional embeddings, max_seq_len must be passed in"
            )

        self.pos_emb = pos_emb
        self.use_rotary_emb = self.pos_emb is not None
        self.max_seq_len = max_seq_len

        # reduce pairs of consecutive feature dimension before doing projection to attention logits
        if pos_emb is None:
            kv_attn_proj_divisor = 1
        else:
            kv_attn_proj_divisor = 2

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias=False)
        self.to_k_attn_logits = nn.Linear(dim_head // kv_attn_proj_divisor, 1, bias=False)

        self.to_r =nn.Linear(dim_head // kv_attn_proj_divisor, dim_head)
        self.to_out = nn.Linear(inner_dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.tensor) -> torch.tensor:
        b, n, _ = x.size()

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q, k, v = map(lambda t: t.contiguous().view(b, self.heads, n, -1), (q,k,v))
        mask_value = -np.inf
        mask = self.mask.unsqueeze(1)


        # relative positional encoding is needed
        if self.use_rotary_emb:
            frequencies = self.pos_emb(torch.arange(self.max_seq_len), cache_key = self.max_seq_len)
            frequencies = frequencies[:n]
            frequencies = frequencies[None, None, ...]
            q_agg, k_agg, v_agg = map(
                lambda t: apply_rotary_emb(frequencies, t), (q, k, v)
            )
        else:
            q_agg, k_agg, v_agg = q, k, v

        # query attention logits
        q_attn_logits = self.to_q_attn_logits(q).squeeze(-1) * self.scale
        #q_attn_logits.masked_fill_(mask, mask_value)
        q_attn = self.softmax(q_attn_logits)

        # # global query token
        global_q = torch.einsum("b h n, b h n d -> b h d", q_attn, q_agg)
        global_q = global_q.unsqueeze(-2)

        # bias keys with global query token
        k = k * global_q

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension
        if self.use_rotary_emb:
            k = reduce(k, "b h n (d r) -> b h n d", "sum", r=2)

        # key attention logits
        k_attn_logits = self.to_k_attn_logits(k).squeeze(-1) * self.scale
        #k_attn_logits.masked_fill_(mask, mask_value)
        k_attn = self.softmax(k_attn_logits)

        # global key token
        global_k = torch.einsum("b h n, b h n d -> b h d", k_attn, k_agg)
        global_k = global_k.unsqueeze(-2)

        # bias the values with global keys
        v = v_agg * global_k

        if self.use_rotary_emb:
            v = reduce(v, "b h n (d r) -> b h n d", "sum", r=2)

        r = self.to_r(v)

        # add queries as a residual
        r = r + q

        # combine heads
        r = r.contiguous().view(b, n, -1)

        return self.to_out(r)

if __name__ == '__main__':
    dim = 32
    x = torch.randn(2, 240, dim)
    print(x.shape)
    mask = torch.ones([1, 240], dtype=torch.bool)

    attn = FastAttention(dim, mask,  max_seq_len=240)
    y = attn(x)
    print(y.shape)
