import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from network.fast_transformer_torch.fast_attention import FastAttention

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.mult = mult

        self.ff = nn.Sequential(
                nn.Linear(dim, dim * mult),
                nn.GELU(),
                nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.ff(x)


class FastTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads=8,
        dim_head=64,
        ff_mult=4,
        absolute_pos_emb=False,
        mask=None,
        patch_height = 16, 
        patch_width = 16,
        image_size = 224
    ):
        super(FastTransformer, self).__init__()

        #self.token_emb = nn.Embedding(num_tokens, dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_width * patch_height * 3, dim),
        )
        num_patches = (image_size // patch_height) * (image_size // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mask = mask
        self.dropout = nn.Dropout(0.1)

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, ( "dimension of the head must be divisible by 4 to use rotary embeddings")
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        fast_tranformer_layers = []

        for _ in range(depth):
            attn = FastAttention(
                dim = dim,
                dim_head=dim_head,
                heads=heads,
                pos_emb=layer_pos_emb,
                max_seq_len=max_seq_len + 1,
                mask = self.mask,
            )
            ff = FeedForward(dim, mult=ff_mult)

            fast_tranformer_layers.append(PreNorm(dim, attn))
            fast_tranformer_layers.append(PreNorm(dim, ff))

        self.fast_tranformer_layers = nn.ModuleList(fast_tranformer_layers)

        first_block = self.fast_tranformer_layers[0]
        for block in self.fast_tranformer_layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        self.to_logits = nn.Sequential(nn.LayerNorm(dim),
                                       nn.Linear(dim, num_tokens),
                                       )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for current_layer in self.fast_tranformer_layers:
            x = current_layer(x) + x

        x = x[:,0]
        return self.to_logits(x)


if __name__ == '__main__':
    mask = torch.ones([16, 197], dtype=torch.bool)
    model = FastTransformer(num_tokens = 1,
                            dim = 512,
                            depth = 4,
                            max_seq_len = 196,
                            absolute_pos_emb = True, # Absolute positional embeddings
                            mask = mask
                            )
    x = torch.rand(16,3,224,224)
    logits = model(x) # (1, 4096, 20000)
    print(logits.shape)