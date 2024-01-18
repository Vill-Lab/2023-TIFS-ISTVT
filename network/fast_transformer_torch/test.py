from fast_transformer_torch import FastTransformer
import torch

mask = torch.ones([16, 3], dtype=torch.bool)
model = ft.FastTransformer(num_tokens = 20000,
                        dim = 512,
                        depth = 2,
                        max_seq_len = 4096,
                        absolute_pos_emb = False, # Absolute positional embeddings
                        mask = mask
                        )

x = torch.rand(16,)

logits = model(x) # (1, 4096, 20000)
print(logits.shape)