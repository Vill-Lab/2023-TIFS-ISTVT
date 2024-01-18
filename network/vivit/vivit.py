import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from network.vivit.module import Attention, PreNorm, FeedForward, TemporalOnlyAttention, SpatialOnlyAttention, TemporalResidualAttention, LocalSpatialAttention
from network.models_copy import model_selection
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 728, depth = 12, heads = 8, pool = 'cls', in_channels = 728, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            #nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
    


class STTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TemporalResidualAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, SpatialOnlyAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn_t, attn_s, ff in self.layers:
            x = attn_s(attn_t(x)) + x
            x = ff(x) + x
        return self.norm(x)

class DSTTr(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 728, depth = 12, heads = 8, pool = 'cls', in_channels = 728, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c h w -> b t (h w) c'), 
            #nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = STTransformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t = t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        cls_temporal_tokens = repeat(self.temporal_token, 't () d -> b t n d', b = b, n = n + 1)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.transformer(x)
        x = rearrange(x, 'b (t n) d -> b t n d', n = 19 * 19 + 1)

        x = x[:, 0, 0]

        return self.mlp_head(x)

class VanillaTr(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 728, depth = 12, heads = 8, pool = 'cls', in_channels = 728, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c h w -> b t (h w) c'),
            nn.Linear(patch_dim, dim),
            Rearrange('b t n d -> b (t n) d'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, (num_frames * num_patches) + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.transformer(x)

        x = x[:, 0]

        return self.mlp_head(x)

class XceptionVidTr(nn.Module):
    def __init__(self):
        super(XceptionVidTr, self).__init__()
        self.xcep = model_selection(modelname= 'xception', num_out_classes = 2, dropout=0.5, batch_size = 1)
        #self.xcep.load_state_dict(torch.load('/mnt/data/DFD/output/xception_nt_lq_5x5/best.pkl'))
        #for child in self.xcep.children():
        #    for param in child.parameters():
        #        param.requires_grad = False
        self.vit = DSTTr(19, 1, 1, 6)
    def forward(self, x):
        b, _, _, _,_ = x.shape
        x = rearrange(x, 'b t c n d -> (b t) c n d')
        x = self.xcep.low_level_features(x)
        x = rearrange(x, '(b t) c n d -> b t c n d', b = b)
        #residual = torch.cat((x[:,0:1], x[:,:-1] - x[:,1:]) , dim = 1)
        return self.vit(x)

if __name__ == "__main__":
    
    img = torch.ones([12, 4, 728, 19, 19])
    
    model = VidTr(19, 1, 1, 4)    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]
