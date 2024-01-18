import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class SpatialOnlyAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b (t hw) (h d) -> b h t hw d', h = h, hw = 19 * 19 + 1), qkv)
        
        dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t hw d -> b (t hw) (h d)')
        out =  self.to_out(out)
        return out


class LocalSpatialAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        kernel_size = 7
        padding = 6
        stride = 3
        b, _, _, h = *x.shape, self.heads
        x = rearrange(x, 'b (t hw) d -> b t hw d', hw = 19 * 19 + 1)
        x = x[:,:,1:,:].squeeze()
        cls_token = x[:,:,0,:]
        x = rearrange(x, 'b t (h w) d -> (b t) d h w', h = 19, w = 19) # (bt, 728, 19, 19)
        x = F.unfold(x,(kernel_size,kernel_size),padding=padding,stride = stride) # (bt, (728*3*3), 441)
        x = rearrange(x, 'bt (d khw) n -> bt n khw d', khw = kernel_size ** 2) # (bt, 441, 9, 728)
        cls_token = repeat(cls_token, 'b t d -> (b t) n x d', n = x.shape[1], x = 1) # (bt, 441, 1, 728)
        x = torch.cat((cls_token,x),dim = 2) #  # (bt, 441, 10, 728)
        qkv = self.to_qkv(x).chunk(3, dim = -1) # (bt, 441, 10, 64 * 8 * 3)
        q, k, v = map(lambda t: rearrange(t, 'bt n khw (h d) -> bt h n khw d', h = h), qkv) # (bt, head, 441, 9, 64)
        
        dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale # (..., 10, 10)

        attn = dots.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v) # (..., 10, 64)
        out = out[:,:,:,1:,:]
        cls_token = out[:,:,:,0:1,:] #(bt, head, 441, 1, 64)
        cls_token = torch.mean(cls_token, dim = 2) # (bt, head, 1, 64)

        out = rearrange(out, 'bt h n khw d -> (bt h) (d khw) n', khw = kernel_size ** 2) # (bt, head, 441, 9, 728) -> (bt*head, (728*3*3), 441)
        out = F.fold(out,(19,19),(kernel_size,kernel_size),padding=padding,stride=stride) # (bt*head, 728, 19, 19)
        out = rearrange(out, '(bt head) d h w -> bt head (h w) d', head = h) #(bt, head, 361, 64)
        out = torch.cat((cls_token,out), dim = 2) #(bt, head, 362, 64)
        out = rearrange(out, '(b t) head hw d -> b (t hw) (head d)', b = b)

        out =  self.to_out(out)
        return out

class TemporalOnlyAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b (t hw) (h d) -> b h hw t d', h = h, hw = 19 * 19 + 1), qkv)
        
        dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h hw t d -> b (t hw) (h d)')
        out =  self.to_out(out)
        return out

class TemporalResidualAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x_rearrange = rearrange(x, 'b (t hw) d -> b t hw d', hw = 19 * 19 + 1)
        residual = torch.cat((x_rearrange[:,0:2,:,:],x_rearrange[:,2:,:,:] - x_rearrange[:,1:-1,:,:]), dim = 1)
        residual = rearrange(residual, 'b t hw d -> b (t hw) d')
        qk = self.to_qk(residual).chunk(2, dim = -1)
        v = self.to_v(x)
        q, k = map(lambda t: rearrange(t, 'b (t hw) (h d) -> b h hw t d', h = h, hw = 19 * 19 + 1), qk)
        v = rearrange(v, 'b (t hw) (h d) -> b h hw t d', h = h, hw = 19 * 19 + 1)
        
        dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h hw t d -> b (t hw) (h d)')
        out =  self.to_out(out)

        return out

class ReAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        #out =  self.to_out(out)
        return out
    
class LeFF(nn.Module):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super().__init__()
        
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                    )
        
        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
        
    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
    
    
class LCAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class TextureEnhance(nn.Module):
    def __init__(self,num_features,num_attentions):
        super().__init__()
        self.output_features=num_features
        self.output_features_d=num_features
        self.conv_extract=nn.Conv2d(num_features,num_features,3,padding=1)
        self.conv0=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,5,padding=2,groups=num_attentions)
        self.conv1=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn1=nn.BatchNorm2d(num_features*num_attentions)
        self.conv2=nn.Conv2d(num_features*2*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn2=nn.BatchNorm2d(2*num_features*num_attentions)
        self.conv3=nn.Conv2d(num_features*3*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn3=nn.BatchNorm2d(3*num_features*num_attentions)
        self.conv_last=nn.Conv2d(num_features*4*num_attentions,num_features*num_attentions,1,groups=num_attentions)
        self.bn4=nn.BatchNorm2d(4*num_features*num_attentions)
        self.bn_last=nn.BatchNorm2d(num_features*num_attentions)
        
        self.M=num_attentions
    def cat(self,a,b):
        B,C,H,W=a.shape
        c=torch.cat([a.reshape(B,self.M,-1,H,W),b.reshape(B,self.M,-1,H,W)],dim=2).reshape(B,-1,H,W)
        return c

    def forward(self,feature_maps,attention_maps=(1,1)):
        B,N,H,W=feature_maps.shape
        if type(attention_maps)==tuple:
            attention_size=(int(H*attention_maps[0]),int(W*attention_maps[1]))
        else:
            attention_size=(attention_maps.shape[2],attention_maps.shape[3])
        feature_maps=self.conv_extract(feature_maps)
        feature_maps_d=F.adaptive_avg_pool2d(feature_maps,attention_size)
        if feature_maps.size(2)>feature_maps_d.size(2):
            feature_maps=feature_maps-F.interpolate(feature_maps_d,(feature_maps.shape[2],feature_maps.shape[3]),mode='nearest')
        attention_maps=(torch.tanh(F.interpolate(attention_maps.detach(),(H,W),mode='bilinear',align_corners=True))).unsqueeze(2) if type(attention_maps)!=tuple else 1
        feature_maps=feature_maps.unsqueeze(1)
        feature_maps=(feature_maps*attention_maps).reshape(B,-1,H,W)
        feature_maps0=self.conv0(feature_maps)
        feature_maps1=self.conv1(F.relu(self.bn1(feature_maps0),inplace=True))
        feature_maps1_=self.cat(feature_maps0,feature_maps1)
        feature_maps2=self.conv2(F.relu(self.bn2(feature_maps1_),inplace=True))
        feature_maps2_=self.cat(feature_maps1_,feature_maps2)
        feature_maps3=self.conv3(F.relu(self.bn3(feature_maps2_),inplace=True))
        feature_maps3_=self.cat(feature_maps2_,feature_maps3)
        feature_maps=F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_),inplace=True))),inplace=True)
        feature_maps=feature_maps.reshape(B,-1,N,H,W)
        return feature_maps,feature_maps_d


if __name__ == "__main__":
    m = LocalSpatialAttention(728)
    print(m(torch.rand(16,1810,728)).shape)