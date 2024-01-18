  
from network.xception_for_dualnet import Xception
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
from attention_lib.attention.OutlookAttention import OutlookAttention
from attention_lib.attention.CoTAttention import CoTAttention
from attention_lib.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
from attention_lib.attention.CBAM import CBAMBlock
from attention_lib.attention.S2Attention import S2Attention
from attention_lib.attention.ShuffleAttention import ShuffleAttention
from attention_lib.attention.SGE import SpatialGroupEnhance
from attention_lib.attention.PSA import PSA
from attention_lib.attention.BAM import BAMBlock
from perceiver_pytorch import Perceiver
# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out

# LFS Module
class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
    
    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8)/2) + 1
        assert size_after == 149

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2,3,4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)   # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out

class DualPerceiver(nn.Module):
    def __init__(self):
        super(DualPerceiver, self).__init__()
        self.model = Perceiver(
            input_channels = 6,          # number of channels for each token of the input
            input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            freq_base = 2,
            depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                         #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 512,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            num_classes = 1,          # output number of classes
            attn_dropout = 0.,
            ff_dropout = 0.2,
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )
    
    def forward(self,x):
        x = torch.cat((x[0],x[1]),1).permute(0,2,3,1)
        return self.model(x), 0 , [] , []

class DualNet(nn.Module):
    def __init__(self, num_classes=1, img_width=300, img_height=300, LFS_window_size=10, LFS_stride=2, LFS_M = 6, mode='FAD', device=None):
        super(DualNet, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode
        self.window_size = LFS_window_size
        self._LFS_M = LFS_M
        #self.mix_block = MixBlock()

        # init branches
        self.FAD_head = FAD_Head(img_size)
        self.init_xcep_FAD()

        self.LFS_head = LFS_Head(img_size, LFS_window_size, LFS_M) 
        self.init_xcep_LFS()

        # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096, 1)
        self.dp = nn.Dropout(p=0.2)

        self.fusion = SequentialPolarizedSelfAttention(channel=1456)
    def init_xcep_FAD(self):
        self.FAD_xcep = Xception(self.num_classes)
        
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        self.FAD_xcep.load_state_dict(state_dict, False)

    def init_xcep_LFS(self):
        self.LFS_xcep = Xception(self.num_classes)
        
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        self.LFS_xcep.load_state_dict(state_dict, False)


    def init_xcep(self):
        self.xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        self.xcep.load_state_dict(state_dict, False)


    def forward(self, x):
        fea_FAD, fea_LFS = x
        fea_FAD_low = self.FAD_xcep.fea_0_7(fea_FAD)
        fea_LFS_low = self.LFS_xcep.fea_0_7(fea_LFS)
    
        #fea_FAD_low, fea_LFS_low = self.mix_block(fea_FAD_low, fea_LFS_low)
        
        #fusion_feat = self.fusion(torch.cat((fea_FAD_low, fea_LFS_low), dim = 1).permute(0,2,3,1)).permute(0,3,1,2)
        fusion_feat = self.fusion(torch.cat((fea_FAD_low, fea_LFS_low), dim = 1))
        fea_FAD_low = fusion_feat[:,0:728,:,:]
        fea_LFS_low = fusion_feat[:,728:,:,:]
    
        fea_FAD = self.FAD_xcep.fea_8_12(fea_FAD_low)
        fea_LFS = self.FAD_xcep.fea_8_12(fea_LFS_low)
        feat = torch.cat((fea_FAD, fea_LFS), dim=1)
        fea_FAD = self._norm_fea(fea_FAD)
        fea_LFS = self._norm_fea(fea_LFS)
    
        y = torch.cat((fea_FAD, fea_LFS), dim=1)
    
        f = self.dp(y)
        f = self.fc(f)
        return f, feat, [], []

    #def forward(self, x):
    #    fea_FAD, fea_LFS = x
    #    fea_FAD_low = self.FAD_xcep.fea_0_4(fea_FAD)
    #    fea_LFS_low = self.LFS_xcep.fea_0_4(fea_LFS)
#
    #    #fea_FAD_low, fea_LFS_low = self.mix_block(fea_FAD_low, fea_LFS_low)
    #    
    #    #fusion_feat = self.fusion(torch.cat((fea_FAD_low, fea_LFS_low), dim = 1).permute(0,2,3,1)).permute(0,3,1,2)
    #    fusion_feat = self.fusion(torch.cat((fea_FAD_low, fea_LFS_low), dim = 1))
    #    fea_FAD_low = fusion_feat[:,0:728,:,:]
    #    fea_LFS_low = fusion_feat[:,728:,:,:]
    #    
    #    fea_FAD_low = self.FAD_xcep.fea_5_8(fea_FAD_low)
    #    fea_LFS_low = self.LFS_xcep.fea_5_8(fea_LFS_low)
    #    fusion_feat = self.fusion(torch.cat((fea_FAD_low, fea_LFS_low), dim = 1))
    #    fea_FAD_low = fusion_feat[:,0:728,:,:]
    #    fea_LFS_low = fusion_feat[:,728:,:,:]
#
    #    fea_FAD = self.FAD_xcep.fea_9_12(fea_FAD_low)
    #    fea_FAD = self._norm_fea(fea_FAD)
    #    fea_LFS = self.FAD_xcep.fea_9_12(fea_LFS_low)
    #    fea_LFS = self._norm_fea(fea_LFS)
#
    #    y = torch.cat((fea_FAD, fea_LFS), dim=1)
#
    #    f = self.dp(y)
    #    f = self.fc(f)
    #    return f, y, [], []

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1,1))
        f = f.view(f.size(0), -1)
        return f

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

def get_xcep_state_dict(pretrained_path='/mnt/data/DFD/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict
    

# overwrite method for xception in LFS branch
# plan A

def new_xcep_features(self, input):
    # x = self.conv1(input)
    # x = self.bn1(x)
    # x = self.relu(x)

    x = self.conv2(input)   # input :[149, 149, 6]  conv2:[in_filter:32]
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

# function for mix block

def fea_0_7(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    return x

def fea_8_12(self, x):
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

class MixBlock(nn.Module):
    # An implementation of the cross attention module in F3-Net
    # Haven't added into the whole network yet
    def __init__(self, c_in = 728, width = 19, height = 19):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        #q = torch.cat([x_FAD, x_LFS], dim=3)
        #k = torch.cat([x_FAD, x_LFS], dim=2)
        #M_query = self.FAD_query(q).view(-1, W, 2*H)
        #M_key = self.FAD_key(k).view(-1, 2*W, H)
        
        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [sBC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS