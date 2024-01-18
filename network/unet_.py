import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from efficientnet_pytorch.model import EfficientNet
from network.unet_nest import UNet_Nested
from network.utils import MemoryEfficientSwish
from network.efficientnet_cdc import EfficientNet_cdc
from network.xception import xception

def return_pytorch04_xception(pretrained=False):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            '/mnt/data/DFD/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model

def get_xception():
    model = return_pytorch04_xception(pretrained=True)
    model.last_linear = None
    return model

class DoubleConv(nn.Module):
    """(convolution =&gt; [BN] =&gt; ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvSwish(nn.Module):
    """(convolution =&gt; [BN] =&gt; ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            MemoryEfficientSwish(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            MemoryEfficientSwish()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpSingle(nn.Module):
    def __init__(self, in_channels, out_channels,mid_channels=None, bilinear=True,use_swish=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if use_swish:
                self.conv = DoubleConvSwish(in_channels, out_channels,mid_channels= mid_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels,mid_channels= mid_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels , kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self,x):
        x = self.up(x)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class ModifiedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,mode='add'):
        super(ModifiedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16) # 48
        self.down1 = Down(16, 32) # 24
        self.down2 = Down(32, 64) # 12
        self.down3 = Down(64, 128) # 6
        factor = 2 if bilinear else 1 
        self.down4 = Down(128, 256 // factor) # 3
        self.up1 = Up(256, 128 // factor, bilinear) # 6
        self.up2 = Up(128, 64 // factor, bilinear)# 12 
        self.up3 = Up(64, 32 // factor, bilinear)# 24
        self.up4 = Up(32, 16, bilinear) # 48
        self.regression = OutConv(16, n_classes)# 48
        self.classification = EfficientNet.from_pretrained('efficientnet-b2',num_classes=2)
        self.mode=mode
    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        d1 = self.up1(e5, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        regression = self.regression(d4)
        if self.mode == 'mul':
            regression = torch.sigmoid(regression)
            classification = self.classification(x * regression)
        elif self.mode == 'add':
            classification = self.classification(x + regression)
        else:
            raise NotImplementedError(self.mode)
        feat = [e5, d1, d2, d3, d4]
        return regression, classification, feat 

class ModifiedUNetResnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ModifiedUNetResnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16) # 48
        self.down1 = Down(16, 32) # 24
        self.down2 = Down(32, 64) # 12
        self.down3 = Down(64, 128) # 6
        factor = 2 if bilinear else 1 
        self.down4 = Down(128, 256 // factor) # 3
        self.up1 = Up(256, 128 // factor, bilinear) # 6
        self.up2 = Up(128, 64 // factor, bilinear)# 12 
        self.up3 = Up(64, 32 // factor, bilinear)# 24
        self.up4 = Up(32, 16, bilinear) # 48
        self.regression = OutConv(16, n_classes)# 48
        self.classification = resnet18(pretrained = True)
        in_features = self.classification.fc.in_features
        self.classification.fc = nn.Linear(in_features, 2)
    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        d1 = self.up1(e5, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        regression = self.regression(d4)
        classification = self.classification(x + regression)
        feat = [e5, d1, d2, d3, d4]
        return regression, classification, feat 

class TripUNetResnet(nn.Module):
    def __init__(self,):
        super(TripUNetResnet, self).__init__()
        self.net = ModifiedUNetResnet(n_channels = 3, n_classes = 3)
    def forward(self,anchor, positive, negative):
        regression_anchor, classification_anchor, feat_anchor = self.net(anchor)
        regression_positive, classification_positive, feat_positive = self.net(positive)
        regression_negative, classification_negative, feat_negative = self.net(negative)
        return  [regression_anchor, regression_positive, regression_negative],\
                [classification_anchor, classification_positive, classification_negative],\
                [feat_anchor, feat_positive, feat_negative]        

    def forward_single(self,x,mode='mul'):
        return self.net(x)[1]


class TripUNet(nn.Module):
    def __init__(self,mode = 'add'):
        super(TripUNet, self).__init__()
        self.net = ModifiedUNet(n_channels = 3, n_classes = 3, mode=mode)
    def forward(self, anchor, positive, negative):
        regression_anchor, classification_anchor, feat_anchor = self.net(anchor)
        regression_positive, classification_positive, feat_positive = self.net(positive)
        regression_negative, classification_negative, feat_negative = self.net(negative)
        return  [regression_anchor, regression_positive, regression_negative],\
                [classification_anchor, classification_positive, classification_negative],\
                [feat_anchor, feat_positive, feat_negative]

    def forward_single(self,x,mode='mul',ret='cla'):
        if ret == 'cla':
            return self.net(x)[1]
        elif ret == 'reg':
            return self.net(x)[0]


class TripEfficientNet(nn.Module):
    def __init__(self,):
        super(TripEfficientNet, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0, in_channels = 3)
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1408,2)
    def forward(self,anchor, positive, negative):
        a_feat = self.pool(self.net.extract_features(anchor)).flatten(start_dim=1)
        p_feat = self.pool(self.net.extract_features(positive)).flatten(start_dim=1)
        n_feat = self.pool(self.net.extract_features(negative)).flatten(start_dim=1)
        a_cla = self.fc(a_feat)
        p_cla = self.fc(p_feat)
        n_cla = self.fc(n_feat)
        return [a_feat,p_feat,n_feat],[a_cla,p_cla,n_cla]
    
    def get_features(self,x):
        return self.pool(self.net.extract_features(x)).flatten(start_dim=1)

    def forward_single(self,x):
        return self.fc(self.pool(self.net.extract_features(x)).flatten(start_dim=1))

class TripEfficientNet_cdc(nn.Module):
    def __init__(self,):
        super(TripEfficientNet_cdc, self).__init__()
        self.net = EfficientNet_cdc.from_pretrained('efficientnet-b2', num_classes = 0, in_channels = 3)
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1408,2)
    def forward(self,anchor, positive, negative):
        a_feat = self.pool(self.net.extract_features(anchor)).flatten(start_dim=1)
        p_feat = self.pool(self.net.extract_features(positive)).flatten(start_dim=1)
        n_feat = self.pool(self.net.extract_features(negative)).flatten(start_dim=1)
        a_cla = self.fc(a_feat)
        p_cla = self.fc(p_feat)
        n_cla = self.fc(n_feat)
        return [a_feat,p_feat,n_feat],[a_cla,p_cla,n_cla]
    
    def get_features(self,x):
        return self.pool(self.net.extract_features(x)).flatten(start_dim=1)

    def forward_single(self,x):
        return self.fc(self.pool(self.net.extract_features(x)).flatten(start_dim=1))

class EnsembleEfficientNet(nn.Module):
    def __init__(self,num_nets = 3):
        super(EnsembleEfficientNet,self).__init__()
        self.nets = nn.ModuleList()
        for _ in range(num_nets):
            self.nets.append(EfficientNet_cdc.from_pretrained('efficientnet-b2', num_classes = 2))
    
    def forward(self,inputs):
        assert len(inputs) == len(self.nets)
        output = []
        for (x,net) in zip(inputs,self.nets):
            output.append(net(x))
        return output

    def get_features(self,inputs):
        assert len(inputs) == len(self.nets)
        output = []
        for (x,net) in zip(inputs,self.nets):
            output.append(net.extract_features(x))
        return output

class JigsawSolver(nn.Module):
    def __init__(self,out_count):
        super(JigsawSolver,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1 = nn.Linear(1408,1408)
        self.fc2 = nn.Linear(1408,1408)
        self.fc3 = nn.Linear(1408,704)
        self.fc4 = nn.Linear(704,out_count*2)
        self.swish = MemoryEfficientSwish()
    
    def forward(self,x):
        x = self.pool(x).flatten(start_dim = 1)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        x = self.swish(x)
        x = self.fc3(x)
        x = self.swish(x)
        x = self.fc4(x)
        return x        

class JigsawSolverConv(nn.Module):
    def __init__(self,out_count,in_channels = 1408):
        super(JigsawSolverConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,1,1)
        self.fc = nn.Linear(100,out_count*2)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.fc(x.flatten(1))
        return x
        


class JigsawSolverDec(nn.Module):
    def __init__(self,out_count):
        super(JigsawSolverDec,self).__init__()
        self.dec = QuadnetDecoder()
        self.net = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 0)
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc = nn.Linear(1280,out_count*2)

    def forward(self,x):
        rec_i = self.dec(x)
        return self.fc(self.pool(self.net.extract_features(rec_i)).flatten(start_dim = 1)), rec_i

class CompressionCls(nn.Module):
    def __init__(self, in_channel = 1408):
        super(CompressionCls,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1 = nn.Linear(1408,1408)
        self.fc2 = nn.Linear(1408,1408)
        self.fc3 = nn.Linear(1408,704)
        self.fc4 = nn.Linear(704,2)
        self.swish = MemoryEfficientSwish()
    
    def forward(self,x):
        x = self.pool(x).flatten(start_dim = 1)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        x = self.swish(x)
        x = self.fc3(x)
        x = self.swish(x)
        x = self.fc4(x)
        return x         

class DomainCls(nn.Module):
    def __init__(self, in_channel = 2048, num_domain = 5):
        super(DomainCls,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1 = nn.Linear(2048,2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,num_domain)
        self.swish = MemoryEfficientSwish()
    
    def forward(self,x):
        x = self.pool(x).flatten(start_dim = 1)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        x = self.swish(x)
        x = self.fc3(x)
        x = self.swish(x)
        x = self.fc4(x)
        return x         

class JigsawEnsembleEN(nn.Module):
    def __init__(self,num_nets = 3):
        super(JigsawEnsembleEN,self).__init__()
        self.nets = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.ccls = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        for _ in range(num_nets):
            self.nets.append(EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0))
            self.fcs.append(nn.Linear(1408,2))
            self.ccls.append(CompressionCls())
        self.solver_3 = JigsawSolver(9)
        self.solver_5 = JigsawSolver(25)
        
    
    def forward(self,inputs):
        clas = []
        idxs = []
        c_clas = []
        for (x,net,fc,solver,c_cls) in zip(inputs,self.nets,self.fcs,[None,self.solver_3,self.solver_5],self.ccls):
                feat = self.pool(net.extract_features(x)).flatten(start_dim = 1)
                clas.append(fc(feat))
                if solver != None:
                    idxs.append(solver(feat))
                c_clas.append(c_cls(feat))
        return clas,idxs

class EnsembleEN(nn.Module):
    def __init__(self,num_nets = 3):
        super(EnsembleEN,self).__init__()
        self.nets = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        for _ in range(num_nets):
            self.nets.append(EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0))
            self.fcs.append(nn.Linear(1408,2))
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,inputs):
        clas = []
        feats = []
        for (x,net,fc,i) in zip(inputs,self.nets,self.fcs,range(3)):
            feat = net.extract_features(x)
            feats.append(feat)
            clas.append(fc(self.dropout(self.pool(feat).flatten(start_dim = 1))))
        return clas,feats
        
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention,self).__init__()
        self.conv = nn.Conv2d(728 * 2, 728, 3, 1, 1)
    def forward(self,f1,f2):
        att = self.conv(torch.cat((f1,f2),1))
        return torch.sigmoid(att)

class EnsembleXcep(nn.Module):
    def __init__(self,num_nets = 3):
        super(EnsembleXcep,self).__init__()
        self.num_nets = num_nets
        self.nets = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        for _ in range(num_nets):
            self.nets.append(get_xception())
            self.fcs.append(nn.Linear(2048,2))
        self.dropout = nn.Dropout(0.5)
    
    def forward(self,inputs):
        clas = []
        feats = []
        for (x,net,fc,i) in zip(inputs,self.nets,self.fcs,range(self.num_nets)):
                feat = net.features(x)
                feats.append(feat)
                clas.append(fc(self.dropout(self.pool(feat).flatten(start_dim = 1))))
        return clas,feats
    
    def feature(self,inputs):
        feats = []
        for (x,net) in zip(inputs,self.nets):
            feat = net.features(x)
            feats.append(self.pool(feat).flatten(start_dim = 1))
        return (feats[0] + feats[1] + feats[2]) / 3

class CrossAttentionXception(nn.Module):
    def __init__(self,num_nets = 2):
        super(CrossAttentionXception,self).__init__()
        self.nets = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.mix_block = CrossAttention()
        for _ in range(num_nets):
            self.nets.append(get_xception())
            self.fcs.append(nn.Linear(2048,2))
        self.dropout = nn.Dropout(0.5)
    
    def forward(self,inputs,return_feat=False):
        clas = []
        feats = []
        low_0 = self.nets[0].feat_low(inputs[0])
        low_1 = self.nets[1].feat_low(inputs[1])
        att = self.mix_block(low_0,low_1)
        #low_0 = low_1 * att + low_0
        #low_1 = low_0 * att + low_1
        feat_0 = self.nets[0].feat_high(low_0)
        feat_1 = self.nets[1].feat_high(low_1) 
        cla_0 = self.fcs[0](self.dropout(self.pool(feat_0).flatten(start_dim = 1)))
        cla_1 = self.fcs[1](self.dropout(self.pool(feat_1).flatten(start_dim = 1)))
        if return_feat:
            return [cla_0,cla_1],[feat_0,feat_1]
        else:
            return [cla_0,cla_1]

class TripEnsembleEfficientNet(nn.Module):
    def __init__(self,num_nets = 3):
        super(TripEnsembleEfficientNet,self).__init__()
        self.nets = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _ in range(num_nets):
            self.nets.append(EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0))
            self.fcs.append(nn.Linear(1408,2))
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,a,p,n):
        feats = []
        clas = []
        for inputs in [a,p,n]:
            feat_l = []
            cla_l = []
            for (x,net,fc) in zip(inputs,self.nets,self.fcs):
                feat = self.pool(net.extract_features(x)).flatten(start_dim = 1)
                cla = fc(self.dropout(feat))
                feat_l.append(feat)
                cla_l.append(cla)
            feats.append(feat_l)
            clas.append(cla_l)
        
        return feats,clas

    def forward_single(self,inputs):
        clas = []
        for (x,net,fc) in zip(inputs,self.nets,self.fcs):
                feat = self.pool(net.extract_features(x)).flatten(start_dim = 1)
                cla = fc(feat)
                clas.append(cla)
        return clas
                


class FusionEfficientNet(nn.Module):
    def __init__(self,num_nets = 3):
        super(FusionEfficientNet,self).__init__()
        self.nets = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _ in range(num_nets):
            self.nets.append(EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0))
            self.fcs.append(nn.Linear(1408,2))
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(2 * num_nets,2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,inputs):
        assert len(inputs) == len(self.nets)
        feat = self.pool(self.nets[0].extract_features(inputs[0])).flatten(start_dim = 1)
        branch_cls = self.fcs[0].forward(feat)
        mid_fcs = [branch_cls]
        for (x,net,fc) in zip(inputs[1:],self.nets[1:],self.fcs[1:]):
            feat = self.pool(net.extract_features(x)).flatten(start_dim = 1)
            mid_fc = fc(self.dropout(feat))
            mid_fcs.append(mid_fc)
            branch_cls = torch.cat((branch_cls,mid_fc),1)

        return self.fc(self.dropout(branch_cls)),mid_fcs

    def forward_single(self,inputs):
        assert len(inputs) == len(self.nets)
        feat = self.pool(self.nets[0].extract_features(inputs[0])).flatten(start_dim = 1)
        branch_cls = self.fcs[0].forward(feat)
        for (x,net,fc) in zip(inputs[1:],self.nets[1:],self.fcs[1:]):
            feat = self.pool(net.extract_features(x)).flatten(start_dim = 1)
            mid_fc = fc(feat)
            branch_cls = torch.cat((branch_cls,mid_fc),1)

        return self.fc(branch_cls)

class QuadpletEfficientNet(nn.Module):
    def __init__(self,):
        super(QuadpletEfficientNet, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0, in_channels = 3)
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1408,2)
    def forward(self,imgs):
        clas = []
        feats = []
        for img in imgs:
            feat = self.pool(self.net.extract_features(img)).flatten(start_dim=1)
            cla = self.fc(feat)
            feats.append(feat)
            clas.append(cla)

        return feats,clas

    def get_features(self,x):
        return self.pool(self.net.extract_features(x)).flatten(start_dim=1)

    def forward_single(self,x):
        return self.fc(self.pool(self.net.extract_features(x)).flatten(start_dim=1))

class QuadEfficientNet(nn.Module):
    def __init__(self):
        super(QuadEfficientNet, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0, in_channels = 3)
        self.swish = MemoryEfficientSwish()
        self.conv_g = nn.Conv2d(1408,704,1)
        self.conv_id = nn.Conv2d(1408,704,1)
        self.bn_g = nn.BatchNorm2d(704,eps=0.001,momentum=0.01)
        self.bn_id = nn.BatchNorm2d(704,eps=0.001,momentum=0.01)
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(704,2)
    
    def forward(self,inputs):
        gs = []
        ids = []
        clas = []
        for x in inputs:
            v = self.net.extract_features(x)

            g = self.swish(self.bn_g(self.conv_g(v)))
            id = self.swish(self.bn_id(self.conv_id(v)))
            ids.append(id.view(x.shape[0],-1))
            gs.append(g.view(x.shape[0],-1))

            cla = self.fc(self.pool(id).view(x.shape[0],-1))
            clas.append(cla)

        return gs,ids,clas

    def forward_single(self,x):
        v = self.net.extract_features(x)
        id = self.swish(self.bn_id(self.conv_id(v)))
        cla = self.fc(self.pool(id).view(x.shape[0],-1))
        return cla


class QuadEfficientNetLite(nn.Module):
    def __init__(self,id_len=704,use_full=False):
        super(QuadEfficientNetLite,self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0, in_channels = 3)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_full:
            self.fc = nn.Linear(1408,2)
        else:
            self.fc = nn.Linear(id_len,2)
        self.id_len = id_len
        self.use_full = use_full

    def forward(self,inputs):
        gs = []
        ids = []
        clas = []
        for x in inputs:
            v = self.pool(self.net.extract_features(x)).flatten(start_dim=1)

            id = v[:,0:self.id_len]
            g = v[:,self.id_len:]
            ids.append(id)
            gs.append(g)

            if self.use_full:
                cla = self.fc(v)
            else:
                cla = self.fc(id)
            clas.append(cla)

        return gs,ids,clas

    def forward_single(self,input):
        v = self.pool(self.net.extract_features(input)).flatten(start_dim=1)
        id = v[:,0:self.id_len]
        if self.use_full:
            return self.fc(v)
            
        return self.fc(id)

    def get_features(self,input):
        v = self.pool(self.net.extract_features(input)).flatten(start_dim=1)
        id = v[:,0:self.id_len]
        g = v[:,self.id_len:]
        
        return id,g

class QuadnetEncoder(nn.Module):
    def __init__(self,id_len=896,use_full=False):
        super(QuadnetEncoder,self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 0, in_channels = 3)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_full:
            self.fc = nn.Linear(1792,2)
        else:
            self.fc = nn.Linear(id_len,2)
        self.id_len = id_len
        self.use_full = use_full

    def forward(self,inputs):
        gs = []
        ids = []
        gs_full = []
        ids_full = []
        clas = []
        for x in inputs:
            v_full = self.net.extract_features(x)
            v = self.pool(v_full).flatten(start_dim=1)

            id_full = v_full[:,0:self.id_len]
            g_full = v_full[:,self.id_len:]            

            id = v[:,0:self.id_len]
            g = v[:,self.id_len:]

            ids_full.append(id_full)
            gs_full.append(g_full)
            ids.append(id)
            gs.append(g)

            if self.use_full:
                cla = self.fc(v)
            else:
                cla = self.fc(id)
            clas.append(cla)

        return gs,ids,clas,gs_full,ids_full

    def forward_single(self,input):
        v = self.pool(self.net.extract_features(input)).flatten(start_dim=1)
        id = v[:,0:self.id_len]
        if self.use_full:
            return self.fc(v)
            
        return self.fc(id)

    def get_features(self,input):
        v = self.pool(self.net.extract_features(input)).flatten(start_dim=1)
        id = v[:,0:self.id_len]
        g = v[:,self.id_len:]
        
        return id,g

class QuadnetReEncoder(nn.Module):
    def __init__(self,id_len=640):
        super(QuadnetReEncoder,self).__init__()
        self.id_len = id_len
        self.net = EfficientNet.from_pretrained('efficientnet-b1', num_classes = 0, in_channels = 3)
    
    def forward(self,x):
        v = self.net.extract_features(x)
        id = v[:,0:self.id_len]
        g = v[:,self.id_len:]
        return id,g

class QuadnetDecoder(nn.Module):
    def __init__(self,use_swish = False):
        super(QuadnetDecoder,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1408,320,1),
            nn.BatchNorm2d(320),
            MemoryEfficientSwish() if use_swish else nn.ReLU(inplace=True),
            UpSingle(320,112,192,use_swish = use_swish),
            UpSingle(112,80,use_swish = use_swish),
            UpSingle(80,40,use_swish = use_swish),
            UpSingle(40,16,24,use_swish = use_swish),
            UpSingle(16,3,32,use_swish = use_swish),
            nn.BatchNorm2d(3),
            MemoryEfficientSwish()
        )
    
    def forward(self,x):
        return self.net(x)

class QuadnetLandmarkDecoder(nn.Module):
    def __init__(self,use_swish = False):
        super(QuadnetLandmarkDecoder,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(896,320,1),
            nn.BatchNorm2d(320),
            MemoryEfficientSwish() if use_swish else nn.ReLU(inplace=True),
            UpSingle(320,112,192,use_swish = use_swish),
            UpSingle(112,80,use_swish = use_swish),
            UpSingle(80,40,use_swish = use_swish),
            UpSingle(40,16,24,use_swish = use_swish),
            UpSingle(16,1,32,use_swish = use_swish),
        )
    
    def forward(self,x):
        return self.net(x)

class QuadnetRecDecoder(nn.Module):
    def __init__(self,use_swish = False):
        super(QuadnetLandmarkDecoder,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(896,320,1),
            nn.BatchNorm2d(320),
            MemoryEfficientSwish() if use_swish else nn.ReLU(inplace=True),
            UpSingle(320,112,192,use_swish = use_swish),
            UpSingle(112,80,use_swish = use_swish),
            UpSingle(80,40,use_swish = use_swish),
            UpSingle(40,16,24,use_swish = use_swish),
            UpSingle(16,1,32,use_swish = use_swish)
        )
    
    def forward(self,x):
        return self.net(x)
        
class QuadEfficientNetLiteDoubleFC(nn.Module):
    def __init__(self,id_len=704,use_full=False):
        super(QuadEfficientNetLiteDoubleFC,self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b2', num_classes = 0, in_channels = 3)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_full:
            self.fc = nn.Linear(1408,512)
        else:
            self.fc = nn.Linear(id_len,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,2)
        self.id_len = id_len
        self.use_full = use_full

    def forward(self,inputs):
        gs = []
        ids = []
        clas = []
        for x in inputs:
            v = self.pool(self.net.extract_features(x)).flatten(start_dim=1)

            id = v[:,0:self.id_len]
            g = v[:,self.id_len:]
            ids.append(id)
            gs.append(g)

            if self.use_full:
                feat = v
            else:
                feat = id
            
            cla = torch.tanh(self.fc(feat))
            cla = torch.tanh(self.fc2(cla))
            cla = self.fc3(cla)
            clas.append(cla)

        return gs,ids,clas

    def forward_single(self,input):
        v = self.pool(self.net.extract_features(input)).flatten(start_dim=1)
        id = v[:,0:self.id_len]
        if self.use_full:
            feat = v
        else:
            feat = id

        cla = torch.tanh(self.fc(feat))
        cla = torch.tanh(self.fc2(cla))
        cla = self.fc3(cla)

        return cla

    def get_features(self,input):
        v = self.pool(self.net.extract_features(input)).flatten(start_dim=1)
        id = v[:,0:self.id_len]
        g = v[:,self.id_len:]
        
        return id,g



class UnetNestEN(nn.Module):
    def __init__(self):
        super(UnetNestEN,self).__init__()
        self.unet = UNet_Nested()
        self.clas = EfficientNet.from_pretrained('efficientnet-b2',num_classes=2)
    
    def forward(self,x):
        reg,feat=self.unet(x)
        cla=self.clas(reg+x)

        return reg,cla,feat

class TripUNetPlus(nn.Module):
    def __init__(self):
        super(TripUNetPlus, self).__init__()
        self.net = UnetNestEN()
    def forward(self, anchor, positive, negative):
        regression_anchor, classification_anchor, feat_anchor = self.net(anchor)
        regression_positive, classification_positive, feat_positive = self.net(positive)
        regression_negative, classification_negative, feat_negative = self.net(negative)
        return  [regression_anchor, regression_positive, regression_negative],\
                [classification_anchor, classification_positive, classification_negative],\
                [feat_anchor, feat_positive, feat_negative]

    def forward_single(self,x,mode='mul'):
        return self.net(x)[1]       

class BranchWeightGenerator(nn.Module):
    def __init__(self):
        super(BranchWeightGenerator, self).__init__()
        self.net = nn.Linear(6144,3)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
    def forward(self, feats):
        f1,f2,f3 = feats
        f = torch.cat((self.pool(f1).flatten(1),self.pool(f2).flatten(1),self.pool(f3).flatten(1)), dim = 1)
        return torch.softmax(self.net(f),1)

if __name__ == "__main__":
    model = UNet(3, 3)
    x = torch.randn(1,3,64,64)
    print(model(x).size())
