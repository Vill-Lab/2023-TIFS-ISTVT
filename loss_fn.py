import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFinetuningLoss(nn.Module):
    def __init__(self):
        super(FeatureFinetuningLoss, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.triplet_loss = nn.TripletMarginLoss()
        self.mse_loss = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.T = 6
    def forward(self,feat,feat_p,qual,avg_feat,label):
        loss = 0
        f = self.pool(feat)
        fp = self.pool(feat_p)
        #f_sm = torch.softmax(f,1)
        #fp_sm = torch.softmax(fp,1)
        f_hq = torch.cat((f[qual==1],fp[qual==0]),0)
        label_hq = torch.cat((label[qual==1],label[qual==0]),0)
        for fhq,lhq in zip(f_hq,label_hq):
            loss = loss + self.triplet_loss(fhq,avg_feat[lhq],avg_feat[1-lhq])
        #f_lq = torch.cat((fp_sm[qual==1],f_sm[qual==0]),0)
        #avg_feat_sm = torch.softmax(avg_feat,1)
        #f_hq = torch.log(f_hq)
        return loss



class RepresentationLoss(nn.Module):
    def __init__(self):
        super(RepresentationLoss,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        #self.kl = nn.KLDivLoss(reduction='batchmean')
        self.loss = nn.MSELoss()
        self.T = 5
    def forward(self,feat,feat_p,sources,targets,qual):
        #f = self.pool(feat)
        #fp = self.pool(feat_p)
        #f = torch.softmax(f,1)
        #fp = torch.softmax(fp,1)
        f_hq = torch.cat((feat[qual==1],feat_p[qual==0]),0)
        f_lq = torch.cat((feat_p[qual==1],feat[qual==0]),0)
        f_hq.detach()
        #f_lq.detach() 
        sources.detach()
        targets.detach()
        sources = self.pool(sources).view(-1,2048)
        targets = self.pool(targets).view(-1,2048)
        batch_size = 1
        sources = torch.cat((sources[qual==1],targets[qual==0]),0)
        targets = torch.cat((targets[qual==1],sources[qual==0]),0)
        loss = 0
        for fh,fl,i in zip (f_hq,f_lq,range(len(sources))):
            #kernels = guassian_kernel(sources[i:i+1], targets[i:i+1],
            #                          kernel_mul=2.0,    
            #                            kernel_num=5,  
            #                          fix_sigma=None)
            #XX = kernels[:batch_size, :batch_size] # Source<->Source
            #YY = kernels[batch_size:, batch_size:] # Target<->Target
            #XY = kernels[:batch_size, batch_size:] # Source<->Target
            #YX = kernels[batch_size:, :batch_size] # Target<->Source
            #loss += torch.mean(XX + YY - XY -YX) * self.loss(fl,fh)
            loss += self.loss(sources[i:i+1],targets[i:i+1]) * self.loss(fl,fh)
        #f_lq = torch.log(f_lq)
        #return self.loss(f_lq,f_hq) * 
        return loss
        
class TripletLoss(nn.Module):
    def __init__(self, margin = 0.2):
        super(TripletLoss,self).__init__()
        self.margin = margin
    def forward(self, f_anchor, f_positive, f_negative): # (-1,c)
        f_anchor, f_positive, f_negative = renorm(f_anchor), renorm(f_positive), renorm(f_negative)
        b = f_anchor.size(0)
        f_anchor = f_anchor.view(b,-1)
        f_positive = f_positive.view(b,-1)
        f_negative = f_negative.view(b, -1)
        with torch.no_grad():
            idx = hard_samples_mining(f_anchor, f_positive, f_negative, self.margin)
        
        d_ap = torch.norm(f_anchor[idx] - f_positive[idx], dim = 1)  # (-1,1)
          
        return torch.clamp(d_ap - d_an + self.margin,0).mean()
        


def hard_samples_mining(f_anchor,f_positive, f_negative, margin):
    d_ap = torch.norm(f_anchor - f_positive, dim = 1)
    d_an = torch.norm(f_anchor - f_negative, dim = 1)
    idx = (d_ap - d_an) < margin
    return idx 

def renorm(x): # Important for training!
    # renorm in batch axis to make sure every vector is in the range of [0,1]
    # important !
    if x.dim() > 1:
        return x.renorm(2,0,1e-5).mul(1e5)
    else:
        return x.unsqueeze(0).renorm(2,0,1e-5).mul(1e5)[0]

class QuadpletClaLoss(nn.Module):
    def __init__(self,margin = 1,lam_t = 1):
        super(QuadpletClaLoss,self).__init__()
        self.trip = nn.TripletMarginLoss(margin = margin)
        self.cla = nn.CrossEntropyLoss()
        self.lam_t = lam_t
    
    def forward(self,feats,clas,label):
        feat_real = torch.cat((feats[0][label==0],feats[2][label==1]),0)
        feat_real_etc = torch.cat((feats[1][label==0],feats[3][label==1]),0)
        feat_fake_r = torch.cat((feats[2][label==0],feats[0][label==1]),0)
        feat_fake_etc = torch.cat((feats[3][label==0],feats[1][label==1]),0)

        t1 = self.trip(feat_real,feat_real_etc,feat_fake_r)
        t2 = self.trip(feat_real_etc,feat_real,feat_fake_etc)
        t3 = self.trip(feat_fake_r,feat_fake_etc,feat_real)
        t4 = self.trip(feat_fake_etc,feat_fake_r,feat_real_etc)

        t = t1 + t2 + t3 + t4

        cla0 = self.cla(clas[0],label.long().cuda())
        cla1 = self.cla(clas[1],label.long().cuda())
        cla2 = self.cla(clas[2],1 - label.long().cuda())
        cla3 = self.cla(clas[3],1 - label.long().cuda())
        
        cla = cla0 + cla1 + cla2 + cla3

        return cla + self.lam_t * t

class QuadTripletLoss(nn.Module):
    def __init__(self,margin = 1):
        super(QuadTripletLoss,self).__init__()
        self.trip=nn.TripletMarginLoss(margin=margin)
    
    def forward(self,gs,ids,label):    
        g_real = torch.cat((gs[0][label==0],gs[2][label==1]),0)
        id_real = torch.cat((ids[0][label==0],ids[2][label==1]),0)

        g_real_etc = torch.cat((gs[1][label==0],gs[3][label==1]),0)
        id_real_etc = torch.cat((ids[1][label==0],ids[3][label==1]),0)

        g_fake_r = torch.cat((gs[2][label==0],gs[0][label==1]),0)
        id_fake_r = torch.cat((ids[2][label==0],ids[0][label==1]),0)

        g_fake_etc = torch.cat((gs[3][label==0],gs[1][label==1]),0)
        id_fake_etc = torch.cat((ids[3][label==0],ids[1][label==1]),0)

        g_real,g_real_etc,g_fake_r,g_fake_etc,id_real,id_real_etc,id_fake_r,id_fake_etc = renorm(g_real),renorm(g_real_etc),renorm(g_fake_r),renorm(g_fake_etc),renorm(id_real),renorm(id_real_etc),renorm(id_fake_r),renorm(id_real_etc)
        
        t1 = self.trip(g_real,g_fake_r,g_fake_etc)
        t2 = self.trip(g_real,g_fake_r,g_real_etc)
        t3 = self.trip(id_real,id_real_etc,id_fake_r)
        t4 = self.trip(id_fake_r,id_fake_etc,id_real)
        return t1 + t2 + t3 + t4

class QuadClassificatonLoss(nn.Module):
    def __init__(self):
        super(QuadClassificatonLoss,self).__init__()
        self.cla=nn.CrossEntropyLoss()
    
    def forward(self,clas,label):
        cla0 = self.cla(clas[0],label.long().cuda())
        cla1 = self.cla(clas[1],label.long().cuda())
        cla2 = self.cla(clas[2],1 - label.long().cuda())
        cla3 = self.cla(clas[3],1 - label.long().cuda())
        return cla0 + cla1 + cla2 + cla3

class QuadLoss(nn.Module):
    def __init__(self,lam = 1):
        super(QuadLoss,self).__init__()
        self.cla = QuadClassificatonLoss()
        self.trip = QuadTripletLoss(margin=1)
        self.lam = lam
    
    def forward(self,gs,ids,clas,label):
        return self.cla(clas,label) + self.lam * self.trip(gs,ids,label) 

class MultiTripLoss(nn.Module):
    def __init__(self,bs,lam = 10,margin = 0.5):
        super(MultiTripLoss,self).__init__()
        self.cla = nn.CrossEntropyLoss()
        self.trip = nn.TripletMarginLoss()
        self.bs = bs
        self.lam = lam
    
    def forward(self,feats,clas,labels):
        trip_loss = 0
        cla_loss = 0
        for stype in range(3):
            cla_loss += self.cla(clas[0][stype],labels) + self.cla(clas[1][stype],labels) + self.cla(clas[2][stype],1 - labels)
            trip_loss += self.trip(feats[0][stype],feats[1][stype],feats[2][stype])
        cla_loss = cla_loss/9
        return cla_loss + self.lam * trip_loss
    
class TotalLoss(nn.Module):
    def __init__(self,margin = 1):
        super(TotalLoss, self).__init__()
        self.margin = margin
        self.trip = TripletLoss(margin)
        self.reg = nn.MSELoss()
        self.cla = nn.CrossEntropyLoss()
        
    def forward(self, regression, classification, feat, labels):
        regression_anchor, regression_positive, regression_negative = regression
        b,c,_,_ = regression_anchor.size()
        classification_anchor, classification_positive, classification_negative = classification
        
        feat_anchor, feat_positive, feat_negative = feat
        reg_loss_1 = self.reg(regression_negative[labels == 1], torch.zeros_like(regression_negative[labels == 1]).cuda())
        reg_loss_2 = self.reg(regression_anchor[labels == 0], torch.zeros_like(regression_anchor[labels == 0]).cuda()) + self.reg(regression_positive[labels == 0], torch.zeros_like(regression_positive[labels == 0]).cuda())
        if torch.isnan(reg_loss_1):
            reg_loss_1 = torch.tensor(0)
        if torch.isnan(reg_loss_2):
            reg_loss_2 = torch.tensor(0)
        reg_loss = reg_loss_1 + reg_loss_2
        cla_losses = [] 
        cla_losses.append(self.cla(classification_anchor[labels==0], torch.tensor([0] * classification_anchor[labels==0].size(0), dtype = torch.long).cuda()))
        cla_losses.append(self.cla(classification_anchor[labels==1], torch.tensor([1] * classification_anchor[labels==1].size(0), dtype = torch.long).cuda()))
        cla_losses.append(self.cla(classification_positive[labels==0], torch.tensor([0] * classification_positive[labels==0].size(0), dtype = torch.long).cuda()))
        cla_losses.append(self.cla(classification_positive[labels==1], torch.tensor([1] * classification_positive[labels==1].size(0), dtype = torch.long).cuda()))
        cla_losses.append(self.cla(classification_negative[labels==0], torch.tensor([1] * classification_negative[labels==0].size(0), dtype = torch.long).cuda()))
        cla_losses.append(self.cla(classification_negative[labels==1], torch.tensor([0] * classification_negative[labels==1].size(0), dtype = torch.long).cuda()))
        inited=False
        for l in cla_losses:
            if not torch.isnan(l):
                if inited:
                    cla_loss += l
                else:
                    cla_loss = l
                    inited=True
        trip_loss = sum([self.trip(a,b,c) for a,b,c in zip(feat_anchor, feat_positive, feat_negative)])
        

        return cla_loss + trip_loss + reg_loss

class ClaTripletLoss(nn.Module):
    def __init__(self,lam_t=0.3):
        super(ClaTripletLoss,self).__init__()
        self.cla = nn.CrossEntropyLoss()
        self.trip = nn.TripletMarginLoss()
        self.lam_t = lam_t

    def forward(self, classification, feature, labels):

        classification_anchor, classification_positive, classification_negative = classification

        cla_loss =  self.cla(classification_anchor[labels==0], torch.tensor([0] * classification_anchor[labels==0].size(0), dtype = torch.long).cuda()) + \
                    self.cla(classification_anchor[labels==1], torch.tensor([1] * classification_anchor[labels==1].size(0), dtype = torch.long).cuda()) +  \
                    self.cla(classification_positive[labels==0], torch.tensor([0] * classification_positive[labels==0].size(0), dtype = torch.long).cuda()) + \
                    self.cla(classification_positive[labels==1], torch.tensor([1] * classification_positive[labels==1].size(0), dtype = torch.long).cuda()) + \
                    self.cla(classification_negative[labels==0], torch.tensor([1] * classification_negative[labels==0].size(0), dtype = torch.long).cuda()) + \
                    self.cla(classification_negative[labels==1], torch.tensor([0] * classification_negative[labels==1].size(0), dtype = torch.long).cuda())
        
        trip_loss = self.trip(feature[0],feature[1],feature[2])
        if torch.isnan(cla_loss):
            cla_loss=0
        if torch.isnan(trip_loss):
            trip_loss=0
        return cla_loss + self.lam_t * trip_loss

class JigsawLoss(nn.Module):
    def __init__(self):
        super(JigsawLoss,self).__init__()
        self.c_dict = {}
        for i in range(20):
            self.c_dict[i*i] = i

    def forward(self,idx_pred,idx):
        loss = 0
        l = idx_pred.shape[1] // 2
        c = self.c_dict[l]
        pred_x = idx_pred[:,0:l]
        pred_y = idx_pred[:,l:] 
        real_x = idx // c
        real_y = idx % c
        loss_jigsaw = torch.sum(torch.sqrt((pred_x - real_x)**2 + (pred_y - real_y)**2)) / (len(idx_pred) * idx_pred.shape[1] / 2)
        return loss_jigsaw

class SingleCenterLoss(nn.Module):
    def __init__(self):
        super(SingleCenterLoss,self).__init__()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差

        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # 合并在一起
   
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|
   
    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
   
    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起
 
def mmd(sources, targets, qual, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    pool = nn.AdaptiveAvgPool2d(output_size = 1)
    sources = pool(sources).squeeze()
    targets = pool(targets).squeeze()
    batch_size = 1
    sources = torch.cat((sources[qual==1],targets[qual==0]),0)
    targets = torch.cat((targets[qual==1],sources[qual==0]),0)
    loss = 0
    for source,target in zip (sources,targets):
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul,    
                                    kernel_num=kernel_num,  
                                  fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size] # Source<->Source
        YY = kernels[batch_size:, batch_size:] # Target<->Target
        XY = kernels[:batch_size, batch_size:] # Source<->Target
        YX = kernels[batch_size:, :batch_size] # Target<->Source
        loss += torch.mean(XX + YY - XY -YX) # 这里是假定X和Y的样本数量是相同的
                                                                            # 当不同的时候，就需要乘上上面的M矩阵
    return loss

if __name__ == "__main__":
    regression = [torch.randn(1,3,24,24), torch.randn(1,3,24,24), torch.randn(1,3,24,24)]
    classification = [torch.randn(1,2), torch.randn(1,2), torch.randn(1,2)]
    feat = [[torch.randn(1,16),torch.randn(1,16)],[torch.randn(1,16),torch.randn(1,16)],[torch.randn(1,16),torch.randn(1,16)]]
    labels = torch.tensor([0],dtype = torch.long)
    loss_fn = TotalLoss()
    res = loss_fn(regression, classification, feat, labels)
