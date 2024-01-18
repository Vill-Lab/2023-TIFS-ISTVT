import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from loss_fn import *
from tqdm import tqdm
import numpy as np

from network.unet import *
from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from network.utils import recombine_features
from dataset.transform import xception_default_data_transforms,xception_default_data_transforms_256,data_transform_aug,data_transforms_shuffle
from dataset.mydataset import MyDataset
from dataset.dataset_video import *
from dataset.dataset_oulu import OULU
from torch.utils.tensorboard import SummaryWriter   



def main():
    writer = SummaryWriter('./log/')

    args = parse.parse_args()
    name = args.name
    device_no=args.run_device
    continue_train = args.continue_train
    train_length = args.train_length
    val_length = args.val_length
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    triplet_type = args.triplet_type
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    use_swish = args.use_swish
    trans=args.transform
    opt = args.optimizer
    save_dir=args.savedir
    lambda_triplet = args.lambda_triplet
    lam_rep = args.lambda_representation
    lam_rec = args.lambda_rec
    lam_vrec = args.lambda_vrec
    lam_adv = args.lambda_adversarial
    re_encode = args.re_encode
    sub_dataset = args.sub_dataset
    test_mode = args.test_mode
    num_multi = args.num_multi
    min_slice = args.shuffle_min_slice
    ex_comp = args.extra_compression
    comp_prarm = args.compress_param
    input_size = args.input_size
    pretrain_epochs = args.pretrain_epochs
    mix = args.mixed_manipulation_type
    data_quality = args.data_quality
    split_train_set = args.split_train_set
    train_set_split_rate = args.train_set_split_rate
    data_type = args.data_type
    dq = args.diverse_quality
    seq_len = args.sequence_length

    os.environ['CUDA_VISIBLE_DEVICES']=device_no
    if save_dir=='same': 
        save_dir = name
    output_path = os.path.join('./output', save_dir)
    if model_name=='tripunet_efficientnet_add' or model_name=='tripunet_efficientnet_mul' or model_name=='unetplus_efficientnet' or model_name=='unet_resnet':
        use_triplet=True
        criterion = TotalLoss()
    elif model_name == 'triplet_efficientnet' or model_name == 'triplet_efficientnet_cdc':
        use_triplet=True
        criterion = ClaTripletLoss(lam_t = lambda_triplet)
    elif model_name == 'quadplet_efficientnet':
        use_triplet = True
        criterion = QuadpletClaLoss(lam_t= lambda_triplet)
    elif model_name == 'quadnet' or model_name == 'quadnet_full' or model_name == 'quadnet_conv' or model_name == 'quadnet_dfc':
        use_triplet = True
        criterion = QuadLoss(lam = lambda_triplet)
    elif model_name == 'quadnet_decoder':
        if re_encode:
            re_encoder = QuadnetReEncoder().cuda()
            re_encoder.train()
        use_triplet = True
        criterion = QuadLoss(lam = lambda_triplet)
        rec_loss = nn.MSELoss()
        decoder = QuadnetDecoder(use_swish = use_swish).cuda()
        decoder.train()
        best_decoder = decoder.state_dict()
    elif model_name == 'quadnet_landmark':
        use_triplet=True
        criterion = QuadLoss(lam = lambda_triplet)
        rec_loss = nn.MSELoss()
        decoder = QuadnetDecoder(use_swish=True).cuda()
        decoder_lm = QuadnetLandmarkDecoder(use_swish=True).cuda()
    elif model_name == 'trip_multi_en':
        use_triplet=True
        criterion = MultiTripLoss(batch_size,lam = lambda_triplet)
    elif model_name == 'jigsaw_multi_en':
        criterion_idx = nn.L1Loss()
        criterion = nn.CrossEntropyLoss()
        use_triplet = False
    elif model_name == 'jigsaw_multi_xcep_adv_pair':
        criterion_idx = JigsawLoss()
        criterion_rec = nn.MSELoss()
        criterion = nn.BCEWithLogitsLoss()
        criterion_adv = nn.CrossEntropyLoss()
        use_triplet = False
        solver = nn.ModuleList()
        use_rec = False
        solver.append(JigsawSolverConv(4,2048))
        solver.append(JigsawSolverConv(9,2048))
        solver = solver.cuda()
        opt_solver = optim.Adam(solver.parameters(), lr=learning_rate * 100, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        comp_clas = nn.ModuleList()
        for i in range(3):
            comp_clas.append(CompressionCls())
        comp_clas = comp_clas.cuda()
        opt_ccls = optim.Adam(comp_clas.parameters(), lr=learning_rate * 10, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        triplet_type = 'Pair'
        criterion_rep = RepresentationLoss()
        criterion_feat = FeatureFinetuningLoss()
    elif model_name == 'jigsaw_multi_xcep_adv':
        criterion_idx = JigsawLoss()
        criterion = nn.BCEWithLogitsLoss()
        use_triplet = False
        solver = nn.ModuleList()
        use_rec = False
        solver.append(JigsawSolverConv(4,2048))
        solver.append(JigsawSolverConv(9,2048))
        solver = solver.cuda()
        opt_solver = optim.Adam(solver.parameters(), lr=learning_rate * 10, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        if ex_comp:
            comp_clas = nn.ModuleList()
            for i in range(3):
                comp_clas.append(CompressionCls())
            comp_clas = comp_clas.cuda()
            opt_ccls = optim.Adam(comp_clas.parameters(), lr=learning_rate * 10, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        scheduler_solver = lr_scheduler.CosineAnnealingLR(opt_solver, 3, eta_min=learning_rate / 10, last_epoch=-1)
        
        
    else:
        use_triplet=False
        criterion = nn.BCEWithLogitsLoss()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark=True
    

    if trans == '299':
        transform=xception_default_data_transforms
    elif trans == '256':
        transform=xception_default_data_transforms_256
    elif trans == 'aug':
        transform=data_transform_aug
    elif trans == 'shuffle':
        transform=data_transforms_shuffle
    if sub_dataset == 'OULU':
        train_dataset = OULU(num_multi = num_multi,mode = 'Train',shuffle_min_slice = min_slice)
        val_dataset = OULU(mode = 'Val',num_multi = num_multi)
    elif sub_dataset == 'Celeb':
        train_dataset = Celeb(num_multi = num_multi,mode = 'Train',shuffle_min_slice = min_slice,require_idx = model_name[0:15] == 'jigsaw_multi_xcep',compress_param = comp_prarm,pair_return = model_name == 'jigsaw_multi_xcep_adv_pair',fixed_qual = True)
        val_dataset = Celeb(mode = 'Test',num_multi = num_multi,compress_param = comp_prarm, random_test_qual = True, pair_return = False)
    elif sub_dataset == 'DFDC':
        train_dataset = Celeb(num_multi = num_multi,mode = 'Train',shuffle_min_slice = min_slice,require_idx = model_name[0:15] == 'jigsaw_multi_xcep',compress_param = comp_prarm,pair_return = model_name == 'jigsaw_multi_xcep_adv_pair',fixed_qual = True)
        val_dataset = Celeb(mode = 'Test',num_multi = num_multi,compress_param = comp_prarm, random_test_qual = True, pair_return = False)
    else:
        train_dataset = VideoSeqDataset(quality = data_quality, transform=transform['train'],get_triplet=triplet_type,subset=None if mix else sub_dataset,require_landmarks= model_name == 'quadnet_landmark',num_multi=num_multi,shuffle_min_slice = min_slice,require_idx = model_name[0:13] == 'jigsaw_multi_',random_compress = ex_comp,compress_param = comp_prarm,size=input_size,mode='Train',dataset_len=60000,frame_type=data_type,diverse_quality = dq, seq_len = seq_len)
        val_dataset = VideoSeqDataset(quality = data_quality, transform=transform['val'],get_triplet='Test',num_multi = num_multi, subset=None if mix else sub_dataset, return_fake_type = mix,dataset_len=20000, mode= 'Test',size=input_size,frame_type=data_type, seq_len = seq_len)
        #train_dataset = MyDataset(index_range=(0,train_length), transform=transform['train'],get_triplet=triplet_type,subset=sub_dataset,require_landmarks=model_name == 'quadnet_landmark')
        #val_dataset = MyDataset(index_range=(train_length,train_length+val_length),transform=transform['val'],get_triplet='Test',subset='Classic',use_white_list=triplet_type=='QuadCirc',num_multi = num_multi)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size if sub_dataset!='OULU' else 1, shuffle=False, drop_last=False, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    num_device = (len(device_no) + 1) // 2
    model = model_selection(modelname=model_name, num_out_classes=1, dropout=0.5,batch_size=batch_size // num_device)
    if continue_train or test_mode:
        model.load_state_dict(torch.load(model_path))

    if len(device_no) > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    if model_name == 'quadnet_decoder':
        if re_encode:
            params = [{'params':re_encoder.parameters()},{'params':decoder.parameters()},{'params':model.parameters()}]
        else:
            params = [{'params':decoder.parameters()},{'params':model.parameters()}]
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())

    if opt == 'Adam':
        optimizer = optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay,momentum = 0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 3, eta_min= learning_rate/100, last_epoch=-1)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    iteration = 0
    for epoch in range(epoches):
        #warm up
        learning_rate = (epoch+1) * 0.0005 if epoch < 20 else (epoch ** -1.5) * 1
        print('Current lr:',learning_rate)
        optimizer.lr = learning_rate
        if epoch == pretrain_epochs - 1: #calculate the running average of feature vectors before pretraining ends
            avg_feat = torch.zeros(3,2,2048).cuda()
            num_feat_0 = 0
            num_feat_1 = 0

        if 'avg_feat' in locals() and epoch == pretrain_epochs:
            avg_feat[:,0,:] /= num_feat_0
            avg_feat[:,1,:] /= num_feat_1 #norm

        if model_name == 'jigsaw_multi_en_adv_pair' and epoch == pretrain_epochs:
            train_dataset.fixed_qual = False

        print('Epoch {}/{}'.format(epoch, epoches))
        print('-'*10)
        model.train()
        try:
            solver.train()
            comp_clas.train()
        except:
            print('No solver!')
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        if not test_mode:
            if not use_triplet:
                if model_name == 'jigsaw_multi_en':
                    for (image,idx,labels) in train_loader:
                        #if not (iteration % 1000):
                        #    cv2.imwrite('./output/'+str(idx[0][4].numpy())+'.jpg',image[1][4].numpy().transpose(1,2,0)*255)
                        #    cv2.imwrite('./output/'+str(idx[0][4].numpy())+'_ori.jpg',image[0][4].numpy().transpose(1,2,0)*255)
                            
                        iter_loss = 0.0
                        iter_corrects = 0.0
                        for i in range(len(image)):
                            image[i] = image[i].cuda()
                        for i in range(len(idx)):
                            idx[i] = idx[i].cuda()
                        labels = labels.cuda()
                        optimizer.zero_grad()
                        clas,idxs = model(image)
                        loss = 0
                        for cla in clas:
                            loss += criterion(cla,labels)
                        for i in range(2):
                            loss += lam_rec * criterion_idx(idxs[i],idx[i])

                        _, preds0 = torch.max(clas[0].data, 1)
                        _, preds1 = torch.max(clas[1].data, 1)        
                        _, preds2 = torch.max(clas[2].data, 1)                                
                        loss.backward()
                        optimizer.step()
                        iter_loss = loss.data.item()
                        train_loss += iter_loss
                        iter_corrects = (torch.sum(preds0 == labels.data).to(torch.float32) + torch.sum(preds1 == labels.data).to(torch.float32) + torch.sum(preds2 == labels.data).to(torch.float32)) / 3
                        train_corrects += iter_corrects
                        iteration += 1

                        if not (iteration % 1000):
                            print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))

                    epoch_loss = train_loss / train_dataset_size
                    epoch_acc = train_corrects / train_dataset_size

                elif model_name in ['jigsaw_multi_en_adv','jigsaw_multi_xcep_adv']:
                    for ret in train_loader:
                        #if not (iteration % 1000):
                        #    cv2.imwrite('./output/'+str(idx[0][4].numpy())+'.jpg',image[1][4].numpy().transpose(1,2,0)*255)
                        #    cv2.imwrite('./output/'+str(idx[0][4].numpy())+'_ori.jpg',image[0][4].numpy().transpose(1,2,0)*255)
                        
                        if split_train_set:
                            image, idx, labels, split_labels = ret
                            #split_labels = split_labels.cuda()
                        else:
                            image, idx, labels = ret

                        iter_loss = 0.0
                        iter_corrects = 0.0
                        if ex_comp:
                            image,qual = image
                            qual = qual.cuda()
                        for i in range(len(image)):
                            image[i] = image[i]['image'].cuda()
                        for i in range(len(idx)):
                            idx[i] = idx[i].cuda()                            
                        
                        batch_len = len(image[0])
                        labels = labels.cuda()
                        clas, feats, cla_fusion, weights = model(image)
                        loss_model = 0
                        loss_solver = 0
                        loss_ccls = 0
                        #train backbone

                        #for cla in clas:
                        #    if split_train_set:
                        #        if sum(split_labels == 0) > 0: #make sure no empty tensor is sent to calculate loss
                        #            loss_model += criterion(cla[split_labels==0],labels[split_labels==0])
                        #    else:
                        #        loss_model += criterion(cla.view(-1),labels.float())
                        loss_model += criterion(clas.view(-1),labels.float())
                        #if split_train_set:
                        #    if sum(split_labels == 1) > 0:
                        #        loss_model += 1 * criterion(cla_fusion[split_labels==1],labels[split_labels==1])
                        #else:
                        #    loss_model += 1 * criterion(cla_fusion,labels)
                        
                        if lam_rec != 0:
                            if epoch >= pretrain_epochs:
                                for i in range(2):
                                    if use_rec:
                                        idx_pred, _ = solver[i].forward(feats[i+1])
                                    else:
                                        idx_pred = solver[i].forward(feats[i+1])
                                    loss_model += lam_rec * criterion_idx(idx_pred,idx[i])

                            if ex_comp:
                                for i in range(3): #adv compression loss
                                    q_pred = comp_clas[i].forward(feats[i])
                                    loss_model += lam_adv * criterion(q_pred,qual)

                        #cla_fusion = clas[0] + clas[1] + clas[2]
                        preds = (clas > 0).int().squeeze().data
                        optimizer.zero_grad()                                
                        loss_model.backward(retain_graph = True)
                        optimizer.step()
                        #train solver
                        if lam_rec != 0:
                            for i in range(2):
                                if use_rec:
                                    idx_pred, rec = solver[i].forward(feats[i+1].detach())
                                else:
                                    idx_pred = solver[i].forward(feats[i+1].detach())
                                loss_solver += criterion_idx(idx_pred,idx[i])
                                if use_rec:
                                    loss_solver += criterion_rec(rec,image[i+1])
                                #if not (iteration % 500):
                                #    print('pred index:',idx_pred[0].data)
                                #    print('real index:',idx[i][0].data)
                            opt_solver.zero_grad()
                            solver.zero_grad()
                            loss_solver.backward()
                            opt_solver.step()
                        #train compression classifer(if exist)
                        if ex_comp:
                            for i in range(3):
                                q_pred = comp_clas[i].forward(feats[i].detach())
                                loss_ccls += criterion(q_pred,qual)
    
                            opt_ccls.zero_grad()
                            comp_clas.zero_grad()
                            loss_ccls.backward()
                            opt_ccls.step()

                        iter_loss = loss_model.data.item()
                        solver_loss = 0 if lam_rec == 0 else loss_solver.data.item()
                        train_loss += iter_loss
                        iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
                        train_corrects += iter_corrects
                        iteration += 1

                        if not (iteration % 1000):
                            print('iteration {} train loss: {:.4f} solver loss: {:.4f}  Acc: {:.4f}'.format(iteration, iter_loss / batch_size, solver_loss / batch_size , iter_corrects / batch_size))

                    epoch_loss = train_loss / train_dataset_size
                    epoch_acc = train_corrects / train_dataset_size    

                elif model_name == 'jigsaw_multi_xcep_adv_pair':
                    for ret in train_loader:
                        #import ipdb
                        #ipdb.set_trace()
                        ([image,qual],idx,labels) = ret
                        #if not (iteration % 1000):
                        #    cv2.imwrite('./output/'+str(idx[0][4].numpy())+'.jpg',image[1][4].numpy().transpose(1,2,0)*255)
                        #    cv2.imwrite('./output/'+str(idx[0][4].numpy())+'_ori.jpg',image[0][4].numpy().transpose(1,2,0)*255)
                        iter_loss = 0.0
                        iter_corrects = 0.0
                        for i in range(len(image)):
                            image[i] = image[i]['image'].cuda()
                        for i in range(len(idx)):
                            idx[i] = idx[i].cuda()
                        labels = labels.cuda()
                        qual = qual.cuda()
                        qual_p = 1 - qual
                        clas,feats,s_attns,c_attns = model(image[0:3])
                        if epoch >= pretrain_epochs:
                            clas_p,feats_p,s_attns_p,c_attns_p = model(image[3:])
                            if iteration % 1000 == 0:
                                np.save('s_attns_npe.npy',s_attns[0].data.cpu().numpy())
                                np.save('s_attns_p_npe.npy',s_attns_p[0].data.cpu().numpy())
                        loss_model = 0
                        loss_solver = 0
                        loss_ccls = 0
                        if epoch < pretrain_epochs:
                            if epoch == pretrain_epochs - 1:
                                for f,idx_branch in zip(feats,range(3)):
                                    pool = nn.AdaptiveAvgPool2d(1)
                                    f = pool(f).view(-1,2048).detach()
                                    if sum(labels==0) > 0:
                                        avg_feat[idx_branch,0] += torch.sum(f[labels==0],dim=0)
                                    if sum(labels==1) > 0:
                                        avg_feat[idx_branch,1] += torch.sum(f[labels==1],dim=0)
                                num_feat_0 += sum(labels==0)
                                num_feat_1 += sum(labels==1)
                            for cla in clas:
                                loss_model += criterion(cla.view(-1),labels.float())
                        else:
                            for cla,cla_p,s,sp,f,fp,idx_branch in zip(clas,clas_p,s_attns,s_attns_p,feats,feats_p,range(3)):
                                #sm = torch.softmax(cla,dim = 1)
                                #sm_p = torch.softmax(cla_p,dim = 1)
                                #ref_feat = sm[:,0],unsqueeze(1) * avg_feat[idx_branch,0] + sm[:,1],unsqueeze(1) * avg_feat[idx_branch,1]
                                #ref_feat_p = sm_p[:,0],unsqueeze(1) * avg_feat[idx_branch,0] + sm_p[:,1],unsqueeze(1) * avg_feat[idx_branch,1]
                                #loss_feat = criterion_feat(f,fp,qual,avg_feat[idx_branch],labels)
                                loss_dstill = criterion_rep(s,sp,f,fp,qual)  #+ criterion_rep(c,cp,qual)
                                loss_model += criterion(cla.view(-1),labels.float()) + criterion(cla_p.view(-1),labels.float()) + lam_rep * loss_dstill #+ 10 * loss_feat
                                #loss_model += criterion(cla[qual == 1],labels[qual == 1]) + criterion(cla_p[qual == 0],labels[qual == 0]) + lam_rep * criterion_rep(f,fp,qual)
                            
                        #ipdb.set_trace(

                        for i in range(2): #adv jigsaw loss
                            idx_pred = solver[i].forward(feats[i+1])
                            if epoch >= pretrain_epochs:
                                idx_pred_p = solver[i].forward(feats_p[i+1])
                            loss_model += lam_rec * (criterion_idx(idx_pred,idx[i]) + ((criterion_idx(idx_pred_p,idx[i])) if epoch>=pretrain_epochs else 0))

                        if epoch >= pretrain_epochs:
                            for i in range(3): #adv compression loss
                                q_pred = comp_clas[i].forward(feats[i])
                                q_pred_p = comp_clas[i].forward(feats_p[i])
                                loss_model += lam_adv * (criterion_adv(q_pred,qual) + criterion_adv(q_pred_p,qual_p))

                        #_, preds = torch.max((clas[0]+clas[1]+clas[2]), 1)
                        preds = ((clas[0]+clas[1]+clas[2])>0).int().squeeze().data
                        if epoch >= pretrain_epochs:
                            preds_p = ((clas_p[0]+clas_p[1]+clas_p[2])>0).int().squeeze().data

                        optimizer.zero_grad()
                        loss_model.backward(retain_graph = True)
                        optimizer.step()
                        #train jigsaw solver
                        for i in range(2):
                            if epoch >= pretrain_epochs:
                                idx_pred_p = solver[i].forward(feats_p[i+1].detach())
                            idx_pred = solver[i].forward(feats[i+1].detach())
                            loss_solver += criterion_idx(idx_pred,idx[i]) + ((criterion_idx(idx_pred_p,idx[i])) if epoch>=pretrain_epochs else 0)
                            #if not (iteration % 500):
                            #    print('pred index:',idx_pred[0].data)
                            #    print('real index:',idx[i][0].data)
                        opt_solver.zero_grad()
                        solver.zero_grad()
                        loss_solver.backward(retain_graph = True)
                        opt_solver.step()
                        #train compression classifier
                        if epoch >= pretrain_epochs:
                            for i in range(3):
                                q_pred = comp_clas[i].forward(feats[i].detach())
                                q_pred_p = comp_clas[i].forward(feats_p[i].detach())
                                loss_ccls += criterion_adv(q_pred,qual) + criterion_adv(q_pred_p,qual_p)

                            opt_ccls.zero_grad()
                            comp_clas.zero_grad()
                            loss_ccls.backward()
                            opt_ccls.step()

                        iter_loss = loss_model.data.item()
                        solver_loss = loss_solver.data.item()
                        distll_loss = loss_dstill.data.item() if epoch >= pretrain_epochs else 0
                        ccls_loss = 0 if epoch<pretrain_epochs else loss_ccls.data.item()
                        train_loss += iter_loss
                        iter_corrects = torch.sum(preds == labels.data).to(torch.float32) + ((torch.sum(preds_p == labels.data).to(torch.float32)) if epoch>=pretrain_epochs else 0)
                        train_corrects += iter_corrects
                        iteration += 1
                        if iteration == 1:
                            import ipdb
                            #ipdb.set_trace()
                        if not (iteration % 1000):
                            print('iteration {} train loss: {:.4f} solver loss: {:.4f} qual. loss:{:.4f} dstill loss:{:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, solver_loss / batch_size , ccls_loss / batch_size, distll_loss / batch_size , iter_corrects / (batch_size *(1 if epoch < pretrain_epochs else 2)))) 

                    epoch_loss = train_loss / train_dataset_size
                    epoch_acc = train_corrects / (train_dataset_size*2 if epoch >= pretrain_epochs else train_dataset_size)
                


                else:
                    #normal baseline training
                    for ret in train_loader:
                        if len(ret) == 2:
                            image, labels = ret
                        elif len(ret) == 3:
                            image, _ ,labels = ret
                        iter_loss = 0.0
                        iter_corrects = 0.0
                        if num_multi == 0:
                            try:
                                image = image.cuda()
                            except:
                                image = image['image'].cuda()
                        else:
                            for i in range(len(image)):
                                image[i] = image[i]['image'].cuda()
                        labels = labels.cuda()
                        optimizer.zero_grad()
                        if model_name == 'fusion_efficientnet':
                            outputs,mfcs=model(image)
                        else:
                            outputs = model(image)

                        if len(outputs) != batch_size and len(outputs) == num_multi:
                            loss = 0
                            for y in outputs:
                                loss += criterion(y,labels)

                            preds = (clas > 0).int().squeeze().data
                        else:
                            loss = criterion(outputs.view(-1),labels.float())
                            preds = (outputs > 0).int().squeeze().data

                        if model_name == 'fusion_efficientnet':
                            for mfc in mfcs:
                                loss += 10 * criterion(mfc,labels)
                        loss.backward()
                        optimizer.step()
                        iter_loss = loss.data.item()
                        train_loss += iter_loss
                        iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
                        train_corrects += iter_corrects
                        iteration += 1
                        #if not (iteration % 100):
                            #e_iter = iteration - (epoch * train_dataset_size / batch_size)
                            #print(preds)
                            #writer.add_scalar('training loss', train_loss/e_iter, iteration)
                            #writer.add_scalar('training accuracy',train_corrects/(e_iter*batch_size), iteration)
                            #writer.add_images('image sequence', image[0], global_step=None, walltime=None, dataformats='NCHW')
                        if not (iteration % 1000):
                            print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))

                    epoch_loss = train_loss / train_dataset_size
                    epoch_acc = train_corrects / train_dataset_size

            elif model_name == 'quadplet_efficientnet':
                for (imgs,labels) in train_loader:
                    iter_loss = 0.0
                    iter_corrects = 0.0

                    for i in range(4):
                        imgs[i]=imgs[i].cuda()

                    labels = labels.cuda()

                    optimizer.zero_grad()
                    feats,clas = model(imgs)

                    for i in range(4):
                        if i >= 2:
                            label = 1-labels.data
                        else:
                            label = labels.data
                        out=clas[i]
                        _, preds = torch.max(out, 1)
                        iter_corrects += torch.sum(preds == label).to(torch.float32)

                    train_corrects += iter_corrects

                    loss = criterion(feats,clas,labels)
                    loss.backward()
                    optimizer.step()

                    iter_loss = loss.data.item()
                    train_loss += iter_loss
                    iteration += 1
                    if not (iteration % 1000):
                        print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / (batch_size*4)))

                epoch_loss = train_loss / (4 * train_dataset_size / batch_size)
                epoch_acc = train_corrects / (4 * train_dataset_size)            

            elif model_name == 'quadnet_landmark':
                for (imgs,labels,landmarks) in train_loader:
                    iter_loss = 0.0
                    iter_corrects = 0.0

                    for i in range(4):
                        imgs[i]=imgs[i].cuda()

                    labels = labels.cuda()

                    optimizer.zero_grad()
                    gs,ids,clas,gsf,idsf = model(imgs)

                    for i in range(4):
                        if i >= 2:
                            label = 1-labels.data
                        else:
                            label = labels.data
                        out=clas[i]
                        _, preds = torch.max(out, 1)
                        iter_corrects += torch.sum(preds == label).to(torch.float32)

                    train_corrects += iter_corrects
                    
                    loss_rec_lm = 0
                    for i in range(4):
                        lm_rec = decoder_lm(gs[i])
                        loss_rec_lm += rec_loss(lm_rec,landmarks[i])
                    
                    g_real,id_fake_r,g_fake_r,id_real,img_seq_fr,img_seq_r = recombine_features(gsf,idsf,imgs,labels)
                    gr_idfr = torch.cat((g_real,id_fake_r),1)
                    gfr_idr = torch.cat((g_fake_r,id_real),1)

                    rec = decoder(torch.cat((gfr_idr,gr_idfr),0))
                    r_rec = rec[0:batch_size]
                    fr_rec = rec[batch_size:]
                    loss_cla_trip = criterion(gs,ids,clas,labels)
                    loss_rec = rec_loss(r_rec,img_seq_r) + rec_loss(fr_rec,img_seq_fr)

                    loss = loss_cla_trip + lam_rec * loss_rec + lam_lmrec * loss_rec_lm
                    loss.backward()
                    optimizer.step()

                    iter_loss = loss_cla_trip.data.item()
                    iter_rec_loss = loss_rec.data.item()
                    iter_vrec_loss = loss_vrec.data.item()
                    train_loss += iter_loss + iter_rec_loss
                    iteration += 1

                    if not (iteration % 1000):
                        print('iteration {} train loss cla+trip: {:.4f} train loss rec: {:.4f} train loss vrec: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_rec_loss/batch_size,iter_vrec_loss/batch_size,iter_corrects / (batch_size*4)))
                        cv2.imwrite("./output/"+save_dir+"/sample_rec_r.png",r_rec[0].detach().cpu().numpy().transpose(1,2,0)*255)
                        cv2.imwrite("./output/"+save_dir+"/sample_rec_fr.png",fr_rec[0].detach().cpu().numpy().transpose(1,2,0)*255)
                        cv2.imwrite("./output/"+save_dir+"/sample_r.png",img_seq_r[0].cpu().numpy().transpose(1,2,0)*255)
                        cv2.imwrite("./output/"+save_dir+"/sample_fr.png",img_seq_fr[0].cpu().numpy().transpose(1,2,0)*255)


                    epoch_loss = train_loss / (4 * train_dataset_size / batch_size)
                    epoch_acc = train_corrects / (4 * train_dataset_size)

            elif model_name == 'quadnet' or model_name == 'quadnet_full' or model_name == 'quadnet_conv' or model_name == 'quadnet_dfc':
                for (imgs,labels) in train_loader:
                    iter_loss = 0.0
                    iter_corrects = 0.0
                    for i in range(4):
                        imgs[i]=imgs[i].cuda()
                    labels = labels.cuda()

                    optimizer.zero_grad()
                    gs,ids,clas = model(imgs)

                    for i in range(4):
                        if i >= 2:
                            label = 1-labels.data
                        else:
                            label = labels.data
                        out=clas[i]
                        _, preds = torch.max(out, 1)
                        iter_corrects += torch.sum(preds == label).to(torch.float32)

                    train_corrects += iter_corrects

                    loss = criterion(gs,ids,clas,labels)
                    loss.backward()
                    optimizer.step()

                    iter_loss = loss.data.item()
                    train_loss += iter_loss
                    iteration += 1
                    if not (iteration % 1000):
                        print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / (batch_size*4)))

                epoch_loss = train_loss / (4 * train_dataset_size / batch_size)
                epoch_acc = train_corrects / (4 * train_dataset_size)

            elif model_name == 'quadnet_decoder':
                for (imgs,labels) in train_loader:
                    iter_loss = 0.0
                    iter_corrects = 0.0
                    for i in range(4):
                        imgs[i]=imgs[i].cuda()
                    labels = labels.cuda()

                    optimizer.zero_grad()
                    gs,ids,clas,gsf,idsf = model(imgs)

                    for i in range(4):
                        if i >= 2:
                            label = 1-labels.data
                        else:
                            label = labels.data
                        out=clas[i]
                        _, preds = torch.max(out, 1)
                        iter_corrects += torch.sum(preds == label).to(torch.float32)

                    train_corrects += iter_corrects

                    g_real,id_fake_r,g_fake_r,id_real,img_seq_fr,img_seq_r = recombine_features(gsf,idsf,imgs,labels)
                    gr_idfr = torch.cat((g_real,id_fake_r),1)
                    gfr_idr = torch.cat((g_fake_r,id_real),1)

                    rec = decoder(torch.cat((gfr_idr,gr_idfr),0))
                    r_rec = rec[0:batch_size]
                    fr_rec = rec[batch_size:]
                    loss_cla_trip = criterion(gs,ids,clas,labels)
                    loss_rec = rec_loss(r_rec,img_seq_r) + rec_loss(fr_rec,img_seq_fr)

                    if re_encode:
                        id_rec,g_rec = re_encoder(rec)
                        idr_rec = id_rec[0:batch_size]
                        idfr_rec = id_rec[batch_size:]
                        gfr_rec = g_rec[0:batch_size]
                        gr_rec = g_rec[batch_size:]

                        loss_vrec = rec_loss(id_real,idr_rec) + rec_loss(g_fake_r,gfr_rec) + rec_loss(id_fake_r,idfr_rec) + rec_loss(g_real,gr_rec)
                    else:
                        loss_vrec = 0

                    loss = loss_cla_trip + lam_rec * loss_rec + lam_vrec * loss_vrec
                    loss.backward()
                    optimizer.step()

                    iter_loss = loss_cla_trip.data.item()
                    iter_rec_loss = loss_rec.data.item()
                    iter_vrec_loss = loss_vrec.data.item()
                    train_loss += iter_loss + iter_rec_loss
                    iteration += 1

                    if not (iteration % 1000):
                        print('iteration {} train loss cla+trip: {:.4f} train loss rec: {:.4f} train loss vrec: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_rec_loss/batch_size,iter_vrec_loss/batch_size,iter_corrects / (batch_size*4)))
                        cv2.imwrite("./output/"+save_dir+"/sample_rec_r.png",r_rec[0].detach().cpu().numpy().transpose(1,2,0)*255)
                        cv2.imwrite("./output/"+save_dir+"/sample_rec_fr.png",fr_rec[0].detach().cpu().numpy().transpose(1,2,0)*255)
                        cv2.imwrite("./output/"+save_dir+"/sample_r.png",img_seq_r[0].cpu().numpy().transpose(1,2,0)*255)
                        cv2.imwrite("./output/"+save_dir+"/sample_fr.png",img_seq_fr[0].cpu().numpy().transpose(1,2,0)*255)


                    epoch_loss = train_loss / (4 * train_dataset_size / batch_size)
                    epoch_acc = train_corrects / (4 * train_dataset_size)            

            elif model_name!='triplet_efficientnet' and model_name!='triplet_efficientnet_cdc' and model_name!='trip_multi_en':
                for (imgs,labels) in train_loader:
                    a,p,n=imgs
                    iter_loss = 0.0
                    iter_corrects = 0.0
                    a=a.cuda()
                    p=p.cuda()
                    n=n.cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()
                    reg,cla,feat = model(a,p,n)

                    for i in range(3):
                        if i == 2:
                            label = 1-labels.data
                        else:
                            label = labels.data
                        out=cla[i]
                        _, preds = torch.max(out, 1)
                        iter_corrects += torch.sum(preds == label).to(torch.float32)

                    train_corrects += iter_corrects

                    loss = criterion(reg,cla,feat,labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss = loss.data.item()
                    train_loss += iter_loss
                    iteration += 1
                    if not (iteration % 1000):
                        print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_correctmuts / (batch_size*3)))

                epoch_loss = train_loss / (3 * train_dataset_size / batch_size)
                epoch_acc = train_corrects / (3 * train_dataset_size)
            else:
                for (imgs,labels) in train_loader:
                    a,p,n=imgs
                    iter_loss = 0.0
                    iter_corrects = 0.0
                    if num_multi == 0:
                        a=a.cuda()
                        p=p.cuda()
                        n=n.cuda()
                    else:
                        for i in range(len(a)):
                            a[i]=a[i].cuda()
                            p[i]=p[i].cuda()
                            n[i]=n[i].cuda()

                    labels = labels.cuda()
                    optimizer.zero_grad()
                    feat,cla = model(a,p,n)

                    for i in range(3):
                        if i == 2:
                            label = 1 - labels.data
                        else:
                            label = labels.data
                        out=cla[i]
                        if num_multi == 0:
                            _, preds = torch.max(out, 1)
                            #print(preds,label)
                            iter_corrects += torch.sum(preds == label).to(torch.float32)
                        else:
                            for o in out:
                                _, preds = torch.max(o, 1)
                                iter_corrects += torch.sum(preds == label).to(torch.float32)


                    train_corrects += iter_corrects

                    loss = criterion(feat,cla,labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss = loss.data.item()
                    train_loss += iter_loss
                    iteration += 1
                    if not (iteration % 1000):
                        print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / (batch_size*3*max(num_multi,1))))

                epoch_loss = train_loss / (max(num_multi,1) * 3 * train_dataset_size / batch_size)
                epoch_acc = train_corrects / (max(num_multi,1) * 3 * train_dataset_size)


            print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


        ################################################################################
        #                                evaluation                                    #
        ################################################################################
        model.eval()
        try:
            solver.eval()
            comp_clas.eval()
        except:
            print('No solver!')
        with torch.no_grad():
            test_multi = False
            if test_multi:
                val_corrects = [0] * 3
                for (image, labels) in val_loader:
                    for i in range(len(image)):
                        image[i] = image[i].cuda()    
                    labels = labels.cuda()
                    outputs = model(image)
                    for i in range(len(outputs)):
                        _, preds = torch.max(outputs[i].data, 1)
                        val_corrects[i] += torch.sum(preds == labels.data).to(torch.float32)
                for i in range(len(val_corrects)):
                    epoch_acc = val_corrects[i] / val_dataset_size
                    print('epoch val Acc: {:.4f}'.format(epoch_acc))    
            elif sub_dataset == 'OULU':
                real_corrects = 0
                attack_corrects = 0
                real_errors = 0
                attack_errors = 0
                for (image, labels) in val_loader:
                    labels = labels.cuda()
                    if len(image) != 1:
                        for i in range(len(image)):
                            image[i] = image[i].cuda()
                        outputs,_ = model(image)

                        output = 0
                        for y in outputs:
                            output += y

                    else:
                        image = image.cuda()
                        output = model.forward_single(image)                

                    _, preds = torch.max(output.data, 1)
                    if labels.data == 0:
                        real_corrects += torch.sum(preds == labels.data).to(torch.float32)
                        real_errors += torch.sum(preds != labels.data).to(torch.float32)
                    elif labels.data == 1:
                        attack_corrects += torch.sum(preds == labels.data).to(torch.float32)
                        attack_errors += torch.sum(preds != labels.data).to(torch.float32)
                apcer = attack_errors / (attack_corrects + attack_errors)
                bpcer = real_errors / (real_corrects + real_errors)
                acer = (apcer + bpcer) / 2
                acc = (real_corrects + attack_corrects) / val_dataset_size
                print(attack_corrects,attack_errors,real_corrects,real_errors)
                print('apcer: {:.4f}'.format(apcer))
                print('bpcer: {:.4f}'.format(bpcer))
                print('acer: {:.4f}'.format(acer))
                print('acc: {:.4f}'.format(acc))
            else:
                avg_acc = 0
                for q in ['lq']:
                    print('==========================================================')
                    print('Testing on ',q,' videos')
                    print('==========================================================')
                    #val_dataset.set_quality(q)
                    val_corrects = 0
                    val_corrects_rgb = 0
                    val_corrects_residual = 0
                    if mix:
                        class_corrects = [0,0,0,0,0]
                        class_all = [0,0,0,0,0]

                    y_labels = []
                    y_preds = []
                    for ret in val_loader:
                        if mix:
                            image,labels,ftype  = ret
                        else:
                            image,labels = ret
                        if num_multi == 0:
                            try:
                                image = image.cuda()
                            except:
                                image = image['image'].cuda()
                        else:
                            for i in range(len(image)):
                                image[i] = image[i]['image'].cuda()        
                        labels = labels.cuda()
                        if model_name == 'xception':
                            res = 0
                            for fidx in range(seq_len):
                                with torch.no_grad():
                                    clas = model(image[:,fidx,:,:,:])
                            res += clas
                        elif not use_triplet and model_name != 'fusion_efficientnet':
                            if model_name in ['jigsaw_multi_en','jigsaw_multi_en_adv','jigsaw_multi_en_adv_pair','jigsaw_multi_xcep_adv','jigsaw_multi_xcep_adv_pair']:
                                with torch.no_grad():
                                    clas, _, _,_= model(image)
                                cla_fusion = sum(clas)
                                #print(weights)
                                #weights = solver[2](feats)
                            else:
                                with torch.no_grad():
                                    clas = model(image)
                        else:
                            clas = model.forward_single(image)

                        #_, preds = torch.max(cla_fusion.data, 1)
                        preds = (clas > 0).int().squeeze().data
                        if data_type == '_residual':
                            preds_rgb = (clas[0] > 0).int().squeeze().data
                            preds_residual = (clas[1] > 0).int().squeeze().data


                        iter_ct = preds == labels.data
                        if data_type == '_residual':
                            iter_ct_rgb = preds_rgb == labels.data
                            iter_ct_residual = preds_residual == labels.data

                        if mix:
                            for i in range(5):
                                if i < 4:
                                    class_all[i] += torch.sum(ftype==i).data.item()
                                    class_corrects[i] += torch.sum(iter_ct[ftype==i]).data.item()
                                else:
                                    class_all[4] += torch.sum(labels==0).data.item()
                                    class_corrects[4] += torch.sum(iter_ct[labels==0]).data.item()

                        val_corrects += torch.sum(iter_ct).to(torch.float32)
                        if data_type == '_residual':
                            val_corrects_rgb += torch.sum(iter_ct_rgb).to(torch.float32)
                            val_corrects_residual += torch.sum(iter_ct_residual).to(torch.float32)
                    epoch_acc = val_corrects / val_dataset_size
                    avg_acc += epoch_acc
                    print('epoch val Acc: {:.4f}'.format(epoch_acc))
                    if data_type == '_residual':
                        print('rgb branch Acc: {:.4f}'.format(val_corrects_rgb / val_dataset_size))
                        print('residual branch Acc: {:.4f}'.format(val_corrects_residual / val_dataset_size))


                    if mix:
                        fake_type=['Deepfakes','NeuralTextures','FaceSwap','Face2Face']
                        for i in range(5):
                            if i<4:
                                print(fake_type[i],'Acc: {:.4f}.'.format(class_corrects[i]/class_all[i]))
                            else:
                                print('Pristine Acc: {:.4f}'.format(class_corrects[4]/class_all[4]))
                avg_acc /= 1
                print('average acc:',avg_acc)
                            
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_model_wts = model.state_dict()
                if model_name == 'quadnet_decoder':
                    best_decoder = decoder.state_dict()
            print('current best Acc: {:.4f}'.format(best_acc))

        if test_mode:
            return 
        #scheduler.step()
        #scheduler_solver.step()
        #if not (epoch % 40):
        if len(device_no) > 1:
            torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name + '.pkl'))
        else:
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name + '.pkl'))

        if model_name == 'quadnet_decoder':
            torch.save(decoder.state_dict(),os.path.join(output_path, str(epoch) + '_' + model_name + '_decoder.pkl'))

    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    if len(device_no) > 1:
        torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    else:
        torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))




if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='xception')
    parse.add_argument('--run_device', '-d', type=str, default='0')
    parse.add_argument('--train_length', '-tl' , type=int, default = 90500)
    parse.add_argument('--val_length', '-vl' , type=int, default = 8000)
    parse.add_argument('--batch_size', '-bz', type=int, default=16)
    parse.add_argument('--epoches', '-e', type=int, default='20')
    parse.add_argument('--model_name', '-mn', type=str, default='xception')
    parse.add_argument('--continue_train', '-ct', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./output/df_xception_c0_299/1_df_c0_299.pkl')
    parse.add_argument('--triplet_type', '-t', type=str, default='False')
    parse.add_argument('--savedir', '-sd', type=str, default='same')
    parse.add_argument('--transform', '-tf', type=str, default='299')
    parse.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parse.add_argument('--weight_decay', '-wd', type=float, default=0)
    parse.add_argument('--optimizer', '-opt', type=str, default='SGD')
    parse.add_argument('--step_size', '-ss', type=int, default=10)
    parse.add_argument('--lambda_triplet', '-lt', type=float, default=1)
    parse.add_argument('--lambda_rec', '-lrc', type=float, default=1)
    parse.add_argument('--use_swish', '-us', type=bool, default=True)
    parse.add_argument('--re_encode', '-re', type=bool, default=False)
    parse.add_argument('--lambda_vrec', '-lvr', type=float, default=1)
    parse.add_argument('--lambda_lmrec', '-llm', type=float, default=1)
    parse.add_argument('--lambda_adversarial','-lad',type=float,default=-1)
    parse.add_argument('--lambda_representation','-lrp',type=float,default=0)
    parse.add_argument('--sub_dataset', '-sds', type=str, default='Classic')
    parse.add_argument('--test_mode','-tm',type=bool,default=False)    
    parse.add_argument('--num_multi','-nm',type=int,default=0)    
    parse.add_argument('--shuffle_min_slice','-sms',type=int,default=1)
    parse.add_argument('--extra_compression','-ec',type=bool,default=False)
    parse.add_argument('--compress_param','-cp',type=float,default=0.8)
    parse.add_argument('--input_size','-is',type=int, default=300)
    parse.add_argument('--pretrain_epochs','-pe',type=int, default=0)
    parse.add_argument('--mixed_manipulation_type','-mmt',type=bool,default=False)
    parse.add_argument('--data_quality','-qual',type=str,default='hq')
    parse.add_argument('--split_train_set','-sts',type=bool,default=False)
    parse.add_argument('--train_set_split_rate','-sr',type=float,default=0.95)
    parse.add_argument('--data_type','-dt',type=str,default='normal')
    parse.add_argument('--diverse_quality','-dq',type=bool,default=False)
    parse.add_argument('--sequence_length','-sl',type=int,default=4)
    main()
