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

def main():
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
        criterion = nn.CrossEntropyLoss()
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
    else:
        train_dataset = MixedVideoDataset(quality = data_quality, transform=transform['train'],get_triplet=triplet_type,subset=None if mix else sub_dataset,require_landmarks= model_name == 'quadnet_landmark',num_multi=num_multi,shuffle_min_slice = min_slice,require_idx = model_name[0:13] == 'jigsaw_multi_',random_compress = ex_comp,compress_param = comp_prarm,size=input_size,mode='Train',dataset_len=60000,frame_type=data_type,diverse_quality = dq)
        val_dataset = MixedVideoDataset(quality = data_quality, transform=transform['val'],get_triplet='Test',num_multi = num_multi, subset=None if mix else sub_dataset, return_fake_type = mix,dataset_len=20000, mode= 'Test',size=input_size,frame_type=data_type)
        #train_dataset = MyDataset(index_range=(0,train_length), transform=transform['train'],get_triplet=triplet_type,subset=sub_dataset,require_landmarks=model_name == 'quadnet_landmark')
        #val_dataset = MyDataset(index_range=(train_length,train_length+val_length),transform=transform['val'],get_triplet='Test',subset='Classic',use_white_list=triplet_type=='QuadCirc',num_multi = num_multi)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size if sub_dataset!='OULU' else 1, shuffle=False, drop_last=False, num_workers=0)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    model = model_selection(modelname=model_name, num_out_classes=2, dropout=0.5)
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
        params = model.parameters()

    if opt == 'Adam':
        optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay,momentum = 0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 3, eta_min= learning_rate/100, last_epoch=-1)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    iteration = 0
    train_loss = 0.0
    train_corrects = 0.0
    val_loss = 0.0
    val_corrects = 0.0

    avg_acc = 0
    for q in ['hq','lq']:
        print('==========================================================')
        print('Testing on ',q,' videos')
        print('==========================================================')
        val_dataset.set_quality(q)
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
                image = image.cuda()
            else:
                for i in range(len(image)):
                    image[i] = image[i]['image'].cuda()        
            labels = labels.cuda()
            if not use_triplet and model_name != 'fusion_efficientnet':
                if model_name in ['jigsaw_multi_en','jigsaw_multi_en_adv','jigsaw_multi_en_adv_pair','jigsaw_multi_xcep_adv','jigsaw_multi_xcep_adv_pair']:
                    with torch.no_grad():
                        clas, f, _,_= model(image)
                    cla_fusion = sum(clas)
                    f = F.relu(f).cpu().numpy()
                    for cidx in range(2048):
                        cv2.imwrite('./visualize/'+str(cidx)+'.png',f[0,cidx,:,:] * 128)
                    import ipdb
                    ipdb.set_trace()
                    #print(weights)
                    #weights = solver[2](feats)
                else:
                    with torch.no_grad():
                        outputs = model(image)
            else:
                outputs = model.forward_single(image)
            if model_name not in ['jigsaw_multi_en','jigsaw_multi_en_adv','jigsaw_multi_en_adv_pair','jigsaw_multi_xcep_adv','multi_xception','jigsaw_multi_xcep_adv_pair']:
                _, preds = torch.max(outputs.data, 1)
            else:
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
    avg_acc /= 2
    print('average acc:',avg_acc)




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
    main()
