"""
Author: Andreas Rössler
"""
import os
import argparse


import torch
#import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from network.xception import xception, xception_concat
from network.unet import *
import math
import torchvision
from efficientnet_pytorch.model import EfficientNet
from network.efficientnet_cdc import EfficientNet_cdc
from network.mesonet import *
from network.dual_net import DualNet,DualPerceiver
from network.fast_transformer_torch.fast_transformer_torch import FastTransformer
from vit_pytorch.cvt import CvT
from vit_pytorch.cross_vit import CrossViT
from network.resnet3d.models import resnet
from network.vivit.vivit import ViViT, XceptionVidTr
from network.xception import return_pytorch04_xception
from tfe.baselines.ViT.ViT_LRP import VisionTransformer

class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.5, batch_size=16):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice in ['xception']:
            self.model = return_pytorch04_xception(pretrained=True)
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'xception_concat':
            self.model = xception_concat()
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'mesonet':
            self.model = Meso4()
        elif modelchoice == 'mesoincep':
            self.model = MesoInception4()
        elif modelchoice == 'efficientnet':
            self.model=EfficientNet.from_pretrained('efficientnet-b2',num_classes=2)
            # Replace fc
            #num_ftrs = self.model.fc.in_features
            #if not dropout:
            #    self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            #else:
            #    self.model.fc = nn.Sequential(
            #        nn.Dropout(p=dropout),
            #        nn.Linear(num_ftrs, num_out_classes)
            #    )

        elif modelchoice == 'efficientnet_cdc':
            self.model=EfficientNet_cdc.from_pretrained('efficientnet-b3',num_classes=2)

        elif modelchoice == 'unet_efficientnet_add':
            self.model = ModifiedUNet(n_channels=3,n_classes=3,mode='add')

        elif modelchoice == 'unet_efficientnet_mul':
            self.model = ModifiedUNet(n_channels=3,n_classes=3,mode='mul')

        elif modelchoice == 'tripunet_efficientnet_add':
            self.model = TripUNet(mode='add')

        elif modelchoice == 'tripunet_efficientnet_mul':
            self.model = TripUNet(mode='mul')

        elif modelchoice == 'triplet_efficientnet':
            self.model = TripEfficientNet()

        elif modelchoice == 'triplet_efficientnet_cdc':
            self.model = TripEfficientNet_cdc()

        elif modelchoice == 'unetplus_efficientnet':
            self.model = TripUNetPlus()
        
        elif modelchoice == 'unet_resnet':
            self.model = TripUNetResnet()

        elif modelchoice == 'quadnet':
            self.model = QuadEfficientNetLite()

        elif modelchoice == 'quadnet_full':
            self.model = QuadEfficientNetLite(use_full=True)
        
        elif modelchoice == 'quadnet_conv':
            self.model = QuadEfficientNet()

        elif modelchoice == 'quadnet_dfc':
            self.model = QuadEfficientNetLiteDoubleFC()

        elif modelchoice == 'decoder':
            self.model = QuadnetDecoder()

        elif modelchoice == 'quadnet_decoder':
            self.model = QuadnetEncoder()
        
        elif modelchoice == 'quadplet_efficientnet':
            self.model = QuadpletEfficientNet()
        
        elif modelchoice == 'multi_efficientnet':
            self.model = EnsembleEfficientNet(num_nets=3)

        elif modelchoice == 'fusion_efficientnet':
            self.model = FusionEfficientNet(num_nets=3)

        elif modelchoice == 'trip_multi_en':
            self.model = TripEnsembleEfficientNet(num_nets=3)

        elif modelchoice == 'multi_xception':
            self.model = VaniTripleXcep()

        elif modelchoice == 'jigsaw_multi_en':
            self.model = JigsawEnsembleEN(num_nets=3)

        elif modelchoice[0:19] == 'jigsaw_multi_en_adv':
            self.model = EnsembleEN(num_nets=3)

        elif modelchoice in ['jigsaw_multi_xcep_adv','jigsaw_multi_xcep_adv_pair'] :
            self.model = DualNet()
            #self.model = DualPerceiver()
        elif modelchoice == 'fastformer':
            #mask = torch.ones([batch_size, 197], dtype=torch.bool).cuda()
            #ph = 8
            #pw = 8
            #seq_len = (224//ph) * (224//pw)
            #mask = torch.ones([batch_size, seq_len], dtype=torch.bool).cuda()
            #self.model = FastTransformer(num_tokens = 1,
            #            dim = 1024,
            #            depth = 32,
            #            max_seq_len = seq_len,
            #            absolute_pos_emb = True, # Absolute positional embeddings
            #            mask = mask,
            #            patch_height = ph,
            #            patch_width = pw
            #            )
            #self.model = CrossViT(image_size = 384, num_classes = 1, sm_dim = 1024, lg_dim = 1024)
            self.model = CvT(num_classes = 1)
        elif modelchoice == 'resnet_3d':
            #self.model = resnet.i3_res50_nl(num_classes = 400)
            #self.model = XceptionVidTr()
            
            
            self.model = VisionTransformer()
        elif modelchoice in ['mixed_xcep','xception_dg']:
            self.model = CrossAttentionXception(num_nets=2)
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True
    def get_model(self):
        return self.model

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self,x):
        return self.model.features(x)

    def extract_features(self,x):
        return self.model.extract_features(x)



def model_selection(modelname, num_out_classes,
                    dropout=None,batch_size=16):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes)
    #    , 299, \True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes)
    #    , \224, True, ['image'], None
    elif modelname == 'xception_concat':
        return TransferModel(modelchoice='xception_concat',
                             num_out_classes=num_out_classes)
    elif modelname == 'efficientnet':
        return TransferModel(modelchoice='efficientnet',
                             num_out_classes=num_out_classes)
    elif modelname == 'unet_efficientnet_add':
        return TransferModel(modelchoice='unet_efficientnet_add',
                             num_out_classes=num_out_classes)
    elif modelname == 'unet_efficientnet_mul':
        return TransferModel(modelchoice='unet_efficientnet_mul',
                             num_out_classes=num_out_classes)
    elif modelname == 'tripunet_efficientnet_add':
        return TransferModel(modelchoice='tripunet_efficientnet_add',
                             num_out_classes=num_out_classes).get_model()
    elif modelname == 'tripunet_efficientnet_mul':
        return TransferModel(modelchoice='tripunet_efficientnet_mul',
                             num_out_classes=num_out_classes).get_model()
    elif modelname == 'triplet_efficientnet':
        return TransferModel(modelchoice='triplet_efficientnet',
                             num_out_classes=num_out_classes).get_model()
    elif modelname == 'unetplus_efficientnet':
        return TransferModel(modelchoice='unetplus_efficientnet',
                             num_out_classes=num_out_classes).get_model()
    elif modelname == 'unet_resnet':
        return TransferModel(modelchoice='unet_resnet',
                             num_out_classes=num_out_classes).get_model()
    else:
        return TransferModel(modelchoice=modelname, num_out_classes=num_out_classes, batch_size=batch_size).get_model()       


if __name__ == '__main__':
    model, image_size, *_ = model_selection('xception', num_out_classes=2)
    print(model)
    model = model.cuda()
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s))