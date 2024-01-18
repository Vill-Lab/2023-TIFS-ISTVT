from network.models import model_selection
import time
import torch
model = model_selection(modelname='jigsaw_multi_xcep_adv', num_out_classes=2, dropout=0.5).cuda()
time_s = time.time()
for i in range(10000):
    _ = model([torch.rand(1,3,300,300).cuda(),torch.rand(1,3,300,300).cuda(),torch.rand(1,3,300,300).cuda()])
time_e = time.time()
print('avg time:',(time_e - time_s)/10000)