import os
import numpy as np
import torch
from data import cf_refinedet as cfg
from data import evalargs as eargs
from models.refinedet import build_refinedet
import time
import warnings
warnings.filterwarnings("ignore")


t0 = time.time()

cuda = True
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

pathes = {
    'weight': './weights/cf_refinedet_20230621/RefineDet128_dscf_s_v0_eph31_itr3000.pth',
    'dataset': './dataset/cf_s_v0/image/',
}

net = build_refinedet('test',128, cfg['128']['num_classes'], cfg, eargs)
net.load_weights(pathes['weight'])
net.cuda()

for idx, _data in enumerate(os.listdir(pathes['dataset'])):
    img = np.load(pathes['dataset']+_data).astype(np.float32)
    img = torch.tensor(img).repeat(1,3,1,1)
    img = img.cuda()
    
    result = net(img)
    
    if idx%100 == 0:
        print('idx, time: ',idx,time.time()-t0)
    
print('all time used', len(os.listdir(pathes['dataset'])), time.time()-t0)