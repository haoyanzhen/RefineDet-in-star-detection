from models.refinedet import build_refinedet
from layers.box_utils import *
from layers.modules import RefineDetMultiBoxLoss
from data import cargs
from data import HOME
from data import cf_refinedet as cfg
import numpy as np
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib import patches as patches
from matplotlib.collections import PatchCollection
import sys
from utils.logging import Logger
import time


#################-*- initial -*-####################
# init definition
pathes = {
    'weight': './weights/cf_refinedet_20230609/RefineDet128_dscf_s_v0_eph189_itr18000.final.pth',
    'dataroot': './dataset/cf_s_v0/',
}

# file list generation based on anno and stamp
fileid_list = list()
for i in os.listdir(pathes['dataroot']+'anno'):
    if i.endswith('fits'):
        fileid_list.append(i[:-5].split('anno_')[1])
date_stamp = pathes['weight'].split('cf_refinedet_')[1].split('/RefineDet128_ds')[0]
eph_stamp = pathes['weight'].split('eph')[1].split('_itr')[0]
dataset_stamp = pathes['weight'].split('ds')[1].split('_eph')[0]
pathes['dataroot'] = './dataset/%s/' % dataset_stamp
check_root = './check/%s_|%s|_epoch%s'%(date_stamp, dataset_stamp, eph_stamp)
if not os.path.exists(check_root):
    os.mkdir(check_root)

# loss curve
fig, ax = plt.subplots(1,1,figsize=(6,6))
iterationloss = np.load(pathes['weight'][:-4].split('RefineDet')[0]+'iterationloss'+pathes['weight'][:-4].split('RefineDet')[1]+'.npy')
ax.plot(iterationloss)
ax.legend()
fig.savefig(check_root+'iteration_loss.png')

# net load
net = build_refinedet('train',128, cfg['128']['num_classes'])
net.load_weights(pathes['weight'])

#################-*- prediction show -*-####################
fig, ((ax0,ax1),(ax2,ax3),(ax4,ax5),(ax6,ax7),(ax8,ax9)) = plt.subplots(5,2,figsize=(8,32))
ax_list = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
# reg generation
for index, testid in enumerate(fileid_list):
    img = np.load(pathes['dataroot']+'/image/img_'+testid+'.npy').astype(np.float32)
    img = torch.tensor(img).repeat(1,3,1,1)
    out = net(img)
    arm_loc, arm_conf, odm_loc, odm_conf, prior = out

    odm_boxes = decode(odm_loc[0],prior,cfg['128']['variance'])
    odm_kind = odm_conf[0].max(1)
    odm_boxes_cs = torch.concat([(odm_boxes[:,2:]+odm_boxes[:,:2])/2,(odm_boxes[:,2:]-odm_boxes[:,:2])],dim=1)

    arm_boxes = decode(arm_loc[0],prior,cfg['128']['variance'])
    arm_kind = arm_conf[0].max(1)
    arm_boxes_cs = torch.concat([(arm_boxes[:,2:]+arm_boxes[:,:2])/2,(arm_boxes[:,2:]-arm_boxes[:,:2])],dim=1)

    prior_sizes = prior[:,2].unique()
    
    truths = np.load(pathes['dataroot']+'/catalog/cat_'+testid+'.npy')
    fig_ = ax_list[index].imshow(np.log(img[0][0]),origin='lower',cmap='ocean')
    rtas_pre = [patches.Rectangle((xc-w/2,yc-h/2),w,h) for xc,yc,w,h in odm_boxes_cs[odm_kind.indices.gt(0)]]
    pc_pre = PatchCollection(rtas_pre,edgecolor='r',facecolor='none',label='prediction')
    ax_list[index].add_collection(pc_pre)
    rtas_tru = [patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin) for xmin,ymin,xmax,ymax in truths]
    pc_tru = PatchCollection(rtas_tru,edgecolor='y',facecolor='none',label='truth')
    ax_list[index].add_collection(pc_tru)
    ax_list[index].set_title('testid')
    fig.colorbar(fig_, ax=ax_list[index])
    ax_list[index].legend()
    
    with open(pathes['dataroot']+'anno/pred_'+testid+'_'+date_stamp+'_eph'+eph_stamp+'_odm.reg','w') as p:
        # # background
        # p.write('global color=cyan dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
        # for i,_i in enumerate(odm_kind.indices.lt(1)):
        #     if _i:
        #         p.write(f'box({odm_boxes_cs[i,0].item()},{odm_boxes_cs[i,1].item()},{odm_boxes_cs[i,2].item()},{odm_boxes_cs[i,3].item()},0) \t' +\
        #                     '# text = {%.2f}\n'%odm_kind.values[i].item())
        # object
        p.write('global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
        for i,_i in enumerate(odm_kind.indices.gt(0)):
            if _i:
                p.write(f'box({odm_boxes_cs[i,0].item()},{odm_boxes_cs[i,1].item()},{odm_boxes_cs[i,2].item()},{odm_boxes_cs[i,3].item()},1) \t' +\
                        '# text = {%.2f}\n'%odm_kind.values[i].item())
                
    with open(pathes['dataroot']+'anno/pred_'+testid+'_'+date_stamp+'_eph'+eph_stamp+'_arm.reg','w') as p:
        # # background
        # p.write('global color=cyan dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
        # for i,_i in enumerate(arm_kind.indices.lt(1)):
        #     if _i:
        #         p.write(f'box({arm_boxes_cs[i,0].item()},{arm_boxes_cs[i,1].item()},{arm_boxes_cs[i,2].item()},{arm_boxes_cs[i,3].item()},0) \t' +\
        #                     '# text = {%.2f}\n'%arm_kind.values[i].item())
        # object
        p.write('global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
        for i,_i in enumerate(arm_kind.indices.gt(0)):
            if _i:
                p.write(f'box({arm_boxes_cs[i,0].item()},{arm_boxes_cs[i,1].item()},{arm_boxes_cs[i,2].item()},{arm_boxes_cs[i,3].item()},0) \t' +\
                        '# text = {%.2f}\n'%arm_kind.values[i].item())


with open(pathes['dataroot']+'anno/pred_'+date_stamp+'_prior.reg','w') as p:
    p.write('global color=#FF0064 dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
    for i,_i in enumerate(prior[:,2]):
        if _i == prior_sizes[0]:
            p.write(f'circle({prior[i,0].item()},{prior[i,1].item()},{prior[i,2].item()/2}) \n')
    p.write('global color=#A0A096 dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
    for i,_i in enumerate(prior[:,2]):
        if _i == prior_sizes[1]:
            p.write(f'circle({prior[i,0].item()},{prior[i,1].item()},{prior[i,2].item()/2}) \n')
    p.write('global color=#A0A0C8 dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
    for i,_i in enumerate(prior[:,2]):
        if _i == prior_sizes[2]:
            p.write(f'circle({prior[i,0].item()},{prior[i,1].item()},{prior[i,2].item()/2}) \n')
    p.write('global color=#A0A0FA dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
    for i,_i in enumerate(prior[:,2]):
        if _i == prior_sizes[3]:
            p.write(f'circle({prior[i,0].item()},{prior[i,1].item()},{prior[i,2].item()/2}) \n')
            
# save reg figure
fig.title('image with truth and prediction')
fig.savefig(check_root+'image_truth_prediction.png')

#################-*- loss-overlap -*-####################
# args of loss
class Args_loss(object):
    def __init__(self):
        self.use_gpu = False
        self.num_classes = 2
        self.threshold = 0.1
        self.background_label = 0
        self.do_neg_mining = True
        self.negpos_ratio = 3
        self.variance = cfg['128']['variance']
        self.conf_thld = 2
        self.is_ODM = False
        
argsloss = Args_loss()

# output repointing
sys.stdout = Logger(os.path.join('./check/pngs/', 'check_matchloss_%s.out'%date_stamp))
print('='*10+'data loading'+'='*10)
print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime()))
print('date_stamp, eph_stamp: ', date_stamp, eph_stamp)


def smooth_l1(diff,beta):
    result = torch.zeros_like(diff)
    result[diff<beta] = 0.5*diff[diff<beta]**2/beta
    result[diff>=beta] = diff[diff>=beta]-0.5*beta
    return result

# check on one slide.
def check_one_slide(net, img, targets, testid):
    # forecasting
    out = net(img)
    arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors = out

    # data selection
    if argsloss.is_ODM:
        loc_data, conf_data = odm_loc_data, odm_conf_data
    else:
        loc_data, conf_data = arm_loc_data, arm_conf_data
    num = loc_data.size(0)  # batch
    priors = priors[:loc_data.size(1), :]  # choose prior boxes in num of prediction boxes
    num_priors = (priors.size(0))  # in fact this is prior boxes after choose
    num_classes = argsloss.num_classes
    print('batchsize, loc_data, conf_data, priors: ', num, loc_data.shape, conf_data.shape, priors.shape)

    # before match
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)
    truths_loc = targets[0][:, :-1].data
    truths_labels = targets[0][:, -1].data

    # fig definition
    gs = GridSpec(7,12)
    gs.update(wspace=0.8)
    fig = plt.figure(figsize=(16,28))
    ax11 = fig.add_subplot(gs[0,:6])
    ax12 = fig.add_subplot(gs[0,6:])
    ax21 = fig.add_subplot(gs[1,:3])
    ax22 = fig.add_subplot(gs[1,3:6])
    ax23 = fig.add_subplot(gs[1,6:9])
    ax24 = fig.add_subplot(gs[1,9:])
    ax31 = fig.add_subplot(gs[2,:6])
    ax32 = fig.add_subplot(gs[2,6:])
    ax41 = fig.add_subplot(gs[3,:4])
    ax42 = fig.add_subplot(gs[3,4:8])
    ax43 = fig.add_subplot(gs[3,8:])
    ax51 = fig.add_subplot(gs[4,:4])
    ax52 = fig.add_subplot(gs[4,4:8])
    ax53 = fig.add_subplot(gs[4,8:])
    ax61 = fig.add_subplot(gs[5,:4])
    ax62 = fig.add_subplot(gs[5,4:8])
    ax63 = fig.add_subplot(gs[5,8:])
    ax71 = fig.add_subplot(gs[6,:4])
    ax72 = fig.add_subplot(gs[6,4:8])
    ax73 = fig.add_subplot(gs[6,8:])


    # overlap
    print('='*10+'matching'+'='*10)
    if not argsloss.is_ODM:
        truths_labels = truths_labels >= 0
        defaults = priors.data
        overlaps = jaccard(truths_loc, point_form(priors))
    print('overlaps: ',overlaps.shape)

    # fig: overlaps
    fig11 = ax11.imshow(overlaps,origin='lower',aspect=overlaps.size(1)/overlaps.size(0)/2,cmap='ocean_r')
    ax11.yaxis.set_minor_locator(MultipleLocator(1))
    ax11.grid(which='both',axis='y')
    ax11.set_title('overlaps')
    fig.colorbar(fig11)

    # overlap check
    overlapindex = overlaps.gt(0).sum(dim=0).gt(0)
    overlaps_ = overlaps[:,overlapindex]
    prior_overlaped = priors[overlapindex,:]
    print('overlapped: ',overlaps_.shape)

    # fig: overlaped
    fig12 = ax12.imshow(overlaps_,origin='lower',aspect=overlaps_.size(1)/overlaps_.size(0)/2,cmap='ocean_r')
    ax12.set_title('overlaped')
    fig.colorbar(fig12)

    # overlaped reg
    with open(pathes['dataroot']+'/anno/overlaped_priors_%s.reg'%testid,'w') as p:
        p.write('global color=#FF0064 dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
        for i,_i in enumerate(priors[overlaps.gt(0).sum(dim=0).gt(0),:]):
            p.write(f'circle({_i[0].item()},{_i[1].item()},{_i[2].item()/2}) \n')
            
    # match
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # best prior for each ground truth
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)  # best ground truth for each prior
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # best_truth_overlap.index_fill_(0, best_prior_idx, 1)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths_loc[best_truth_idx]          # Shape: [num_priors,4]

    # fig: matches x,y hist
    ax21.hist((matches[:,2]+matches[:,0])/2,bins=100)
    ax21.set_title('matches x hist')
    ax22.scatter(priors[:,0]-(matches[:,2]+matches[:,0])/2, priors[:,1]-(matches[:,3]+matches[:,1])/2, s=1)
    ax22.set_title('x,y differs')

    # match just overlapped priors
    best_prior_overlap_, best_prior_idx_ = overlaps_.max(1, keepdim=True)
    best_truth_overlap_, best_truth_idx_ = overlaps_.max(0, keepdim=True)
    best_truth_idx_.squeeze_(0)
    best_truth_overlap_.squeeze_(0)
    best_prior_idx_.squeeze_(1)
    best_prior_overlap_.squeeze_(1)
    best_truth_overlap_
    # best_truth_overlap_.index_fill_(0, best_prior_idx_, 1)  # ensure best prior
    for j in range(best_prior_idx_.size(0)):
        best_truth_idx_[best_prior_idx_[j]] = j
    matches_ = truths_loc[best_truth_idx_]          # Shape: [num_priors,4]
    print('matches_, best_truth_idx_',matches_.shape, best_truth_idx_.shape)

    # fig: matches x,y hist just with overlapped priors
    ax23.hist((matches_[:,2]+matches_[:,0])/2,bins=100)
    ax23.set_title('matches x hist overlapped')
    ax24.scatter(prior_overlaped[:,0]-(matches_[:,2]+matches_[:,0])/2, prior_overlaped[:,1]-(matches_[:,3]+matches_[:,1])/2, s=1)
    ax24.set_title('x,y differs overlapped')

    # encode and write in
    print('='*10+'encoding'+'='*10)
    mkind = truths_labels[best_truth_idx]         # Shape: [num_priors]  这是先基于面积匹配得到了框，然后直接将与匹配度最高的真值框的种类赋予。
    loc = encode(matches, priors, cfg['128']['variance'])
    mkind[best_truth_overlap < argsloss.threshold] = 0  # label as background
    loc_t[0] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[0] = mkind  # [num_priors] top class label for each prior
    print('loc.max, loc.min, abs(loc).min: ',loc.max(0), loc.min(0), torch.abs(loc).min(0),end='\n\n')
    print('mkind over threshold: ', mkind.shape,end='\n\n')

    # encode just with overlapped priors
    mkind_ = truths_labels[best_truth_idx_]         # Shape: [num_priors]  这是先基于面积匹配得到了框，然后直接将与匹配度最高的真值框的种类赋予。
    loc_ = encode(matches_, prior_overlaped, cfg['128']['variance'])
    mkind_[best_truth_overlap_ < argsloss.threshold] = 0  # label as background
    print('loc_.max, loc_.min, abs(loc_).min: ',loc_.max(0), loc_.min(0), torch.abs(loc_).min(0),end='\n\n')
    print('mkind_ over threshold: ', mkind_.shape,end='\n\n')

    # fig: IOU-each prior with over threshold
    overlap_over = best_truth_overlap >= argsloss.threshold
    ax31.scatter(torch.where(overlap_over)[0], best_truth_overlap[overlap_over],s=1)
    ax31.scatter(best_prior_idx, best_truth_overlap[best_prior_idx],s=1,c='r')
    ax31.set_title('IOU-each prior with over threshold')

    #################-*- loss-posneg -*-####################
    # def: num of match pos with threshold
    def matched_positive_with_threshold(threshold):
        if len(threshold) == 1:
            mkinded = truths_labels[best_truth_idx_]         # Shape: [num_priors]  这是先基于面积匹配得到了框，然后直接将与匹配度最高的真值框的种类赋予。
            mkinded[best_truth_overlap_ < threshold[0]] = 0  # label as background
            result = mkinded.sum()
        elif len(threshold) > 1:
            result = []
            for i in threshold:
                mkinded = truths_labels[best_truth_idx_]         # Shape: [num_priors]  这是先基于面积匹配得到了框，然后直接将与匹配度最高的真值框的种类赋予。
                mkinded[best_truth_overlap_ < i] = 0  # label as background
                result.append(mkinded.sum())
        return result

    # fig: num of matched_positive - threshold
    tl = np.linspace(-10,0,101)
    tl = 2 ** tl
    ax32.plot(tl,matched_positive_with_threshold(tl))
    ax32.grid()
    ax32.set_title('num of matched_positive - threshold')

    # loc_t, conf_t setting
    if argsloss.use_gpu:
        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
    loc_t.requires_grad = False
    conf_t.requires_grad = False

    # pos definition
    print('='*10+'lossing'+'='*10)
    if argsloss.is_ODM:
        P = F.softmax(arm_conf_data, 2)
        arm_conf_tmp = P[:,:,1]
        object_score_index = arm_conf_tmp <= argsloss.theta
        pos = conf_t > 0
        pos[object_score_index.data] = 0
    else:
        pos = conf_t > 0
    print('pos.shape, pos.sum', pos.shape, pos.sum())

    # loc loss: Smooth L1 loss
    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
    loc_prediction = loc_data[pos_idx].view(-1, 4)
    loc_truth = loc_t[pos_idx].view(-1, 4)
    print('loc_prediction.shape, loc_truth.shape, loc_prediction, loc_truth, loc_truth.max(0): ',\
            loc_prediction.shape, loc_truth.shape, loc_prediction, loc_truth, loc_truth.max(0))
    loss_l = F.smooth_l1_loss(loc_prediction, loc_truth, reduction='sum')
    print('>>> loss_loc: ',loss_l.item())

    # loss_conf pre-calculation
    batch_conf = conf_data.view(-1, argsloss.num_classes)
    loss_conf = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  # gather(dim, indexes)

    # hard negative mining
    loss_conf[pos.view(-1,1)] = 0  # filter out pos boxes for now
    loss_conf = loss_conf.view(num, -1)  # batch, num_priors
    _, loss_idx = loss_conf.sort(1, descending=True)   # 一次排序的idx得到原张量从大到小排序每个值对应的序列号
    _, idx_rank = loss_idx.sort(1)                  # 二次排序的idx得到原张量从大到小排序每个值对应的序列号的从大到小排序，即每个值从大到小的排名
    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(argsloss.negpos_ratio*num_pos, max=pos.size(1)-1)
    neg = idx_rank < num_neg.expand_as(idx_rank)  # 这一步也可以通过对loss_idx索引前num_neg条实现，若如此则不需要二次排序
    print('num of pos/neg/all: ', num_pos, num_neg, (pos+neg).sum())

    # kind loss: cross entropy
    pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    neg_idx = neg.unsqueeze(2).expand_as(conf_data)
    conf_prediction = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, argsloss.num_classes)
    targets_weighted = conf_t[(pos+neg).gt(0)]
    loss_c = F.cross_entropy(conf_prediction, targets_weighted, reduction='sum')
    loss_lm = loss_l / num_pos.data.sum().float()
    loss_cm = loss_c / (pos+neg).sum().float()
    loss_conf = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  # 重计算
    loss_conf = loss_conf.reshape(1,-1)
    print('loss_l, loss_c, loss_lm, loss_cm: ', loss_l.item(), loss_c.item(), loss_lm.item(), loss_cm.item())
    print('loss_c pos neg: ', loss_conf[pos].sum().item(), loss_conf[neg].sum().item())

    # overlapped priors selection
    posneg = (pos+neg).gt(0)
    loss_c_overlaped = loss_conf.squeeze()[overlapindex]
    loss_c_unoverlaped = loss_conf.squeeze()[overlapindex.lt(1)]

    # fig: overlap-unoverlap-all loss_c
    ax41.hist(loss_c_overlaped.squeeze().detach())
    ax42.hist(loss_c_unoverlaped.squeeze().detach())
    ax43.hist(loss_conf.squeeze().detach())
    ax41.set_title('overlaped conf loss hist')
    ax42.set_title('unoverlaped conf loss hist')
    ax43.set_title('total conf loss hist')
    fig51 = ax51.scatter(batch_conf[overlapindex,0].detach(),batch_conf[overlapindex,1].detach(),s=1,c=torch.log(loss_c_overlaped.detach()))
    fig52 = ax52.scatter(batch_conf[overlapindex.lt(1),0].detach(),batch_conf[overlapindex.lt(1),1].detach(),s=1,c=torch.log(loss_c_unoverlaped.squeeze()).detach())
    fig53 = ax53.scatter(batch_conf[:,0].detach(),batch_conf[:,1].detach(),s=1,c=torch.log(loss_conf.squeeze()).detach())
    fig.colorbar(fig51, ax=ax51)
    fig.colorbar(fig52, ax=ax52)
    fig.colorbar(fig53, ax=ax53)
    ax51.set_title('overlaped conf')
    ax52.set_title('unoverlaped conf')
    ax53.set_title('total conf')

    # fig: pos/neg in overlap-unoverlap-all loss_c
    fig61 = ax61.scatter(batch_conf[overlapindex,0].detach(),batch_conf[overlapindex,1].detach(),s=1,c=torch.log(loss_c_overlaped.detach()))
    ax61.scatter(batch_conf[overlapindex&neg.squeeze(),0].detach(),batch_conf[overlapindex&neg.squeeze(),1].detach(),s=1,c='r',label='neg')
    ax61.scatter(batch_conf[overlapindex&pos.squeeze(),0].detach(),batch_conf[overlapindex&pos.squeeze(),1].detach(),s=1,c='y',label='pos')
    ax61.legend()
    fig62 = ax62.scatter(batch_conf[overlapindex.lt(1),0].detach(),batch_conf[overlapindex.lt(1),1].detach(),s=1,c=torch.log(loss_c_unoverlaped.squeeze()).detach())
    ax62.scatter(batch_conf[overlapindex.lt(1)&neg.squeeze(),0].detach(),batch_conf[overlapindex.lt(1)&neg.squeeze(),1].detach(),s=1,c='r',label='neg')
    ax61.scatter(batch_conf[overlapindex.lt(1)&pos.squeeze(),0].detach(),batch_conf[overlapindex.lt(1)&pos.squeeze(),1].detach(),s=1,c='y',label='pos')
    ax62.legend()
    fig63 = ax63.scatter(batch_conf[:,0].detach(),batch_conf[:,1].detach(),s=1,c=torch.log(loss_conf.squeeze()).detach())
    ax63.scatter(batch_conf[neg.squeeze(),0].detach(),batch_conf[neg.squeeze(),1].detach(),s=1,c='r',label='neg')
    ax63.scatter(batch_conf[pos.squeeze(),0].detach(),batch_conf[pos.squeeze(),1].detach(),s=1,c='y',label='pos')
    ax63.legend()
    ax61.set_title('pos/neg in overlaped conf')
    ax62.set_title('pos/neg in unoverlaped conf')
    ax63.set_title('pos/neg in total conf')
    fig.colorbar(fig61, ax=ax61)
    fig.colorbar(fig62, ax=ax62)
    fig.colorbar(fig63, ax=ax63)
    fig.savefig('check/pngs/loss_c-overlap_selection_pos.png')

    # fig: xc,yc,w,h loss_l
    diff = loc_prediction-loc_truth
    diff = diff.detach()
    with torch.no_grad():
        ax71.scatter(diff[:,0], smooth_l1(torch.abs(diff[:,0]),1), s=1, c='r', label='xc_encode')
        ax71.scatter(diff[:,1], smooth_l1(torch.abs(diff[:,1]),1), s=1, c='g', label='yc_encode')
        ax71.scatter(diff[:,2], smooth_l1(torch.abs(diff[:,2]),1), s=1, c='c', label='w_encode')
        ax71.scatter(diff[:,3], smooth_l1(torch.abs(diff[:,3]),1), s=1, c='b', label='h_encode')
        ax71.legend()
        ax71.set_title('overlap-unoverlap loss_l')
        # fig: overlap/unoverlap loss_l
        diff = loc_data-loc_t
        diff = diff.view(-1, 4)
        #TODO: x
        ax72.scatter(diff.squeeze()[overlapindex,0], smooth_l1(torch.abs(diff.squeeze()[overlapindex]),1).sum(1), s=1, c='r', label='overlap')
        ax72.scatter(diff.squeeze()[overlapindex.lt(1),0], smooth_l1(torch.abs(diff.squeeze()[overlapindex.lt(1)]),1).sum(1), s=1, c='g', label='unoverlap')
        ax72.legend()
        ax72.set_title('overlap/unoverlap loss_l')
        # fig: pos/neg loss_l
        ax73.scatter(diff[posneg.squeeze().lt(1),0], smooth_l1(torch.abs(diff[posneg.squeeze().lt(1)]),1).sum(1), s=1, c='c', label='others')
        ax73.scatter(diff[neg.squeeze(),0], smooth_l1(torch.abs(diff[neg.squeeze()]),1).sum(1), s=1, c='g', label='neg')
        ax73.scatter(diff[pos.squeeze(),0], smooth_l1(torch.abs(diff[pos.squeeze()]),1).sum(1), s=1, c='r', label='pos')
        ax73.legend()
        ax73.set_title('pos/neg loss_l')

    fig.suptitle('check_matchloss\n| dataset: %s | date: %s | epoch: %s | id: %s |'%
                (pathes['dataroot'].split('dataset/')[1][:-1], date_stamp, eph_stamp, testid))
    fig.savefig(check_root+'check_matchloss_|%s|.png'%testid)
    # fig.show()

    
# data loading
# for idx in fileid_list:
print('='*10+'data loading'+'='*10)
for testid in fileid_list:
    print('testid: ',testid)
    img = np.load(pathes['dataroot']+'/image/img_'+testid+'.npy').astype(np.float32)
    img = torch.tensor(img).repeat(1,3,1,1)
    targets = np.load(pathes['dataroot']+'/catalog/cat_'+testid+'.npy').astype(np.float16)
    targets = torch.tensor(targets).repeat(1,1,1)
    print('truth: ', targets.size(1))
    check_one_slide(net, img, targets, testid)
