"""
created by hyz in 2023.06.14
for crowded field detection
"""

from data import *
from data import cf_refinedet
from data import evalargs as eargs
from layers.box_utils import jaccard
from models.refinedet import build_refinedet
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches as patches
from matplotlib.collections import PatchCollection


# init definition
pathes = {
    'weight': eargs.weight,
    'dataroot': '',
    'evalroot': ''
}

'''
# ap score function
def test_eval_cf(boxes, truth, eval_overlap_threshold):
    """calculate fp tp list and ap score.
    Args:
        boxes (_type_): prediction boxes in min-max
        truth (_type_): truth boxes in min-max
        fp_threshold (_type_): overlap threshold to believe it is true
    """
    # IoU calculation and threshold and match
    overlap = jaccard(boxes,truth)  # [len_boxes, len_truth]
    overlap_ebox_max, overlap_ebox_idx = overlap.max(0, keepdim=True)       # box
    overlap_etruth_max, overlap_etruth_idx = overlap.max(1, keepdim=True)   # truth
    
    # evaluation of 4 
    tp, fp = torch.zeros(len(boxes)), torch.zeros(len(boxes))  # truth-positive_pred, false-positive_pred. choose from predictions
    tn = torch.zeros(len(truth))  # truth-negative_pred, false-negative_pred（not exist）. choose from truths
    tp_mask = overlap_ebox_idx[overlap_ebox_max > eval_overlap_threshold]
    tp[tp_mask] = 1
    fp[tp_mask] = 1
    fp = 1 - fp
    tn_mask = overlap_etruth_idx[overlap_etruth_max < eval_overlap_threshold]
    tn[tn_mask] = 1
    
    # recall, precision and ap
    recall = tp                 # recall = tp/(tp+fn)
    precision = tp/(tp+fp)      # precision = tp/(tp+fp) = tp/p
    tt1 = tp/(tp+tn)
    tt2 = tn/(tp+fp)
    tt3 = fp/(tp+fp)
    
    # 累计分布
    accu_recall = torch.zeros(len(recall))          # accumulative recall
    accu_precision = torch.zeros(len(precision))    # accumulative precision
    ap = 0
    if len(accu_recall) != len(accu_precision):
        print('Warning: len of accu_recall and accu_precision is not same')
    for idx, _ in accu_recall:
        accu_recall[idx] = recall[:idx].sum()
        accu_precision[idx] = precision[:idx].sum()
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.step(accu_recall, accu_precision, where='post')
    ax.set_title('precision-recall curve')
    fig.savefig(pathes['evalroot']+'precision-recall.png')
    
    # 分段插值
    sarl = [sar1, sar2, sar3, sar4] = torch.split(accu_recall, len(accu_recall)//4+1)
    inter_arec = torch.cat(sarl)
    inter_apre = torch.empty(0)
    for sar in sarl:
        inter_ap = torch.cat([inter_apre, F.interpolate(sar,scale_factor=2,mode='linear',align_corners=False)])
    
    # 积分ap
    ap = (inter_apre[1:]+inter_apre[:-1])/2 * (inter_arec[1:]-inter_arec[:-1])
    ap = ap.sum()
    
    # 输出最终的准确率和漏查率
    value_precision = tp.sum() / (tp.sum()+fp.sum())
    fail_detection_ratio = tn.sum() / (tn.sum()+tp.sum())
    
    return value_precision, fail_detection_ratio, ap
'''

# accuracy and precision
def test_eval_cf(boxes, truth, eval_overlap_threshold):
    """calculate fp tp list and ap score.
    Args:
        boxes (_type_): prediction boxes in min-max
        truth (_type_): truth boxes in min-max
        fp_threshold (_type_): overlap threshold to believe it is true
    """
    # IoU calculation and threshold and match
    overlap = jaccard(boxes,truth)  # [len_boxes, len_truth]
    overlap_ebox_max, overlap_ebox_idx = overlap.max(0, keepdim=True)       # box
    overlap_etruth_max, overlap_etruth_idx = overlap.max(1, keepdim=True)   # truth
    
    # accuracy
    accu = overlap_ebox_max > eval_overlap_threshold
    accu = accu.squeeze()
    accu = accu.sum()/len(accu)
    # precision
    prec = overlap_etruth_max > eval_overlap_threshold
    prec = prec.squeeze()
    prec = prec.sum()/len(prec)
    
    return accu, prec

def test_cf(eval_overlap_threshold, if_pred_only=False):
    # data loading
    date_stamp = pathes['weight'].split('cf_refinedet_')[1].split('/RefineDet128_ds')[0]
    eph_stamp = pathes['weight'].split('eph')[1].split('_itr')[0]
    dataset_stamp = pathes['weight'].split('_ds')[1].split('_eph')[0]
    pathes['dataroot'] = './dataset/' + dataset_stamp + '/'
    pathes['evalroot'] = './eval/' + dataset_stamp + '_' + date_stamp + '_test%s'%eargs.testdate + '/'
    if not os.path.exists(pathes['evalroot']):
        os.mkdir(pathes['evalroot'])
    
    fileid_list = list()
    for i in os.listdir(pathes['dataroot']+'anno'):
        if i.endswith('fits'):
            fileid_list.append(i[:-5].split('anno_')[1])

    # net load
    cfg_json = (eargs.cfg_json if os.path.exists(eargs.cfg_json) 
                else os.path.split(pathes['weight'])[0]+'/cf_refinedet.json')
    cfg = json.load(open(cfg_json,'r'))
    net = build_refinedet('test',128, cfg['128']['num_classes'], cfg, eargs)
    net.load_weights(pathes['weight'])

    # fig define
    fig, ((ax0,ax1),(ax2,ax3),(ax4,ax5),(ax6,ax7),(ax8,ax9)) = plt.subplots(5,2,figsize=(8,32))
    ax_list = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
        
    for index, testid in enumerate(fileid_list):
        # forward
        img = np.load(pathes['dataroot']+'/image/img_'+testid+'.npy').astype(np.float32)
        img = torch.tensor(img).repeat(1,3,1,1)
        target = np.load(pathes['dataroot']+'/catalog/cat_'+testid+'.npy')
        truth = torch.tensor(target[:,:-1])  # min-max
        truth_cs = torch.cat([(truth[:,2:]+truth[:,:2])/2, truth[:,2:]-truth[:,:2]], dim=1)
        out = net(img)  # batch, class, (boxes, score+loc)
        boxes = out[0,1,:]  # 0:background 1:star
        scores = boxes[:,0]
        boxes = boxes[:,1:]
        nonzero = int((scores != 0).sum())
        boxes = boxes[:nonzero]
        scores = scores[:nonzero]
        print('testid, boxes size, truth size:  ', testid, len(boxes), len(truth))
        
        # 假设是xmin,ymin,xmax,ymax形式
        boxes_cs = torch.cat([(boxes[:,2:]+boxes[:,:2])/2,(boxes[:,2:]-boxes[:,:2])], dim=1)
        # reg file generation
        with open(pathes['evalroot']+testid+'.reg','w') as p:
            p.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
            for i,_score in enumerate(scores):
                p.write(f'box({boxes_cs[i,0].item()},{boxes_cs[i,1].item()},{boxes_cs[i,2].item()},{boxes_cs[i,3].item()},0) \t' +\
                        '# text = {%.2f}\n'%_score.item())
            p.write('global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nphysical \n')
            for i in truth_cs:
                p.write(f'box({i[0].item()},{i[1].item()},{i[2].item()},{i[3].item()},0) \t\n')
        
        # evaluation
        # value_precision, fail_detection_ratio, ap = test_eval_cf(boxes, truth, eval_overlap_threshold)
        accu, prec = test_eval_cf(boxes, truth, eval_overlap_threshold)
        print("testid, cover, precision:  ",testid, accu, prec)
    
        # png file generation
        fig_ = ax_list[index].imshow(np.log(img[0][0]),origin='lower',cmap='ocean')
        rtas_pre = [patches.Rectangle((xc-w/2,yc-h/2),w,h) for xc,yc,w,h in boxes_cs.detach()]
        pc_pre = PatchCollection(rtas_pre,edgecolor='y',facecolor='none',label='prediction')
        ax_list[index].add_collection(pc_pre)
        if not if_pred_only:
            rtas_tru = [patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin) for xmin,ymin,xmax,ymax in truth.detach()]
            pc_tru = PatchCollection(rtas_tru,edgecolor='r',facecolor='none',label='truth')
            ax_list[index].add_collection(pc_tru)
        ax_list[index].set_title('%s\n|num| pred:%d truth:%d\naccu:%.4f  prec:%.4f'%
                                (testid, len(boxes), len(truth), accu, prec))
        fig.colorbar(fig_, ax=ax_list[index])
        # ax_list[index].legend()
        
    if if_pred_only:
        fig.suptitle('evaluation with prediction only')
        fig.savefig(pathes['evalroot']+'eval_prediction.png')
    else:
        fig.suptitle('evaluation with truth and prediction')
        fig.savefig(pathes['evalroot']+'eval_truth_prediction.png')
    return
    
if __name__ == '__main__':
    test_cf(eargs.eval_overlap_threshold)
    test_cf(eargs.eval_overlap_threshold, if_pred_only=True)
