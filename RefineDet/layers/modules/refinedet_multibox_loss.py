# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import cf_refinedet as cfg
from data import HOME
from data import args
from ..box_utils import match, log_sum_exp, refine_match


class RefineDetMultiBoxLoss(nn.Module):
    """RefineDet Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) calculate loss of cat and regression, and add them with power.
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Strategy:
        hard negative mining: only choose the max several negative to calculate loss
            (default negative:positive ratio 3:1)
    """

    def __init__(self, num_classes, img_size, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, theta=0.01, is_ODM=False):
        super(RefineDetMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg[img_size]['variance']
        self.theta = theta
        self.is_ODM = is_ODM

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors = predictions

        # check point
        # torch.save(predictions,HOME+'/check/predictions.pt')
        # torch.save(targets,HOME+'/check/targets.pt')

        # initialize
        if self.is_ODM:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        num = loc_data.size(0)  # batch
        priors = priors[:loc_data.size(1), :]  # choose prior boxes in num of prediction boxes
        num_priors = (priors.size(0))  # in fact this is prior boxes after choose
        num_classes = self.num_classes
        #print(loc_data.size(), conf_data.size(), priors.size())

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if not self.is_ODM:
                labels = labels >= 0
            defaults = priors.data
            if self.is_ODM:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx, arm_loc_data[idx].data)
                # print(args.ischeck)
                if args.ischeck:
                    torch.save(loc_t,HOME+'/check/loc_t_odm.pt')
                    torch.save(conf_t,HOME+'/check/conf_t_odm.pt')
            else:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)
                if args.ischeck:
                    torch.save(loc_t,HOME+'/check/loc_t_arm.pt')
                    torch.save(conf_t,HOME+'/check/conf_t_arm.pt')
        
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        #print(loc_t.size(), conf_t.size())

        # pos definition
        if self.is_ODM:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_tmp = P[:,:,1]
            object_score_index = arm_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0
        #print(pos.size())
        #num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # 损失指的是最佳先验匹配和预测框的损失
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # print(loc_data.device,pos_idx.device)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        # 这里的loss_c是临时的
        batch_conf = conf_data.view(-1, self.num_classes)
        # print(batch_conf.size,conf_t.size)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #print(loss_c.size())

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)  # batch, num_priors
        _, loss_idx = loss_c.sort(1, descending=True)   # 一次排序的idx得到原张量每个idx对应的大小排名
        _, idx_rank = loss_idx.sort(1)                  # 二次排序的idx得到原张量每个值从大到小的idx
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print(num_pos.size(), num_neg.size(), neg.size())

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_prediction = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # print(pos_idx.size(), neg_idx.size(), conf_p.size(), targets_weighted.size())

        # check point
        if args.ischeck:
            if self.is_ODM:
                torch.save(conf_prediction,HOME+'/check/conf_prediction_odm.pt')
                torch.save(targets_weighted,HOME+'/check/target_weighted_odm.pt')
            else:
                torch.save(conf_prediction,HOME+'/check/conf_prediction_arm.pt')
                torch.save(targets_weighted,HOME+'/check/target_weighted_arm.pt')
        loss_c = F.cross_entropy(conf_prediction, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum().float()
        # N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        #print(N, loss_l, loss_c)
        return loss_l, loss_c
