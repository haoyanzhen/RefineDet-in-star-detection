'''
0
backup by hyz in 2023.04.14
changed way of generation of prior boxes to fit mission of cf separation.
'''

from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.feeling_field = cfg['feeling_field']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect = cfg['aspect']
        self.star_gal_ratio = cfg['star_gal']
        self.aspect_ratios = cfg['aspect_ratios']
        self.mbox = cfg['mbox']
        self.offset_ratio = cfg['offset_ratio']
        self.size_offset = cfg['size_offset']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        output = torch.empty((0,4))
        # 空间均匀产生先验框
        for k, f in enumerate(self.feature_maps):
            # for i, j in product(range(f), repeat=2):
                # f_k = self.image_size / self.steps[k]
                # # unit center x,y
                # cx = (j + 0.5) / f_k
                # cy = (i + 0.5) / f_k

                # # aspect_ratio: 1
                # # rel size: min_size
                # s_k = self.min_sizes[k]/self.image_size
                # mean += [cx, cy, s_k, s_k]

                # # aspect_ratio: 1
                # # rel size: sqrt(s_k * s_(k+1))
                # if self.max_sizes:
                #     s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                #     mean += [cx, cy, s_k_prime, s_k_prime]

                # # rest of aspect ratios
                # for ar in self.aspect_ratios[k]:
                #     mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                #     mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                
            # unit center x,y
            center_axis = torch.arange(f) + 0.5
            center_x, center_y = torch.meshgrid(center_axis,center_axis)
            center_grid = torch.vstack((center_x.reshape(-1),center_y.reshape(-1))).T
            center_k = center_grid
            
            # size
            size_k = torch.ones_like(center_k) * self.min_sizes[k]
            if self.size_offset:
                size_min = (0 if k == 0 else self.feeling_field[k-1]/2)
                size_max = self.feeling_field[k]
                offset_min = self.min_sizes[k] - size_min
                offset_max = size_max - self.min_sizes[k]
                size_offset_ratio = (  0.3 if self.mbox[k] <= 5 else 0.7/((self.mbox[k]-1)//2)  )
                offset_l = [[i*(size_offset_ratio)*offset_max for i in torch.arange(self.mbox[k])[::2]/2],
                            [i*(-size_offset_ratio)*offset_min for i in (torch.arange(self.mbox[k])[1::2]+1)/2]]
                if len(offset_l[0]) != len(offset_l[1]):
                    offset_l[1].append(0)
                offset_l = torch.tensor(offset_l).T.reshape(-1)
                size_kp = torch.empty(0)
                for i in offset_l[:self.mbox[k]]:
                    size_kp = torch.cat([size_kp, size_k+i])
            else:
                size_kp = size_k.repeat(self.mbox[k],1)
            
            # mbox
            mbox_offset_list = [(0,0),(1,1),(1,-1),(-1,-1),(-1,1),(1,0),(-1,0),(0,1),(0,-1)]
            center_kp = torch.empty(0)
            for _m in range(self.mbox[k]):
                center_kpi = center_k + torch.tensor(mbox_offset_list[_m]).expand_as(center_k) * self.offset_ratio
                center_kp = torch.cat([center_kp,center_kpi],dim=0)
            # center_k *= self.steps[k]
            center_kp *= self.steps[k]
            
            # output
            # output_k = torch.concat((center_k,size_k),dim=1)
            output_kp = torch.concat((center_kp,size_kp),dim=1)
            output = torch.concat((output,output_kp),dim=0)
                
        # output = output * self.image_size  # 未发现任何有必要归一的地方，因此在此处乘以图像尺度
        # back to torch land
        output = output.view(-1, 4)
        if self.clip:
            output.clamp_(max=self.image_size, min=0)  # 同时也改了这个
        return output
