# a spatial matching with kdtree.
# extremely matching stars with two cat, each experiencing a sole match.
# created by hyz in 2023.03.21

from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

def kdmatching(cat_sex,cat_raw,reg0=None,reg1=None):
    _acat = cat_sex
    sex_cat = fits.open(_acat)[2].data
    sex_cat = Table(sex_cat)

    _catr = cat_raw
    with open(_catr,'r') as f:
        names = f.readlines()[0].strip()[1:].split(' ')[1:]
    cat_raw = pd.read_csv(_catr,delim_whitespace=True,names=names,comment='#')
    lenc = len(sex_cat)
    _reg = _acat[0:-4] + '_sex.reg'
    with open(_reg, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nimage\n")
        for i in range(lenc):
            f.writelines("ellipse("+str(sex_cat['XWIN_IMAGE'][i])+","+str(sex_cat['YWIN_IMAGE'][i])+
                        ","+str(4*sex_cat['AWIN_IMAGE'][i])+","+str(4*sex_cat['BWIN_IMAGE'][i])+","
                        +str(sex_cat['THETAWIN_IMAGE'][i])+")\n")
    _reg = _acat[0:-4] + '_sex_circle.reg'
    with open(_reg, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nimage\n")
        for i in range(lenc):
            f.writelines("circle("+str(sex_cat['XWIN_IMAGE'][i])+","+str(sex_cat['YWIN_IMAGE'][i])+","+
                         str((sex_cat['ISOAREAF_IMAGE'][i]/np.pi)**0.5)+")\n")
    sex_cat[:2]
    lencr = len(cat_raw)
    _regr = _acat[0:-4] + '_raw.reg'
    with open(_regr, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nimage\n")
        for i in range(lencr):
            f.writelines("circle("+str(cat_raw['xImage'][i])+","+str(cat_raw['yImage'][i])+","+str((28-cat_raw['mag'][i])**0.5)+")\n")
    from scipy import spatial

    cat0 = sex_cat.to_pandas()[['XWIN_IMAGE','YWIN_IMAGE','AWIN_IMAGE','BWIN_IMAGE','THETAWIN_IMAGE','ISOAREAF_IMAGE']]
    cat0 = cat0.to_numpy()
    tree0 = spatial.KDTree(cat0[:,:2])
    cat0nb = tree0.query(cat0[:,:2],k=[2])
    cat0nb = np.array(cat0nb)
    cat0nb = cat0nb.squeeze()
    index0 = np.where(cat0nb[0,:]>cat0[:,2])[0]
    cat0s = cat0[index0]
    tree0s = spatial.KDTree(cat0s[:,:2])

    cat1 = cat_raw[['xImage','yImage','mag','teff','logg','feh']].to_numpy()
    tree1 = spatial.KDTree(cat1[:,:2])
    cat1nb = tree1.query(cat1[:,:2],k=[2])
    cat1nb = np.array(cat1nb)
    cat1nb = cat1nb.squeeze()
    index1 = np.where(cat1nb[0,:]>(28-cat1[:,2])**0.5)[0]
    cat1s = cat1[index1]
    tree1s = spatial.KDTree(cat1s[:,:2])
    cat01nb = tree1s.query(cat0s[:,:2],k=[1])
    cat01nb = np.array(cat01nb)
    cat01nb = cat01nb.squeeze()
    index01 = np.where(cat01nb[0,:]<cat0s[:,2])[0]

    cat0ss = cat0s[index01]
    cat1ss = cat1s[np.int32(cat01nb[1,index01])]
    len0 = len(cat0ss)
    if reg0 == None:
        _reg0 = './__data__/nearby_0.reg'
    else:
        _reg0 = reg0
    with open(_reg0, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nimage\n")
        for i in range(len0):
            f.writelines("ellipse("+str(cat0ss[i,0])+","+str(cat0ss[i,1])+
                        ","+str(4*cat0ss[i,2])+","+str(4*cat0ss[i,3])+","
                        +str(cat0ss[i,4])+")\n")

    len1 = len(cat1ss)
    if reg1 == None:
        _reg1 = './__data__/nearby_1.reg'
    else:
        _reg1 = reg1
    with open(_reg1, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nimage\n")
        for i in range(len1):
            f.writelines("circle("+str(cat1ss[i,0])+","+str(cat1ss[i,1])+","+str((28-cat1ss[i,2])**0.5)+")\n")
    return cat0ss,cat1ss

def mag_side(mag,a,b,c,d,e):
    return ((np.exp(a*c-a*mag)-e*np.exp(-a*c+a*mag))/(e*np.exp(-a*c+a*mag)+np.exp(a*c-a*mag)))*b+d


class Mag2Side(nn.Module):
    def __init__(self):
        super(Mag2Side,self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,1)
        self.a1 = nn.Sigmoid()
        self.a2 = nn.Sigmoid()
        self.a3 = nn.Sigmoid()
        # self.a4 = nn.ReLU()
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.a1(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.a2(x)
        x = self.fc3(x)
        # x = self.relu(x)
        # x = self.a3(x)
        x = self.fc4(x)
        # x = self.relu(x)
        # x = self.a4(x)
        return x






if __name__ == '__main__':
    cat_sex = './__data__/test.cat'
    cat_raw = '/media/hyz/dwarfcave/data/csst_simulation/simulation_work/work_dir/Ant_simulation20230310/MSC_0000000/MSC_100000000_chip_08_filt_g.cat'
    solestar = kdmatching(cat_sex,cat_raw)