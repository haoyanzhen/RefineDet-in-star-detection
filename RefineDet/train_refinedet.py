from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import RefineDetMultiBoxLoss
#from ssd import build_ssd
from models.refinedet import build_refinedet
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.logging import Logger
import time



def stuctured_experiment():
    """
    因为调用栈的存在，本方法并不适用。
    找到了解决办法！
    """
    if args.expstructured:
        num_exp = 4
        cfr = {
            "128":{'num_classes': [1,2,3,4],
            }
        }
        cfp = {'negpos_ratio': [0,1,5,10],}
        ar  = {'save_folder':['weights/cf_refinedet_20230618/',
                            'weights/cf_refinedet_20230618p/',
                            'weights/cf_refinedet_20230618pp/',
                            'weights/cf_refinedet_20230618p3/']}
    if args.expstructured:
        temp_config = ''
        cfr = cfr[args.input_size]
        with open('./data/config.py', 'r') as _c:
            lines = _c.readlines()
            for keys in cfr[args.input_size]:
                if len(cfr[keys]) not in [1,num_exp]:
                    print('WARNING: the lens of cfr are not same with experiment times')
                    return
            for _idx, _l in enumerate(lines):
                if 'cf_refinedet = ' in _l:
                    for __l in lines[_idx:]:
                        if '%s: {'%args.input_size in __l:
                            for line in lines[_idx:]:
                                if keys in line:
                                    _old = line.split("'%s': "%keys)[1]
                                    _new = (f'{cfr[keys]}' + '#'.ljust(49) + _old.split('#')[1] 
                                            if '#' in _old else f'{cfr[keys]}')
                                    line.replace(_old)
                        _l = (_l.replace(_l.split(keys)[1],'')    if keys in _l else _l)
                if not _l.startswith(' '):
                    break
                
                
    from importlib import reload
    data = reload(data)
                
        
    


st = time.time()
# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")

# change default tensor type with cuda
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# torch.multiprocessing.set_start_method('spawn')  # To use CUDA with multiprocessing.

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# 将所有标准输出重定向到文件中
sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

def train():
    # dataset
    if args.dataset == 'cf01' or 'cf02':
        cfg = cf_refinedet[args.input_size]
        dataset = CFDetection(dataset=args.dataset, cuda=args.cuda)
        print('dataset creation')

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()
    
    # net
    refinedet_net = build_refinedet('train', cfg['min_dim'], cfg['num_classes'], cf_refinedet)
    net = refinedet_net
    # print(net)
    net.use_nnpack = False
    print('build refinedet')
    print(net)
    # print(net)
    if args.cuda:
        print('xx')
        net = torch.nn.DataParallel(refinedet_net)
        cudnn.benchmark = True

    # resume loading
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet_net.load_weights(args.resume)
    else:
        #vgg_weights = torch.load(args.save_folder + args.basenet)
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        refinedet_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        refinedet_net.extras.apply(weights_init)
        refinedet_net.arm_loc.apply(weights_init)
        refinedet_net.arm_conf.apply(weights_init)
        refinedet_net.odm_loc.apply(weights_init)
        refinedet_net.odm_conf.apply(weights_init)
        #refinedet_net.tcb.apply(weights_init)
        refinedet_net.tcb0.apply(weights_init)
        refinedet_net.tcb1.apply(weights_init)
        refinedet_net.tcb2.apply(weights_init)

    # optimizer and loss function
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(2, args.input_size, cf_parameters['overlap_threshold'], True, 
                                          0, True, cf_parameters['negpos_ratio'], 0.5,
                                        False, args.cuda, cf_parameters['arm_conf_softmax_threshold'])
    odm_criterion = RefineDetMultiBoxLoss(cfg['num_classes'], args.input_size, cf_parameters['overlap_threshold'], True, 
                                          0, True, cf_parameters['negpos_ratio'], 0.5,
                                        False, args.cuda, cf_parameters['arm_conf_softmax_threshold'], is_ODM=True)

    net.train()
    # loss counters
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(vars(args))

    step_index = 0

    # create visdom plot
    if args.visdom:
        if args.resume:
            iteration = int(os.path.split(args.resume)[1].split('itr')[1].split('.pth')[0])
            epoch = int(os.path.split(args.resume)[1].split('eph')[1].split('_')[0])
        else:
            iteration, epoch = 0, 0
        vis_title = dataset.name + ' | ' + args.save_folder.split('cf_refinedet_')[1].split('/')[0] + \
                    ' | ' + time.strftime('%H:%M:%S',time.localtime())
        vis_legend = ['Loc Loss arm', 'Conf Loss arm', 'Loc Loss odm', 'Conf Loss odm', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend, iteration)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend, epoch)
    
    # generator will generate number in device randomly, so generator should keep in together with tensors
    if args.cuda:
        generator=torch.Generator(device = 'cuda')
    else:
        generator=None
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,  # 用于生成batch数据，对同尺寸数据使用默认即可，此处框的数量不固定，在data.__init__修改
                                  pin_memory=False,              # torch.cuda.floattensor是不能pin的
                                  generator=generator)  # 防止iter那里device错误
    # create batch iterator
    batch_iterator = iter(data_loader)

    # if resume, load resumed loss
    if args.resume:
        resume_dir = os.path.split(args.resume)[0]
        resume_id = os.path.split(args.resume)[1].split('Det')[1].split('.pth')[0]
        resume_epochloss = os.path.join(resume_dir, 'epochloss'+resume_id+'.npy')
        resume_itrloss = os.path.join(resume_dir, 'iterationloss'+resume_id+'.npy')
        epochloss_array = np.load(resume_epochloss)
        iterationloss_array = np.load(resume_itrloss)
    else:
        epochloss_array = np.empty((0,5))
        iterationloss_array = np.empty((0,5))
    
    # decide iteration stop
    itr_stop = min(args.start_iter+args.iteration, cfg['max_iter'])

    print('initial complete: ',time.time()-st)
    for iteration in range(args.start_iter, itr_stop+1):  # 这里的iteration其实是每次batch迭代的iteration，即=epoch*num_batches, 如batch_size=16,len_data=16000,epoch=100,iteration应为100,000
        # print('ite: ', iteration)

        if args.visdom and iteration != args.start_iter:
            update_vis_plot(iteration, arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item(), loss.item(),
                            iter_plot, 'append')  # iteration plot. epoch plot is at the head.
        
        if iteration % epoch_size == 0:  # every epoch save and reset loss counters
            loss_arrayi = np.array([arm_loc_loss, arm_conf_loss, odm_loc_loss, odm_conf_loss,])
            loss_arrayi = np.hstack([loss_arrayi,np.sum(loss_arrayi)])
            epochloss_array = np.vstack([epochloss_array,loss_arrayi])
            if args.visdom and iteration != args.start_iter:
                epoch += 1
                print('-*- epoch: %d -*-'%epoch)
                update_vis_plot(epoch, arm_loc_loss, arm_conf_loss, odm_loc_loss, odm_conf_loss, _loss,
                                epoch_plot, 'append', epoch_size)  # epoch plot
            arm_loc_loss = 0
            arm_conf_loss = 0
            odm_loc_loss = 0
            odm_conf_loss = 0
            _loss = 0

        if iteration == itr_stop:
            break 

        if iteration in np.array(cfg['lr_steps']*args.iteration,dtype=int):
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        # print('setting before calculation: ',time.time()-st)
        
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = images
            targets = [ann for ann in targets]

        # forward
        t0 = time.time()
        images = images.repeat(3,1,1,1).transpose(0,1)  # 迫不得已hh
        out = net(images)
        # print('forward: ',time.time()-st)

        # loss 
        # 此处的loss相当于加权(1,1,1,1)
        optimizer.zero_grad()
        arm_loss_l, arm_loss_c = arm_criterion(out, targets)
        odm_loss_l, odm_loss_c = odm_criterion(out, targets)
        arm_loss = arm_loss_l + arm_loss_c
        odm_loss = odm_loss_l + odm_loss_c
        # loss = arm_loss + odm_loss
        loss = args.loss_weight[0]*arm_loss_l + \
               args.loss_weight[1]*arm_loss_c + \
               args.loss_weight[2]*odm_loss_l + \
               args.loss_weight[3]*odm_loss_c
        # print('loss calculation: ',time.time()-st)

        # backward
        loss.backward()
        optimizer.step()
        t1 = time.time()
        # print('backward: ',time.time()-st)

        # record
        arm_loc_loss += arm_loss_l.item()
        arm_conf_loss += arm_loss_c.item()
        odm_loc_loss += odm_loss_l.item()
        odm_conf_loss += odm_loss_c.item()
        _loss += loss.item()


        # show
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f ||' \
                    % (arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item()))
            loss_arrayi = np.array([arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item(), loss.item()])
            iterationloss_array = np.vstack([iterationloss_array,loss_arrayi])

                
        # auto save
        if iteration != 0 and iteration % args.save_step == 0:
            print('Saving state, iter:', iteration)
            torch.save(refinedet_net.state_dict(), args.save_folder 
                            + '/RefineDet{}_ds{}_eph{}_itr{}.pth'.format(args.input_size, args.dataset, 
                            repr(epoch), repr(iteration)))
            np.save(args.save_folder+'/iterationloss{}_ds{}_eph{}_itr{}.npy'.format(args.input_size, args.dataset, 
                            repr(epoch), repr(iteration)), iterationloss_array)
            np.save(args.save_folder+'/epochloss{}_ds{}_eph{}_itr{}.npy'.format(args.input_size, args.dataset, 
                            repr(epoch), repr(iteration)), epochloss_array)
            torch.save(out,args.save_folder+'prediction{}_ds{}_eph{}_itr{}.pt'.format(args.input_size, args.dataset, 
                            repr(epoch), repr(iteration)))
    with open(args.save_folder+'/cf_refinedet.json','w') as f:
        json.dump(cf_refinedet, f)
    torch.save(refinedet_net.state_dict(), args.save_folder
                        + '/RefineDet{}_ds{}_eph{}_itr{}.final.pth'.format(args.input_size, args.dataset, 
                        repr(epoch), repr(iteration)))
    np.save(args.save_folder+'/iterationloss{}_ds{}_eph{}_itr{}.final.npy'.format(args.input_size, args.dataset, 
                        repr(epoch), repr(iteration)), iterationloss_array)
    np.save(args.save_folder+'/epochloss{}_ds{}_eph{}_itr{}.final.npy'.format(args.input_size, args.dataset, 
                        repr(epoch), repr(iteration)), epochloss_array)
    torch.save(out,args.save_folder+'prediction{}_ds{}_eph{}_itr{}.final.pt'.format(args.input_size, args.dataset, 
                            repr(epoch), repr(iteration)))
    for file in os.listdir(args.save_folder):
        if 'final' not in file and '.json' not in file:
            os.remove('%s/%s'%(args.save_folder,file))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend, _start):
    return viz.line(
        X=torch.ones((1,)).cpu() * _start,
        Y=torch.zeros((1, 5)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc_arm, conf_arm, loc_odm, conf_odm, loss, window, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 5)).cpu() * (iteration-1),
        Y=torch.Tensor([loc_arm, conf_arm, loc_odm, conf_odm, loss]).unsqueeze(0).cpu() / epoch_size,
        win=window,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 5)).cpu(),
            Y=torch.Tensor([loc_arm, conf_arm, loc_odm, conf_odm, loss]).unsqueeze(0).cpu(),
            win=window,
            update=True
        )


if __name__ == '__main__':
    # stuctured_experiment()
    train()
