import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from datetime import datetime
import argparse
import logging
from network.UFPN_p import UFPN
from tqdm import tqdm
from utils import dataloader
from utils import dataloader_v2
from utils.utils import *
from utils.metrics import Evaluator
from utils.tools import adjust_lr
import warnings
warnings.filterwarnings('ignore')

def entropy(prob_map):
    return -torch.sum(prob_map * torch.log(prob_map + 1e-8), dim=1, keepdim=True)

# ----------------------------------------------------------------------------------------------------------------------

def Train(train_loader, CD_Model, CD_Model_optimizer, epoch, Eva, confidence_threshold=0.3, alpha=0.5):
    CD_Model.train()
    print('UACD Learning Rate: {}'.format(CD_Model_optimizer.param_groups[0]['lr']))

    for i, sample in enumerate(tqdm(train_loader), start=1):
        CD_Model_optimizer.zero_grad()

        image, gts = sample['Image'], sample['label'] 
        image, gts = image.cuda(), gts.cuda()

        gts = gts.unsqueeze(1)

        hard_labels = (gts > 0.5).long().squeeze(1) 

        gts_prob = torch.cat([(1 - gts), gts], dim=1)  

        p = CD_Model(image)  
        prob = F.softmax(p, dim=1) 
        pred = prob.argmax(dim=1) 

        entropy_map = entropy(prob).squeeze(1)  
        high_conf_mask = (entropy_map < confidence_threshold).float()
        low_conf_mask = (entropy_map >= confidence_threshold).float()

        ce_loss = F.cross_entropy(p, hard_labels, reduction='none')  
        ce_loss = (ce_loss * high_conf_mask).mean() 

        kl_loss = F.kl_div(F.log_softmax(p, dim=1), gts_prob, reduction='none')  
        kl_loss = kl_loss.sum(dim=1) 
        kl_loss = (kl_loss * low_conf_mask).mean() 

        loss = ce_loss + alpha * kl_loss
        loss.backward()
        CD_Model_optimizer.step()

        output = pred.data.cpu().numpy().squeeze()
        target = hard_labels.cpu().numpy()
        Eva.add_batch(target, output.astype(np.int64))

    IoU = Eva.Intersection_over_Union()
    print('Epoch [{:03d}/{:03d}], \n[Training] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}'.format(epoch, opt.epoch, IoU[0], IoU[1]))
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], \n[Training] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}'.format(epoch, opt.epoch, IoU[0], IoU[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr_uacd', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--load', type=str, help='pretrained_path')
    parser.add_argument('--segclass', type=int, default=2,
                        help='')
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                        help='')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='')
    parser.add_argument('--save_path', type=str,
                            default='./Pseudo_output/UFPN/')
    parser.add_argument('--img_root', type=str,
                            default=None)
    parser.add_argument('--gt_root', type=str,
                            default=None)
    
    opt = parser.parse_args()

    save_path = opt.save_path + '/' + opt.backbone + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('UACD Learning Rate: {}'.format(opt.lr_uacd))

    CD_Model = UFPN(backbone_name=opt.backbone, num_classes=opt.segclass)
    CD_Model.cuda()
    CD_Model_params = CD_Model.parameters()
    CD_Model_optimizer = torch.optim.Adam(CD_Model_params, opt.lr_uacd)

    if opt.load:
        CD_Model.load_state_dict(torch.load(opt.load), strict=False)
        logging.info(f'Loaded model from {opt.load}')

    train_loader = dataloader_v2.get_prob_loader(img_root = opt.img_root, gt_root = opt.gt_root, trainsize = opt.trainsize,  mode ='train', batchsize = opt.batchsize,  num_workers=4, shuffle=True, pin_memory=True)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("UFPN-Train")
    logging.info("Config")
    logging.info('epoch:{}; lr_uabcd:{}; batchsize:{}; trainsize:{}; save_path:{}'.
                format(opt.epoch, opt.lr_uacd, opt.batchsize, opt.trainsize, save_path))

    # loss function
    CE_loss = torch.nn.CrossEntropyLoss().cuda()
    print("Let's go!")
    best_iou = 0
    best_epoch = 0
    Eva_tr = Evaluator(opt.segclass)
    for epoch in range(1, (opt.epoch+1)):
        Eva_tr.reset()
        uabcd_lr = adjust_lr(CD_Model_optimizer, opt.lr_uacd, epoch, 0.1, opt.decay_epoch)
        Train(train_loader, CD_Model, CD_Model_optimizer, epoch, Eva_tr, opt.threshold)
        if epoch % 10 == 0:
            torch.save(CD_Model.state_dict(), save_path + 'Seg_epoch_' + str(epoch) + '_Seg.pth')



