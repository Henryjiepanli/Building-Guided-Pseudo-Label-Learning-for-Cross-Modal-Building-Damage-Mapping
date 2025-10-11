import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
import numpy as np
from datetime import datetime
import argparse
import logging
from network.UABCD_v2 import UABCD_v2
from tqdm import tqdm
import warnings
from utils import dataloader_v3
from utils import dataloader_v2
from utils.metrics import Evaluator
from utils.func import adjust_lr, AvgMeter, print_network, poly_lr
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------------------------------------------------

   
def entropy(prob):
    return -torch.sum(prob * torch.log(prob + 1e-8), dim=1, keepdim=True) 

def Train(train_loader, CD_Model, CD_Model_optimizer, epoch, Eva, alpha=0.5, confidence_threshold=0.2):
    CD_Model.train()
    print('UACD Learning Rate: {}'.format(CD_Model_optimizer.param_groups[0]['lr']))

    for i, sample in enumerate(tqdm(train_loader), start=1):
        CD_Model_optimizer.zero_grad()

        A, B, probs, buidling_gts = sample['A'], sample['B'], sample['prob'], sample['building'] 
        A, B, probs, buidling_gts = Variable(A).cuda(), Variable(B).cuda(), Variable(probs).cuda(), Variable(buidling_gts).cuda()

        p = CD_Model(A, B, buidling_gts.unsqueeze(1)) 
        prob = F.softmax(p, dim=1) 
        pred = prob.argmax(dim=1) 

        entropy_map = entropy(prob).squeeze(1) 
        high_conf_mask = (entropy_map < confidence_threshold).float() 
        low_conf_mask = (entropy_map >= confidence_threshold).float() 

        hard_labels = probs.argmax(dim=1) 
        ce_loss = F.cross_entropy(p, hard_labels, reduction='none')  
        ce_loss = (ce_loss * high_conf_mask).mean() 

        kl_loss = F.kl_div(F.log_softmax(p, dim=1), probs, reduction='none')  
        kl_loss = kl_loss.sum(dim=1) 
        kl_loss = (kl_loss * low_conf_mask).mean() 

        loss = ce_loss + alpha * kl_loss
        loss.backward()
        CD_Model_optimizer.step()


        output = pred.data.cpu().numpy() 
        target = hard_labels.cpu().numpy() 
        Eva.add_batch(target, output.astype(np.int64))  

    IoU = Eva.Intersection_over_Union()
    print('Epoch [{:03d}/{:03d}], \n[Training] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}, Class 2 IoU: {:.4f}, Class 3 IoU: {:.4f}'.format(
        epoch, opt.epoch, IoU[0], IoU[1], IoU[2], IoU[3]))

    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], \n[Training] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}, Class 2 IoU: {:.4f}, Class 3 IoU: {:.4f}'.format(
        epoch, opt.epoch, IoU[0], IoU[1], IoU[2], IoU[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr_uacd', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--load', type=str, help='pretrained_path')
    parser.add_argument('--segclass', type=int, default=4,
                        help='')
    parser.add_argument('--save_path', type=str,
                            default='./Experiment/UABCD_v2/')
    parser.add_argument('--train_root', type=str,
                            default='None')
    opt = parser.parse_args()
    save_path = opt.save_path + '/' + opt.data_name + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('UACD Learning Rate: {}'.format(opt.lr_uacd))

    # build models
    CD_Model = UABCD_v2(num_classes=opt.segclass)
    CD_Model.cuda()
    CD_Model_params = CD_Model.parameters()
    CD_Model_optimizer = torch.optim.Adam(CD_Model_params, opt.lr_uacd)

    if opt.load:
        CD_Model.load_state_dict(torch.load(opt.load), strict=False)
        logging.info(f'Loaded model from {opt.load}')


    train_loader = dataloader_v2.get_loader(img_A_root = opt.train_root + 'pre-event/', img_B_root = opt.train_root + 'post-event/', prob_root = opt.train_root + 'prob/', building_root = opt.train_root + 'building_pred/', trainsize = opt.trainsize, batchsize = opt.batchsize, num_workers=4, shuffle=True, pin_memory=True)
    total_step = len(train_loader)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("UACD-Train")
    logging.info("Config")
    logging.info('epoch:{}; lr_uabcd:{}; batchsize:{}; trainsize:{}; save_path:{}'.
                format(opt.epoch, opt.lr_uacd, opt.batchsize, opt.trainsize, save_path))

    logging.shutdown()
    sys.stdout.flush()

    # loss function
    CE_loss = nn.CrossEntropyLoss().cuda()
    print("Let's go!")
    best_iou = 0
    best_epoch = 0
    Eva_tr = Evaluator(opt.segclass)
    for epoch in range(1, (opt.epoch+1)):
        Eva_tr.reset()
        uabcd_lr = adjust_lr(CD_Model_optimizer, opt.lr_uacd, epoch, 0.1, opt.decay_epoch)
        Train(train_loader, CD_Model, CD_Model_optimizer, epoch, Eva_tr)
        if epoch % 10 == 0:
            torch.save(CD_Model.state_dict(), save_path + 'Seg_epoch_best.pth')



