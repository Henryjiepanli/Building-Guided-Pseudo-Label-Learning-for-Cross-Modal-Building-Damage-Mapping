import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from datetime import datetime
from utils import dataloader
from utils.metrics import Evaluator
from utils.tools import adjust_lr, AvgMeter, print_network, poly_lr
import argparse
import logging
from network.UFPN import UFPN
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------------------------------------------------

def Train(train_loader, CD_Model, CD_Model_optimizer, epoch, Eva):
    CD_Model.train()

    print('UACD Learning Rate: {}'.format(CD_Model_optimizer.param_groups[0]['lr']))
   
    for i, sample in enumerate(tqdm(train_loader), start=1):
        CD_Model_optimizer.zero_grad()
       
        image, mask = sample['Image'], sample['label']
        image = Variable(image)

        gts = Variable(mask)
        image = image.cuda()
        Y = gts.cuda()
        p = CD_Model(image)
        seg_loss = CE_loss(p, Y.long())
        seg_loss.backward()
        CD_Model_optimizer.step()
            
        output = p.argmax(dim=1).data.cpu().numpy().squeeze()
        target = Y.cpu().numpy()
        Eva.add_batch(target, output.astype(np.int64))
    IoU = Eva.Intersection_over_Union()
    print('Epoch [{:03d}/{:03d}], \n[Training] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}'.format(epoch, opt.epoch, IoU[0], IoU[1]))

    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], \n[Training] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}'.format(epoch, opt.epoch, IoU[0], IoU[1]))

def Val(test_loader, CD_Model, epoch, Eva, save_path):
    global best_iou, best_epoch
    CD_Model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            image, mask = sample['Image'], sample['label']
            image = Variable(image)

            gts = Variable(mask)
            image = image.cuda()
            Y = gts.cuda()
            res = CD_Model(image)
            output = res.argmax(dim=1).data.cpu().numpy().squeeze()
            target = Y.cpu().numpy()
            # Add batch sample into evaluator
            Eva.add_batch(target, output.astype(np.int64))
    IoU = Eva.Intersection_over_Union()
    mIou = IoU[1]

    print('Epoch [{:03d}/{:03d}], \n[Validing] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}'.format(epoch, opt.epoch, IoU[0], IoU[1]))
    logging.info('#Val#:Epoch [{:03d}/{:03d}], \n[Validing] Class 0 IoU: {:.4f}, Class 1 IoU: {:.4f}'.format(epoch, opt.epoch, IoU[0], IoU[1]))
    new_iou = mIou
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        print('Best Model Iou :%.4f; Best epoch : %d' % (mIou, best_epoch))

        torch.save(CD_Model.state_dict(), save_path + 'Seg_epoch_best.pth')

    logging.info('#TEST#:Epoch:{} Iou:{} bestEpoch:{} bestIou:{}'.format(epoch, mIou, best_epoch, best_iou))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=40, help='epoch number')
    parser.add_argument('--lr_uacd', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--load', type=str, help='prertained_path')
    parser.add_argument('--segclass', type=int, default=2,
                        help='')
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                        help='')
    parser.add_argument('--save_path', type=str,
                            default='./output/UFPN/')
    parser.add_argument('--train_root', type=str,
                            default='./train_test_set/train/')
    parser.add_argument('--val_root', type=str,
                            default='./train_test_set/test/')
    opt = parser.parse_args()

    save_path = opt.save_path + opt.backbone + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('UACD Learning Rate: {}'.format(opt.lr_uacd))

    # build models
    CD_Model = UFPN(backbone_name=opt.backbone, num_classes=opt.segclass)
    CD_Model.cuda()
    CD_Model_params = CD_Model.parameters()
    CD_Model_optimizer = torch.optim.Adam(CD_Model_params, opt.lr_uacd)

    if opt.load:
        CD_Model.load_state_dict(torch.load(opt.load), strict=False)
        logging.info(f'Loaded model from {opt.load}')


    train_loader = dataloader.get_loader(img_root = opt.train_root + 'images/', gt_root = opt.train_root + 'labels/', trainsize = opt.trainsize,  mode ='train', batchsize = opt.batchsize,  num_workers=4, shuffle=True, pin_memory=True)
    test_loader = dataloader.get_loader(img_root = opt.val_root + 'images/',  gt_root = opt.val_root + 'labels/', trainsize = opt.trainsize, mode ='val', batchsize = opt.batchsize, num_workers=4, shuffle=False, pin_memory=True)
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
    Eva_val = Evaluator(opt.segclass)
    for epoch in range(1, (opt.epoch+1)):
        Eva_tr.reset()
        Eva_val.reset()
        uabcd_lr = poly_lr(CD_Model_optimizer, opt.lr_uacd, epoch, opt.epoch)
        Train(train_loader, CD_Model, CD_Model_optimizer, epoch, Eva_tr)
        Val(test_loader, CD_Model, epoch, Eva_val, save_path)



