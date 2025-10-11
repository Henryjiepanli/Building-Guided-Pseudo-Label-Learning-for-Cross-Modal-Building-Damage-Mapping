import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from utils import testloader
from tqdm import tqdm
import argparse
from network.UFPN import UFPN
import ttach as tta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--segclass', type=int, default=2, help='')
    parser.add_argument('--model_path_1', type=str, default='None', help='model path')
    parser.add_argument('--model_path_2', type=str, default='None', help='retrained model path')
    parser.add_argument('--test_root', type=str, default='None', help='retrained model path')
    parser.add_argument('--save_path', type=str, default='None', help='result save path')
    opt = parser.parse_args()
    test_root = opt.test_root
    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        nets = []
        backbones = ['pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4']
        
        for backbone in backbones:
            net = UFPN(backbone_name=backbone, num_classes=opt.segclass).cuda()
            model_path = os.path.join(opt.model_path_1, backbone, 'Seg_epoch_best.pth')
            net.load_state_dict(torch.load(model_path))
            net.eval()
            nets.append(net)

            net_retrain = UFPN(backbone_name=backbone, num_classes=opt.segclass).cuda()
            retrain_model_path = os.path.join(opt.model_path_2, backbone, 'Seg_epoch_best.pth')
            net_retrain.load_state_dict(torch.load(retrain_model_path))
            net_retrain.eval()
            nets.append(net_retrain)

        test_load = testloader.get_loader(img_root=test_root + 'pre-event/',
                                          trainsize=opt.trainsize,
                                          batchsize=opt.batchsize, num_workers=4, shuffle=False, pin_memory=True)

        print("Start Testing!")
        test(test_load, nets, save_path)

def test(test_load, nets, save_path):
    pred_save_path = os.path.join(save_path, 'pred/')
    os.makedirs(pred_save_path, exist_ok=True)
    
    vis_save_path = os.path.join(save_path, 'vis/')
    os.makedirs(vis_save_path, exist_ok=True)
    
    prob_save_path = os.path.join(save_path, 'prob/') 
    os.makedirs(prob_save_path, exist_ok=True)

    transforms = tta.Compose([tta.HorizontalFlip()])

    for i, (sample, filename) in enumerate(tqdm(test_load)):
        img = sample['Image'].cuda()

        outs = []

        for net in nets:
            for transformer in transforms:
                augmented_image = transformer.augment_image(img)
                out = F.softmax(transformer.deaugment_mask(net(augmented_image)), dim=1)
                outs.append(out)

        out = torch.mean(torch.stack(outs), dim=0)  
        output = out.argmax(dim=1).cpu().numpy().squeeze() 
        building_prob = out[:, 1, :, :].cpu().numpy().squeeze()  

        final_pred_savepath = os.path.join(pred_save_path, f'{filename[0]}_building_damage.png')
        final_vis_savepath = os.path.join(vis_save_path, f'{filename[0]}_building_damage.png')
        im = Image.fromarray(output.astype(np.uint8))
        im.save(final_pred_savepath)

        vis_im = Image.fromarray((output * 255).astype(np.uint8)) 
        vis_im.save(final_vis_savepath)

        prob_save_filepath = os.path.join(prob_save_path, f'{filename[0]}_building_damage.npy')
        np.save(prob_save_filepath, building_prob)

if __name__ == '__main__':
    main()
