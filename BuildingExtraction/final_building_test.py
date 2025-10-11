import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import testloader
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from network.UFPN import UFPN
import ttach as tta

def landcover_to_colormap(pred):
    """
    Convert prediction array to a color map for visualization.

    Parameters:
    pred (numpy array): 2D array of class predictions.

    Returns:
    numpy array: 3D array representing the color map.
    """
    # Define the colormap as a dictionary of class index to RGB values
    class_to_color = {
        0: [0,0, 0],   # Background (#000000)
        1: [255,255,255]}

    # Create an empty color map
    colormap = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    # Assign RGB colors to the colormap based on the prediction classes
    for class_index, color in class_to_color.items():
        colormap[pred == class_index] = color

    return colormap
def onehot_to_mask(semantic_map, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    #x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    semantic_map = np.uint8(colour_codes[semantic_map])
    return semantic_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--segclass', type=int, default=2,
                        help='')
    parser.add_argument('--save_path', type=str, default=None,
                        help='')
    parser.add_argument('--test_root', type=str, default=None,
                        help='')
    parser.add_argument('--model_path_1', type=str, default=None,
                        help='')
    parser.add_argument('--model_path_2', type=str, default=None,
                        help='')
    opt = parser.parse_args()

    test_root = opt.test_root
    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)
    nets = []
    with torch.no_grad():
        net_1 = UFPN(backbone_name='pvt_v2_b2', num_classes=opt.segclass).cuda()
        net_2 = UFPN(backbone_name='pvt_v2_b3', num_classes=opt.segclass).cuda()
        net_1.load_state_dict(torch.load(opt.model_path_1))
        net_2.load_state_dict(torch.load(opt.model_path_2))
        nets.append(net_1)
        nets.append(net_2)
        test_load = testloader.get_loader(img_root = test_root + 'pre-event/',\
                                        trainsize = opt.trainsize,\
                                        batchsize = opt.batchsize, num_workers=4, shuffle=False, pin_memory=True)
    
        print("Start Testing!")
        test(test_load, nets, save_path)
def test(test_load, nets, save_path):
    pred_save_path = save_path+'pred/'
    os.makedirs(pred_save_path,exist_ok=True)
    vis_save_path = save_path+'vis/'
    os.makedirs(vis_save_path,exist_ok=True)
    for net in nets:
        net.train(False)
        net.eval()
    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),    
    ])
    for i, (sample, filename) in enumerate(tqdm(test_load)):
        
        img = sample['Image']
        img = img.cuda()

        outs = []

        for net in nets:
            for transformer in transforms: 
                augmented_image = transformer.augment_image(img)
                out = F.softmax(transformer.deaugment_mask(net(augmented_image)), dim=1)
                outs.append(out)
        output = torch.mean(torch.stack(outs), dim=0).argmax(dim=1).cpu().numpy().squeeze()
        final_pred_savepath = pred_save_path + '/' + filename[0] + '_building_damage.png'
        final_vis_savepath = vis_save_path + '/' + filename[0] + '_building_damage.png'
        im = Image.fromarray((output).astype(np.uint8))
        im.save(final_pred_savepath)
        vis_im = Image.fromarray(landcover_to_colormap(output).astype(np.uint8))
        vis_im.save(final_vis_savepath)




if __name__ == '__main__':
    main()