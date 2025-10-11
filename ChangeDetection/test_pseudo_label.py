import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import testloader_v2
import numpy as np
from PIL import Image
from utils.metrics import Evaluator
from tqdm import tqdm
import argparse
import ttach as tta
from network.UABCD_v2 import UABCD_v2

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
        0: [255, 255, 255],   # Background (#000000)
        1: [108,178,125],       # Intact (#6CB27D)
        2: [219,190,144],    # Damaged (#DBBE90)
        3: [163,78,73],   # Destroyed (#A34E49)
    }

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
    parser.add_argument('--segclass', type=int, default=4,
                        help='')
    parser.add_argument('--model_path', type=str, default=None,
                        help='')
    parser.add_argument('--test_root', type=str, default=None,
                        help='')
    parser.add_argument('--building_root', type=str, default=None,
                        help='')
    parser.add_argument('--save_path', type=str, default=None,
                        help='')
    opt = parser.parse_args()

    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        net = UABCD_v2(num_classes=opt.segclass).cuda()
        model_path = opt.model_path

        net.load_state_dict(torch.load(model_path))
        test_load = testloader_v2.get_loader(img_A_root = opt.test_root + 'pre-event/', img_B_root = opt.test_root + 'post-event/',\
                                        building_root = opt.building_root, trainsize = opt.trainsize,\
                                        batchsize = opt.batchsize, num_workers=4, shuffle=False, pin_memory=True)
    
        print("Start Testing!")
        test(test_load, net, save_path)
def test(test_load, net, save_path):
    pred_save_path = save_path+'pred/'
    os.makedirs(pred_save_path,exist_ok=True)
    vis_save_path = save_path+'vis/'
    os.makedirs(vis_save_path,exist_ok=True)
    net.train(False)
    net.eval()
    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
    ])
    for i, (sample, filename) in enumerate(tqdm(test_load)):
        A, B, building = sample['A'], sample['B'], sample['building']
        A = A.cuda()
        B = B.cuda()
        building = building.cuda()

        outs = []

        for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform()   
                # augment image
            augmented_A = transformer.augment_image(A)
            augmented_B = transformer.augment_image(B)
            augmented_building = transformer.augment_image(building.unsqueeze(1))
            preds = net(augmented_A, augmented_B, augmented_building)
            out = F.softmax(transformer.deaugment_mask(preds), dim=1)
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