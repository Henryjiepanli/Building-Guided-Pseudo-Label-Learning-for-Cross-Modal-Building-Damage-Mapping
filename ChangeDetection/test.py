import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import testloader
import numpy as np
from PIL import Image
from utils.metrics import Evaluator
from tqdm import tqdm
import argparse
from network.UABCD import UABCD
import ttach as tta

# Define colormap for visualization
def landcover_to_colormap(pred):
    """
    Convert prediction array to a color map for visualization.

    Parameters:
    pred (numpy array): 2D array of class predictions.

    Returns:
    numpy array: 3D array representing the color map.
    """
    class_to_color = {
        0: [255, 255, 255],   # Background
        1: [108, 178, 125],    # Intact
        2: [219, 190, 144],    # Damaged
        3: [163, 78, 73],      # Destroyed
    }
    colormap = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_index, color in class_to_color.items():
        colormap[pred == class_index] = color
    return colormap

def onehot_to_mask(semantic_map, palette):
    """
    Converts a one-hot encoded mask to RGB format using a provided palette.
    """
    colour_codes = np.array(palette)
    return np.uint8(colour_codes[semantic_map])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='image size for training')
    parser.add_argument('--segclass', type=int, default=4, help='number of segmentation classes')
    parser.add_argument('--model_path', type=str, default=None, help='directory for model output')
    parser.add_argument('--save_path', type=str, default=None, help='directory for save path')
    parser.add_argument('--test_root', type=str, default=None, help='directory for test root')
    parser.add_argument('--epoch', type=int, nargs='+', default=[1], help='epochs to evaluate')
    opt = parser.parse_args()

    epoch = '_'.join(str(e) for e in opt.epoch)
    save_path = os.path.join(opt.save_path, epoch)
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        Eva = Evaluator(opt.segclass)
        nets = []
        
        for e in opt.epoch:
            model_path = os.path.join(opt.model_path, f'Seg_epoch_{e}_Seg.pth')
            net = UABCD(num_classes=opt.segclass).cuda()
            net.load_state_dict(torch.load(model_path))
            net.eval()
            nets.append(net)
        
        test_load = testloader.get_loader(
            img_A_root=os.path.join(opt.test_root, 'pre-event/'),
            img_B_root=os.path.join(opt.test_root, 'post-event/'),
            trainsize=opt.trainsize,
            batchsize=opt.batchsize, 
            num_workers=4, 
            shuffle=False, 
            pin_memory=True
        )

        print("Start Testing!")
        test(test_load, nets, Eva, save_path)

def test(test_load, nets, Eva, save_path):
    pred_save_path = os.path.join(save_path, 'pred')
    vis_save_path = os.path.join(save_path, 'vis')
    prob_save_path = os.path.join(save_path, 'prob')
    
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(vis_save_path, exist_ok=True)
    os.makedirs(prob_save_path, exist_ok=True)

    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
    ]
)
    
    for net in nets:
        net.eval()

    for i, (sample, filename) in enumerate(tqdm(test_load)):
        A, B = sample['A'].cuda(), sample['B'].cuda()
        outs = []

        # Collect predictions from each network
        for net in nets:
            for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform()   
                # augment image
                augmented_A = transformer.augment_image(A)
                augmented_B = transformer.augment_image(B)
                preds = net(augmented_A, augmented_B)
                if isinstance(preds, tuple):
                    out = F.softmax(transformer.deaugment_mask(preds[-1]), dim=1)
                else:
                    out = F.softmax(transformer.deaugment_mask(preds), dim=1)
                outs.append(out)

        # Combine predictions by averaging
        cd_prob = torch.mean(torch.stack(outs), dim=0).cpu().numpy().squeeze()  # 提取建筑物类别的概率（类别1）
        output = torch.mean(torch.stack(outs), dim=0).argmax(dim=1).cpu().numpy().squeeze()

        # Save predicted mask
        final_pred_savepath = os.path.join(pred_save_path, f'{filename[0]}_building_damage.png')
        im = Image.fromarray(output.astype(np.uint8))
        im.save(final_pred_savepath)

        # Save visualized image
        vis_im = Image.fromarray(landcover_to_colormap(output).astype(np.uint8))
        final_vis_savepath = os.path.join(vis_save_path, f'{filename[0]}_building_damage.png')
        vis_im.save(final_vis_savepath)

        prob_save_filepath = os.path.join(prob_save_path, f'{filename[0]}_building_damage.npy')
        np.save(prob_save_filepath, cd_prob)



if __name__ == '__main__':
    main()