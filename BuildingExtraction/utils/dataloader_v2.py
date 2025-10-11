import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['Image']
        mask = sample['label']
        
        img = np.array(img).astype(np.float32) / 255.0
        img -= self.mean
        img /= self.std

        return {'Image': img, 'label': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        img = sample['Image'].transpose((2, 0, 1))  # HWC -> CHW
        mask = sample['label']

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'Image': img, 'label': mask}

class Prob_Segmentation_Dataset(data.Dataset):
    def __init__(self, img_root, gt_root, trainsize, mode):
        self.trainsize = trainsize
        self.image_root = img_root
        self.gt_root = gt_root
        self.mode = mode

        self.images = sorted([os.path.join(img_root, f) for f in os.listdir(img_root) if f.endswith(('.png', '.tif'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.npy')])

        self.filter_files()
        self.size = len(self.images)

    def __getitem__(self, index):
        image, mask = self.load_img_and_mask(index)
        sample = {'Image': image, 'label': mask}

        if self.mode == 'train':
            return self.transform_tr(sample)
        
    def load_img_and_mask(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = np.load(self.gts[index]).astype(np.float32)  # 直接加载 npy
        return image, mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensor()
        ])
        return composed_transforms(sample)

    def filter_files(self):
        assert len(self.images) == len(self.gts), "Image and label count mismatch!"

        valid_images = []
        valid_gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            try:
                img = Image.open(img_path)
                mask = np.load(gt_path)
                if img.size == mask.shape:
                    valid_images.append(img_path)
                    valid_gts.append(gt_path)
            except Exception as e:
                print(f"Skipping {img_path} or {gt_path} due to error: {e}")

        self.images = valid_images
        self.gts = valid_gts

    def __len__(self):
        return self.size

def get_prob_loader(img_root, gt_root, trainsize, mode, batchsize, num_workers=4, shuffle=True, pin_memory=True):
    dataset = Prob_Segmentation_Dataset(img_root, gt_root, trainsize, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader
