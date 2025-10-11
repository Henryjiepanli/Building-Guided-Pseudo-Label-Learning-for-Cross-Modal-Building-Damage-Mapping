import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_A = sample['A']
        img_B = sample['B']
        
        img_A = np.array(img_A).astype(np.float32)
        img_B = np.array(img_B).astype(np.float32)
        
        img_A /= 255.0
        img_A -= self.mean
        img_A /= self.std

        img_B /= 255.0
        img_B -= self.mean
        img_B /= self.std

        return {'A': img_A, 'B': img_B}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_A = sample['A']
        img_B = sample['B']
        
        img_A = np.array(img_A).astype(np.float32).transpose((2, 0, 1))
        img_B = np.array(img_B).astype(np.float32).transpose((2, 0, 1))
        
        img_A = torch.from_numpy(img_A).float()
        img_B = torch.from_numpy(img_B).float()
        

        return {'A': img_A, 'B': img_B}


class Multi_Class_Segmentation_Dataset(data.Dataset):
    def __init__(self, img_A_root, img_B_root, trainsize):
        self.trainsize = trainsize
        self.image_A_root = img_A_root
        self.image_B_root = img_B_root
        
        self.images_A = [self.image_A_root + f for f in os.listdir(self.image_A_root) if f.endswith('.png') or f.endswith('.tif')]
        self.images_B = [self.image_B_root + f for f in os.listdir(self.image_B_root) if f.endswith('.png') or f.endswith('.tif')]
        
        self.images_A = sorted(self.images_A)
        self.images_B = sorted(self.images_B)
    
        self.filter_files()
        self.size = len(self.images_A)

    def __getitem__(self, index):
        
        image_A, image_B = self.load_img_and_mask(index)

        sample = {'A': image_A, 'B': image_B}

        file_name = self.images_A[index].split('/')[-1][:-len("_pre_disaster.tif")]
        return self.transform_test(sample),file_name
        
    def load_img_and_mask(self, index):
        image_A = Image.open(self.images_A[index]).convert('RGB')
        image_B = Image.open(self.images_B[index]).convert('RGB')
       
        return image_A, image_B

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensor()])

        return composed_transforms(sample)


    def filter_files(self):
        assert len(self.images_A) == len(self.images_B)
        images_A = []
        images_B = []
        for img_A_path, img_B_path in zip(self.images_A, self.images_B):
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            if img_A.size == img_B.size:
                images_A.append(img_A_path)
                images_B.append(img_B_path)

        self.images_A = images_A
        self.images_B = images_B


    def __len__(self):
        return self.size

def get_loader(img_A_root, img_B_root, trainsize, batchsize,  num_workers=4, shuffle=True, pin_memory=True):

    dataset = Multi_Class_Segmentation_Dataset(img_A_root = img_A_root, img_B_root = img_B_root, trainsize = trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



# if __name__ == '__main__':
#     root = '/home/lijiepan/multi_class_semantic_segementation/drive_dataset/train/'
#     batchsize = 8
#     trainsize = 512
#     palette = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
#     data = get_loader(root, batchsize, trainsize, palette, num_workers=4, shuffle=True, pin_memory=True)
#     for x,y in data:
#         print(x)
#         print(y)



