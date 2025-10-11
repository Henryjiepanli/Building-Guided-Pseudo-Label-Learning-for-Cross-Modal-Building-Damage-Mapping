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
        img = sample['Image']
       
        img = np.array(img).astype(np.float32)
      
        img /= 255.0
        img -= self.mean
        img /= self.std


        return {'Image': img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['Image']
       
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        
        img = torch.from_numpy(img).float()
       

        return {'Image': img}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['Image']
       
    
        img = img.resize(self.size, Image.BILINEAR)

        return {'Image': img}


class Multi_Class_Segmentation_Dataset(data.Dataset):
    def __init__(self, img_root, trainsize):
        self.trainsize = trainsize
        self.image_root = img_root
     
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.png') or f.endswith('.tif')]
        
        self.images = sorted(self.images)
        self.filter_files()
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.load_img_and_mask(index)

        sample = {'Image': image}
        file_name = self.images[index].split('/')[-1][:-len("_pre_disaster.tif")]
        return self.transform_test(sample),file_name

    def load_img_and_mask(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        return image
        

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            FixedResize(self.trainsize),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensor()])

        return composed_transforms(sample)


    def filter_files(self):
       
        images = []
        for img_path in self.images:
            img = Image.open(img_path)
            images.append(img_path)

        self.images = images

    def __len__(self):
        return self.size

def get_loader(img_root,  trainsize,  batchsize, num_workers=4, shuffle=True, pin_memory=True):

    dataset = Multi_Class_Segmentation_Dataset(img_root = img_root, trainsize = trainsize)
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



