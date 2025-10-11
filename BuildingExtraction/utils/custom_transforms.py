import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

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
       
        mask = sample['label']
        img = np.array(img).astype(np.float32)
      
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std


        return {'Image': img,  'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['Image']
       
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
       
        mask = torch.from_numpy(mask).float()

        return {'Image': img,  'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['Image']
       
        mask = sample['label']
       
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'Image': img,  'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['Image']
       
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'Image': img,  'label': mask}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['Image']
       
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'Image': img,  'label': mask}




class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['Image']
       
        mask = sample['label']

        assert img.size == mask.size
    
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'Image': img,  'label': mask}