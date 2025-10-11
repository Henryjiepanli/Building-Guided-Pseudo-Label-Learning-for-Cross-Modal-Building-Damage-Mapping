import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from utils import custom_transforms as tr

def randomCrop_Mosaic(image_A, image_B, label, crop_win_width, crop_win_height):
    image_width = image_A.size[0]
    image_height = image_B.size[1]
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image_A.crop(random_region), image_B.crop(random_region), label.crop(random_region)


def mask_to_onehot(mask, palette):

    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1) #单通道有索引使用
        # class_map = equality.astype(int)  #单通道无索引使用
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = np.argmax(semantic_map,axis=-1)
    return semantic_map


class Multi_Class_Segmentation_Dataset(data.Dataset):
    def __init__(self, img_root, gt_root, trainsize,  mode):
        self.trainsize = trainsize
        self.image_root = img_root
        self.gt_root = gt_root
        self.mode = mode
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.png') or f.endswith('.tif')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.tif') \
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

    def __getitem__(self, index):
        image, mask = self.load_img_and_mask(index)


        sample = {'Image': image, 'label': mask}

        if self.mode == 'train':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        elif self.mode == 'test':
            file_name = self.images[index].split('/')[-1][:-len(".tif")]
            return self.transform_test(sample),file_name
        
    def load_img_and_mask(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = np.array(Image.open(self.gts[index]).convert('L'))
        # 二值化处理
        mask[mask > 0] = 1
        mask = Image.fromarray(mask.astype(np.uint8))
        return image, mask

    
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.FixedResize(self.trainsize),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(self.trainsize),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(self.trainsize),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def filter_files(self):
        assert len(self.images) == len(self.gts)
       
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
           
            gt = Image.open(gt_path)
            if img.size == img.size:
                
                images.append(img_path)
                gts.append(gt_path)

        self.images = images
        self.gts = gts


    def __len__(self):
        return self.size

def get_loader(img_root, gt_root, trainsize, mode, batchsize, num_workers=4, shuffle=True, pin_memory=True):

    dataset = Multi_Class_Segmentation_Dataset(img_root = img_root, gt_root = gt_root, trainsize = trainsize, mode = mode)
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



