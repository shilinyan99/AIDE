# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch
from .dct import DCT_base_Rec_Module
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        if'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root

            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):
                
                    assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                 
                    for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
            
            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        else:

            for filename in os.listdir(root):

                file_path = os.path.join(root, filename)

                if '0_real' not in os.listdir(file_path):
                    for folder_name in os.listdir(file_path):
                    
                        assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                        for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                    
                        for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
                
                else:
                    for image_path in os.listdir(os.path.join(file_path, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                    for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
                
        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


        image = transform_before(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        


        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        file_path = root

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
    
                assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
        
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})


        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')

        image = transform_before_test(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)


        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        
        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

