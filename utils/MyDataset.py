import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
#import PIL.Image as Image
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
import os
import io
# import imutils

class MyDataset(Dataset):
    def __init__(self, txt, trans=None, option=None, type=None, size=None, augment=None):
        self.root = txt
        assert(type!=None)
        assert(size!=None)
        self.option = option
        # if type == "DALLE":
        #     if option is not None:
        #         self.Neg_root = os.path.join("/data/xiziyi/DALLE/DA_"+ str(size) + "_JPG_postpro/DALLE/", option) + '/'
        #         self.Pos_root = os.path.join("/data/xiziyi/DALLE/DA_" + str(size) + "_JPG_postpro/ALASKA/", option) + '/'
        #     else:
        #         self.Neg_root = "/data/xiziyi/DALLE/DALLE_" + str(size) + "_JPG/"
        #         self.Pos_root = "/data/xiziyi/DALLE/ALASKA_" + str(size) + "_JPG/"
        # elif type == "DreamStudio":
        #     if option is not None:
        #         self.Neg_root = os.path.join("/data/xiziyi/ds/DS_"+ str(size) +"_JPG_postpro/DS/", option) + '/'
        #         self.Pos_root = os.path.join("/data/xiziyi/ds/DS_"+ str(size) + "_JPG_postpro/ALASKA/", option) + '/'
        #     else:
        #         self.Neg_root = "/data/xiziyi/ds/DS_" + str(size) + "_JPG/"
        #         self.Pos_root = "/data/xiziyi/DALLE/ALASKA_"+ str(size) + "_JPG/"
        # elif (type == "SPL2018" or type == 'DsTok') and option is not None:
        #     self.Neg_root = "/data/xiziyi/" + type + '/' + "split0/224_JPG_postpro/" + option + '/' + 'CG/'
        #     self.Pos_root = "/data/xiziyi/" + type + '/' + "split0/224_JPG_postpro/" + option + '/' + 'PG/'
        # elif (type == "LSCGB") and option is not None:
        #     self.Neg_root = "/data/hjk/" + type + '/' + "split0/224_JPG_postpro/" + option + '/'
        #     self.Pos_root = "/data/hjk/" + type + '/' + "split0/224_JPG_postpro/" + option + '/'
            #print(self.Neg_root)
            #print(self.Pos_root)
        self.type = type
        self.image_list = self.get_dataset_info()
        self.augment = augment
        random.shuffle(self.image_list)
        self.tensor = transforms.ToTensor()
        if transforms:
            self.transforms = trans

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_path, label = self.image_list[item]
        image = Image.open(img_path).convert('RGB')
        if (self.type == 'SPL2018' or self.type == 'DsTok' or self.type == 'LSCGB'):
            image = self.post_process(image, self.option)
        
        image = self.transforms(image)
        label = int(label)
        #torch.as_tensor(image)
        label = torch.as_tensor(label)
        return image, label

    def get_dataset_info(self):
        image_list = []
        with open(self.root,'r') as file:
            lines = file.read().splitlines()
        # if (self.type == 'SPL2018' or self.type == 'DsTok' or self.type == 'LSCGB') and (self.option is None):
        if (self.type == 'SPL2018' or self.type == 'DsTok' or self.type == 'LSCGB'):
            for line in lines:
                image_path = line[:-2]
                label = line[-1]
                image_list.append([image_path, label])
        else:
            #if self.type == 'DALLE' or self.type == 'DreamStudio':
            for line in lines:
                name = os.path.basename(line[:-2])
                #print(name)
                #if '.JPG' in name:
                #    name = name.replace('.JPG', '.jpg')
                # image_path.replace("openai_90","openai_75to85")
                label = line[-1]
                # 0: dalle
                if label == '0':
                    image_path = self.Neg_root + name
                # 1: alaska
                else:
                    image_path = self.Pos_root + name
                image_list.append([image_path, label])
        return image_list
    
    def post_process(self, image: Image, option):
        if option == None:
            pass
        elif option == 'jpeg95':
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=95)
            image = Image.open(stream)
        elif option == 'jpeg75':
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=75)
            image = Image.open(stream)
        elif option == 'scale300':
            image = transforms.Resize(300)(image)
        elif option == 'scale1000':
            image = transforms.Resize(1000)(image)
        elif option == 'color':
            num = round(random.uniform(0.5, 2.5), 2)
            image = ImageEnhance.Color(image).enhance(num)
        elif option == 'brightness':
            num = round(random.uniform(0.5, 2.5), 2)
            image = ImageEnhance.Brightness(image).enhance(num)
        elif option == 'contrast':
            num = round(random.uniform(0.5, 2.5), 2)
            image = ImageEnhance.Contrast(image).enhance(num)
        elif option == 'sharpness':
            num = round(random.uniform(0.5, 2.5), 2)
            image = ImageEnhance.Sharpness(image).enhance(num)
        elif option == 'median3x3':
            np_img = np.array(image)
            np_img = cv2.medianBlur(np_img, 3)
            image = Image.fromarray(np_img)
        elif option == 'median5x5':
            np_img = np.array(image)
            np_img = cv2.medianBlur(np_img, 5)
            image = Image.fromarray(np_img)
        elif option == 'mean3x3':
            np_img = np.array(image)
            np_img = cv2.blur(np_img, (3, 3))
            image = Image.fromarray(np_img)
        elif option == 'mean5x5':
            np_img = np.array(image)
            np_img = cv2.blur(np_img, (5, 5))
            image = Image.fromarray(np_img)
        elif option == 'gb3x3':
            np_img = np.array(image)
            np_img = cv2.GaussianBlur(np_img, (3, 3), 0)
            image = Image.fromarray(np_img)
        elif option == 'gb5x5':
            np_img = np.array(image)
            np_img = cv2.GaussianBlur(np_img, (5, 5), 0)
            image = Image.fromarray(np_img)
        elif option == 'noise1':
            pass
        elif option == 'noise1.5':
            pass
        else:
            return

        return image