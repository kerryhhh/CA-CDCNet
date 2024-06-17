import math
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
#import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
import json
import time
import warnings
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from PIL import Image
from PIL import ImageFile

from models.models import Net
import argparse
import logging
from tqdm import tqdm
from tqdm.contrib import tenumerate
from torch.utils.tensorboard import SummaryWriter
from utils.utils import Logger
import pprint
from utils.loss import Loss_Function
import torchvision
from torchsummary import summary
from utils.MyDataset import *
import shutil


def parse_args():
    parse = argparse.ArgumentParser(description='test')
    # the path of inpainting images
   # parse.add_argument('-img_path', '--image_path', default= "/data/xiziyi/CG/SPL2018/raw/", type=str)
    # the path of masks
    #parse.add_argument('-msk_path', '--mask_path', default='/data/xiziyi/Inpainting/Mask_4w/', type=str)
    # the path of flist ./Flist/train.txt ./Flist/val.txt ./Flist/test.txt
    #parse.add_argument('--flist', default='./Flist/', type=str)
    # the path of save log and model params
    parse.add_argument('-c', '--checkpoints', default='/data/hjk/result/AIGID/Net/LSCGB_224/')
    parse.add_argument('--gpu', default='0,1', type=str, required=False)
    # the params of train
    parse.add_argument('-bs', '--batch_size', default=16, type=int, required=False)
    parse.add_argument('-e', '--epoch', default=120, type=int, required=False)
    # the learning rate
    parse.add_argument('--lr', default=2e-4, type=float, required=False)
    parse.add_argument('--model', default=Net, required=False)
    parse.add_argument('--type', default="LSCGB", required=False)
    parse.add_argument('--size', default=224, type=int, required=False)
    parse.add_argument('--split', default="split0", required=False)
    parse.add_argument('--decay', default=30, type=int, required=False)
    parse.add_argument('--decay_rate', default=0.1, type=float, required=False)
    #parse.add_argument('--size', default=128, type=int, required=False)
    # parse.add_argument('--optim',default='Adam',choices=['Adam','SGD'])
    args = parse.parse_args()
    return args


args = parse_args()
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.type == "DreamStudio":
    txt1 = './data/Flistds/' + args.split + '/'
elif args.type == 'DALLE':
    txt1 = './data/FlistDA/' + args.split + '/'
elif args.type == 'DsTok':
    txt1 = "./data/FlistDsTok/" + args.split + '/'
elif args.type == 'SPL2018':
    txt1 = "./data/FlistSPL2018/" + args.split + '/'
elif args.type == 'LSCGB':
    txt1 = "./data/LSCGB/" + args.split + '/'

class Rgb2Gray(object):
    def __init__(self):
        '''rgb_image = object.astype(np.float32)
        if len(rgb_image.shape)!= 3 or rgb_image.shape[2] != 3:
            raise ValueError("input image is not a rgb image!")'''

    def __call__(self, img):
        L_image = img.convert('L')
        return L_image
data_transform = {
    "train": transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(512),
            transforms.RandomCrop(args.size, pad_if_needed=True),
            # transforms.CenterCrop(args.size),
            transforms.ToTensor()
        ]),
    "test": transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(512),
            transforms.CenterCrop(args.size),
            transforms.ToTensor()
        ])
}

log_dir = os.path.join(args.checkpoints + 'tensorboard', 'train')
# train_writer = SummaryWriter(log_dir=log_dir)
logger_test = Logger('test', args.checkpoints + 'test_log.log')


# test
def test(name, option=None):
    # print(image_path + '/' +
    # logger_test.info(summary(net,))
    test_dataset = MyDataset(txt=txt1+'test.txt',
                             trans=data_transform['test'],
                             option=option,
                             type=args.type,
                             size=args.size)
    test_num = len(test_dataset)
    print(test_num)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              drop_last=False,
                                              num_workers=4)
    net = args.model()
    # net = models.resnet18(pretrained=False)
    # net = EfficientNet.from_pretrained('efficientnet-b4', 'models/backbone/efficientnet-b4-6ed6700e.pth', num_classes=2)
    # net = DualNet()
    # net = Generator()
    '''
    dicts = torch.load(name)['net']
    mynet_dict = net.state_dict()
    new_dict = {k: v for k, v in dicts.items() if k in mynet_dict.keys()}
    mynet_dict.update(new_dict)
    net.load_state_dict(mynet_dict)
    '''
    net.load_state_dict(torch.load(name)['net'])
    net = nn.DataParallel(net)
    #net.load_state_dict(torch.load(name['net']))
    #net = nn.DataParallel(net)
    net.cuda()
    print("model has been imported successfully!")

    net.eval()
    loss_function = nn.CrossEntropyLoss()
    test_loss = 0.0
    acc = 0.0
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    time_start = time.perf_counter()
    with torch.no_grad():
        for step, test_data in tenumerate(test_loader, start=0, dynamic_ncols=True):
            #print(test_data)
            test_images, test_labels = test_data
            test_images = test_images.cuda()
            test_labels = test_labels.cuda()
            outputs = net(test_images)
            b, _ = outputs.size()
            # bsx1xhxw
            result = torch.max(outputs, dim=1)[1]
            for i in range(b):
                if result[i] == test_labels[i] and test_labels[i] == 0:
                    TN += 1
                if result[i] == test_labels[i] and test_labels[i] == 1:
                    TP += 1
                if result[i] != test_labels[i]:
                    if result[i] == 1:
                        #target_pth = target1 + os.path.basename(img_path[i])
                        #shutil.copyfile(img_path[i], target_pth)
                        FP += 1
                    else:
                        #target_pth = target2 + os.path.basename(img_path[i])
                        #shutil.copyfile(img_path[i], target_pth)
                        FN += 1
            loss1 = loss_function(outputs, test_labels)
            test_loss += loss1.item()
        TPR = TP / (TP+FN)
        TNR = TN / (TN+FP)
        test_acc = (TP+TN) / test_num
        logger_test.info(
            f"[dataset {args.type}]: {args.size}x{args.size} "
            f"time_waste:{time.perf_counter() - time_start:.3f} "
            f"test: {option} |"
            f"TPR: {TPR:.4f} |"
            f"TNR: {TNR:.4f} |"
            f"loss: {(test_loss / step):.4f} |"
            f"acc: {test_acc:.4f} |"
        )


if __name__ == '__main__':
    #bacc = args.checkpoints + 'DsTok.pth'
    bacc = args.checkpoints + 'model_best_acc.pt'
    bloss = args.checkpoints + 'model_best_loss.pt'
    test(bacc)
    test(bloss)
    #test(bloss, "col")
    #options = ["brightness", "contrast", "sharp", "rotate", "gb5x5", "avg5x5"]
    #for op in options:
    #    test(bacc, op)
    #test(blacc, "rotate")
    #test(bacc, "brightness")
    #test(bacc, "gs3x3")
    #test(bacc, "gs5x5")
    #test(bacc, "median3x3")
    #test(bacc, "median5x5")
    #test(bacc, "avg3x3")
    #test(bacc, "avg5x5")
    # options = ["color", "brightness", "contrast", "sharp", "rotate", "gb5x5", "avg5x5"]
    # for op in options:
    #     test(bacc, op)
    #     test(bloss, op)
    # options = ['jpeg95', 'jpeg75', 'scale300', 'scale1000', 'color', 'brightness', 'contrast', 'sharpness',
    #            'median3x3', 'median5x5', 'mean3x3', 'mean5x5','gb3x3', 'gb5x5']
    # for op in options:
    #     test(bacc, op)
    # for op in options:
    #     test(bloss, op)
