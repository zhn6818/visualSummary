import os
import torch
import torch.nn as nn
import torchvision
import cv2
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from torchvision import transforms 
import time
from PIL import Image
from cls import names, JFDetDataset, img_path, input_h, input_w

modelPath = '/data1/zhn/xianghao/modelzoo/cls_model/20210518/epoch_1980.pth'

if __name__ == "__main__":
    
    train_dataset = JFDetDataset(img_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=1, 
                                            num_workers =4,
                                            shuffle=False)
    
    
    nclass = len(names)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features,nclass,bias=False) 
    model.load_state_dict(torch.load(modelPath))
    model.cuda()
    model.eval()
    img_list = []
    with open(img_path,"r",encoding="utf-8") as f:
        img_list = f.readlines()
    print(len(img_list))
    # for j, (inputs, labels) in enumerate(train_loader):
    for file in img_list:
        
        
        filename = file.strip("\n")
        print(filename)
        img = cv2.imread(filename)
        inputs = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
        inputs = inputs.astype(np.float32) / 255.
        inputs = inputs.transpose(2, 0, 1)
        inputs = torch.Tensor(inputs)
        inputs = inputs.unsqueeze(0).cuda()
        # inputs = Variable(inputs).cuda()
        outputs = model(inputs)
        _, index = torch.max(outputs, 1)
        index = index.detach().cpu()
        print(names[index])
        print('')
        
    