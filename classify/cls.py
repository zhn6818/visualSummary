# -*- coding: utf-8 -*
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
# os.environ['CUDA_VISIBLE_DEVICES']="1,2,4"
os.environ['CUDA_VISIBLE_DEVICES']="0"


## dfg

'''
sleep
'''
# names = ["awake","sleep"]
# img_path ="/opt/data/jp/datasets/classification/fatigue/sleep/train.txt"
# save_path = "/opt/data/jp/modelzoo/cls/sleep/"
# input_h = 224
# input_w = 224
# input_channel = 3

'''
eye
'''
# names = ["open","close"]
# img_path = "/opt/data/jp/datasets/classification/fatigue/eye/train.txt"
# save_path = "/opt/data/jp/modelzoo/cls/eye/"
# input_h = 150
# input_w = 150
# input_channel = 3

'''
staff
'''
names =  ["cats", "dogs"]

img_path = "/data1_dev/zhn/dogcat/train/train.txt"
save_path = "/data1_dev/zhn/dogcat/models/"
input_h = 96
input_w = 96
input_channel = 3

print(img_path)
print(save_path)

nclass = len(names)
batch_size = 512
max_epoches = 2000
lr = 0.001


class JFDetDataset(data.Dataset):
    def __init__(self,img_path,transform=None):        
        with open(img_path,"r",encoding="utf-8") as f:
            self.img_list = f.readlines()
            self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_info = self.img_list[index].strip("\n")
        img = cv2.imread(img_info)
        inputs = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
        inputs = inputs.astype(np.float32) / 255.
        inputs = inputs.transpose(2, 0, 1)
        # inputs = Image.fromarray(inputs.astype('uint8')).convert('RGB')
        # inputs = self.transform(inputs)
        for ind, each_name in enumerate(names):
            if names[ind] == img_info.split("/")[-2]: # 各类图片应按文件夹存储，文件夹名为类比名
                labels = np.array(ind,dtype=np.int64)
        return inputs, labels
            



if __name__ == "__main__":
    transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5)
    ])


    train_dataset = JFDetDataset(img_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            num_workers =4,
                                            shuffle=True)

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features,nclass,bias=True) # resnet18
    # model.classifier[1]=nn.Linear(model.classifier[1].in_features,nclass,bias=False)  # mobilenetv2

    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_in')

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # torch.optim.Adam

    criterion = nn.CrossEntropyLoss()
    for i in range(max_epoches):
        for j, (inputs, labels) in enumerate(train_loader):
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            # while 1:
            #     torch.cuda.synchronize()
            #     t1 = time.time()
            #print(inputs.shape[0], inputs.shape[1], inputs.shape[2],inputs.shape[3])
            # if inputs.shape[0] != 512:
            #     continue

            outputs = model(inputs)
                # print(outputs.shape)
                # torch.cuda.synchronize()
                # t2 = time.time()
                # print("time cost: {:.2f}ms".format( (t2-t1)*1000))
            loss = criterion(outputs, labels)
            print("Epoch{}, Batch {}: tot loss {:.6f}.".format(i,j,loss.data))
            loss.backward()
            optimizer.step()
            
        if i%2 == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.2

            torch.save(model.state_dict(),save_path+"/epoch_bias_"+str(i)+".pth")


