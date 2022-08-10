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
from cls import names, input_h, input_w

modelPath = '/data1/zhn/xianghao/modelzoo/cls_model/20211108/epoch_1782.pth'

def convert_model(model, input_h, input_w, modelpath, savepath):
    '''
    description: convert model to onnx
    '''
    fname = os.path.splitext(modelpath.split("/")[-1])[0]
    savepath = savepath +""+fname+".onnx"
    model.eval()
    x = torch.ones(1,3,input_h,input_w).cuda()  # initialize a indentity matrix
    torch.onnx.export(model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                savepath,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : [0 ],    # variable lenght axes
                                'output' : [0 ]}
                # dynamic_axes={'input' : {0:"1"},    # variable lenght axes
                #              'output' : {0:"1"}}
                            )
    print("convert finished, model saved to ", savepath)



if __name__ == "__main__":
    
    nclass = len(names)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features,nclass,bias=False) 
    model.load_state_dict(torch.load(modelPath))
    model.cuda()
    model.eval()
    convert_model(model, input_h, input_w, modelPath, './')
    