import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary
from cls import names, input_h, input_w

modelPath = '/data1_dev/zhn/dogcat/models/epoch_bias_2.pth'

##testsdf


def main():
    print('cuda device count: ', torch.cuda.device_count())
    
    
    nclass = len(names)
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features,nclass,bias=True) 
    net.load_state_dict(torch.load(modelPath))
    net = net.to('cuda:0')
    net.eval()
    print('model: ', net)
    #print('state dict: ', net.state_dict().keys())
    tmp = torch.ones(1, 3, input_h, input_w).to('cuda:0')
    print('input: ', tmp)
    out = net(tmp)
    print('output:', out)

    summary(net, (3, input_h, input_w))
    #return
    f = open("/data1_dev/zhn/dogcat/models/resnet18_bias.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        # print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()

