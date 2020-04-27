"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['asnet']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import resnet50, Bottleneck
import copy
import math
import random
from .nn_utils import *
from .pc import *

class DimReduceLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AsNet(nn.Module):

    def __init__(self, num_classes, loss=None, dropout_p=None, **kwargs):
        super(AsNet, self).__init__()
        
        resnet_ = resnet50(pretrained=True)

        self.num_parts = 3
        
        self.loss = loss
        
        self.layer0 = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.pc1 = PC_Module(512)
        self.layer3 = resnet_.layer3
 
        # self.pc2 = PC_Module(1024)
        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())
        
        self.layer4_trunk = nn.Sequential(copy.deepcopy(layer4))
        self.layer4_branch = nn.Sequential(copy.deepcopy(layer4))
        # self.layer4 = nn.Sequential(copy.deepcopy(layer4))

        self.layer5_trunk = Bottleneck(2048, 512)
        # self.layer5_branch = Bottleneck(2048, 512)
        # self.attention_pc_branch = PC_Module(1024)

        # self.res_part1 = Bottleneck(2048, 512) 
        # self.res_part2 = Bottleneck(2048, 512)  
                
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.parts_pool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        # self.dropout = nn.Dropout(p=0.5)

        self.bn_trunk = nn.BatchNorm2d(2048)    #20191226_1
        # self.bn_branch = nn.BatchNorm2d(2048)   #20191226_1
        # self.fc = nn.Linear(2048, 2048)
        # self.bn = nn.BatchNorm2d(2048)

        # self.dim_red = DimReduceLayer(2048, 256)
        # self.dim_red_trunk = DimReduceLayer(2048, 256)
        self.dim_red_branch = DimReduceLayer(2048, 256)
        # self.dim_red_branch = DimReduceLayer(1024, 256)
        # self.dim_red_branch = nn.ModuleList([DimReduceLayer(2048, 256) for _ in range(self.num_parts)])
        # self.conv5 = DimReduceLayer(2048, 2048) # 20191221_3

        # self.classifier_trunk = nn.Linear(256, num_classes)
        self.classifiers = nn.ModuleList([nn.Linear(256, num_classes) for _ in range(self.num_parts)])
        # self.classifiers = nn.ModuleList([nn.Linear(256, num_classes) for _ in range(self.num_parts+1)])
        # self.classifiers = nn.ModuleList([nn.Linear(2048, num_classes) for _ in range(self.num_parts)])
        self.classifier_trunk = nn.Linear(2048, num_classes)
        # self.classifier1 = nn.Linear(2048, num_classes) # 20191221_3
        # self.classifier = nn.ModuleList([nn.Linear(256, num_classes) for _ in range(6)])    # 20191221_2
        
        self._init_params()

    def _init_params(self):
        # self.init_struct(self.dim_red)
        # self.init_struct(self.dim_red_trunk)
        # for i in range(self.num_parts):
        #     self.init_struct(self.dim_red_branch[i])
        init_struct(self.dim_red_branch)
        init_struct(self.layer5_trunk)
        # self.init_struct(self.layer5_trunk)

        # self.init_bn(self.bn)
        init_bn(self.bn_trunk) #20191226_1
        # self.init_bn(self.bn_branch)    #20191226_1
        
        # nn.init.normal_(self.classifier_trunk.weight, 0, 0.01)
        # if self.classifier_trunk.bias is not None:
        #     nn.init.constant_(self.classifier_trunk.bias, 0)

        # ''' 20191221_2
        nn.init.normal_(self.classifier_trunk.weight, 0, 0.01)
        if self.classifier_trunk.bias is not None:
            nn.init.constant_(self.classifier_trunk.bias, 0)
        for c in self.classifiers:
            nn.init.normal_(c.weight, 0, 0.01)
            if c.bias is not None:
                nn.init.constant_(c.bias, 0)
        # '''

    def featuremaps(self, x):
        x = self.layer0(x)  #[batch, 64, 96, 32]
        x = self.layer1(x)  #[batch, 256, 96, 32]
        x = self.layer2(x)  #[batch, 512, 48, 16]
        x = self.pc1(x)
        x = self.layer3(x)  #[batch, 1024, 24, 8]

        return x

    def forward(self, x):
        x = self.featuremaps(x)
        
        # f_trunk = self.layer4(x)
        # f_branch = f_trunk
        f_trunk = self.layer4_trunk(x)  #[batch, 2048, 24, 8]
        # f_branch = self.attention_pc_branch(x)
        # f_branch = self.layer4_branch(f_branch)
        f_branch = self.layer4_branch(x)    #[batch, 2048, 24, 8]
        # f_branch = x

        f_trunk = self.layer5_trunk(f_trunk)    #[batch, 2048, 24, 8]
        # f_branch = self.attention_pc_branch(f_branch)    #[batch, 2048, 24, 8]
        # f_branch = self.layer5_branch(f_branch)

        f_trunk = self.global_avgpool(f_trunk)  # [batch, 2048, 1, 1]
        f_branch = self.parts_pool(f_branch)    # [batch, 2048, self.parts, 1]
        
        # f = F.normalize(f, p=2, dim=1)
        # f = torch.cat([f_trunk, f_branch], 2)
        f_train = torch.cat([f_trunk, f_branch], 2)
        # f_train = torch.cat([f_trunk.view(f_trunk.size(0), -1), f_branch.view(f_branch.size(0), -1)], 1)
        # f_trunk_train = F.normalize(f_trunk, p=2, dim=1)

        f_trunk = self.bn_trunk(f_trunk)
        f = torch.cat([f_trunk, f_branch], 2)
        # f = torch.cat([f_trunk.view(f_trunk.size(0), -1), f_branch.view(f_branch.size(0), -1)], 1)

        if not self.training:
            # f_trunk = F.normalize(f_trunk, p=2, dim=1)
            # f_branch = F.normalize(f_branch, p=2, dim=1)
            f = F.normalize(f, p=2, dim=1)
            # f = F.normalize(f.view(f.size(0), -1), p=2, dim=1)
            # f_trunk = self.bn_trunk(f_trunk)    #20191226_1
            # f_branch = self.bn_branch(f_branch) #20191226_1
            # f = self.bn(f)
            # f = torch.cat([f_trunk.view(f_trunk.size(0), -1), f_branch.view(f_branch.size(0), -1)], 1)

            return f.view(f.size(0), -1)
            # return f

        # f_trunk = self.dropout(f_trunk)   #???
        # f_branch = self.dropout(f_branch)

        # f_short = self.dim_red(f)
        # f_trunk_short = self.dim_red_trunk(f_trunk)
        f_branch_short = self.dim_red_branch(f_branch)
        # f_branch_short = []
        # for i in range(self.num_parts):
        #     f_i = f_branch[:, :, i, :].view(f_branch.size(0), f_branch.size(1), 1, 1)
        #     f_i = self.dim_red_branch[i](f_i)
        #     f_branch_short.append(f_i)
        
        y = []
        # f_trunk_short = f_trunk_short.view(f_trunk_short.size(0), -1)
        f_trunk = f_trunk.view(f_trunk.size(0), -1)
        # y_trunk = self.classifiers[self.num_parts](f_trunk_short)
        y_trunk = self.classifier_trunk(f_trunk)

        # y_trunk = self.classifiers[3](f_trunk)
        # f_i = f_branch_short[:, :, 0, :]
        # f_i = f_i.view(f_i.size(0), -1)
        # y_i = self.classifiers[0](f_i)
        y.append(y_trunk)

        for i in range(self.num_parts):
            # if i < self.num_parts-1:
            # y_trunk = y_trunk+self.classifiers[i](f_trunk_short)
            # f_i = f_branch[:, :, i, :]
            f_i = f_branch_short[:, :, i, :]
            f_i = f_i.view(f_i.size(0), -1)
            # f_i = f_branch_short[i].view(f_branch_short[i].size(0), -1)
            y_i = self.classifiers[i](f_i)
            y.append(y_i)

        # y.append(y_trunk/(self.num_parts-1))        

        if self.loss == 'softmax':
            return y
        elif self.loss == 'engine_as_net':
            # f = torch.cat([f_trunk, f_branch], 2)
            # f_branch = F.normalize(f_branch, p=2, dim=1)

            # f = torch.cat([f_trunk_train.view(f_trunk_train.size(0), -1), f_branch.view(f_branch.size(0), -1)], 1)

            f = F.normalize(f_train, p=2, dim=1)
            # f = F.normalize(f_train.view(f_train.size(0), -1), p=2, dim=1)
            return y, f.view(f.size(0), -1)
            # return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def asnet(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = AsNet(
        num_classes=num_classes,
        loss=loss,
        dropout_p=None,
        **kwargs
    )
    return model

