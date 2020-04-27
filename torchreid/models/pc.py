###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import math
import torch
from torch.nn import Module, Conv2d, Parameter, Softmax, Conv1d
import torch.nn as nn
from .nn_utils import *

import logging

class PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim, dropout=False):
    # def __init__(self, in_dim, dropout=True):   #20191217_2
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        t = int(abs(math.log(self.channel_in, 2)+1)/2)
        self.k = t if t % 2 else t+1

        self.query_conv = Conv1d(in_channels=1, out_channels=1, kernel_size=self.k, padding=int(self.k/2), bias=False)
        self.key_conv = Conv1d(in_channels=1, out_channels=1, kernel_size=self.k, padding=int(self.k/2), bias=False)
        # self.value_conv = Conv1d(in_channels=1, out_channels=1, kernel_size=self.k, padding=int(self.k/2), bias=False)

        # self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1) #20191212_3
        # self.res_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1) #20191215_2

        self.gamma = Parameter(torch.zeros(1))
        # self.gamma = Parameter(torch.zeros(1)) 20191212_2

        self.softmax = Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)

        if dropout:
            # self.dropout = nn.Dropout(0.3) # 20191215_1
            self.dropout = nn.Dropout(0.1) # 20191216_3

        if dropout:
            self.is_dropout = True
        else:
            self.is_dropout = False

        # self.scale = torch.sqrt(torch.FloatTensor([in_dim // 8]))   #20191212_4
        self._init_param()

    def _init_param(self):
        # init_conv2d(self.key_conv)
        # init_conv2d(self.value_conv)
        # init_conv2d(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)
        init_conv(self.query_conv)

        init_bn(self.bn)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(x.size())

        x_proj = x.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous().view(-1, 1, C)
        # print(x_proj.size())

        proj_query = self.query_conv(x_proj).view(m_batchsize, -1, C)
        # print(proj_query.size())

        proj_key = self.key_conv(x_proj).view(m_batchsize, -1, C).permute(0, 2, 1)
        # print(proj_key.size())
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)        
        proj_value = self.value_conv(x) #20191212_3
        # proj_value = self.value_conv(x_proj).view(m_batchsize, -1, C).permute(0, 2, 1) #20200114_1
        
        energy = torch.bmm(proj_query, proj_key)
        # energy = torch.bmm(proj_query, proj_key)/self.scale.to(x.device)    #20191212_4
        attention = self.softmax(energy)
       
        # x_view = x.view(m_batchsize, -1, width * height)
        x_view = proj_value.view(m_batchsize, -1, width * height) #20191212_3
        # x_view = proj_value #20200114_1

                # out = torch.bmm(self.dropout(x_view), self.dropout(attention.permute(0, 2, 1))) # 20191215_1

        # if self.is_dropout:
        #     out = torch.bmm(x_view, self.dropout(attention.permute(0, 2, 1))) #20191215_8
        # else:
        out = torch.bmm(x_view, attention.permute(0, 2, 1))

        # out = torch.bmm(self.dropout(x_view), attention.permute(0, 2, 1)) #20191215_9
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask
        # out = attention_mask    #20191212_2
        # out = attention_mask*0.01 #20191215_2
        # out = self.res_conv(attention_mask) #20191215_4

        out = self.bn(out) #20191215_3
        # out = self.res_conv(out) #20191216_1        
        # out = self.bn(out+x) #20191215_5

        if self.is_dropout:
            out = self.dropout(out) + x # 20191215_1
        else:
            out = out + x

        return out

class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        # self.gamma = Parameter(torch.zeros(1))    #20191212_2

        self.softmax = Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        self._init_param()

    def _init_param(self):
        init_bn(self.bn)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention, proj_query)
        out = out.view(m_batchsize, C, height, width)

        gamma = self.gamma.to(out.device)
        out = gamma * out 
        # gamma = self.gamma.to(out.device) #20191212_2
        # out = gamma * out     #20191212_2
        # out = 0.01*out #20191215_2

        # out = self.bn(out) #20191215_3
        out = self.bn(out)
        out = out + x
        # out = self.bn(out+x) #20191215_5
        return out
        

class PC_Module(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_channel = in_dim
        # self.pam = PAM_Module(in_dim)
        self.pam = PAM_Module(in_dim)  #20191216_4
        self.cam = CAM_Module(in_dim)

        # self.cam = CAM_Module(in_dim) 20191212_1

    def forward(self, x):
        out = self.pam(x)
        out = self.cam(out)
        # out = self.cam(out) 20191212_1
        return out