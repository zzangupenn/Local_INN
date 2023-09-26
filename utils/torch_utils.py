import torch
import torch.nn as nn
import numpy as np

BN_MOMENTUM = 0.1

class LinearLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, is_relu=True):
        super().__init__()
        # print('LinearLayer', in_channels, out_channels, BN_MOMENTUM)
        self.add_module('layer1', nn.Linear(in_channels, out_channels))
        self.add_module('norm', nn.BatchNorm1d(out_channels, momentum=BN_MOMENTUM))
        if is_relu:
            self.add_module('relu', nn.ReLU())
    def forward(self, x):
        return super().forward(x)
    
    
class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 is_relu=True, stride = 1, padding = 0,
                 add_layer = True, add_layer_kernel_size=3):
        super().__init__()
        # print('Conv2DLayer', in_channels, out_channels, BN_MOMENTUM)
        # self.add_module('layer3', nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding='same'))
        if add_layer:
            self.add_module('layer2', nn.Conv2d(in_channels, in_channels, add_layer_kernel_size, 1, padding='same'))
            self.add_module('norm2', nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM))
            self.add_module('relu2', nn.ReLU())
            
        self.add_module('layer1', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module('norm', nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
        if is_relu:
            self.add_module('relu', nn.ReLU())
    def forward(self, x):
        return super().forward(x)
    
class TransConv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, is_relu=True, stride = 1, 
                 padding = 0, output_padding = 0, dilation=1,
                 add_layer = True, add_layer_kernel_size=3):
        super().__init__()
        # print('Conv2DLayer', in_channels, out_channels, BN_MOMENTUM)
        if add_layer:
            self.add_module('layer2', nn.Conv2d(in_channels, in_channels, add_layer_kernel_size, 1, padding='same'))
            self.add_module('norm2', nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM))
            self.add_module('relu2', nn.ReLU())
        
        self.add_module('layer1', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, \
                                                     padding, output_padding=output_padding, dilation=dilation))
        self.add_module('norm', nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
        if is_relu:
            self.add_module('relu', nn.ReLU())
    def forward(self, x):
        return super().forward(x)

class PositionalEncoding():
    def __init__(self, L):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = np.array(self.val_list)

    def encode(self, x):
        return np.sin(self.val_list * np.pi * x), np.cos(self.val_list * np.pi * x)
    
    def encode_even(self, x):
        return np.sin(self.val_list * np.pi * 2 * x), np.cos(self.val_list * np.pi * 2 * x)
    
    def decode(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / (np.pi)
        if np.isscalar(atan2_value) == 1:
            if atan2_value > 0:
                return atan2_value
            else:
                return 1 + atan2_value
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            return atan2_value
        
    def decode_even(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / np.pi/2
        if np.isscalar(atan2_value) == 1:
            if atan2_value < 0:
                atan2_value = 1 + atan2_value
            if np.abs(atan2_value - 1) < 0.001:
                atan2_value = 0
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            atan2_value[np.where(np.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value
    
class PositionalEncoding_torch():
    def __init__(self, L, device):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = torch.Tensor(self.val_list)[None, :].to(device)
        self.pi = torch.Tensor([3.14159265358979323846]).to(device)
        
    def encode(self, x):
        return torch.sin(x * self.val_list * self.pi), torch.cos(x * self.val_list * self.pi)
    
    def encode_even(self, x):
        return torch.sin(x * self.val_list * self.pi * 2), torch.cos(x * self.val_list * self.pi * 2)
    
    def batch_encode(self, batch):
        batch_encoded_list = []
        for ind in range(3):
            if ind == 2:
                encoded_ = self.encode_even(batch[:, ind, None])
            else:
                encoded_ = self.encode(batch[:, ind, None])
            batch_encoded_list.append(encoded_[0])
            batch_encoded_list.append(encoded_[1])
        batch_encoded = torch.stack(batch_encoded_list)
        batch_encoded = batch_encoded.transpose(0, 1).transpose(1, 2).reshape((batch_encoded.shape[1], self.L * batch_encoded.shape[0]))
        return batch_encoded

    def batch_decode(self, sin_value, cos_value):
        atan2_value = torch.arctan2(sin_value, cos_value) / (self.pi)
        sub_zero_inds = torch.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        return atan2_value

    def batch_decode_even(self, sin_value, cos_value):
        atan2_value = torch.arctan2(sin_value, cos_value) / self.pi/2
        sub_zero_inds = torch.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        atan2_value[torch.where(torch.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value

def mmd_multiscale(x, y, device):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape, device=device),
                torch.zeros(xx.shape, device=device),
                torch.zeros(xx.shape, device=device))

    # kernel computation
    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)