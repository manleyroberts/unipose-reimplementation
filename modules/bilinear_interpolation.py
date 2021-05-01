import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearInterpolation(nn.Module):
    def __init__(self, output_size):
        super(BilinearInterpolation, self).__init__()
        self.interpolate_func = F.interpolate
        self.output_size = output_size
    
    def forward(self, x):
        return self.interpolate_func(x, size=self.output_size, mode='bilinear', align_corners=True)

    def __repr__(self):
        rep = f'BilinearInterpolation({self.output_size})'
        return rep
        