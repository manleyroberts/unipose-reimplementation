
import torch
import torch.nn as nn
import numpy as np
from ..bilinear_interpolation import BilinearInterpolation

class DistributionDifferenceLoss(nn.Module):
    '''
    Joint Max MSE Loss
    input: [
        - (N, K, H, W) (N, features, H, W) - predicted feature maps
        - (N, K, 2)    (N, features, 2)    - expected maxes
    ]

    MSE between softmaxed feature map output of UniPose and the KPT Gaussian

    output dim: (1) Loss
    '''
    def __init__(self, device, stride=1, sigma=3, input_shape=(1,16,46,46)):
        super(DistributionDifferenceLoss, self).__init__()
        self.device = device
        self.stride = stride
        self.sigma = sigma
        self.input_shape = input_shape

        self.bilinear = BilinearInterpolation(output_size=(self.input_shape[2]//self.stride, self.input_shape[3]//self.stride))
        self.softmax = nn.Softmax(dim=2)
        self.loss = nn.MSELoss()

    def forward(self, predicted, expected_maps):
        # if stride != 1 rescale predicted
        rescaled = self.bilinear(predicted)

        # get Softmax over 2D image channels
        # input_view = rescaled.view(rescaled.shape[0], rescaled.shape[1], -1)
        # output_view = self.softmax(input_view)
        # softmax_output = output_view.view(rescaled.shape)

        # get MSELoss between predicted and expected
        adjusted_expected = torch.where(expected_maps < 0, rescaled, expected_maps)
        loss = self.loss(rescaled, adjusted_expected)
        return loss


