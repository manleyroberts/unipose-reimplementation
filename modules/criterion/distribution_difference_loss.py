import sys
sys.path.insert(1, '../')

import torch
import torch.nn as nn
import numpy as np
from bilinear_interpolation import BilinearInterpolation

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
    def __init__(self, stride=8, sigma=3, input_shape=(1,16,720,960)):
        super(DistributionDifferenceLoss, self).__init__()
        self.stride = stride
        self.sigma = sigma
        self.input_shape = input_shape

        self.bilinear = BilinearInterpolation(output_size=(self.input_shape[2]//self.stride, self.input_shape[3]//self.stride))
        self.softmax = nn.Softmax()
        self.loss = nn.MSELoss()

    def gaussian_kernel(self, h, w, center):
        # Build a gaussian map
        x,y = center
        h, w, x, y = h//self.stride, w//self.stride, x//self.stride, y//self.stride
        ycoords, xcoords = np.mgrid[0:h, 0:w]
        num = -1 * (np.square(ycoords - y) + np.square(xcoords - x))
        den = 2 * np.square(self.sigma)
        ans = np.exp(num/den)
        normalized = ans/np.sum(ans)
        return torch.Tensor(ans)

    def expected_to_gaussian(self, expected_list):
        # Build gaussian maps for all expected
        N, K, H, W = self.input_shape
        heatmap = torch.zeros((N, K, H//self.stride, W//self.stride))
        for n in range(N):
            for k in range(K):
                heatmap[n, k, :, :] = self.gaussian_kernel(H, W, expected_list[n][k])
        return heatmap

    def forward(self, predicted, expected):
        # if stride != 1 rescale predicted
        rescaled = self.bilinear(predicted)

        # get Softmax over 2D image channels
        input_view = predicted.view(predicted.shape[0], predicted.shape[0], 1, -1)
        output_view = self.softmax(input_view)
        softmax_output = output_view.view(predicted.shape)

        # get gaussian maps for expected
        expected_maps = self.expected_to_gaussian(expected)

        # get MSELoss between predicted and expected
        loss = self.loss(rescaled, expected_maps)
        
        return loss


