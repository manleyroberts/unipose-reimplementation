
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
    def __init__(self, device, stride=8, sigma=3, input_shape=(1,16,720,960)):
        super(DistributionDifferenceLoss, self).__init__()
        self.device = device
        self.stride = stride
        self.sigma = sigma
        self.input_shape = input_shape

        self.bilinear = BilinearInterpolation(output_size=(self.input_shape[2]//self.stride, self.input_shape[3]//self.stride))
        self.softmax = nn.Softmax(dim=2)
        self.loss = nn.MSELoss()

#     def gaussian_kernel(self, h, w, center):
#         # Build a gaussian map
#         x,y = center
#         h, w, x, y = h//self.stride, w//self.stride, np.array(x//self.stride), np.array(y//self.stride)
#         ycoords, xcoords = np.mgrid[0:h, 0:w]
#         num = -1 * (np.square(ycoords - y) + np.square(xcoords - x))
#         den = 2 * np.square(self.sigma)
#         ans = np.exp(num/den)
#         normalized = ans/np.sum(ans)
#         return torch.Tensor(ans).to(self.device)

#     def expected_to_gaussian(self, expected_list):
#         # Build gaussian maps for all expected
#         _, K, H, W = self.input_shape
#         N = expected_list.shape[0]
#         heatmap = torch.zeros((N, K, H//self.stride, W//self.stride)).to(self.device)
#         for n in range(N):
#             for k in range(K):
#                 heatmap[n, k, :, :] = self.gaussian_kernel(H, W, expected_list[n][k])
#         return heatmap

    def forward(self, predicted, expected_maps):
        # if stride != 1 rescale predicted
        rescaled = self.bilinear(predicted)

        # get Softmax over 2D image channels
        input_view = rescaled.view(rescaled.shape[0], rescaled.shape[0], 1, -1)
        output_view = self.softmax(input_view)
        softmax_output = output_view.view(rescaled.shape)

        # get gaussian maps for expected
#         expected_maps = self.expected_to_gaussian(expected)

        # get MSELoss between predicted and expected
        loss = self.loss(softmax_output, expected_maps)
        
        return loss


