import torch
import torch.nn as nn
import numpy as np

class Gaussians(nn.Module):

    def __init__(self, device, stride=8, sigma=3, input_shape=(1,16,720,960)):
        self.device = device
        self.stride = stride
        self.sigma = sigma
        self.input_shape = input_shape

    def gaussian_kernel(self, h, w, center):
        # Build a gaussian map
        x,y = center
        h, w, x, y = h//self.stride, w//self.stride, np.array(x//self.stride), np.array(y//self.stride)
        ycoords, xcoords = np.mgrid[0:h, 0:w]
        num = -1 * (np.square(ycoords - y) + np.square(xcoords - x))
        den = 2 * np.square(self.sigma)
        ans = np.exp(num/den)
        normalized = ans/np.sum(ans)
        return torch.Tensor(ans).to(self.device)

    def expected_to_gaussian(self, expected_list):
        # Build gaussian maps for all expected
        _, K, H, W = self.input_shape
        N = expected_list.shape[0]
        heatmap = torch.zeros((N, K, H//self.stride, W//self.stride)).to(self.device)
        for n in range(N):
            for k in range(K):
                heatmap[n, k, :, :] = self.gaussian_kernel(H, W, expected_list[n][k])
        return heatmap