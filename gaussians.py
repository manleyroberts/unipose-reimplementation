import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import normalize

class Gaussians(nn.Module):

    def __init__(self, stride=8, sigma=1, input_shape=(1,16,368,368)):
        self.stride = stride
        self.sigma = sigma
        self.input_shape = input_shape

    def gaussian_kernel(self, h, w, center):
        # Build a gaussian map
        x,y = center
        h, w, x, y = h//self.stride, w//self.stride, np.array(x//self.stride), np.array(y//self.stride)
        if x < 0 or y < 0:
            ans = np.ones((h, w)) * -1
            print(f'Image with -1')
        else:
            ycoords, xcoords = np.mgrid[0:h, 0:w]
            num = -1 * (np.square(ycoords - y) + np.square(xcoords - x))
            den = 2 * np.square(self.sigma)
            ans = np.exp(num/den)
            ans = ans/(np.sum(ans))
            #ans = normalize(ans)
        tensor_out = torch.HalfTensor(ans)
        tensor_out.requires_grad = False
        return tensor_out

    def expected_to_gaussian(self, expected_list):
        # Build gaussian maps for all expected
        _, K, H, W = self.input_shape
        N = len(expected_list)
        heatmap = torch.zeros((N, K, H//self.stride, W//self.stride), dtype=torch.half)
        for n in range(N):
            for k in range(K):
                heatmap[n, k, :, :] = self.gaussian_kernel(H, W, expected_list[n][k])
        heatmap.requires_grad = False
        return heatmap