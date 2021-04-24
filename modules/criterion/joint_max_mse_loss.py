import torch
import torch.nn as nn

class JointMaxMSELoss(nn.Module):
    '''
    Joint Max MSE Loss
    input: [
        - (N, K, W, H) (N, features, w, h) - predicted feature maps
        - (N, K, 2)    (N, features, 2)    - expected maxes
    ]

    Mean across all examples of MSE between max of output map from UniPose and given max

    output dim: (1) Loss
    '''
    def __init__(self):
        super(JointMaxMSELoss, self).__init__()


    def forward(self, predicted, expected):
        # Get tensor of max indices for feature maps
        maxes, row_i = predicted.max(dim=2)
        col_i = maxes.argmax(dim=2)
        row_i_maxes = row_i[torch.arange(1),torch.arange(1),col_i].unsqueeze(2)
        col_i_maxes = col_i.unsqueeze(2)
        # Shape of max_indices should be (N, K, 2)
        max_indices = torch.cat((row_i_maxes,col_i_maxes), dim=2)

        # get difference between max_indices and expected
        squared_diffs = torch.pow(max_indices - expected, 2)
        
        # sum across dims of image
        squared_errors = torch.sum(squared_diffs, dim=2)

        # mean across feature maps and across examples
        mean_squared_error = torch.mean(squared_errors)
        return mean_squared_error


