import torch
import torch.nn as nn

from decoder import Decoder
from wasp import WASP
from resnet_wrapper import ResNetWrapper

class UniPose(nn.Module):
    '''
    UniPose Architecture
    input: Images (N, 3, 720, 960)

    Figure 2: https://openaccess.thecvf.com/content_CVPR_2020/papers/Artacho_UniPose_Unified_Human_Pose_Estimation_in_Single_Images_and_Videos_CVPR_2020_paper.pdf

    output: Joint heat maps (N, K, 720, 960)
    '''
    def __init__(self, pretrained_resnet=True):
        super(UniPose, self).__init__()
        
        self.resnet = ResNetWrapper(pretrained=pretrained_resnet)
        self.wasp = WASP()
        self.decoder = Decoder()

    def forward(self, x):
        final_feats, low_level_feats = self.resnet(x)
        wasp_feats = self.wasp(final_feats)
        output = self.decoder(wasp_feats, low_level_feats)
        return output