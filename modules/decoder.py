import torch
import torch.nn as nn
from .bilinear_interpolation import BilinearInterpolation

class Decoder(nn.Module):
    '''
    Decoder for UniPose Architecture
    input: [
        - ResNet Low Level Features (N, 256, 180, 240)
        - WASP Score Maps (N, 256, 90, 120)
    ]

    Figure 3: https://openaccess.thecvf.com/content_CVPR_2020/papers/Artacho_UniPose_Unified_Human_Pose_Estimation_in_Single_Images_and_Videos_CVPR_2020_paper.pdf

    K = output joint # (16)
    output dim: (N, K, 720, 960)
    '''
    def __init__(self, low_level_features_shape=(1, 256, 92, 92), wasp_score_maps_shape=(1, 256, 46, 46), output_dim=(1, 16, 368, 368), low_level_concat_features=48, hidden_size=256, dropout=0.2):
        super(Decoder, self).__init__()
        
        # Create ResNet LowLevel stream
        self.low_level_stream = nn.Sequential(
            nn.Conv2d(low_level_features_shape[1], low_level_concat_features, kernel_size=1),
            nn.MaxPool2d(2, stride=2)
        )

        # Create WASP Score Maps stream
        self.wasp_score_maps_stream = nn.Sequential(
            BilinearInterpolation(output_size=(46,46))
        )

        # Create combined stream
        self.combined_stream = nn.Sequential(
            nn.Conv2d(low_level_concat_features + wasp_score_maps_shape[1], hidden_size, kernel_size=3, padding=1),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden_size, output_dim[1], kernel_size=1),
            BilinearInterpolation(output_dim[2:])
        )


    def forward(self, wasp_score_maps, low_level_features):
        # LowLevel stream
        low_level_stream_output = self.low_level_stream(low_level_features)

        # WASP Score Maps stream
        wasp_score_maps_output = self.wasp_score_maps_stream(wasp_score_maps)

        # Concatenation
        concatenated = torch.cat((low_level_stream_output, wasp_score_maps_output), dim=1)

        # Combined stream
        score_maps = self.combined_stream(concatenated)
        return score_maps