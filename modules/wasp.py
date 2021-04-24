import torch
import torch.nn as nn

class WASP(nn.Module):
    '''
    Waterfall Atrous Spatial Pooling
    input dim: (N, 1280, 90, 120) (w, h, features)

    Figure 4: https://openaccess.thecvf.com/content_CVPR_2020/papers/Artacho_UniPose_Unified_Human_Pose_Estimation_in_Single_Images_and_Videos_CVPR_2020_paper.pdf

    output dim: (N, 256, 90, 120) (w, h, features)
    '''
    def __init__(self, input_channels=1280, total_output_channels=256, output_channels=(60,60,60,60), rates=(6,12,18,24), core_kernel=3, hidden_size=256, avg_pool_kernel=3, avg_pool_output_channels=16):
        super(WASP, self).__init__()

        self.core_kernel = core_kernel

        # Every 3x3 conv "core" must have an output channel associated
        assert(len(output_channels)==len(rates))

        # Final output must be a stack of all outputs from convs and the avg pool outputs
        assert(sum(output_channels) + avg_pool_output_channels == total_output_channels)

        # core kernel should be odd and greater than 1
        assert(core_kernel > 1 and core_kernel % 2 == 1)

        # first conv has different input channels than others
        # Total size of FOV covered by dilated kernel is (rates[0] * core_kernel - rates[0]  + 1)
        first_conv = nn.ModuleDict({
            f'{core_kernel}x{core_kernel}': nn.Conv2d(input_channels, total_output_channels, kernel_size=core_kernel, padding=((rates[0] * core_kernel - rates[0]  + 1) // 2), dilation=rates[0]),
            '1x1': nn.Sequential(
                nn.Conv2d(total_output_channels, hidden_size, kernel_size=1),
                nn.Conv2d(hidden_size, output_channels[0], kernel_size=1)
            )
        })

        other_convs = [
            # each "core" has a 3x3 layer, followed by 2 1x1
            # Total size of FOV covered by dilated kernel is (rates[rate_i] * core_kernel - rates[rate_i]  + 1)
            nn.ModuleDict({
                f'{core_kernel}x{core_kernel}': nn.Conv2d(total_output_channels, total_output_channels, kernel_size=core_kernel, padding=((rates[rate_i] * core_kernel - rates[rate_i]  + 1) // 2), dilation=rates[rate_i]),
                '1x1': nn.Sequential(
                    nn.Conv2d(total_output_channels, hidden_size, kernel_size=1),
                    nn.Conv2d(hidden_size, output_channels[rate_i], kernel_size=1)
                )
            })
            # for each of the given rates except first
            for rate_i in range(1, len(rates))
        ]

        # list of all conv cores
        all_convs = [first_conv] + other_convs
        self.convolutions = nn.ModuleList(all_convs)

        self.avg_pool = nn.Sequential(
            # avg pool
            nn.AvgPool2d(kernel_size=avg_pool_kernel, stride=1, padding=(avg_pool_kernel // 2), count_include_pad=False),
            # 1x1 conv to reduce channels
            nn.Conv2d(input_channels, avg_pool_output_channels, kernel_size=1)
        )

    def forward(self, x):
        outputs = []
        layer_input = x
        for conv_core in self.convolutions:
            # pass through core_kernel x core_kernel layer
            conv_output = conv_core[f'{self.core_kernel}x{self.core_kernel}'](layer_input)
            # pass on to next core
            layer_input = conv_output
            # pass on this output through 1x1 and add to output list
            final_1x1_output = conv_core['1x1'](conv_output)
            outputs.append(final_1x1_output)

        # add avg pool to outputs
        outputs.append(self.avg_pool(x))

        # cat all outputs together in channel dimension
        return torch.cat(outputs, dim=1)