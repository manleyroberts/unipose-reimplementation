import torch
import torch.nn as nn

class ResNetWrapper(nn.Module):
    '''
    Wrapper to configure ResNet101 as needed for our architecture.
    input dim: (N, 3, 960, 720) (w, h, features)

    Structure:

    UNALTERED TORCHHUB RESNET: [
        - pre-layer:
            conv1(x)
            bn1(x)
            relu(x)
            maxpool(x)
        - layer 1:
            3 sublayers of Bottleneck
        - layer 2:
            4 sublayers of Bottleneck
        - layer 3:
            23 sublayers of Bottleneck
    ]
    REMOVED FROM TORCHHUB RESNET: [
        - layer 4:
            3 sublayers of Bottleneck
        - avgpool
        - fc (used to map output into 1000 classes for ImageNet)
    ]
    CUSTOM ADDED TO END: [
        - upsample layer:
            dilated version of Bottleneck used to upsample from layer 3 output of shape (N, 1024, 60, 45) to (N, 1280, 120, 90)
    ]
    
    Notice that image has been downsampled 8 times.
    output dim: (N, 256, 120, 90) (w, h, features)
    '''
    def __init__(self, pretrained=True):
        super(ResNetWrapper, self).__init__()

        # Preload base from TorchHub
        self.net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=pretrained)

        # Remove unnecessary layers
        del self.net.layer4
        del self.net.fc
        del self.net.avgpool

        # add custom layers
        upsample_layer = self.build_upsample_layer()

        self.net.add_module("upsample_layer", upsample_layer)

    def build_upsample_layer(self):
        return nn.Sequential(
            nn.Conv2d(1024, 1280, kernel_size=1)
        )

    # def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
    #                 stride: int = 1, dilate: bool = False) -> nn.Sequential:
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     in_planes = 1024
    #     out_planes = 1280
    #     planes = 512

    #     layers = []
    #     layers.append(block(in_planes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=nn.BatchNorm2d))
# 
        # return nn.Sequential(*layers)


    def forward(self, x):
        # Unaltered TorchHub layers
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        layer1_output = x
        x = self.net.layer2(x)
        x = self.net.layer3(x)

        # Removed TorchHub Layers
        # x = self.net.layer4(x)

        # x = self.net.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.net.fc(x)

        # Added Layers
        # x = self.net.upsample_layer(x)

        return x, layer1_output
