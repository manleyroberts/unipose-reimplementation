import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, resnet101

class ResNetWrapper(nn.Module):
    '''
    Wrapper to configure ResNet101 as needed for our architecture.
    input dim: (N, 3, 368, 368) (batch, features, h, w)

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
            dilated version of Bottleneck used to upsample from layer 3 output of shape (N, 1024, 45, 60) to (N, 1280, 90, 120)
    ]
    
    Notice that image has been downsampled 8 times.
    output dim: (N, 256, 90, 120) (w, h, features)
    '''
    def __init__(self, pretrained=True):
        super(ResNetWrapper, self).__init__()

        # Preload base from TorchHub
        self.net = resnet101(pretrained=pretrained, progress=False)

        # Remove unnecessary layers
        del self.net.layer4
        del self.net.fc
        del self.net.avgpool

        # add custom layer
        upsample_layer = self.build_upsample_layer()
        self.net.add_module("upsample_layer", upsample_layer)

    def build_upsample_layer(self):
        # Configure necessary input sizes for new layer 4
        self.net.inplanes = 1024
        self.net.dilation = 1024

        # Build Layer
        upsample_layer = nn.Sequential(
            # build replacement layer 4 with dilation enabled to preserve size while mimicking ResNet101 behavior closely
            self.net._make_layer(Bottleneck, 512, 3, stride=2, dilate=True),

            # Conv2d to scale up to necessary output and correct channel count
            # Goes from (N, 2048, 45, 60) to (N, 1280, 90, 120)
            nn.ConvTranspose2d(2048, 1280, kernel_size=2, stride=2)
        )
        return upsample_layer

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
        x = self.net.upsample_layer(x)

        return x, layer1_output
