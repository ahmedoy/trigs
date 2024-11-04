from typing import Any
from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, conv1x1, conv3x3


class FlexResNet(nn.Module):
    """
    Flexible ResNet model that allows for changing the settings of the first convolutional layer and the number of
    planes input to the first layer. Note that the conv1 and bn1 layers of the base resnet model will be replaced by
    new ones that are randomly initialized. Therefore, pretrained weights in these layers, if chosen, will be lost.

    This class is created to allow using the existing ResNet models in PyTorch with older settings that have been
    replaced in the current implementation, or with custom settings.
    """
    def __init__(self,
                 resnet_model: ResNet = None,
                 inplanes: int = 64,
                 conv1_args: dict[str, Any] = None,
                 **resnet_args
                 ) -> None:
        super().__init__()

        # Create the wrapped resnet model if it is not provided
        self.resnet_model = resnet_model or ResNet(**resnet_args)
        if conv1_args is None:
            assert inplanes == 64
            return  # No need to do anything else

        assert conv1_args['out_channels'] == inplanes, "ResNet's conv1 out_channels must equal inplanes"

        # Replace the first convolutional layer with one with the given arguments
        self.resnet_model.conv1 = nn.Conv2d(**conv1_args)
        self.init_module_for_resnet(self.resnet_model.conv1)

        if inplanes != 64:  # If inplanes is different from the default, a lot of more work is needed.
            norm_layer = self.resnet_model.bn1.__class__

            # Create and initialize a new normalization layer for bn1
            self.resnet_model.bn1 = norm_layer(inplanes)
            self.init_module_for_resnet(self.resnet_model.bn1)

            # Create a downsampler for the first layer's first block if it does not exist,
            # or update its convolutional part if it already exists
            downsample_outplanes = 64 * self.resnet_model.layer1[0].expansion
            downsample_conv = conv1x1(inplanes, downsample_outplanes)
            self.init_module_for_resnet(downsample_conv)
            if self.resnet_model.layer1[0].downsample is None:
                self.resnet_model.layer1[0].downsample = nn.Sequential(
                    downsample_conv,
                    norm_layer(downsample_outplanes),
                )
                self.init_module_for_resnet(self.resnet_model.layer1[0].downsample[1])
            else:
                self.resnet_model.layer1[0].downsample[0] = downsample_conv

            # Finally, create and initialize the first convolutional layer of the first layer's first block
            if isinstance(self.resnet_model.layer1[0], BasicBlock):
                self.resnet_model.layer1[0].conv1 = conv3x3(inplanes, 64)
            elif isinstance(self.resnet_model.layer1[0], Bottleneck):
                self.resnet_model.layer1[0].conv1 = conv1x1(inplanes, self.resnet_model.layer1[0].conv1.out_channels)
            else:
                raise ValueError(f"Unsupported resnet block class {self.resnet_model.layer1[0].__class__}")
            self.init_module_for_resnet(self.resnet_model.layer1[0].conv1)

    @staticmethod
    def init_module_for_resnet(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.resnet_model(x)
        return x

    def load_state_dict(self, state_dict, strict: bool = True):
        new_state_dict={}
        for key, value in state_dict.items():
            if not key.startswith('resnet_model.'):
                new_key = f"resnet_model.{key}"
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        return super().load_state_dict(new_state_dict, strict=strict)

def resnet18_mod(num_classes=100, weights=None, progress=True, **kwargs):
    return FlexResNet(conv1_args={
                          "in_channels": 3,
                          "out_channels": 64,
                          "kernel_size": 3,
                          "stride": 1,
                          "padding": 1,
                          "bias": False
                      },
                      block=BasicBlock,
                      layers=[1, 1, 1, 1],
                      num_classes=num_classes,
                      **kwargs)


def baseline_classifier(num_channels, pretrained=False, progress=True, **kwargs):
    base_model = models.resnext50_32x4d()
    inplanes = num_channels // 3
    conv1_args = {
        "in_channels": num_channels,
        "out_channels": inplanes,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "groups": inplanes,
    }
    print(f"--> Number of input channels in convolutional model is {num_channels}")
    return FlexResNet(resnet_model=base_model, inplanes=inplanes, conv1_args=conv1_args,
                      pretrained=pretrained, progress=progress, **kwargs)
