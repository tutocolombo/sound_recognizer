import argparse
from typing import Any, Dict

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CustomResnet(nn.Module):
    """Pretrained resnet modified to fit ESC50 data"""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        input_channels = self.data_config["input_dims"][0]
        num_classes = len(self.data_config["mapping"])

        custom_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        custom_resnet.conv1 = nn.Conv2d(
            input_channels,
            custom_resnet.conv1.out_channels,
            kernel_size=custom_resnet.conv1.kernel_size,
            stride=custom_resnet.conv1.stride,
            padding=custom_resnet.conv1.padding,
        )
        num_features = custom_resnet.fc.in_features
        custom_resnet.fc = nn.Linear(num_features, num_classes)

        self.__dict__.update(custom_resnet.__dict__)
        self.forward = custom_resnet.forward

    @staticmethod
    def add_to_argparse(parser):
        return parser
