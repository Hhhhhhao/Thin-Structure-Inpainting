import numpy as np
import torch
import torch.nn as nn
from base import BaseModel


def conv_block(input_dim, output_dim, kernel_size=3, stride=1):
    seq = []
    if kernel_size != 1:
        padding = int(np.floor((kernel_size-1)/2))
        seq += [nn.ReflectionPad2d(padding)]

    seq += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()]
    return nn.Sequential(*seq)


def upsample_conv_block(input_dim, output_dim, kernel_size=3):
    seq = []
    seq += [nn.UpsamplingNearest2d(scale_factor=2)]
    if kernel_size != 1:
        padding = int(np.floor((kernel_size-1)/2))
        seq += [nn.ReflectionPad2d(padding)]

    seq += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()]

    return nn.Sequential(*seq)


class Unet(BaseModel):
    """
    A Unet like model
    """
    def __init__(self):
        super(Unet, self).__init__()
        self.conv_block_1 = conv_block(input_dim=1, output_dim=16, kernel_size=1, stride=1)
        self.conv_block_2 = conv_block(input_dim=16, output_dim=32, kernel_size=3, stride=2)
        self.conv_block_3 = conv_block(input_dim=32, output_dim=64, kernel_size=3, stride=1)
        self.conv_block_4 = conv_block(input_dim=64, output_dim=128, kernel_size=3, stride=2)
        self.conv_block_5 = conv_block(input_dim=128, output_dim=128, kernel_size=3, stride=1)
        self.conv_block_6 = conv_block(input_dim=128, output_dim=256, kernel_size=3, stride=2)
        self.conv_block_7 = conv_block(input_dim=256, output_dim=256, kernel_size=3, stride=1)
        self.conv_block_8 = conv_block(input_dim=256, output_dim=512, kernel_size=3, stride=2)
        self.conv_block_9 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)

        self.up_conv_block_1 = upsample_conv_block(input_dim=256, output_dim=256, kernel_size=3)
        self.conca_conv_block_1 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)
        self.up_conv_block_2 = upsample_conv_block(input_dim=256, output_dim=128, kernel_size=3)
        self.conca_conv_block_2 = conv_block(input_dim=256, output_dim=128, kernel_size=3, stride=1)
        self.up_conv_block_3 = upsample_conv_block(input_dim=128, output_dim=64, kernel_size=3)
        self.conca_conv_block_3 = conv_block(input_dim=96, output_dim=64, kernel_size=3, stride=1)
        self.up_conv_block_4 = upsample_conv_block(input_dim=64, output_dim=32, kernel_size=3)
        self.conv_block_10 = conv_block(input_dim=32, output_dim=16, kernel_size=3, stride=1)
        self.conv_block_11 = conv_block(input_dim=16, output_dim=2, kernel_size=3, stride=1)
        self.conv_block_12 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax2d()
        self.monte_carlo_count = 5

        self.latent_conv = conv_block(input_dim=512, output_dim=256, kernel_size=1, stride=1)
        # self.residul_block_2 = Residule_Block(inplanes=512, planes=128, stride=2)
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.similarity_liner = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.conv_block_1(images)
        x = self.conv_block_2(x)
        skip_1 = x

        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        skip_2 = x

        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        skip_3 = x

        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = self.conv_block_9(x)

        # Upsample
        x = self.up_conv_block_1(x)
        x = torch.cat([x, skip_3], 1)
        x = self.conca_conv_block_1(x)

        x = self.up_conv_block_2(x)
        x = torch.cat([x, skip_2], 1)
        x = self.conca_conv_block_2(x)

        x = self.up_conv_block_3(x)
        x = torch.cat([x, skip_1], 1)
        x = self.conca_conv_block_3(x)

        x = self.up_conv_block_4(x)
        x = self.conv_block_10(x)
        x = self.conv_block_11(x)
        x = self.conv_block_12(x)

        prob = self.softmax(x)
        return x, prob

    def inference(self, images):
        outputs, prob = self.forward(images)
        return prob # self.softmax(outputs)
