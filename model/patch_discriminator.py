from torch import nn
from base import BaseModel
from torchgan.layers import SpectralNorm2d


class Local_Discriminator(BaseModel):
    """
    Local discriminator
    """
    def __init__(self, input_nc=1, ndf=64, n_layers=5):
        super(Local_Discriminator, self).__init__()
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        use_bias=True

        kw = 4
        padw = 1
        sequence = [SpectralNorm2d(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                SpectralNorm2d(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                          bias=use_bias)),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            SpectralNorm2d(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.BatchNorm2d(ndf * nf_mult),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.linear = nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, padding=0, bias=use_bias) # nn.Linear(25088, 1)  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        x = self.model(input)
        # x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output.view(x.size(0), -1)


if __name__ == '__main__':
    discriminator = Local_Discriminator(input_nc=1, ndf=32, n_layers=5)
    print(discriminator)
    import torch
    inputs = torch.ones(16, 1, 128, 128)
    outputs = discriminator(inputs)