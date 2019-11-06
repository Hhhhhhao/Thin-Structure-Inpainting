from torch import nn
from base import BaseModel
from torchgan.layers import SpectralNorm2d
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Global_Discriminator(BaseModel):
    """
    Global discriminator
    """
    def __init__(self, input_nc=1, ndf=64, n_layers=6):
        super(Global_Discriminator, self).__init__()
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        use_bias=True

        kw = 5
        padw = 2
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
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.AdaptiveAvgPool2d((1, 1))]
        self.f = nn.Sequential(*sequence)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_full_images, target_full_images):
        """
        normal forawrd on two batch of images
        :param input_full_images: inpainted patches or full images
        :param target_full_images: ground truth patches or full images
        :return: a similarity score over the batch
        """
        # transform them into feature vectors
        input_feature = self.f(input_full_images)
        target_feature = self.f(target_full_images)
        input_feature = input_feature.view(input_feature.size(0), -1, 1)
        target_feature = target_feature.view(input_feature.size(0), -1, 1)

        # compute the similarity score
        dot_product = torch.bmm(input_feature.permute(0, 2, 1), target_feature)
        similarity = self.sigmoid(dot_product.view(dot_product.size(0), -1))
        return similarity

    def reward_forward(self, prob, locations, orig_window_length, full_image, other_full_image):
        """
        forward with policy gradient
        :param prob: probability maps
        :param locations: locations recording where the patches are extracted
        :param orig_window_length: original patches length to calculat the replication times
        :param full_image: ground truth full image
        :param other_full_image: another ground truth full image
        :return:
        """
        # Bernoulli samoling
        batch_size = prob.size(0)
        bernoulli_dist = Bernoulli(prob)
        samples = bernoulli_dist.sample()
        log_probs = bernoulli_dist.log_prob(samples)

        # put back
        with torch.no_grad():
            repeat_times = int(np.ceil(batch_size / orig_window_length))

            target_full_images = other_full_image.repeat(repeat_times, 1, 1, 1)
            inpaint_full_images = full_image.repeat(repeat_times, 1, 1, 1)

            # j th full image
            j = 0
            for batch_idx in range(batch_size):
                sample = samples[batch_idx]
                y1, x1, y2, x2 = locations[batch_idx]
                # sample = torch.where(sample >= 0.5, torch.ones_like(sample), torch.zeros_like(sample))
                inpaint_full_images[j, :, y1:y2, x1:x2] = sample.detach()

                if (batch_idx+1) % orig_window_length == 0:
                    j += 1

            # calculate the reward over the re-composed root and ground truth root
            rewards = self.forward(inpaint_full_images, target_full_images)
            # broadcast the rewards to each element of the feature maps
            broadcast_rewards = torch.zeros(batch_size, 1)
            broadcast_rewards = broadcast_rewards.to(device)
            # j th full image
            j = 0
            for batch_idx in range(batch_size):
                broadcast_rewards[batch_idx] = rewards[j]
                if (batch_idx+1) % orig_window_length == 0:
                    j += 1

        broadcast_rewards = broadcast_rewards.view(broadcast_rewards.size(0), 1, 1, 1)
        image_size = prob.size(2)
        broadcast_rewards = broadcast_rewards.repeat(1, 1, image_size, image_size)

        return log_probs, broadcast_rewards


if __name__ == '__main__':
    discriminator = Global_Discriminator(input_nc=1, ndf=64, n_layers=7)
    print(discriminator)
    import torch
    inputs = torch.ones(1, 1, 1280, 1280)
    targets = torch.ones(1, 1, 1280, 1280)
    outputs = discriminator(inputs, targets)
