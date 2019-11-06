import torch.nn.functional as F
from torch import nn
import torch

def cross_entropy2d(output, target, weight=None, size_average=True):
    """
    2D cross entropu loss
    :param output: generator output
    :param target: ground truth
    :return: loss
    """
    n, c, h, w = output.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        raise RuntimeError("inconsistent dimension of outputs and targets")

    output = output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        output, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


class Masked_CrossEntropy(nn.Module):
    """
    Masked corss entropy loss
    """
    def __init__(self):
        super(Masked_CrossEntropy, self).__init__()

    def forward(self, output, target, mask):
        """
        calculate the loss
        :param output: generator output
        :param target: ground truth
        :param mask: masks indicating the missing pixels
        :return: a loss
        """
        masked_output = output * mask
        masked_target = target * mask[:, 0, :, :]

        if torch.cuda.is_available():
            masked_target = masked_target.type(torch.cuda.LongTensor)
        else:
            masked_target = masked_target.type(torch.LongTensor)

        loss = cross_entropy2d(masked_output, masked_target)
        return loss


class PG_Loss(torch.nn.Module):
    """
    Policy gradient loss: reward * prob
    """
    def __init__(self):
        super(PG_Loss, self).__init__()

    def forward(self, reward, log_prob):
        """
        calculate the loss
        :param reward: reward from the global discriminator
        :param log_prob: logarithm of the probability maps from the generator
        :return: a loss
        """
        loss = torch.bmm(reward.squeeze(1), log_prob.squeeze(1))
        loss = -torch.mean(loss.view(loss.size(0), -1))
        return loss