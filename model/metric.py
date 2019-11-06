import torch


def accuracy(output, target):
    with torch.no_grad():
        output = output[:, 0, :, :]
        pred = torch.argmax(output, dim=2)
        pred = torch.argmax(pred, dim=3)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target) * pred.size(-1) ** 2


