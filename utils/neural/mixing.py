import numpy as np
import torch


def mixup(data, targets, alpha=1.0):
    # when alpha = 1 then Beta(1, 1) = U(0, 1)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size()[0]
    if data.is_cuda is True:
        indices = torch.randperm(batch_size).cuda()
    else:
        indices = torch.randperm(batch_size)

    mixed_data = lam * data + (1 - lam) * data[indices, :]
    shuffled_targets = targets[indices]
    return mixed_data, targets, shuffled_targets, lam
