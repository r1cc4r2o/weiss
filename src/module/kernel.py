# This file contains the code for the IMQ kernel used in the MMD loss.
# 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 are the scales used in the kernel.

import torch

def imq_kernel(z1, z2):
    batch_size = z1.size(0)
    dists_z1 = ((z1[:, None] - z1[None]) ** 2).sum(-1)
    dists_z2 = ((z2[:, None] - z2[None]) ** 2).sum(-1)
    dists_z1_z2 = ((z1[:, None] - z2[None]) ** 2).sum(-1)
    I_matrix = torch.eye(batch_size).to(z1)
    stats = 0.0
    z_dim = z1.size(-1)
    for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        C = 2 * z_dim * 1.0 * scale
        res1 = C / (C + dists_z1) + C / (C + dists_z2)
        res1 = (1.0 - I_matrix) * res1
        res1 = res1.sum() / (batch_size - 1)

        res2 = C / (C + dists_z1_z2)
        res2 = res2.sum() * 2.0 / (batch_size)
        stats += res1 - res2
    return stats / batch_size
