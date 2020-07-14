import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):

    def __init__(self, max_batch_size, scale=16, device="cuda"):
        super(SimCLRLoss, self).__init__()
        self.scale = float(scale)
        self.batch_sample_index = []
        max_batch_size = max_batch_size * 2
        for b_id in range(max_batch_size):
            sample_index = torch.arange(max_batch_size, device=device)
            sample_index = torch.cat((sample_index[:b_id], sample_index[b_id+1:]))
            self.batch_sample_index.append(sample_index)
        self.batch_sample_index = torch.stack(self.batch_sample_index)

    def forward(self, feature, label, rand_idx=None):
        # feature is l2 normalized
        B2, D = feature.size()
        sample_index = self.batch_sample_index[:B2][:, :B2-1]
        # if rand_idx is not None:
            # sample_index = sample_index[rand_idx]
            # print(self.batch_sample_index)
        # print(sample_index)
        feature = F.normalize(feature, dim=1)

        logits_mat = torch.mm(feature, feature.T)
        # print(logits_mat)

        logits_mat = torch.gather(logits_mat, dim=1, index=sample_index) * self.scale
        # print(logits_mat)
        # print(label)
        # print(logits_mat[torch.arange(len(logits_mat)), label])
        # print(label.max())
        # print(logits_mat.shape)
        cls_loss = F.cross_entropy(logits_mat, label, reduction="mean")

        return cls_loss, logits_mat
