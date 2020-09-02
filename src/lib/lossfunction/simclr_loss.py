import torch
import torch.nn as nn
import torch.nn.functional as F
from .metric import calc_mahalanobis_torch


class SimCLRLoss(nn.Module):

    def __init__(self, max_batch_size, scale=16, device="cuda", is_gaussian_prob=False):
        super(SimCLRLoss, self).__init__()
        self.scale = float(scale)
        self.batch_sample_index = []
        self.gaussian_prob = is_gaussian_prob
        max_batch_size = max_batch_size * 2
        for b_id in range(max_batch_size):
            sample_index = torch.arange(max_batch_size, device=device)
            sample_index = torch.cat(
                (sample_index[:b_id], sample_index[b_id+1:]))
            self.batch_sample_index.append(sample_index)
        self.batch_sample_index = torch.stack(self.batch_sample_index)

    def forward(self, feature, label, rand_idx=None):
        # feature is l2 normalized
        B2, D = feature.size()
        sample_index = self.batch_sample_index[:B2][:, :B2-1]
        # print(sample_index.size(), feature.size())
        # if rand_idx is not None:
        # sample_index = sample_index[rand_idx]
        # print(self.batch_sample_index)
        # print(sample_index)
        feature = F.normalize(feature, dim=1)

        logits_mat = torch.mm(feature, feature.T)
        # print(logits_mat)

        logits_mat = torch.gather(
            logits_mat, dim=1, index=sample_index) * self.scale
        # print(logits_mat)
        # print(label)
        # print(logits_mat[torch.arange(len(logits_mat)), label])
        # print(label.max())
        # print(logits_mat.shape)
        cls_loss = F.cross_entropy(logits_mat, label, reduction="mean")

        return cls_loss, logits_mat


class SimCLRLossV(nn.Module):

    def __init__(self, max_batch_size, scale=16, device="cuda", feature_dim=512):
        super(SimCLRLossV, self).__init__()
        self.scale = float(scale)
        self.batch_sample_index = []
        max_batch_size = max_batch_size * 2
        for b_id in range(max_batch_size):
            sample_index = torch.arange(max_batch_size, device=device)
            sample_index = torch.cat(
                (sample_index[:b_id], sample_index[b_id+1:]))
            self.batch_sample_index.append(sample_index)
        self.batch_sample_index = torch.stack(self.batch_sample_index)
        self.batch_sample_index2 = self.batch_sample_index[:, :, None].repeat(
            1, 1, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, feature, sigma_v, label, rand_idx=None):

        cls_loss, logits_mat = self.calc_gaussian_prob_rate_simclr(
            feature, sigma_v, label)
        return cls_loss, logits_mat

    def calc_gaussian_prob_rate_simclr(self, mean_v, sigma_v, label):
        B2, D = mean_v.size()
        # print(mean_v)
        assert self.feature_dim == D, (self.feature_dim, D)
        gather_index = self.batch_sample_index[:B2][:, :B2-1]
        gather_index2 = self.batch_sample_index2[:B2][:, :B2-1]

        # exp((x-mc)^T Sc^-1 (x-mc) - (x-mi)^T Si^-1 (x-mi))
        dist_mat = calc_mahalanobis_torch(mean_v, mean_v, sigma_v)
        dist_mat = torch.gather(dist_mat, dim=1, index=gather_index)
        gt_logit = dist_mat[torch.arange(len(label)), label][:, None]
        logit_mat = gt_logit - dist_mat
        logit_mat *= 0.1
        # print(logit_mat.max(), logit_mat.min())
        logit_mat = logit_mat.exp()
        
        N = sigma_v.size(0)
        s = sigma_v.unsqueeze(0).repeat(N, 1, 1)

        s = s.gather(dim=1, index=gather_index2)
        s = s[torch.arange(len(label)), label]
        
        # det(S_c) / det(S_i)
        ratio = sigma_v[:, None, :] / s
        ratio = ratio.sqrt().prod(dim=2)
        ratio = ratio.gather(dim=1, index=gather_index)
        
        logit_mat = ratio * logit_mat
        # print(logit_mat)
        # print(logit_mat[torch.arange(len(logit_mat)), label])

        loss = logit_mat.sum(dim=1)
        loss = loss.log().mean()
        return loss, logit_mat
