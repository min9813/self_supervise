import torch
import numpy as np


def mseloss(y_hat, y):
    diff = y_hat - y
    # loss = torch.sum(torch.mean(diff * diff, dim=0))
    loss = torch.mean(diff * diff)
    return loss


def calc_gaussian_prob_rate(x, mean_v, sigma_v, label):
    dist_mat = calc_mahalanobis_torch(x, mean_v, sigma_v)
    gt_logit = dist_mat[torch.arange(len(label)), label][:, None]
    logit_mat = gt_logit - dist_mat
    logit_mat = logit_mat.exp()
    ratio = sigma_v[[0]] / sigma_v
    ratio = ratio.sqrt().prod(dim=1)[None, :]
    logit_mat = ratio * logit_mat
    logit_mat = logit_mat.sum(dim=1)
    loss = logit_mat.log().mean()
    return loss
    


def calc_mahalanobis_numpy(feature1, feature2, sigma):
    XX = feature1 * feature1
    XXs = XX[:, None, :] / sigma[None, :, :]
    XXs = np.sum(XXs, axis=2)
    XY = np.dot(feature1, (feature2 / sigma).T)
    YY = (feature2 * feature2 / sigma).sum(axis=1)
    dist = XXs - 2*XY + YY
    return dist


def calc_mahalanobis_torch(feature1, feature2, sigma):
    XX = feature1 * feature1
    XXs = XX[:, None, :] / sigma[None, :, :]
    XXs = torch.sum(XXs, dim=2)
    XY = torch.mm(feature1, (feature2 / sigma).T)
    YY = (feature2 * feature2 / sigma).sum(dim=1)
    dist = XXs - 2*XY + YY
    return dist


def calc_mahalanobis_naive(features1, features2, sigma):
    dist_mat = np.zeros((len(features1), len(features2)))
    for idx, feat1 in enumerate(features1):
        for idx2, feat2 in enumerate(features2):
            diff = feat1 - feat2
            dist = (diff * diff / sigma[idx2]).sum()
            dist_mat[idx, idx2] = dist
    return dist_mat


if __name__ == "__main__":
    x = np.array([[1, 2.], [1, 0.]])
    y = np.array([[1, 4], [0, 1.], [2, 2]])
    sigma = np.array([[1, 3], [1, 2.], [2, 4.]])

    x_t = torch.Tensor(x)
    y_t = torch.Tensor(y)
    s_t = torch.Tensor(sigma)
    print(calc_mahalanobis_torch(x, y, sigma),
          calc_mahalanobis_naive(x, y, sigma))
