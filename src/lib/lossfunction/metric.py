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


def calc_cossim_dist_numpy(feature1, feature2):
    feature1 = feature1 / \
        np.sqrt(np.sum(feature1 * feature1, axis=1, keepdims=True))
    feature2 = feature2 / \
        np.sqrt(np.sum(feature2 * feature2, axis=1, keepdims=True))
    distance_mat = np.dot(feature1, feature2.T)
    return distance_mat


def calc_l2_dist_numpy(feature1, feature2):
    # (Qn, D), (Sn, D)
    XX = np.sum(feature1*feature1, axis=1, keepdims=True)
    XY = np.dot(feature1, feature2.T)
    YY = np.sum(feature2*feature2, axis=1, keepdims=True).T
    # print(XX.shape, XY.shape, YY.shape, feature1.shape)

    dist = XX - 2 * XY + YY
    dist = np.sqrt(dist)

    return -dist


def calc_l2_dist_torch(feature1, feature2, dim=1):
    XX = torch.sum(feature1*feature1, axis=dim, keepdims=True)
    if dim == 1:
        XY = torch.mm(feature1, feature2.T)
        YY = torch.sum(feature2*feature2, axis=dim, keepdims=True).T
    else:
        XY = torch.bmm(feature1, feature2.permute(0, 2, 1))
        YY = torch.sum(feature2*feature2, axis=dim, keepdims=True).permute(0, 2, 1)
    # print(XX.shape, XY.shape, YY.shape, feature1.shape)

    dist = XX - 2 * XY + YY
    dist = torch.sqrt(dist)
    print(dist.shape)
    
    return -dist


if __name__ == "__main__":
    x = np.array([[1, 2.], [1, 0.]])
    y = np.array([[1, 4], [0, 1.], [2, 2]])
    sigma = np.array([[1, 3], [1, 2.], [2, 4.]])

    x_t = torch.Tensor(x)
    y_t = torch.Tensor(y)
    s_t = torch.Tensor(sigma)
    print(calc_mahalanobis_torch(x, y, sigma),
          calc_mahalanobis_naive(x, y, sigma))
