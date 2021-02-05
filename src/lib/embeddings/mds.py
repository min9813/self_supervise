import torch
import torch.nn as nn
import numpy as np
try:
    import lib.lossfunction.metric as metric
except ImportError:
    import sys
    sys.path.append("../")
    from lossfunction import metric


class MDSNumpy:

    def __init__(self, metric_type="euclidean"):
        self.metric = metric_type

    def euclidean(self, x, y, is_sqrt=False):
        # (Qn, D), (Sn, D)
        XX = np.sum(x*x, axis=1, keepdims=True)
        XY = np.dot(x, y.T)
        YY = np.sum(y*y, axis=1, keepdims=True).T
        # print(XX.shape, XY.shape, YY.shape, x.shape)

        dist = XX - 2 * XY + YY
        dist = dist.clip(min=0)
        if is_sqrt:
            dist = np.sqrt(dist)

        return dist

    def fit_transform(self, x, D=None, n_components=2):
        if self.metric == "euclidean":
            D = self.euclidean(x, x)

        else:
            assert D is not None

        N, dim = x.shape
        H = np.eye(N) - np.full((N, N), 1) / N
        DH = np.dot(D, H)
        K = - 0.5 * np.dot(H, DH)
        # print(K - K.T)
#         print(np.max(np.abs(K - K.T)))

#         U, S, V = np.linalg.svd(K)
        V, P = np.linalg.eig(K)
        # np.dot(P*V[None, :], P.T)- K = 0
        V = V.real
        P = P.real

        sorted_indices = np.argsort(-np.abs(V))
        pick_indices = sorted_indices[:n_components]
        S = np.sqrt(V[pick_indices])
        # print(S)
        P = P.T[pick_indices, :]
#         print(S.shape, V.shape)
        X_hat = S[:, None] * P
        X_hat = X_hat.T

        return X_hat


class MDSTorch(nn.Module):

    def __init__(self, metric_type="euclidean"):
        super(MDSTorch, self).__init__()
        self.metric = metric_type

    def euclidean(self, x, y):
        dist = metric.calc_l2_dist_torch(
            feature1=x,
            feature2=y,
            dim=1,
            is_sqrt=False,
            is_neg=False
        )

        return dist

    def fit_transform(self, x, D=None, n_components=2, method="naive"):
        if D is None:
            if self.metric == "euclidean":
                D = self.euclidean(x, x)
            else:
                raise NotImplementedError

        else:
            assert D is not None

        N, dim = x.shape
        H = torch.eye(N, device=x.device, dtype=x.dtype) - \
            torch.full((N, N), 1, device=x.device, dtype=x.dtype) / N
        DH = torch.mm(D, H)
        K = - 0.5 * torch.mm(H, DH)
#         print(np.max(np.abs(K - K.T)))

#         U, S, V = np.linalg.svd(K)
        V, P = torch.symeig(K, eigenvectors=True)
        # torch.mm(P*V[None, :], P.T) - K = 0
        # print(V)
        # print(P)
        sorted_indices = torch.argsort(-V.abs())
        pick_indices = sorted_indices[:n_components]
        # V = V.real
        # P = P.real
        # print(P.shape)
        S = torch.sqrt(V[pick_indices])
        P = P.T[pick_indices, :]
#         print(S.shape, V.shape)
        X_hat = S[:, None] * P
        X_hat = X_hat.T

        return X_hat

    def forward(self, support_vector, query_vector, method="naive"):
        concat_data = torch.cat((support_vector, query_vector))
        embedding = self.fit_transform(concat_data, method=method)

        support_num = support_vector.shape[0]
        support_embedding = embedding[:support_num]
        query_embedding = embedding[support_num:]

        return support_embedding, query_embedding


if __name__ == "__main__":
    mds_torch = MDSTorch()
    mds_numpy = MDSNumpy()
    x = torch.randn(10, 20, requires_grad=True)
    out1 = mds_torch.fit_transform(x)
    print(out1)
    # out2 = mds_numpy.fit_transform(x.numpy())
