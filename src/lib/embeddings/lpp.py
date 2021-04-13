import torch
import torch.nn as nn
import numpy as np
import scipy


class LPPTorch(nn.Module):

    def __init__(self):
        super(LPPTorch, self).__init__()
        self.components_ = None

    def euclidean(self, x, y, is_sqrt=False):
        # (Qn, D), (Sn, D)
        XX = torch.sum(x*x, dim=1, keepdim=True)
        XY = torch.mm(x, y.T)
        YY = torch.sum(y*y, dim=1, keepdim=True).T
        # print(XX.shape, XY.shape, YY.shape, x.shape)

        dist = XX - 2 * XY + YY
        dist = dist.clamp(min=0)
        if is_sqrt:
            dist = torch.sqrt(dist)

        return dist

    def gaussian(self, X, Y, sigma=10):
        euc_dist = self.euclidean(X, Y, is_sqrt=True)
        dist = torch.exp(-euc_dist / sigma)

        return dist

    def fit(self, X, distance="", n_components=2, reg=1e-4):
        X = X - torch.mean(X, dim=0, keepdim=True)
        W = self.gaussian(X, X)
        D = self.get_degree_matrix(W)
        L = D - W
        A = X.T @ L @ X
        B = X.T @ D @ X
        # print(B)
#         print(D)
        # print(A)
#         print(L)
#         print(np.linalg.eigvals(A))

        indices = torch.arange(A.shape[0], device=A.device)

#         print(A)
#         A_ = A[None, :]
#         B_ = B[None, :]
        succeed = False

        while not succeed:
            A_ = A.detach().clone()
            B_ = B.detach().clone()
            A_[indices, indices] += reg
            B_[indices, indices] += reg
            try:
                eig_val, eig_vec = scipy.linalg.eigh(A.detach().cpu().numpy(), B.detach().cpu().numpy())
                succeed = True
            except np.linalg.LinAlgError:
                succeed = False
            reg *= 10

            if reg >= 10:
                raise np.linalg.LinAlgError(reg)
        eig_val = torch.from_numpy(eig_val).type(A.dtype).to(A.device)
        eig_vec = torch.from_numpy(eig_vec).type(A.dtype).to(A.device)

        index = torch.argsort(eig_val)
        eig_vec = eig_vec[:, index]

        components = eig_vec / \
            torch.sqrt(torch.sum(eig_vec*eig_vec, dim=0, keepdim=True))

        self.components_ = components[:, :n_components]

        embedding = X @ self.components_
#         embedding2 = np.dot(X, self.components_)
#         print(np.all(np.abs(embedding - embedding2)<=1e-6))

        return embedding

    def transform(self, X):
        assert self.components_ is not None

        embedding = X @ self.components_
        return embedding

    def forward(self, support_vector, query_vector, method="naive", n_components=2):
        if method == "naive":
            concat_data = torch.cat((support_vector, query_vector))
            embedding = self.fit(concat_data)

            support_num = support_vector.shape[0]
            support_embedding = embedding[:support_num]
            query_embedding = embedding[support_num:]

        elif method == "supervise_only":
            support_embedding = self.fit(
                support_vector, n_components=n_components)
            query_embedding = self.transform(query_vector)

        else:
            raise NotImplementedError

        return support_embedding, query_embedding

    def get_degree_matrix(self, W):
        diag_elements = torch.sum(W, dim=1)
        matrix = torch.diag(diag_elements)
        return matrix
