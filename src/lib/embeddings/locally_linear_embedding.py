import os
import inspect
import torch
import torch.nn as nn
import numpy as np
try:
    import lib.lossfunction.metric as metric
except ImportError:
    import sys
    sys.path.append("../")
    import lossfunction.metric as metric


def location():
    frame = inspect.currentframe().f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno


def barycenter_weights_torch(X, Y, indices, reg=1e-3):
    #     n_samples, n_neighbors = indices.shape
    n_samples = X.shape[0]
    weight_matrix = torch.zeros(
        (n_samples, n_samples), device=X.device, dtype=X.dtype)
#     weight_matrix = []

    for data_index, neighb_indice in enumerate(indices):
        x_neighbors = Y[neighb_indice]

#         v = np.ones(n_neighbors)

        res_to_x = x_neighbors - X[[data_index]]
        # res_to_x = (num_neighbor, dim)

        cov_matrix = torch.mm(res_to_x, res_to_x.T)
        trace = torch.trace(cov_matrix)
        if trace > 0:
            R = reg * trace

        else:
            R = reg

#         cov_matrix.flat[::n_neighbors+1] += R
        eye_indices = torch.arange(cov_matrix.shape[0])
        cov_matrix[eye_indices, eye_indices] += R
#         weight = np.linalg.inv(cov_matrix)
        weight = torch.inverse(cov_matrix)
        weight = torch.sum(weight, dim=1)
#         weight = np.sum(weight, axis=1)

        weight = weight / torch.sum(weight)

        weight_matrix[data_index, neighb_indice] = weight

    return weight_matrix


def barycenter_weights_numpy(X, Y, indices, reg=1e-3):
    n_samples, n_neighbors = indices.shape
    weight_matrix = np.zeros((n_samples, n_samples))

    v = np.ones(n_neighbors)

    for data_index, neighb_indice in enumerate(indices):
        x_neighbors = Y[neighb_indice]
        res_to_x = x_neighbors - X[[data_index]]

        cov_matrix = np.dot(res_to_x, res_to_x.T)
        trace = np.trace(cov_matrix)
        if trace > 0:
            R = reg * trace

        else:
            R = reg

        cov_matrix.flat[::n_neighbors+1] += R
        weight = np.linalg.inv(cov_matrix)
        weight = np.sum(weight, axis=1)

        weight = weight / np.sum(weight)

        weight_matrix[data_index, neighb_indice] = weight

    return weight_matrix


def naive_neighbor_graph(data, n_neighbors, reg=1e-3):
    dist_mat = metric.calc_l2_dist_torch(data, data, is_neg=False)
    neighb_indices = torch.argsort(dist_mat, dim=1)

    neighb_indices = neighb_indices[:, 1:n_neighbors+1]

    return neighb_indices


def supervise_naive_graph(n_class, n_support, n_query, reg=1e-3, device="cpu"):
    """
    data: [batch_size, dimension]
    """
#     labeled = data[:n_support*n_class]
#     unlabeld = data[n_support*n_class:]

#     assert len(unlan_beld) == (n_class * n_query)
    num_data = n_class * (n_support + n_query)
    neighb_indices_base = torch.arange(n_support)
#     neighb_indices = torch.zeros(data.shape[0], n_support-1,  device=data.device, dtype=data.dtype)

    one_class_neighbors = torch.zeros(
        n_support, n_support-1,  device=device, dtype=torch.long)
    for i in range(n_support):
        indices1 = neighb_indices_base[:i]
        indices2 = neighb_indices_base[i+1:]

        indices = torch.cat([indices1, indices2])
        one_class_neighbors[i, :n_support-1] = indices

    all_class_neighbors = [one_class_neighbors]
    for i in range(1, n_class):
        each_class_neighbors = one_class_neighbors + n_support * i
        all_class_neighbors.append(each_class_neighbors)

    all_class_neighbors = torch.cat(all_class_neighbors, dim=0)

    label_num = all_class_neighbors.shape[0]
    unlabel_num = num_data - label_num
    label_to_unlabel_neighbors = torch.arange(label_num, num_data,
                                              device=all_class_neighbors.device,
                                              dtype=all_class_neighbors.dtype
                                              )
    label_to_unlabel_neighbors = label_to_unlabel_neighbors[None, :].repeat(
        label_num, 1)

    all_class_neighbors = torch.cat(
        (all_class_neighbors, label_to_unlabel_neighbors), dim=1)
    all_to_unlabel_neighbors = torch.arange(
        num_data, device=device, dtype=all_class_neighbors.dtype).unsqueeze(0)
    all_to_unlabel_neighbors = all_to_unlabel_neighbors.repeat(unlabel_num, 1)

    neighb_indices = []
    for i in range(num_data):
        if i < all_class_neighbors.shape[0]:
            neighb_indices.append(all_class_neighbors[i])

        else:
            neighb_indices.append(
                all_to_unlabel_neighbors[i-all_class_neighbors.shape[0]])
#     all_class_neighbors = torch.cat((all_class_neighbors, all_to_unlabel_neighbors), dim=1)

    return neighb_indices


def lle(data, n_neighbors, n_class, n_support, n_query, method="naive", target_dim=2):
    weight_matrix = barycenter_weights_graph_torch(
        data=data,
        n_neighbors=n_neighbors,
        n_class=n_class,
        n_support=n_support,
        n_query=n_query,
        method=method
    )

    device = weight_matrix.device
    dtype = weight_matrix.dtype

    M = torch.eye(weight_matrix.shape[0],
                  device=device, dtype=dtype) - weight_matrix
    M = torch.mm(M.T, M)

    D, V = torch.eig(M, eigenvectors=True)
    D = D[:, 0]
    indices = torch.argsort(D.abs())

    pick_v = V[:, indices[1:target_dim+1]]

    return pick_v


class LLE(nn.Module):

    def __init__(self, n_neighbors, n_class, n_support, n_query, device):
        super(LLE, self).__init__()
        self.n_class = n_class
        self.n_support = n_support
        self.n_query = n_query
        self.n_neighbors = n_neighbors
        self.neighb_indices = supervise_naive_graph(
            n_class=self.n_class,
            n_support=self.n_support,
            n_query=self.n_query,
            device=device
        )

    def barycenter_weights_graph_torch(self, data, n_neighbors, method="naive"):
        if method == "naive":
            neighb_indices = naive_neighbor_graph(
                data, n_neighbors=n_neighbors)
        elif method == "supervise_naive":
            neighb_indices = self.neighb_indices
            if data.device != self.neighb_indices.device:
                neighb_indices = neighb_indices.to(data.device)

        output_torch = barycenter_weights_torch(data, data, neighb_indices)

        return output_torch

    def lle(self, data, method="naive", target_dim=2):
        weight_matrix = self.barycenter_weights_graph_torch(
            data=data,
            n_neighbors=self.n_neighbors,
            method=method
        )

        device = weight_matrix.device
        dtype = weight_matrix.dtype

        M = torch.eye(weight_matrix.shape[0],
                      device=device, dtype=dtype) - weight_matrix
        M = torch.mm(M.T, M)

        # print("start eig", M.shape, location())
        D, V = torch.symeig(M, eigenvectors=True)
        # print("end eig")
        # D = D[:, 0
        indices = torch.argsort(D.abs())
        # print("finish eig value:", location())

        pick_v = V[:, indices[1:target_dim+1]]

        return pick_v

    def forward(self, support_vector, query_vector, method="naive"):
        concat_data = torch.cat((support_vector, query_vector))
        embedding = self.lle(concat_data, method=method)

        support_num = support_vector.shape[0]
        support_embedding = embedding[:support_num]
        query_embedding = embedding[support_num:]

        return support_embedding, query_embedding


if __name__ == "__main__":
    n_neighbors = 3
    n_class = 3
    n_support = 5
    n_query = 5
    lle = LLE(
        n_neighbors=n_neighbors,
        n_query=n_query,
        n_class=n_class,
        n_support=n_support,
        device="cpu"
    )

    data = torch.randn(n_support, 3)

    data = [data]
    for class_i in range(1, n_class):
        data_i = data[0] + 5 * class_i
        data.append(data_i)

    data = torch.cat(data, dim=0)

    unlabel_data = data + torch.Tensor([[5, -5, 5]])

    support_embedding, query_embedding = lle(
        support_vector=data,
        query_vector=unlabel_data,
        method="naive"
    )
    print(support_embedding.shape)
    print(query_embedding.shape)