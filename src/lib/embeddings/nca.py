import numpy as np
import torch
import torch.nn as nn
import lib.lossfunction.metric as metric


class NCA(nn.Module):

    def __init__(self, input_dim, output_dim=2, init_method="random", max_batch_size=128, scale=1, device="cpu", distance_method="euclidean"):
        super(NCA, self).__init__()
        self.init_method = init_method
        self.scale = float(scale)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device(device)
        self.euclidean = euclidean

        self._init_transformation()
        self._init_indices(max_batch_size=max_batch_size)

    def _init_indices(self, max_batch_size):
        self.batch_sample_index = []
#         max_batch_size = max_batch_size * 2
        for b_id in range(max_batch_size):
            sample_index = torch.arange(max_batch_size, device=self.device)
            sample_index = torch.cat(
                (sample_index[:b_id], sample_index[b_id+1:]))
            self.batch_sample_index.append(sample_index)
        self.batch_sample_index = torch.stack(self.batch_sample_index)

    def _init_transformation(self):
        """Initialize the linear transformation A.
        """
        if self.input_dim is None:
            self.input_dim = self.output_dim
        if self.init_method == "random":
            print('using random init')
            a = torch.randn(self.input_dim, self.output_dim,
                            device=self.device) / np.sqrt(self.input_dim)
            self.A = torch.nn.Parameter(a)
        elif self.init_method == "identity":
            a = torch.eye(self.input_dim, self.output_dim, device=self.device)
            self.A = torch.nn.Parameter(a)
        else:
            raise ValueError(
                "[!] {} initialization is not supported.".format(self.init))

    def forward(self, x, y, check=False):
        this_batch_size = x.shape[0]
        transformed_x = torch.mm(x, self.A)
#         print(transformed_x)

        logits_mat = metric.calc_l2_dist_torch(
            feature1=transformed_x,
            feature2=transformed_x,
            dim=1,
            is_sqrt=False,
            is_neg=True
        )
        sample_indices = self.batch_sample_index[:
                                                 this_batch_size][:, :this_batch_size-1]

        logits_mat = torch.gather(
            logits_mat, dim=1, index=sample_indices) * self.scale


#         logits_mat_exp = logits_mat.exp()
#         prob = logits_mat_exp / torch.sum(logits_mat_exp, dim=1, keepdim=True)

#         print(prob)
        logits_mat_stable = logits_mat - \
            torch.max(logits_mat, dim=1, keepdim=True)[0]
        logits_mat_exp = logits_mat_stable.exp()
        p_ij = logits_mat_exp / torch.sum(logits_mat_exp, dim=1, keepdim=True)

        y_mask = y[:, None] == y[None, :]
        y_mask = torch.gather(
            y_mask,
            dim=1,
            index=sample_indices
        )
#         print(y_mask)
#         print(p_ij)
        p_ij_mask = p_ij * y_mask.float()
        p_i = p_ij_mask.sum(dim=1)

#         print(p_i)
        p_i = p_i.clamp(min=1e-5, max=1-1e-5)
        classification_loss = - torch.log(p_i).mean()

        return classification_loss, transformed_x

    def transform(self, x):
        return torch.mm(x, self.A)


class NCATrainer:

    def __init__(self, input_dim, output_dim=2, is_instanciate_each_iter=True, init_method="random", max_batch_size=128, scale=1, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_instanciate_each_iter = is_instanciate_each_iter

        if not is_instanciate_each_iter:
            self.nca = NCA(
                input_dim=input_dim,
                output_dim=output_dim,
                init_method=init_method,
                max_batch_size=max_batch_size,
                scale=scale,
                device=device
            )

    def __call__(self, support_vector, query_vector, query_label, init_method="random", max_batch_size=128, distance_method="euclidean", lr=0.01, max_iter=50, stop_diff=1e-4, scale=1.):
        """
        support_vectors: torch.Tensor, (num_class, num_support, D)
        query_vectors: torch.Tensor, (num_class*num_query, D)
        """

        if self.is_instanciate_each_iter:
            num_class, num_support, D = support_vector.shape
            label_for_support = torch.arange(num_class).reshape(-1, 1)
            label_for_support = label_for_support.repeat(
                1, num_support).reshape(-1)

            support_vector = support_vector.reshape(-1, D)

            nca = NCA(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                init_method=init_method,
                max_batch_size=max_batch_size,
                scale=scale,
                device=support_vector.device,
                distance_method=distance_method
            )
            for i in range(max_iter):
                loss, _ = nca(support_vector, label_for_support, check=True)
                gradients = torch.autograd.grad(
                    loss, nca.A, create_graph=True)[0]

                prev_nca_A = nca.A.data

                nca.A.data = nca.A - gradients * lr
                diff = torch.abs(prev_nca_A - nca.A.data)

                if diff <= stop_diff:
                    break

            transformed_logit = nca.transform(
                x=query_vector
            )

        else:
            assert len(query_vector.shape) == 2, query_vector.shape
            assert len(query_label.shape) == 1, query_label.shape
            loss, transformed_logit = self.nca(
                query_vector, query_label, check=True)

        return loss, transformed_logit
