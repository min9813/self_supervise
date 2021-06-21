import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import lib.lossfunction.metric as metric


class NCA(nn.Module):

    def __init__(self, input_dim, output_dim=2, init_method="random", max_batch_size=128, scale=1, device="cpu", distance_method="euclidean"):
        super(NCA, self).__init__()
        self.init_method = init_method
        self.scale = float(scale)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device(device)
        self.distance_method = distance_method

        if distance_method not in ("euclidean", ):
            raise NotImplementedError

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
            # print('using random init')
            a = torch.randn(self.input_dim, self.output_dim,
                            device=self.device) / np.sqrt(self.input_dim)
            self.A = torch.nn.Parameter(a)
        elif self.init_method == "identity":
            a = torch.eye(self.input_dim, self.output_dim, device=self.device)
            self.A = torch.nn.Parameter(a)

        elif self.init_method == "pca":
            self.A = None

        else:
            raise ValueError(
                "[!] {} initialization is not supported.".format(self.init))

    def _pca(self, x):
        torch.svd(x)

    def forward(self, x, y, check=False):
        this_batch_size = x.shape[0]

        if self.A is None and self.init_method == "pca":
            self._pca()

        transformed_x = torch.mm(x, self.A)
#         print(transformed_x)

        if self.distance_method == "euclidean":
            logits_mat = metric.calc_l2_dist_torch(
                feature1=transformed_x,
                feature2=transformed_x,
                dim=1,
                is_sqrt=False,
                is_neg=True
            )

        else:
            raise NotImplementedError

        sample_indices = self.batch_sample_index[:
                                                 this_batch_size][:, :this_batch_size-1]

        # print(logits_mat.shape, transformed_x.shape)
        # print(sample_indices)
        logits_mat = torch.gather(
            logits_mat, dim=1, index=sample_indices) * self.scale
        # print(logits_mat.shape)


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

    def __init__(self, input_dim, output_dim=2, is_instanciate_each_iter=True, init_method="random", max_batch_size=128, scale=1., device="cpu", distance_method="euclidean"):
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
                device=device,
                distance_method=distance_method
            )

        self.base_nca_state_dict = None

    def register_nca_state_dict(self, base_nca_state_dict):
        self.base_nca_state_dict = base_nca_state_dict

    def __call__(self, support_vector, query_vector, query_label, init_method="random", max_batch_size=128, distance_method="euclidean", lr=0.01, max_iter=50, stop_diff=1e-4, scale=1., get_feats=False, stop_criteria="l1norm"):
        """
        support_vectors: torch.Tensor, (num_class, num_support, D)
        query_vectors: torch.Tensor, (num_class*num_query, D)
        """

        if self.is_instanciate_each_iter:
            num_class, num_support, D = support_vector.shape
            # support_vector = support_vector.cuda()

            if "cuda" not in str(support_vector.device):
                # support_vector = support_vector.cuda()
                # query_vector = query_vector.cuda()

                # if query_label is not None:
                #     query_label = query_label.cuda()

                # turn_to_cpu = True
                turn_to_cpu = False

            else:
                turn_to_cpu = False

            label_for_support = torch.arange(num_class, device=support_vector.device).reshape(-1, 1)
            label_for_support = label_for_support.repeat(
                1, num_support).reshape(-1)

            support_vector = support_vector.reshape(-1, D)
            # print(support_vector.device)

            nca = NCA(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                init_method=init_method,
                max_batch_size=max_batch_size,
                scale=scale,
                device=support_vector.device,
                distance_method=distance_method
            )

            if self.base_nca_state_dict is not None:
                nca.load_state_dict(self.base_nca_state_dict)

            first_data = nca.A.data
            for i in range(max_iter):
                loss, _ = nca(support_vector, label_for_support, check=True)
                gradients = torch.autograd.grad(
                    loss, nca.A, create_graph=True)[0]

                prev_nca_A = nca.A.data

                nca.A.data = nca.A - gradients * lr
                diff = torch.abs(prev_nca_A - nca.A.data)
                
                if stop_criteria == "l1norm":
                    diff = diff.sum()
                elif stop_criteria == "l2norm":
                    diff = (diff * diff).sum().sqrt()
                else:
                    raise NotImplementedError(stop_criteria)
                # print("{} th iter diff: {}, loss={}".format(i, diff, loss))

                if diff <= stop_diff:
                    break
            # final_diff = torch.abs(nca.A.data - first_data)
            # final_diff = (final_diff * final_diff).sum().sqrt()
            # print("final diff:", final_diff, "max iter={}".format(max_iter))

            # with torch.no_grad():
            num_query = query_vector.shape[0]
            concated_vector = torch.cat((support_vector, query_vector), axis=0)
            transformed_logit = nca.transform(
                x=concated_vector
            )
            transformed_support = transformed_logit[:-num_query]
            transformed_query = transformed_logit[-num_query:]
            NB, D2 = transformed_query.shape

            transformed_support = transformed_support.reshape(num_class, num_support, D2)
            transformed_support_mean_feats = transformed_support.mean(dim=1)

            # print(transformed_query.shape)
            # print(transformed_support_mean_feats.shape)

            if distance_method == "euclidean":
                transformed_logit = metric.calc_l2_dist_torch(
                    transformed_query, transformed_support_mean_feats, dim=1
                )
            else:
                raise NotImplementedError

            if turn_to_cpu:
                transformed_logit = transformed_logit.cpu()
                transformed_query = transformed_query.cpu()
                transformed_support = transformed_support.cpu()
            # transformed_logit = torch.mm(transformed_query, transformed_support_mean_feats.permute(1, 0))

        else:
            if get_feats:
                loss = 0
                num_class, num_support, D = support_vector.shape
                support_vector = support_vector.reshape(-1, D)

                num_query = query_vector.shape[0]
                concated_vector = torch.cat((support_vector, query_vector), axis=0)
                transformed_logit = self.nca.transform(
                    x=concated_vector
                )
                transformed_support = transformed_logit[:-num_query]
                transformed_query = transformed_logit[-num_query:]
                NB, D2 = transformed_query.shape

                transformed_support = transformed_support.reshape(num_class, num_support, D2)
                transformed_support_mean_feats = transformed_support.mean(dim=1)

                # print(transformed_query.shape)
                # print(transformed_support_mean_feats.shape)

                if distance_method == "euclidean":
                    transformed_logit = metric.calc_l2_dist_torch(
                        transformed_query, transformed_support_mean_feats, dim=1
                    )
                else:
                    raise NotImplementedError

            else:
                assert len(query_vector.shape) == 2, query_vector.shape
                assert len(query_label.shape) == 1, query_label.shape
                loss, transformed_logit = self.nca(
                    query_vector, query_label, check=True)

        if get_feats:
            return loss, transformed_logit, transformed_support, transformed_query

        else:
            return loss, transformed_logit


def fit_nca_with_sklearn(features, labels, args):
    nca = NeighborhoodComponentsAnalysis(
        max_iter=args.MODEL.nca_max_iter, 
        tol=args.MODEL.nca_stop_diff, 
        n_components=args.MODEL.embedding_n_components,
        init="pca",
        verbose=1
        )

    nca.fit(
        X=features,
        y=labels
    )

    device = args.MODEL.trn_embedder.nca.A.device
    dtype = args.MODEL.trn_embedder.nca.A.dtype

    components = nca.components_.T
    components = torch.from_numpy(components.astype(np.float32)).type(dtype).to(device)

    args.MODEL.trn_embedder.nca.A.data = components
    args.MODEL.val_embedder.nca.A.data = components
