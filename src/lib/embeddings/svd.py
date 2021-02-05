import torch
import torch.nn as nn
# import lib.lossfunction.metric as metric


class SVDTorch(nn.Module):

    def __init__(self):
        super(SVDTorch, self).__init__()

    def fit_transform(self, x, D=None, n_components=2, method="naive"):
        U, D, V = self.fit(x)

        embedding = self.transform(x, D=D, V=V, n_components=n_components)

        return embedding

    def fit(self, x):
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True)

        x = (x - x_mean) / (x_std + 1e-4)

        U, D, V = torch.svd(
            x
        )
        # torch.mm(a, V[:, :2]) - torch.mm(U[:, :2], torch.diag(D[:2])) = 0

        return U, D, V

    def transform(self, x, D, V, n_components):
        sorted_indices = torch.argsort(-D.abs())
        pick_indices = sorted_indices[:n_components]

        V = V[:, pick_indices]
        embedding = torch.mm(x, V)

        return embedding

    def forward(self, support_vector, query_vector, method="naive", n_components=2):
        concat_data = torch.cat((support_vector, query_vector))
        if method == "naive":
            embedding = self.fit_transform(concat_data, method=method)

        elif "supervise_naive":
            U, D, V = self.fit(support_vector)
            embedding = self.transform(
                x=concat_data,
                D=D,
                V=V,
                n_components=n_components
            )

        support_num = support_vector.shape[0]
        support_embedding = embedding[:support_num]
        query_embedding = embedding[support_num:]

        return support_embedding, query_embedding


if __name__ == "__main__":
    svd = SVDTorch()
    x = torch.randn(10, 20)
    y = torch.randn(10, 20)
    out = svd(x, y, method="supervise_naive")
    print(out[0].shape, out[1].shape)
