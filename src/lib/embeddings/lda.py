import torch
import numpy as np


def LDAloss(H, label, lamb=0.1, epsilon=1, need_v=False):
    if len(H.shape) == 2:
        H = H[:, :, None, None]

    N, C, d, _ = H.shape
    N_new = N * d * d
    H = H.permute(0, 2, 3, 1)
    H = torch.reshape(H, (N_new, C))
    H_bar = H - torch.mean(H, 0, True)
    # print(H_bar.shape, "H_bar shape")
    label = label.view(N, 1)
    unique_labels = torch.unique(label)
    labels = torch.reshape(label * torch.Tensor().new_ones((N, d * d),
                                                           device=H.device, dtype=torch.long), (N_new,))
    S_w = torch.Tensor().new_zeros((C, C), device=H.device, dtype=H.dtype)
    S_t = H_bar.t().matmul(H_bar) / (N_new - 1)
    for i in unique_labels:
        H_i = H[torch.nonzero(labels == i).view(-1)]

        H_i_bar = H_i - torch.mean(H_i, 0, True)
        N_i = H_i.shape[0]
        if N_i == 0:
            continue
        S_w += H_i_bar.t().matmul(H_i_bar) / (N_i - 1) / len(unique_labels)
    S_w_reg = (S_w + lamb * torch.diag(torch.Tensor().new_ones((C),
                                                               device=H.device, dtype=H.dtype)))
    # print(S_w_reg.shape, H.shape)
    # lamb, v = np.linalg.eigh(S_t - 0.5 * S_w)
    temp = S_w_reg.pinverse()
    temp = temp.matmul(S_t - 0.5 * S_w)
    w, v = torch.symeig(temp, eigenvectors=True)

    w = w.detach()
    v = v.detach()
    # print(w)
    # sfda

    index = len(unique_labels) - 2

    # mask = w <= w[-index]
    pick_v_large_mask = w >= w[-index-1]
    pick_cls_v = v[:, torch.nonzero(pick_v_large_mask).view(-1)]
    if need_v:
        return pick_cls_v

    pick_v_large_mask = w >= w[-index]
    pick_v_small_mask = w <= w[-index] + epsilon
    mask = (pick_v_large_mask * pick_v_small_mask)

    pick_v = v[:, torch.nonzero(mask).view(-1)]
    loss = (pick_v.t().matmul(temp).matmul(pick_v)).sum()
    loss = loss / v.shape[1]
    # return -loss / (index+1), w
    # loss_w = w[mask]
    # loss_w = torch.mean(loss_w)

    return -loss, w, pick_cls_v


def lda_prediction(train_h, train_l, test_h, test_l, v):
    if len(train_h.shape) == 2:
        train_h = train_h[:, :, None, None]

    N, C, d, _ = train_h.shape
    N_new = N * d * d
    train_h = train_h.permute(0, 2, 3, 1)
    train_h = torch.reshape(train_h, (N_new, C))

    # H_bar = H - torch.mean(H, 0, True)
    train_l = train_l.view(N, 1)
    unique_labels = torch.unique(train_l)
    train_labels = torch.reshape(
        train_l * torch.Tensor().new_ones((N, d * d),
                                          device=train_h.device,
                                          dtype=torch.long), (N_new,)
    )

    all_mean_vectors = []
    for i in unique_labels:
        train_h_i = train_h[torch.nonzero(train_labels == i).view(-1)]
        class_mean_h = torch.mean(train_h_i, 0, True)
        all_mean_vectors.append(class_mean_h)

    all_mean_vectors = torch.cat(all_mean_vectors, dim=0)
    index = len(unique_labels) - 1

    pick_v = v

    # (C, C-1) = (C, d) * (d, C-1)
    hyperplane_vectors = torch.mm(all_mean_vectors, pick_v)
    # (C, d) = (C, C-1) * (C-1, d)
    hyperplane_vectors = torch.mm(hyperplane_vectors, pick_v.T)
    # print(hyperplane_vectors.shape)

    bias = all_mean_vectors * hyperplane_vectors
    bias = torch.sum(bias, dim=1)

    logit = torch.mm(test_h, hyperplane_vectors.T)
    logit = logit - bias

    # logit = torch.sigmoid(logit)
    # logit = logit / torch.sum(logit, dim=1, keepdim=True)

    return logit


def lda_prediction_main(train_feats, train_labels, test_feats=None, test_labels=None, need_v=False, lamb=0.001):
    if test_feats is None:
        splitted_train_feats, new_train_labels, splitted_test_feats, new_test_labels = split_and_get_data(
            train_feats=train_feats,
            train_labels=train_labels
        )

        train_v = LDAloss(
            H=splitted_train_feats,
            label=new_train_labels,
            lamb=lamb,
            need_v=True
        )

        logit1 = lda_prediction(
            train_h=splitted_train_feats,
            train_l=new_train_labels,
            test_h=splitted_test_feats,
            test_l=new_test_labels,
            v=train_v
        )

        train_v = LDAloss(
            H=splitted_test_feats,
            label=new_test_labels,
            lamb=lamb,
            need_v=True
        )

        logit2 = lda_prediction(
            train_h=splitted_test_feats,
            train_l=new_test_labels,
            test_h=splitted_train_feats,
            test_l=new_train_labels,
            v=train_v
        )

        logit = torch.cat(
            [logit1, logit2]
        )
        labels = torch.cat(
            [new_test_labels, new_train_labels],
            dim=0
        )

    else:
        v = LDAloss(
            H=train_feats,
            label=train_labels,
            lamb=lamb,
            need_v=True
        )
        logit = lda_prediction(
            train_h=train_feats,
            train_l=train_labels,
            test_h=test_feats,
            test_l=test_labels,
            v=v
        )
        labels = test_labels

    if need_v:
        return logit, labels, v

    else:
        return logit, labels


def split_and_get_data(train_feats, train_labels):
    unique_labels = torch.unique(train_labels)
    splitted_train_feats = []
    splitted_test_feats = []

    new_train_labels = []
    new_test_labels = []
    for i in unique_labels:
        mask = train_labels == i
        pick_feats = train_feats[mask]

        split_index = len(pick_feats)//2
        splitted_train_feats.append(pick_feats[:split_index])
        splitted_test_feats.append(pick_feats[split_index:])

        new_train_labels.append(
            train_labels[mask][:split_index]
        )
        new_test_labels.append(
            train_labels[mask][split_index:]
        )

    splitted_train_feats = torch.cat(splitted_train_feats, dim=0)
    splitted_test_feats = torch.cat(splitted_test_feats, dim=0)

    new_train_labels = torch.cat(new_train_labels, dim=0)
    new_test_labels = torch.cat(new_test_labels, dim=0)

    train_random_indices = torch.randperm(splitted_train_feats.shape[0])

    splitted_train_feats = splitted_train_feats[train_random_indices]
    new_train_labels = new_train_labels[train_random_indices]

    test_random_indices = torch.randperm(splitted_test_feats.shape[0])

    splitted_test_feats = splitted_test_feats[test_random_indices]
    new_test_labels = new_test_labels[test_random_indices]

    return splitted_train_feats, new_train_labels, splitted_test_feats, new_test_labels


def lda_for_episode(support_vector, query_vector, lamb=0.001, method="naive"):
    """
    support_vector: torch.Tensor, (num_class, num_support, D)
    query_vector: torch.Tensor, (num_class*num_query, D)
    """
    n_class, n_support, D = support_vector.shape

    class_labels = torch.arange(n_class,
                                device=support_vector.device,
                                dtype=support_vector.dtype
                                )
    class_labels = class_labels[:, None]
    class_labels = class_labels.repeat(1, n_support).reshape(-1)

    support_vector = support_vector.reshape(n_class*n_support, D)

    lda_loss, w, pick_v = LDAloss(
        H=support_vector,
        label=class_labels,
        lamb=lamb
    )

    logit = lda_prediction(
        train_h=support_vector,
        train_l=class_labels,
        test_h=query_vector,
        test_l=None,
        v=pick_v
    )

    return lda_loss, logit, pick_v
