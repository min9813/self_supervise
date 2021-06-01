import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import lib.lossfunction.metric as metric
    # import lib.embeddings.locally_linear_embedding as locally_linear_embedding
except ImportError:
    import sys
    sys.path.append("../")
    import lossfunction.metric as metric
    # import embeddings.locally_linear_embedding as locally_linear_embedding


class LossWrap(torch.nn.Module):
    def __init__(self, args, model, head, criterion, trn_embedding=None, val_embedding=None):
        self.args = args
        super(LossWrap, self).__init__()
        self.model = model
        self.head = head
        self.criterion = criterion
        self.trn_embedding = trn_embedding
        self.val_embedding = val_embedding

    def forward(self, input, label):
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        raw_logits = self.model(input)
        output = self.head(raw_logits)
        # print(label)

        loss = self.criterion(output, label)

        return loss, output


class LossWrapEpisode(LossWrap):

    def forward(self, input, label):
        # input = (B, n_class*n_sample, C, W, H)
        if self.training:
            self.n_way = self.args.TRAIN.n_way
            n_support = self.args.TRAIN.n_support
            n_query = self.args.TRAIN.n_query
            n_sample = n_support + n_query
        else:
            self.n_way = self.args.TEST.n_way
            n_support = self.args.TEST.n_support
            n_query = self.args.TEST.n_query
            n_sample = n_support + n_query

        # if self.training:
        B, CXN, C, W, H = input.size()
        input = input.reshape(-1, C, W, H)
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        shuffle_index = torch.randperm(B*CXN, device=input.device)
        shuffled_input = input[shuffle_index]
        raw_logits = self.model(shuffled_input)
        # raw_logits = self.model(input)
        # raw_logits = raw_logits[shuffle_index]
        raw_logits[shuffle_index] = raw_logits.clone()

        raw_logits = raw_logits.reshape(
            B, self.n_way, n_sample, -1)

        support_feats = raw_logits[:, :, :n_support]

        query_feats = raw_logits[:, :, n_support:]
        query_feats = query_feats.reshape(
            B, self.n_way*n_query, -1)
        support_mean_feats = torch.mean(
            support_feats, dim=2)  # (B, n_class, feat_dim)
        if self.args.MODEL.embedding_flag:
            support_feats_all = []
            query_feats_all = []
            for data_index in range(B):
                support_data = support_feats[data_index]
                support_mean_data = support_mean_feats[data_index]
                query_data = query_feats[data_index]

                support_embedding, query_embedding = self.embedding_vectors(
                    support_vectors=support_data,
                    support_mean_vectors=support_mean_data,
                    query_vectors=query_data
                )
                support_mean_embedding = torch.mean(support_embedding, dim=1)

                support_feats_all.append(support_mean_embedding)
                query_feats_all.append(query_embedding)

            support_mean_feats = torch.stack(support_feats_all, dim=0)
            query_feats = torch.stack(query_feats_all, dim=0)
        else:
            pass

        label_episode = torch.arange(self.n_way,
                                     dtype=torch.long, device=query_feats.device)
        label_episode = label_episode.view(-1,
                                           1)
        label_episode = label_episode.repeat(B, n_query).reshape(-1)

        if self.args.TRAIN.meta_mode == "cossim":
            logit = self.compute_cosine_similarity(
                support_mean_feats, query_feats)
        elif self.args.TRAIN.meta_mode == "euc":
            logit = metric.calc_l2_dist_torch(
                query_feats, support_mean_feats, dim=2
            )
            # logit = logit.permute(0, 2, 1)
            logit = logit.reshape(-1, self.n_way)
        else:
            raise NotImplementedError

        loss = self.criterion(logit, label_episode)

        return loss, logit, label_episode, raw_logits

    def compute_cosine_similarity(self, support_feats, query_feats):
        query_feats = F.normalize(query_feats, dim=2)
        support_feats = F.normalize(support_feats, dim=2)
        # print(query_feats.size(), support_feats.size())
        cossim = torch.bmm(query_feats, support_feats.permute(0, 2, 1))
        # assert cossim.max() <= 1.001, cossim.max()
        cossim = cossim * self.args.TRAIN.logit_scale
        # order in one batch = (C1, C1, C1, ..., C2, C2, C2, ...)
        # cossim = cossim.permute(0, 2, 1)
        # print(cossim.shape, query_feats.shape)
        cossim = cossim.reshape(-1, self.n_way)

        return cossim

    def embedding_vectors(self, support_vectors, support_mean_vectors, query_vectors, use_mean=False):
        """
        support_vectors: torch.Tensor, (num_class, num_support, D)
        query_vectors: torch.Tensor, (num_class*num_query, D)
        """

        n_class, n_support, D = support_vectors.shape
        if "lda" not in self.trn_embedding.__class__.__name__:
            support_vectors = support_vectors.reshape(n_class*n_support, D)

        if self.training:
            support_embedding, query_embedding = self.trn_embedding(
                support_vector=support_vectors,
                query_vector=query_vectors,
                method=self.args.MODEL.embedding_method
            )

        else:
            support_embedding, query_embedding = self.val_embedding(
                support_vector=support_vectors,
                query_vector=query_vectors,
                method=self.args.MODEL.embedding_method
            )

        support_embedding = support_embedding.reshape(n_class, n_support, -1)

        return support_embedding, query_embedding


class LossWrapEpisodeLDA(LossWrap):

    def forward(self, input, label):
        # input = (B, n_class*n_sample, C, W, H)
        if self.training:
            self.n_way = self.args.TRAIN.n_way
            n_support = self.args.TRAIN.n_support
            n_query = self.args.TRAIN.n_query
            n_sample = n_support + n_query
        else:
            self.n_way = self.args.TEST.n_way
            n_support = self.args.TEST.n_support
            n_query = self.args.TEST.n_query
            n_sample = n_support + n_query

        # if self.training:
        B, CXN, C, W, H = input.size()
        input = input.reshape(-1, C, W, H)
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        # shuffle_index = torch.randperm(B*CXN, device=input.device)
        # shuffled_input = input[shuffle_index]
        raw_logits = self.model(input)
        # print(torch.any(torch.isnan(raw_logits)))
        # raw_logits = self.model(input)
        # raw_logits = raw_logits[shuffle_index]
        # raw_logits[shuffle_index] = raw_logits.clone()

        raw_logits_2 = raw_logits.reshape(
            B, self.n_way, n_sample, -1)

        support_feats = raw_logits_2[:, :, :n_support]

        query_feats = raw_logits_2[:, :, n_support:]
        query_feats = query_feats.reshape(
            B, self.n_way*n_query, -1)
        support_mean_feats = torch.mean(
            support_feats, dim=2)  # (B, n_class, feat_dim)
        if self.args.MODEL.embedding_flag:
            # support_feats_all = []
            # query_feats_all = []
            query_logit = []
            lda_loss = 0
            for data_index in range(B):
                support_data = support_feats[data_index]
                support_mean_data = support_mean_feats[data_index]
                query_data = query_feats[data_index]

                try:
                    each_lda_loss, logit, pick_v = self.embedding_vectors(
                        support_vectors=support_data,
                        support_mean_vectors=support_mean_data,
                        query_vectors=query_data
                    )
                except RuntimeError:
                    is_support_nan = torch.any(torch.isnan(support_mean_data))
                    is_query_nan = torch.any(torch.isnan(query_data))
                    raise RuntimeError("is support nan:{} is query nan:{}".format(is_support_nan, is_query_nan))
                lda_loss += each_lda_loss
                query_logit.append(logit)

            lda_loss = lda_loss / B
            if not self.args.TRAIN.lda_loss:
                lda_loss = lda_loss.detach()

            query_logit = torch.cat(query_logit)

        else:
            lda_loss = torch.tensor(0,
                                    device=support_mean_feats.device,
                                    dtype=support_mean_feats.dtype
                                    )

        label_episode = torch.arange(self.n_way,
                                     dtype=torch.long, device=query_feats.device)
        label_episode = label_episode.view(-1,
                                           1)
        label_episode = label_episode.repeat(B, n_query).reshape(-1)
        # print(support_data.shape, query_logit.shape, label_episode.shape)
        output = {
            "loss_lda": lda_loss.detach()
        }

        if self.args.TRAIN.lda_cls_loss:
            lda_cls_loss = self.criterion(query_logit, label_episode)
            output["loss_lda_logit"] = lda_cls_loss.detach()

        else:
            lda_cls_loss = torch.tensor(
                0, device=support_feats.device, dtype=support_feats.dtype)

        total_loss = lda_cls_loss + lda_loss

        if self.args.TRAIN.is_normal_cls_loss:
            if self.args.TRAIN.meta_mode == "cossim":
                logit = self.compute_cosine_similarity(
                    support_mean_feats, query_feats)
            elif self.args.TRAIN.meta_mode == "euc":
                logit = metric.calc_l2_dist_torch(
                    query_feats, support_mean_feats, dim=2
                )
                # logit = logit.permute(0, 2, 1)
                logit = logit.reshape(-1, self.n_way)
            else:
                raise NotImplementedError

            loss = self.criterion(logit, label_episode)
            total_loss += loss

            output["loss_cls"] = loss
            output["normal_logit"] = logit

        output["loss_total"] = total_loss
        output["lda_logit"] = query_logit
        output["label_episode"] = label_episode
        output["features"] = raw_logits

        return output

    def compute_cosine_similarity(self, support_feats, query_feats):
        query_feats = F.normalize(query_feats, dim=2)
        support_feats = F.normalize(support_feats, dim=2)
        # print(query_feats.size(), support_feats.size())
        cossim = torch.bmm(query_feats, support_feats.permute(0, 2, 1))
        # assert cossim.max() <= 1.001, cossim.max()
        cossim = cossim * self.args.TRAIN.logit_scale
        # order in one batch = (C1, C1, C1, ..., C2, C2, C2, ...)
        # cossim = cossim.permute(0, 2, 1)
        # print(cossim.shape, query_feats.shape)
        cossim = cossim.reshape(-1, self.n_way)

        return cossim

    def embedding_vectors(self, support_vectors, support_mean_vectors, query_vectors, use_mean=False):
        """
        support_vectors: torch.Tensor, (num_class, num_support, D)
        query_vectors: torch.Tensor, (num_class*num_query, D)
        """

        n_class, n_support, D = support_vectors.shape

        if self.training:
            lda_loss, logit, pick_v = self.trn_embedding(
                support_vector=support_vectors,
                query_vector=query_vectors,
                method=self.args.MODEL.embedding_method,
                is_svd=self.args.MODEL.is_lda_svd,
                svd_dim=self.args.MODEL.lda_svd_dim,
                lamb=self.args.MODEL.lda_lamb
            )

        else:
            lda_loss, logit, pick_v = self.val_embedding(
                support_vector=support_vectors,
                query_vector=query_vectors,
                method=self.args.MODEL.embedding_method,
                is_svd=self.args.MODEL.is_lda_svd,
                svd_dim=self.args.MODEL.lda_svd_dim,
                lamb=self.args.MODEL.lda_lamb
            )

        # support_embedding = support_embedding.reshape(n_class, n_support, -1)

        return lda_loss, logit, pick_v


class LossWrapEpisodeNCA(LossWrap):

    def forward(self, input, label):
        # input = (B, n_class*n_sample, C, W, H)
        if self.training:
            self.n_way = self.args.TRAIN.n_way
            n_support = self.args.TRAIN.n_support
            n_query = self.args.TRAIN.n_query
            n_sample = n_support + n_query
        else:
            self.n_way = self.args.TEST.n_way
            n_support = self.args.TEST.n_support
            n_query = self.args.TEST.n_query
            n_sample = n_support + n_query

        # if self.training:
        B, CXN, C, W, H = input.size()
        input = input.reshape(-1, C, W, H)
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        # shuffle_index = torch.randperm(B*CXN, device=input.device)
        # shuffled_input = input[shuffle_index]
        raw_logits = self.model(input)
        # print(torch.any(torch.isnan(raw_logits)))
        # raw_logits = self.model(input)
        # raw_logits = raw_logits[shuffle_index]
        # raw_logits[shuffle_index] = raw_logits.clone()

        raw_logits_2 = raw_logits.reshape(
            B, self.n_way, n_sample, -1)

        support_feats = raw_logits_2[:, :, :n_support]

        query_feats = raw_logits_2[:, :, n_support:]
        query_feats = query_feats.reshape(
            B, self.n_way*n_query, -1)
        support_mean_feats = torch.mean(
            support_feats, dim=2)  # (B, n_class, feat_dim)

        label_episode = torch.arange(self.n_way,
                                     dtype=torch.long, device=query_feats.device)
        label_episode = label_episode.view(-1,
                                           1)
        label_episode = label_episode.repeat(B, n_query).reshape(-1)

        if self.args.MODEL.embedding_flag:
            # support_feats_all = []
            # query_feats_all = []
            query_logit = []
            nca_loss = 0
            for data_index in range(B):
                support_data = support_feats[data_index]
                support_mean_data = support_mean_feats[data_index]
                query_data = query_feats[data_index]

                try:
                    each_nca_loss, logit = self.embedding_vectors(
                        support_vectors=support_data,
                        support_mean_vectors=support_mean_data,
                        query_vectors=query_data,
                        query_label=label_episode[data_index]
                    )
                except RuntimeError:
                    is_support_nan = torch.any(torch.isnan(support_mean_data))
                    is_query_nan = torch.any(torch.isnan(query_data))
                    raise RuntimeError("is support nan:{} is query nan:{}".format(is_support_nan, is_query_nan))
                nca_loss += each_nca_loss
                query_logit.append(logit)

            nca_loss = nca_loss / B
            if not self.args.TRAIN.nca_loss:
                nca_loss = nca_loss.detach()

            query_logit = torch.cat(query_logit)

        else:
            nca_loss = torch.tensor(0,
                                    device=support_mean_feats.device,
                                    dtype=support_mean_feats.dtype
                                    )


        # print(support_data.shape, query_logit.shape, label_episode.shape)
        output = {
            "loss_lda": nca_loss.detach()
        }

        # if self.args.TRAIN.lda_cls_loss:
        lda_cls_loss = self.criterion(query_logit, label_episode)
        output["loss_lda_logit"] = lda_cls_loss.detach()

        total_loss = lda_cls_loss

        if self.args.TRAIN.is_normal_cls_loss:
            if self.args.TRAIN.meta_mode == "cossim":
                logit = self.compute_cosine_similarity(
                    support_mean_feats, query_feats)
            elif self.args.TRAIN.meta_mode == "euc":
                logit = metric.calc_l2_dist_torch(
                    query_feats, support_mean_feats, dim=2
                )
                # logit = logit.permute(0, 2, 1)
                logit = logit.reshape(-1, self.n_way)
            else:
                raise NotImplementedError

            loss = self.criterion(logit, label_episode)
            total_loss += loss

            output["loss_cls"] = loss
            output["normal_logit"] = logit

        output["loss_total"] = total_loss
        output["nca_logit"] = query_logit
        output["label_episode"] = label_episode
        output["features"] = raw_logits

        return output

    def compute_cosine_similarity(self, support_feats, query_feats):
        query_feats = F.normalize(query_feats, dim=2)
        support_feats = F.normalize(support_feats, dim=2)
        # print(query_feats.size(), support_feats.size())
        cossim = torch.bmm(query_feats, support_feats.permute(0, 2, 1))
        # assert cossim.max() <= 1.001, cossim.max()
        cossim = cossim * self.args.TRAIN.logit_scale
        # order in one batch = (C1, C1, C1, ..., C2, C2, C2, ...)
        # cossim = cossim.permute(0, 2, 1)
        # print(cossim.shape, query_feats.shape)
        cossim = cossim.reshape(-1, self.n_way)

        return cossim

    def embedding_vectors(self, support_vectors, support_mean_vectors, query_vectors, query_label, use_mean=False):
        """
        support_vectors: torch.Tensor, (num_class, num_support, D)
        query_vectors: torch.Tensor, (num_class*num_query, D)
        """

        n_class, n_support, D = support_vectors.shape

        if self.training:
            lda_loss, logit = self.trn_embedding(
                support_vector=support_vectors,
                query_vector=query_vectors,
                query_label=query_label,
                init_method=self.args.MODEL.init_nca_method,
                max_batch_size=math.ceil(n_class*n_support)+10,
                distance_method=self.args.MODEL.mds_metric_type,
                lr=self.args.MODEL.nca_lr,
                max_iter=self.args.MODEL.nca_max_iter,
                stop_diff=self.args.MODEL.nca_stop_diff,
                scale=self.args.MODEL.nca_scale
            )

        else:
            lda_loss, logit = self.val_embedding(
                support_vector=support_vectors,
                query_vector=query_vectors,
                init_method=self.args.MODEL.init_nca_method,
                max_batch_size=math.ceil(n_class*n_support)+10,
                distance_method=self.args.MODEL.mds_metric_type,
                lr=self.args.MODEL.nca_lr,
                max_iter=self.args.MODEL.nca_max_iter,
                stop_diff=self.args.MODEL.nca_stop_diff,
                scale=self.args.MODEL.nca_scale
            )

        # support_embedding = support_embedding.reshape(n_class, n_support, -1)

        return lda_loss, logit


class LossWrapLinear(torch.nn.Module):

    def __init__(self, args, model, criterion):
        self.args = args
        super(LossWrapLinear, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, input, label):
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        raw_logits = self.model(input)

        loss = self.criterion(raw_logits, label)

        return loss, raw_logits


class LossWrapSimCLR(nn.Module):

    def __init__(self, args, model, head, criterion):
        super(LossWrapSimCLR, self).__init__()
        self.args = args
        self.model = model
        self.head = head
        self.criterion = criterion

    def forward(self, input_x, label):
        # input_x = torch.cat((input_x1, input_x2), dim=0)
        # if self.args.TRAIN.shuffle_simclr:
        #     rand_idx = torch.randperm(len(label))
        #     print("rand idx:", rand_idx)
        #     label = label[rand_idx]
        #     input_x = input_x[rand_idx]
        # else:
        #     rand_idx = None
        rand_idx = None
        if self.args.multi_gpus:
            input_x = input_x.cuda()
            label = label.cuda()
        else:
            input_x = input_x.to(self.args.device)
            label = label.to(self.args.device)

        # print("org label:", label)
        # if self.args.TRAIN.shuffle_simclr:
        #     rand_idx = torch.randperm(len(label))
        #     print("rand idx:", rand_idx)
        #     label = label[rand_idx]
        #     input_x = input_x[rand_idx]
        # else:
        #     rand_idx = None
        raw_logits = self.model(input_x)

        # this is not L2 Normalized
        output = self.head(raw_logits)
        # print("permuted :", label)
        # print(output)
        # print(output.size())

        loss, logits = self.criterion(output, label, rand_idx)

        return loss, logits


class LossWrapVAE(nn.Module):

    def __init__(self, args, model, vae, head, rec_loss, kl_loss, cont_loss):
        super(LossWrapVAE, self).__init__()
        self.args = args
        self.model = model
        self.vae = vae
        self.head = head
        self.rec_loss = rec_loss
        self.kl_loss = kl_loss
        self.cont_loss = cont_loss

    def forward(self, input_x, label):
        rand_idx = None
        if self.args.multi_gpus:
            input_x = input_x.cuda()
            label = label.cuda()
        else:
            input_x = input_x.to(self.args.device)
            label = label.to(self.args.device)

        # with torch.no_grad():
        # with torch.tra
        raw_logits = self.model(input_x)

        # this is not L2 Normalized
        z, mean_v, sigma_v = self.vae(raw_logits)
        # rec_loss = self.rec_loss(rec_logits, raw_logits)
        if self.args.TRAIN.prior_agg:
            # index = torch.arange(len(label), device=label.device, dtype=label.dtype)
            mean_v2 = mean_v[label]
            sigma_v2 = sigma_v[label]
            mean_v2 = (mean_v + mean_v2) * 0.5
            sigma_v2 = (sigma_v + sigma_v2) * 0.5
            kl_loss = self.kl_loss(mean_v, sigma_v, mean_v2, sigma_v2)
            # mean_v2 = torch.cat([mean_v, mean_v2])
            # sigma_v2 = torch.cat([])
        else:
            kl_loss = self.kl_loss(mean_v, sigma_v)
            # kl_loss = 0

        logits = self.head(z)

        # cont_loss, logits = self.cont_loss(mean_v, sigma_v, label, rand_idx)
        cont_loss, logits = self.cont_loss(logits, label, rand_idx)
        # rec_loss = 0
        # kl_loss = 0

        return kl_loss, cont_loss, logits


def debug_simclr():
    class R:

        def __init__(self):
            self.shuffle_simclr = True

    class T:
        def __init__(self):
            self.multi_gpus = False
            self.device = "cpu"
            self.TRAIN = R()

    def shuffle(input_x, label):
        rand_idx = torch.randperm(len(label))
        input_x = input_x[rand_idx]
        org2rand = {}
        for idx, r_idx in enumerate(rand_idx):
            org2rand[r_idx.item()] = idx
        # print("org label:", label)
        # print("org2rand:", org2rand)
        # print("rand idx:", rand_idx)
        copied_labels = label.clone()
        # print(copied_labels)
        for idx, l_ in enumerate(copied_labels):
            rand_label = org2rand[l_.item()]
            input_idx = org2rand[idx]
            # print("l:", l_, "idx:", idx)
            # print("input idx:", input_idx, "rand label:", rand_label)
            if input_idx == rand_label:
                jkfdhgksdfh
            if(input_idx <= rand_label):
                label[input_idx] = rand_label - 1
            else:
                label[input_idx] = rand_label
        # print("permuted label:", label)
        return input_x, label, rand_idx

    args = T()
    import sys
    sys.path.append("../")
    from lossfunction.simclr_loss import SimCLRLoss
    from resnet import resnet18
    from head import MLP
    data_b = 5
    simclr_loss = SimCLRLoss(10, device="cpu")
    model = resnet18()
    head = MLP(512, 32, hidden_dims=[32])
    x = torch.randn(data_b, 3, 32, 32)
    # x1 = torch.randn(data_b, 3, 32, 32)
    label = torch.arange(len(x))
    label = torch.cat((label+len(label), label))

    x = torch.cat((x, x), dim=0)
    x, label, rand_idx = shuffle(x, label)
    wrap = LossWrapSimCLR(args, model, head, simclr_loss)
    # label = torch.arange(data_b)
    a, b = wrap(x, label)
    print(a)
    print("logits:", b.size())


def debug_simclr_vae():

    class R:

        def __init__(self):
            self.shuffle_simclr = True
            self.prior_agg = True

    class T:
        def __init__(self):
            self.multi_gpus = False
            self.device = "cpu"
            self.TRAIN = R()

    def shuffle(input_x, label):
        rand_idx = torch.randperm(len(label))
        input_x = input_x[rand_idx]
        org2rand = {}
        for idx, r_idx in enumerate(rand_idx):
            org2rand[r_idx.item()] = idx
        # print("org label:", label)
        # print("org2rand:", org2rand)
        # print("rand idx:", rand_idx)
        copied_labels = label.clone()
        # print(copied_labels)
        for idx, l_ in enumerate(copied_labels):
            rand_label = org2rand[l_.item()]
            input_idx = org2rand[idx]
            # print("l:", l_, "idx:", idx)
            # print("input idx:", input_idx, "rand label:", rand_label)
            if input_idx == rand_label:
                jkfdhgksdfh
            if(input_idx <= rand_label):
                label[input_idx] = rand_label - 1
            else:
                label[input_idx] = rand_label
        # print("permuted label:", label)
        return input_x, label, rand_idx

    args = T()
    import sys
    sys.path.append("../")
    from lossfunction.simclr_loss import SimCLRLoss, SimCLRLossV
    from lossfunction.kl_div import kl_div_normal_2
    from resnet import resnet18
    from vae import VAE
    from head import MLP
    data_b = 5
    fd = 64
    # simclr_loss = SimCLRLoss(10, device="cpu")
    simclr_loss = SimCLRLossV(10, device="cpu", feature_dim=fd)
    model = resnet18()
    head = MLP(128, fd, hidden_dims=[32])
    vae = VAE(512, z_dim=fd, layers=[256, 128])
    x = torch.randn(data_b, 3, 84, 84)
    # x1 = torch.randn(data_b, 3, 32, 32)
    label = torch.arange(len(x))
    label = torch.cat((label+len(label), label))

    x = torch.cat((x, x+1), dim=0)
    x, label, rand_idx = shuffle(x, label)
    wrap = LossWrapVAE(args, model, vae, head, None,
                       kl_div_normal_2, simclr_loss)
    # label = torch.arange(data_b)
    _, a, b = wrap(x, label)
    b.backward()
    print(a)
    print(b)


def debug_meta():
    class R:

        def __init__(self):
            # self.n = 10
            self.n_way = 5
            self.n_support = 5
            self.n_query = 5
            self.logit_scale = 8
            # self.meta_mode = "euc"
            self.meta_mode = "cossim"

    class T:
        def __init__(self):
            self.multi_gpus = False
            self.device = "cpu"
            # self.TRAIN = R()
            self.DATA = R()
            self.TRAIN = R()
            self.TEST = R()

    args = T()
    import sys
    sys.path.append("../")
    from lossfunction.simclr_loss import SimCLRLoss, SimCLRLossV
    from lossfunction.kl_div import kl_div_normal_2
    from resnet import resnet18
    from vae import VAE
    from head import MLP
    import numpy as np
    data_b = 10
    fd = 64
    model = resnet18().eval()
    x = torch.randn(1, 3, 84, 84)
    x = torch.cat([x for i in range(args.TRAIN.n_support+args.TRAIN.n_query)])
    print(x.size())
    x = [x+i for i in range(args.TRAIN.n_way)]
    x = torch.cat(x, dim=0)
    # y = torch.randn_like(x)
    x = torch.stack((x, x), dim=0)
    # x1 = torch.randn(data_b, 3, 32, 32)
    label = torch.arange(len(x))
    label = torch.cat((label+len(label), label))
    criterion = nn.CrossEntropyLoss()
    wrap = LossWrapEpisode(args, model, None, criterion).eval()

    loss, output, b = wrap(x, label)
    with torch.no_grad():
        _, pred = torch.max(output, dim=1)
        correct = np.mean(pred.cpu().numpy() == b.numpy())
        print(correct)
    # loss.backward()
    # print(model.conv1.weight.grad)

    # print(loss, loss_2)
    # print(output)
    print(b)


if __name__ == "__main__":
    # debug_simclr()
    # debug_simclr_vae()
    debug_meta()
