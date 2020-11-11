import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import lib.lossfunction.metric as metric
except ImportError:
    import sys
    sys.path.append("../")
    import lossfunction.metric as metric


class LossWrap(torch.nn.Module):
    def __init__(self, args, model, head, criterion):
        self.args = args
        super(LossWrap, self).__init__()
        self.model = model
        self.head = head
        self.criterion = criterion

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
        B, CXN, C, W, H = input.size()
        input = input.reshape(-1, C, W, H)
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        # shuffle_index = torch.randperm(B*CXN, device=input.device)
        # shuffled_input = input[shuffle_index]
        # raw_logits = self.model(shuffled_input)
        raw_logits = self.model(input)
        # raw_logits[shuffle_index] = raw_logits
        raw_logits = raw_logits.reshape(
            B, self.args.DATA.n_class_train, self.args.DATA.nb_sample_per_class, -1)

        support_feats = raw_logits[:, :, :self.args.DATA.n_support]

        support_feats = torch.mean(raw_logits, dim=2)  # (B, n_class, feat_dim)

        n_query = self.args.DATA.nb_sample_per_class - self.args.DATA.n_support
        query_feats = raw_logits[:, :, self.args.DATA.n_support:]
        query_feats = query_feats.reshape(
            B, self.args.DATA.n_class_train*n_query, -1)
        label_episode = torch.arange(self.args.DATA.n_class_train,
                                     dtype=torch.long, device=query_feats.device)
        label_episode = label_episode.view(-1,
                                           1).repeat(B, n_query).reshape(-1)

        if self.args.TRAIN.meta_mode == "cossim":
            logit = self.compute_cosine_similarity(support_feats, query_feats)
        elif self.args.TRAIN.meta_mode == "euc":
            logit = metric.calc_l2_dist_torch(
                query_feats, support_feats, dim=2
            )
            logit = logit.reshape(-1, self.args.DATA.n_class_train)
            print(logit)
        else:
            raise NotImplementedError

        loss = self.criterion(logit, label_episode)

        return loss, logit, label_episode

    def compute_cosine_similarity(self, support_feats, query_feats):
        query_feats = F.normalize(query_feats, dim=2)
        support_feats = F.normalize(support_feats, dim=2)
        # print(query_feats.size(), support_feats.size())
        cossim = torch.bmm(query_feats, support_feats.permute(0, 2, 1))
        assert cossim.max() <= 1.001, cossim.max()
        cossim = cossim * self.args.TRAIN.logit_scale
        cossim = cossim.reshape(-1, self.args.DATA.n_class_train)

        return cossim


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
    print(a)
    print(b)

def debug_meta():
    class R:

        def __init__(self):
            self.nb_sample_per_class = 10
            self.n_class_train = 5
            self.n_support = 5
            self.logit_scale = 1
            self.meta_mode = "euc"

    class T:
        def __init__(self):
            self.multi_gpus = False
            self.device = "cpu"
            # self.TRAIN = R()
            self.DATA = R()
            self.TRAIN = R()


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
    x = torch.cat([x for i in range(args.DATA.nb_sample_per_class)])
    print(x.size())
    x = [x+i for i in range(5)]
    x = torch.cat(x, dim=0)
    # y = torch.randn_like(x)
    x = torch.stack((x, x), dim=0)
    # x1 = torch.randn(data_b, 3, 32, 32)
    label = torch.arange(len(x))
    label = torch.cat((label+len(label), label))
    criterion = nn.CrossEntropyLoss()
    wrap = LossWrapEpisode(args, model, None, criterion)

    loss, output, b = wrap(x, label)
    with torch.no_grad():
        _, pred = torch.max(output, dim=1)
        correct = np.mean(pred.cpu().numpy() == b.numpy())
        print(correct)

    print(loss)
    print(b)


if __name__ == "__main__":
    # debug_simclr()
    # debug_simclr_vae()
    debug_meta()
