import torch
import torch.nn as nn
import torch.nn.functional as F


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

        loss = self.criterion(output, label)

        return loss, output


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



if __name__ == "__main__":
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
    a, b, c = wrap(x, label)
    print(a)
    print("logits:", b.size())
    print("label:", c)
