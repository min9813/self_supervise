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

        return loss
