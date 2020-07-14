import torch


def shuffle(input_x, label):
    rand_idx = torch.randperm(len(label))
    input_x = input_x[rand_idx]
    org2rand = {}
    for idx, r_idx in enumerate(rand_idx):
        org2rand[r_idx.item()] = idx
    copied_labels = label.clone()
    # print(copied_labels)
    for idx, l_ in enumerate(copied_labels):
        rand_label = org2rand[l_.item()]
        input_idx = org2rand[idx]
        if input_idx == rand_label:
            raise ValueError("something wrong")
        if(input_idx <= rand_label):
            label[input_idx] = rand_label - 1
        else:
            label[input_idx] = rand_label
    return input_x, label


def setup_simclr(input_x1, input_x2):
    label = torch.arange(len(input_x1))
    label = torch.cat((label+len(label), label))
    input_x = torch.cat((input_x1, input_x2), dim=0)
    input_x, label = shuffle(input_x, label)

    return input_x, label
    