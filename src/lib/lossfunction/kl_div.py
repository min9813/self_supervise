import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div_normal(q_mean, q_var):
    # q_var = q_log_var.exp()
    q_log_var = q_var.log()
    # print(q_var.mean(), (q_mean*q_mean).mean(), q_log_var.mean())
    # kl = torch.mean(torch.sum(q_log_var - q_mean*q_mean - q_var.add(1.), dim=1)).mul(-0.5)
    kl = torch.mean(q_log_var - q_mean*q_mean - q_var.add(1.)).mul(-0.5)
    # print("kl:", kl, "log var:", q_log_var, q_var)

    return kl


def kl_div_normal_2(q_mean, q_var, p_mean, p_var):
    q_log_var = q_var.log()
    p_log_var = p_var.log()
    diff = q_mean - p_mean
    kl = torch.mean(torch.sum((-diff*diff - q_var) /
                              p_var + q_log_var - p_log_var, dim=1)).mul(-0.5)

    return kl


class KLDiv(nn.Module):

    def __init__(self):
        super(KLDiv, self).__init__()

        self.constant = nn.Parameter(torch.ones(1), requires_grad=False)
