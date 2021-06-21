import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm


def nullspace(tensor, tol=1e-5):
    u, s, vh = torch.svd(tensor)
    nnz = (s >=  tol).sum()
    ns = vh[nnz:].conj().T

    return ns


class TaskAdaptiveProjection(nn.Module):

    def __init__(self, input_dim, n_class):
        super(TaskAdaptiveProjection, self).__init__()
        self.phi_projector = nn.Linear(input_dim, n_class)

    def projection_space(self, average_key, batchsize, nb_class, phi_ind=None, is_backward_m=False):
        c_t = average_key
        eps = 1e-6
        
        if self.training:
            phi_tmp = self.phi_projector.W
        else:
            phi_data = self.phi_projector.W.data
            phi_tmp = phi_data[phi_ind, :]

            # phi_tmp = chainer.Variable(phi_data[phi_ind,:])
        for i in range(nb_class):
            if i == 0:
                phi_sum = phi_tmp[i]
            else:
                phi_sum += phi_tmp[i]
        # phi = nb_class*(phi_tmp)-F.broadcast_to(phi_sum,(nb_class,self.dimension))
        phi = nb_class * phi_tmp - phi_sum[None, :]
                    
        power_phi = torch.sqrt(torch.sum(phi*phi, axis=1, keepdim=True))
        # power_phi = F.transpose(F.broadcast_to(power_phi, [self.dimension,nb_class]))
        
        phi = phi/(power_phi+eps)
        
        power_c = torch.sqrt(torch.sum(c_t*c_t, axis=1, keepdim=True))
        # power_c = F.transpose(F.broadcast_to(power_c, [self.dimension,nb_class]))
        c_tmp = c_t/(power_c+eps)
        
        null = phi - c_tmp

        if not is_backward_m:
            null = null.detach()

        M = nullspace(null)
        # M = M.repeat(batchsize, 1, 1)

        return M

    def compute_power(self, batchsize, key, M, nb_class, train=True, phi_ind=None):
        if self.training:
            phi_tmp = self.phi_projector.W
        else:
            phi_data = self.phi_projector.W.data
            phi_tmp = phi_data[phi_ind, :]

        phi_m = torch.mm(phi_tmp, M)
        phi_m_power = torch.sum(phi_m*phi_m, axis=1)

        key_m = torch.mm(key, M)
        key_m_power = torch.sum(key_m*key_m, axis=1)

        power_m = phi_m_power + key_m_power

        return power_m

    def compute_logit(self, r_t, pow_t):
        dot_m = 2 * self.phi_projector(r_t) - pow_t

        return dot_m
