import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from tqdm import tqdm


def knn_hard_from_distmat(distance_mat, labels, topk, method="partition"):
    N1, N2 = distance_mat.shape
    assert N2 == len(labels), (N2, len(labels))

    # if method == "partition":
    sorted_indices = np.argpartition(distance_mat, topk, axis=1)[:, :topk]
    # else:
        # sorted_indices = np.argsort(distance_mat, axis=1)[:, :topk]
#     print(sorted_indices)
    pred_labels = labels[sorted_indices]
    unique_label = np.unique(labels)

    label_count_mat = np.zeros((N1, len(unique_label)))
    for index, label in enumerate(unique_label):
        num_this_l = pred_labels == label
        num_this_l = np.sum(num_this_l, axis=1)
        label_count_mat[:, index] = num_this_l
        
    pred_label_index = np.argmax(label_count_mat, axis=1)
    pred_label = unique_label[pred_label_index]

    return pred_label
    