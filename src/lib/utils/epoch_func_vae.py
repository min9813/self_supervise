import os
import sys
import pathlib
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import logging
import numpy as np
import torch
import torchvision
import lib.utils.average_meter as average_meter
import lib.utils.simclr_utils as simclr_utils
import lib.evaluation.few_shot_eval as few_shot_eval
from tqdm import tqdm
from sklearn import linear_model
try:
    from apex import amp
except ImportError:
    pass


def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


