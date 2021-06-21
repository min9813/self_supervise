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
import math
import gzip
import numpy
import yaml
from easydict import EasyDict as edict


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def load_json(path, print_path=False):
    if print_path:
        print("load json from", path)

    path = str(path)

    with open(path, "r") as f:
        data = json.load(f)

    return data


def load_pickle(path, print_path=False, is_gzip=False):
    if print_path:
        print("load pickle from", path)

    path = str(path)

    if is_gzip:
        with gzip.open(path, "rb") as pkl:
            data = pickle.load(pkl)

    else:
        with open(path, "rb") as pkl:
            data = pickle.load(pkl)

    return data


def save_json(data, path, print_path=False, is_gzip=False):
    if print_path:
        print("save json to", path)

    path = str(path)
    with open(path, "w") as f:
        json.dump(data, f, cls=MyEncoder)


def save_pickle(data, path, print_path=False, is_gzip=False):
    if print_path:
        print("save pickle to", path)

    path = str(path)

    if is_gzip:
        with gzip.open(path, "wb") as pkl:
            pickle.dump(data, pkl)

    else:
        with open(path, "wb") as pkl:
            pickle.dump(data, pkl)

    return data


def load_yaml(filename):
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    return yaml_cfg


def append_json_to_file(data, path_file):
    os.makedirs(str(pathlib.Path(path_file).parent), exist_ok=True)
    if os.path.exists(path_file) is False:
        with open(path_file, "w") as f:
            f.write('')
    with open(path_file, 'ab+') as f:              # ファイルを開く
        f.seek(0,2)                                # ファイルの末尾（2）に移動（フォフセット0）  
        if f.tell() == 0 :                         # ファイルが空かチェック
            f.write(json.dumps([data], cls=MyEncoder).encode())   # 空の場合は JSON 配列を書き込む
        else :
            f.seek(-1,2)                           # ファイルの末尾（2）から -1 文字移動
            f.truncate()                           # 最後の文字を削除し、JSON 配列を開ける（]の削除）
            f.write(' , '.encode())                # 配列のセパレーターを書き込む
            f.write(json.dumps(data, cls=MyEncoder).encode())     # 辞書を JSON 形式でダンプ書き込み
            f.write(']'.encode()) 