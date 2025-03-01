"""
@file   : utils.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-07
"""
import numpy as np
import scipy.stats
import random
import torch


def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_pearsonr(x, y):
    return scipy.stats.perasonr(x, y)[0]

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True