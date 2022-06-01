from __future__ import annotations

import difflib
import os
import re
import sys
from difflib import SequenceMatcher
from warnings import simplefilter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

plt.rcParams.update(
    {
        'figure.figsize': (20, 16),
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'figure.dpi': 200,
        'legend.fontsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'savefig.bbox': 'tight',
    },
)

pd.options.mode.chained_assignment = None
plt.rcParams.update({'font.size': 12})

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)


def make_name(s):
    s = s.replace("'", '')
    s = re.sub(r"\:|\-|\s|\&|\@|\.|\|\?|\'|\#|\,|\(|\)", '_', s)
    s = re.sub(r'_{1,}', '_', s)
    s = s.strip('_')
    return s.lower()


def escape(s):
    s = re.sub(r"(\:|\-|\s|\&|\@|\.|\|\?|\')", r'\\\1', s)
    return s.lower()


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def seq_dist(s1, s2):
    sm = difflib.SequenceMatcher(None, s1, s2)
    return sm.ratio()


def table_sparsity(df):
    df_array = np.asarray(df)
    sparsity = 1.0 - np.count_nonzero(df_array) / df_array.size
    return sparsity


def get_corr_table(df):
    c = df.corr().abs()
    s = c.unstack().reset_index()
    s.columns = ['level_0', 'level_1', 'corr']
    s = s[s['level_0'] != s['level_1']]
    s['pair'] = s[['level_0', 'level_1']].values.tolist()
    s['pair'] = [sorted(x) for x in s['pair']]
    s['level_0'] = [x[0] for x in s['pair']]
    s['level_1'] = [x[1] for x in s['pair']]
    s['pair'] = [''.join(sorted(x)) for x in s['pair']]
    s = s.drop_duplicates(subset='pair')
    s = s[['level_0', 'level_1', 'corr']]
    s = s.sort_values(by='corr', ascending=False)
    return s


def norm(data):
    """Normalize data set"""
    train_stats = data.describe()
    train_stats = train_stats.transpose()
    return (data - train_stats['mean']) / train_stats['std']
