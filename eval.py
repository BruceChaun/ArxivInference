# Usage: python3 eval.py <pickle-file> <prefix>
import torch as T
import numpy as np
import pickle
import sys
from collections import OrderedDict, Counter
from apk import apk
from dataset import *
import os
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

v_dataset = test_dataset if os.getenv('TEST', None) else valid_dataset

def summarize(l):
    print('Min', np.min(l))
    print('Max', np.max(l))
    print('Mean', np.mean(l))
    print('Std', np.std(l))
    print('Median', np.median(l))

def dot(A, b):
    return A @ b

def jsd(A, b):
    m = (A + b) / 2
    kl_am = entropy(A.T, m.T)
    kl_bm = entropy(b[:, np.newaxis, m.T])
    return (kl_am + kl_bm) / 2

if len(sys.argv) > 3 and sys.argv[3] == 'jsd':
    score = jsd
else:
    score = dot

prefix = sys.argv[2]
U = T.load('%s-U.p' % prefix)
W = T.load('%s-W.p' % prefix)
vocab = T.load('%s-vocab-selected.p' % prefix)
users = T.load('%s-users-selected.p' % prefix)

print('Evaluate diversity of top-K ranked words')
wi_counters = {k: Counter() for k in [1, 5, 10]}

for i, u in enumerate(U[1:]):
    s = score(W[1:], u)
    si = np.argsort(s)
    for k in wi_counters:
        wi_counters[k].update(si[-k:])
    if i % 1000 == 0:
        print('%d/%d' % (i, U.shape[0] - 1))

for k in wi_counters:
    wi_count = wi_counters[k]
    wi_count_total = sum(wi_count.values())
    wi_freq = [wi_count[wi] / wi_count_total for wi in wi_count]
    entropy = sum(-freq * np.log(freq) for freq in wi_freq)
    print('Entropy of top-%d ranking:' % k, entropy)
    print('Number of different words occurred in top-%d ranking:' % k,
          len(wi_count))

print('Evaluating author prediction given a paper')
apks = {k: [] for k in [1, 5, 10, 50]}
author_ranks = []
author_scores = []
author_true_labels = []
for i, (wi, ui, _) in enumerate(v_dataset):
    w = ((W[list(wi[0])] * np.array(wi[1]).reshape(-1, 1)).sum(axis=0) /
         np.array(wi[1]).sum())
    s = score(U, w)
    s_ranking = np.argsort(s)[::-1]
    author_scores.append(s)
    label = np.zeros((U.shape[0],))
    label[list(ui[0])] = 1
    author_true_labels.append(label)
    for k in apks:
        apks[k].append(apk(ui[0], s_ranking, k))
    author_ranks.append([])
    for u in ui[0]:
        author_ranks[-1].append(list(s_ranking).index(u))
    if i % 1000 == 0:
        print('%d/%d' % (i, len(v_dataset)))
for k in apks:
    print('Average precision @ %d:' % k)
    summarize(apks[k])
author_ranks_flattened = sum(author_ranks, [])
author_scores = np.reshape(author_scores, (-1,))
author_true_labels = np.reshape(author_true_labels, (-1,))
print('Author ranks:')
summarize(author_ranks_flattened)
print('AUC:', roc_auc_score(author_true_labels, author_scores))
