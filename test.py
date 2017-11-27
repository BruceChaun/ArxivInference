# Usage: python3 test.py <pickle-file> <prefix>
import nltk
import torch as T
import pickle
import numpy as np
from collections import OrderedDict
import sys
import readline
import matplotlib.pyplot as PL
from sklearn.manifold import TSNE

prefix = sys.argv[2]
U = T.load('%s-U.p' % prefix)
W = T.load('%s-W.p' % prefix)
vocab = T.load('%s-vocab-selected.p' % prefix)
users = T.load('%s-users-selected.p' % prefix)
#U = T.load('U.p')
#W = T.load('W.p')
#vocab = T.load('vocab-selected.p')
#users = T.load('users-selected.p')
with open(sys.argv[1], 'rb') as f:
    db = pickle.load(f)

vocab = list(vocab.keys())
usercounts = users
users = list(users.keys())

def get_user_index(author):
    if author in users:
        author_idx = users.index(author)
    else:
        candidate_users = [
                (u, nltk.edit_distance(u, author)) for u in users
                ]
        candidate_users = list(sorted(candidate_users, key=lambda x: x[1]))[:20]
        candidate_users = [u[0] for u in candidate_users]
        for i, u in enumerate(candidate_users):
            print('[%d] %s' % (i, u))
        print('[-1] cancel')
        author_idx = int(input('? '))
        if author_idx == -1:
            return -1
        author_idx = users.index(candidate_users[author_idx])

    return author_idx

def get_word_index(word):
    return vocab.index(word) if word in vocab else -1


while True:
    inp = input('> ')
    if inp == 'q':
        break
    elif inp[0] == 't':
        fU = open('U.tsv', 'w')
        inps = inp.split(' ', 1)
        cmin = 0 if len(inps) == 1 else int(inps[1])
        print('Tabulating all authors with >%d occurrences...' % cmin)
        fusers = open('users.tsv', 'w')
        for user, u in zip(users, U[1:]):
            if usercounts[user] > cmin:
                fU.write('\t'.join(str(x) for x in u) + '\n')
                fusers.write(user + '\n')
        fU.close()
        fusers.close()
        continue
    elif inp == '?':
        print('t [c]\tTabulate the author vectors and names into U.tsv and')
        print('\tusers.tsv.  If c is provided, only tabulate the authors')
        print('\tthat appeared more than c times.  The files can be then')
        print('\tloaded into Tensorflow Embedding Projector.')
        print('u au\tDisplay the top-ranked words corresponding to the')
        print('\tauthor <au>, as well as the author embedding and norm.')
        print('w wd\tDisplay the word embedding and norm of <wd>.')
        print('s wd au\tCompute score of a word and an author.')
        print('f au\tList all the paper IDs with author <au>.')
        print('p id au\tCompute score of a paper and an author.')
        print('d id\tDump the content of paper <id>.')
        print('q\tQuit')
        continue

    cmd, inp = inp.split(' ', 1)

    if cmd == 'u':
        author_idx = get_user_index(inp)
        if author_idx == -1:
            continue
        s = W @ U[author_idx + 1]
        si = np.argsort(s[1:])[-10:]
        v = [vocab[i] for i in si]

        print(v)
        print(U[author_idx + 1], np.linalg.norm(U[author_idx + 1]))
    elif cmd == 'w':
        wi = get_word_index(inp)
        print(W[wi + 1], np.linalg.norm(W[wi + 1]))
    elif cmd == 's':
        word, author = inp.split(' ', 1)
        wi = get_word_index(word)
        ui = get_user_index(author)
        if ui == -1:
            continue
        print(U[ui + 1] @ W[wi + 1])
    elif cmd == 'f':
        author = inp
        print([k for k in db if author in db[k]['authors']])
    elif cmd == 'p':
        paper_id, author = inp.split(' ', 1)
        abstract = db[paper_id]['abstract']
        wi = [get_word_index(word) for word in abstract]
        wi = [w + 1 for w in wi if w != -1]
        w = W[wi].mean(0)
        u = U[get_user_index(author) + 1]
        print(u @ w)
    elif cmd == 'd':
        paper_id = inp
        print(' '.join(db[paper_id]['abstract']))
