import torch as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial

class PaperDataset(Dataset):
    def __init__(self, db, vocab, users):
        self.vocab = vocab
        self.users = users
        self.vocab_map = dict(enumerate(vocab))
        self.users_map = dict(enumerate(users))
        self.vocab_imap = {v: k for k, v in self.vocab_map.items()}
        self.users_imap = {v: k for k, v in self.users_map.items()}
        charset = set().union(*[set(w) for w in self.vocab_imap])
        self.char_map = dict(enumerate(charset))
        self.char_imap = {v: k for k, v in self.char_map.items()}
        self.max_word_len = max(len(w) for w in self.vocab_imap)

        db = [(k, {
            'abstract': v['abstract'],
            'cleaned': [w for w in v['abstract'] if w in self.vocab_imap],
            'authors': [u for u in v['authors'] if u in self.users_imap],
            }) for k, v in db]
        db = [(k, v) for k, v in db if len(v['authors']) > 1]
        self.db = db

    def __len__(self):
        return len(self.db)

    def get_raw_item(self, i):
        v = self.db[i][1]
        return v['abstract'], v['cleaned'], v['authors']

    def get_mapped_item(self, i):
        _, abstract, authors = self.get_raw_item(i)
        abstract = [self.vocab_imap[w] + 1 for w in abstract
                    if w in self.vocab_imap]
        authors = [self.users_imap[u] + 1 for u in authors]
        return abstract, authors

class MappedPaperDataset(PaperDataset):
    def __getitem__(self, i):
        abstract, authors = self.get_mapped_item(i)
        author_pos = np.asscalar(np.random.choice(authors))
        author_neg = np.random.choice(len(self.users_map))
        return abstract, author_pos, author_neg

def collate_mapped(samples):
    abstract, author_pos, author_neg = [list(x) for x in zip(*samples)]
    max_word_len = max(len(a) for a in abstract)
    lengths = [len(a) for a in abstract]
    for i in range(len(abstract)):
        abstract[i] = np.pad(
                abstract[i],
                (0, max_word_len - len(abstract[i])),
                'constant',
                constant_values=0
                )

    return (T.autograd.Variable(T.LongTensor(np.array(abstract))),
            T.autograd.Variable(T.LongTensor(np.array(author_pos))),
            T.autograd.Variable(T.LongTensor(np.array(author_neg))),
            T.autograd.Variable(T.LongTensor(np.array(lengths))))

class CharMappedPaperDataset(PaperDataset):
    def __getitem__(self, i):
        raise NotImplementedError

MappedDataLoader = partial(
        DataLoader,
        shuffle=False,
        collate_fn=collate_mapped,
        drop_last=True,
        )
