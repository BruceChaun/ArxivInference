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
        db = [(k, v) for k, v in db if len(v['authors']) >= 1]
        self.db = db
        self.max_doc_len = max(len(v['abstract']) for k, v in self.db)
        self.max_cleaned_doc_len = max(len(v['cleaned']) for k, v in self.db)

    def __len__(self):
        return len(self.db)

    @property
    def db_len(self):
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

    def translate(self, abstract, authors):
        abstract = [self.vocab_map[w - 1] for w in abstract]
        authors = [self.users_map[u - 1] for u in authors]
        return abstract, authors

class MappedPaperDataset(PaperDataset):
    def __init__(self, db, vocab, users):
        PaperDataset.__init__(self, db, vocab, users)

    def __getitem__(self, i):
        abstract, authors = self.get_mapped_item(i)
        abstract_neg, _ = self.get_mapped_item(np.random.choice(len(self)))
        return abstract, authors, abstract_neg

def collate_mapped(samples):
    abstract, authors, abstract_neg = [list(x) for x in zip(*samples)]
    vocab = [list(set(a)) for a in abstract]

    def collate_documents(docs):
        max_word_len = max(len(a) for a in docs)
        lengths = [len(a) for a in docs]
        for i in range(len(docs)):
            docs[i] = np.pad(
                    docs[i],
                    (0, max_word_len - len(docs[i])),
                    'constant',
                    constant_values=0
                    )
        return lengths

    lengths = collate_documents(abstract)
    lengths_neg = collate_documents(abstract_neg)
    n_authors = collate_documents(authors)
    vocab_size = collate_documents(vocab)

    return (T.autograd.Variable(T.LongTensor(np.array(abstract))),
            T.autograd.Variable(T.LongTensor(np.array(abstract_neg))),
            T.autograd.Variable(T.LongTensor(np.array(authors))),
            T.autograd.Variable(T.LongTensor(np.array(lengths))),
            T.autograd.Variable(T.LongTensor(np.array(lengths_neg))),
            T.autograd.Variable(T.LongTensor(np.array(n_authors))),
            T.autograd.Variable(T.LongTensor(np.array(vocab))),
            T.autograd.Variable(T.LongTensor(np.array(vocab_size))),
            )

class CharMappedPaperDataset(PaperDataset):
    def __getitem__(self, i):
        raise NotImplementedError

MappedDataLoader = partial(
        DataLoader,
        shuffle=False,
        collate_fn=collate_mapped,
        drop_last=True,
        )
