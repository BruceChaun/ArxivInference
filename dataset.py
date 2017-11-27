import torch as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
from nltk.corpus import stopwords
import pickle
from collections import OrderedDict, Counter
import sys

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

class BOWPaperDataset(MappedPaperDataset):
    def __init__(self, db, vocab, users):
        MappedPaperDataset.__init__(self, db, vocab, users)
        for k, v in self.db:
            v['cleaned-bow'] = Counter([self.vocab_imap[w] + 1 for w in v['cleaned']])
            v['author-bow'] = Counter([self.users_imap[u] + 1 for u in v['authors']])

    def get_mapped_item(self, i):
        v = self.db[i][1]
        bow = v['cleaned-bow']
        author_bow = v['author-bow']
        words, word_occs = zip(*bow.items())
        authors, author_occs = zip(*author_bow.items())
        return (words, word_occs), (authors, author_occs)

def collate_bow(samples):
    words, authors, words_neg = [list(x) for x in zip(*samples)]
    wi, wo = [list(x) for x in zip(*words)]
    ui, uo = [list(x) for x in zip(*authors)]
    wi_p, wo_p = [list(x) for x in zip(*words_neg)]

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

    collate_documents(ui)
    collate_documents(uo)
    collate_documents(wi)
    collate_documents(wo)
    collate_documents(wi_p)
    collate_documents(wo_p)

    return (T.autograd.Variable(T.LongTensor(np.array(ui))),
            T.autograd.Variable(T.LongTensor(np.array(uo))),
            T.autograd.Variable(T.LongTensor(np.array(wi))),
            T.autograd.Variable(T.LongTensor(np.array(wo))),
            T.autograd.Variable(T.LongTensor(np.array(wi_p))),
            T.autograd.Variable(T.LongTensor(np.array(wo_p))),
            )

BOWDataLoader = partial(
        DataLoader,
        shuffle=False,
        collate_fn=collate_bow,
        drop_last=True,
        )

stopword_list = stopwords.words('english') + [
        "n't",
        "'ll",
        "'s",
        "'d",
        ]

with open(sys.argv[1], 'rb') as f:
    db = pickle.load(f)

db = list(db.items())
db_len = len(db)
train_split = db_len * 4 // 5
valid_split = db_len // 10
train_db = db[:train_split]
valid_db = db[train_split:train_split+valid_split]
test_db = db[train_split+valid_split:]

vocab = Counter()
users = Counter()
for k, v in train_db:
    vocab.update(v['abstract'])
    users.update(v['authors'])

vocab = OrderedDict(
        (w, c) for w, c in vocab.items()
        if (w not in stopword_list and len(w) > 1))
users = OrderedDict((u, c) for u, c in users.items() if c >= 3)

train_dataset = BOWPaperDataset(train_db, vocab, users)
valid_dataset = BOWPaperDataset(valid_db, vocab, users)
test_dataset = BOWPaperDataset(test_db, vocab, users)

print('Vocabulary size:', len(vocab))
print('Number of authors:', len(users))
print('Training set size:', len(train_dataset))
print('Validation set size:', len(valid_dataset))
print('Test set size:', len(test_dataset))
