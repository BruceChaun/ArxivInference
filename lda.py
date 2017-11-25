import sys
import pickle
from nltk.corpus import stopwords
from collections import OrderedDict, Counter
import logging
from gensim import models, corpora

N_TOPICS = 20

logging.getLogger().setLevel('INFO')

stopword_list = stopwords.words('english') + [
        "n't",
        "'ll",
        "'s",
        "'d",
        ]

with open('newdb.p', 'rb') as f:
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
vocab_map = dict(enumerate(vocab))
users_map = dict(enumerate(users))
vocab_imap = {w: i for i, w in enumerate(vocab)}
users_imap = {u: i for i, u in enumerate(users)}

def filter_db(db, vocab, users):
    db = [(k, {
        'abstract': v['abstract'],
        'cleaned': [w for w in v['abstract'] if w in vocab],
        'authors': [u for u in v['authors'] if u in users],
        }) for k, v in db]
    db = [(k, v) for k, v in db if len(v['authors']) >= 1]
    return db

train_db = filter_db(train_db, vocab, users)
valid_db = filter_db(valid_db, vocab, users)
test_db = filter_db(test_db, vocab, users)

def db_to_corpus(db):
    author2doc = {}
    for i, (k, v) in enumerate(db):
        for a in v['authors']:
            if not a in author2doc:
                author2doc[a] = []
            author2doc[a].append(i)

    docs = [v['cleaned'] for k, v in db]
    dic = corpora.Dictionary(docs)
    _ = dic[0]  # ugly
    corpus = [dic.doc2bow(doc) for doc in docs]

    return corpus, author2doc, dic.id2token

train_corpus, author2doc, id2word = db_to_corpus(train_db)

lda = models.AuthorTopicModel(
        train_corpus,
        num_topics=N_TOPICS,
        author2doc=author2doc,
        id2word=id2word,
        )
lda.save('lda.mdl')
