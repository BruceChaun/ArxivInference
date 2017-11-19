
import pickle
from dataset import MappedPaperDataset, MappedDataLoader
import torch as T
import viz
import numpy as np
from collections import OrderedDict
import copy
from nltk.corpus import stopwords

batch_size = 64
stopword_list = stopwords.words('english') + [
        "n't",
        "'ll",
        "'s",
        "'d",
        ]

with open('newdb.p', 'rb') as f:
    db = pickle.load(f)
with open('vocab.p', 'rb') as f:
    vocab = pickle.load(f)
with open('users.p', 'rb') as f:
    users = pickle.load(f)

db = list(db.items())
db_len = len(db)
train_split = db_len * 4 // 5
valid_split = db_len // 10
train_db = db[:train_split]
valid_db = db[train_split:train_split+valid_split]
test_db = db[train_split+valid_split:]

vocab = OrderedDict(
        (w, c) for w, c in vocab.items() if w not in stopword_list)
users = OrderedDict((u, c) for u, c in users.items() if c >= 3)

train_dataset = MappedPaperDataset(train_db, vocab, users)
valid_dataset = MappedPaperDataset(valid_db, vocab, users)
test_dataset = MappedPaperDataset(test_db, vocab, users)

train_dataloader = MappedDataLoader(train_dataset, batch_size=batch_size)
valid_dataloader = MappedDataLoader(valid_dataset, batch_size=batch_size)
test_dataloader = MappedDataLoader(test_dataset, batch_size=batch_size)


model = 'bowrank'

bestmodel = None
best_valid_loss = np.inf

wm = viz.VisdomWindowManager()

if model == 'bowrank':
    embed_size = 10

    from bowrank import BOWRanker
    model = BOWRanker(len(vocab), len(users), embed_size)

opt = T.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

for epoch in range(400):
    train_batches = 0
    train_loss = 0
    for w, u, u_p, l in train_dataloader:
        loss = model.loss(u, u_p, w, l)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss = np.asscalar(loss.data.numpy())
        train_loss = ((train_loss * train_batches) + loss) / (train_batches + 1)

    valid_loss = 0
    valid_batches = 0
    for w, u, u_p, l in valid_dataloader:
        loss = np.asscalar(model.loss(u, u_p, w, l).data.numpy())
        valid_loss = ((valid_loss * valid_batches) + loss) / (valid_batches + 1)
        valid_batches += 1
    wm.append_scalar(
            'loss',
            [train_loss, valid_loss],
            opts=dict(legend=['train', 'validation']),
            )

    if (epoch + 1) % 100 == 0:
        for pg in opt.param_groups:
            pg['lr'] /= 2

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        bestmodel = copy.deepcopy(model)

T.save(bestmodel, 'model.p')
T.save(vocab, 'vocab-selected.p')
T.save(users, 'users-selected.p')
