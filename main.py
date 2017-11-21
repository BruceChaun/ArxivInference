
import pickle
from dataset import MappedPaperDataset, MappedDataLoader
import torch as T
import viz
import numpy as np
from collections import OrderedDict, Counter
import copy
from nltk.corpus import stopwords

batch_size = 256
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

train_dataset = MappedPaperDataset(train_db, vocab, users)
valid_dataset = MappedPaperDataset(valid_db, vocab, users)
test_dataset = MappedPaperDataset(test_db, vocab, users)

train_dataloader = MappedDataLoader(train_dataset, batch_size=batch_size, num_workers=2)
valid_dataloader = MappedDataLoader(valid_dataset, batch_size=batch_size, num_workers=2)


model = 'bowrank'

bestmodel = None
best_valid_loss = np.inf

thres = 1
lambda_ = 1e-5
lambda_u = 1e-5

wm = viz.VisdomWindowManager(server='http://log-0', port='8098')

if model == 'bowrank':
    embed_size = 20

    from bowrank import BOWRanker
    model = BOWRanker(len(vocab), len(users), embed_size)

opt = T.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0)

for epoch in range(10000):
    model.train()

    train_batches = 0
    train_loss = 0
    for w, u, l, n, v, vs in train_dataloader:
        loss, reg, reg_u = model.loss(u, w, l, n, v, vs, thres=thres)
        total_loss = loss + lambda_ * reg + lambda_u * reg_u
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        loss = np.asscalar(loss.data.numpy())
        train_loss = ((train_loss * train_batches) + loss) / (train_batches + 1)

    model.eval()

    valid_loss = 0
    valid_batches = 0
    for w, u, l, n, v, vs in valid_dataloader:
        loss, _, _ = model.loss(u, w, l, n, v, vs, thres=thres)
        loss = loss.data.numpy()
        valid_loss = ((valid_loss * valid_batches) + loss) / (valid_batches + 1)
        valid_batches += 1

    wm.append_scalar(
            'loss',
            [train_loss, valid_loss],
            opts=dict(
                legend=['train', 'validation']
                ),
            )

    '''
    for pg in opt.param_groups:
        pg['lr'] *= 0.999
        pg['lr'] = max(pg['lr'], 1e-4)
    '''
    if (epoch + 1) % 50 == 0:
        T.save(bestmodel.U.weight.data.numpy(), 'U.p')
        T.save(bestmodel.W.weight.data.numpy(), 'W.p')
        T.save(vocab, 'vocab-selected.p')
        T.save(users, 'users-selected.p')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        bestmodel = copy.deepcopy(model)
