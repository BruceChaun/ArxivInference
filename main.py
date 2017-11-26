import sys
from dataset import *
import torch as T
import viz
import numpy as np
import copy

def anynan(v):
    s = (v.data != v.data).long().sum()
    return s != 0

batch_size = 256

train_dataloader = MappedDataLoader(train_dataset, batch_size=batch_size, num_workers=2)
valid_dataloader = MappedDataLoader(valid_dataset, batch_size=batch_size, num_workers=2)

model_name = sys.argv[1]
prefix = sys.argv[2]

bestmodel = None
best_valid_loss = np.inf

thres = 1
lambda_ = 1e-5
lambda_u = 1e-5
rho = float(sys.argv[3]) if len(sys.argv) > 3 else 0        # Specificity
order = int(sys.argv[4]) if len(sys.argv) > 4 else 2        # Also specificity

wm = viz.VisdomWindowManager(server='http://log-0', port='8098', env=prefix)

if model_name == 'bowrank':
    embed_size = 20

    from bowrank import BOWRanker
    model = BOWRanker(len(vocab), len(users), embed_size)

    opt = T.optim.Adamax(model.parameters())
elif model_name == 'tfidfrank':
    embed_size = 10

    from bowrank import TFIDFRanker
    from tfidf import TFIDF
    model = TFIDFRanker(len(vocab), len(users), embed_size)
    tf_idf = TFIDF(db, train_dataset.vocab_imap, train_dataset.vocab_map, stopword_list)

    opt = T.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

for epoch in range(10000):
    model.train()
    train_batches = 0
    train_loss = 0
    for w, w_p, u, l, l_p, n, v, vs in train_dataloader:
        if model_name == 'bowrank':
            loss, reg, reg_u = model.loss(u, w, w_p, l, l_p, n, v, vs, thres=thres, rho=rho)
        elif model_name == 'tfidfrank':
            weight = tf_idf.get_tfs(w.data.numpy(), batches * batch_size)
            weight = T.autograd.Variable(T.Tensor(weight)).unsqueeze(1)
            loss, reg, reg_u = model.loss(u, w, l, n, v, vs, weight, thres=thres)

        total_loss = loss + lambda_ * reg + lambda_u * reg_u
        assert not anynan(total_loss)

        opt.zero_grad()
        total_loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not anynan(p.grad)
        opt.step()
        loss = np.asscalar(loss.data.numpy())
        train_loss = ((train_loss * train_batches) + loss) / (train_batches + 1)

    model.eval()

    valid_loss = 0
    valid_batches = 0
    for w, w_p, u, l, l_p, n, v, vs in valid_dataloader:
        if model_name == 'bowrank':
            loss, _, _ = model.loss(u, w, w_p, l, l_p, n, v, vs, thres=thres, rho=rho)
            loss = loss.data.numpy()
        elif model_name == 'tfidfrank':
            weight = tf_idf.get_tfs(w.data.numpy(), valid_batches * batch_size)
            weight = T.autograd.Variable(T.Tensor(weight)).unsqueeze(1)
            loss = np.asscalar(model.loss(u, w, l, n, v, vs, weight, thres=thres).data.numpy())

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

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        bestmodel = copy.deepcopy(model)
        T.save(bestmodel.U.weight.data.numpy(), '%s-U.p' % prefix)
        T.save(bestmodel.W.weight.data.numpy(), '%s-W.p' % prefix)
        T.save(vocab, '%s-vocab-selected.p' % prefix)
        T.save(users, '%s-users-selected.p' % prefix)
