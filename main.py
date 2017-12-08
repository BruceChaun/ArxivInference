# Usage:
# python3 main.py <pickle-file> bowrank <prefix> [<rho>] [<order>]
import sys
from dataset import *
import torch as T
import viz
import numpy as np
import copy

def anynan(v):
    s = (v.data != v.data).long().sum()
    return s != 0

batch_size = 32

train_dataloader = BOWDataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
valid_dataloader = BOWDataLoader(valid_dataset, batch_size=batch_size, num_workers=0)

model_name = sys.argv[2]
prefix = sys.argv[3]

bestmodel = None
best_valid_loss = np.inf

thres = 1
lambda_ = 1e-5
lambda_u = 1e-5
rho = float(sys.argv[4]) if len(sys.argv) > 4 else 0        # Specificity
order = int(sys.argv[5]) if len(sys.argv) > 5 else 2        # Also specificity

wm = viz.VisdomWindowManager(server='http://log-1', port='8098', env=prefix)

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


cuda = T.cuda.is_available()
if cuda:
    model.cuda()


for epoch in range(1000):
    model.train()
    train_batches = 0
    train_loss = 0
    print('Epoch', epoch)

    for ui, uo, wi, wo, wi_p, wo_p in train_dataloader:
        if cuda:
            ui = ui.cuda()
            uo = uo.cuda()
            wi = wi.cuda()
            wo = wo.cuda()
            wi_p = wi_p.cuda()
            wo_p = wo_p.cuda()

        if model_name == 'bowrank':
            loss, reg, reg_u = model.loss(ui, uo, wi, wo, wi_p, wo_p, thres=thres, rho=rho, order=order)
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
        loss = np.asscalar(loss.data.cpu().numpy())
        train_loss = ((train_loss * train_batches) + loss) / (train_batches + 1)
        train_batches += 1

        if train_batches % 1000 == 0:
            print('Train %d/%d' % (train_batches, len(train_dataloader)))

    model.eval()

    valid_loss = 0
    valid_batches = 0
    for ui, uo, wi, wo, wi_p, wo_p in valid_dataloader:
        if cuda:
            ui = ui.cuda()
            uo = uo.cuda()
            wi = wi.cuda()
            wo = wo.cuda()
            wi_p = wi_p.cuda()
            wo_p = wo_p.cuda()

        if model_name == 'bowrank':
            loss, _, _ = model.loss(ui, uo, wi, wo, wi_p, wo_p, thres=thres, rho=rho, order=order)
            loss = loss.data.cpu().numpy()
        elif model_name == 'tfidfrank':
            weight = tf_idf.get_tfs(w.data.numpy(), valid_batches * batch_size)
            weight = T.autograd.Variable(T.Tensor(weight)).unsqueeze(1)
            loss = np.asscalar(model.loss(u, w, l, n, v, vs, weight, thres=thres).data.numpy())

        valid_loss = ((valid_loss * valid_batches) + loss) / (valid_batches + 1)
        valid_batches += 1
        if valid_batches % 1000 == 0:
            print('Valid %d/%d' % (valid_batches, len(train_dataloader)))

    print('Train loss %f  Valid loss %f' % (train_loss, valid_loss))

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
        T.save(bestmodel.U.weight.data.cpu().numpy(), '%s-U.p' % prefix)
        T.save(bestmodel.W.weight.data.cpu().numpy(), '%s-W.p' % prefix)
        T.save(vocab, '%s-vocab-selected.p' % prefix)
        T.save(users, '%s-users-selected.p' % prefix)
