import torch as T
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import os

class BOWRanker(nn.Module):
    def __init__(self, n_vocab, n_users, embed_size):
        nn.Module.__init__(self)
        self.n_users = n_users
        self.U = nn.Embedding(n_users + 1, embed_size, padding_idx=0)
        self.W = nn.Embedding(n_vocab + 1, embed_size, padding_idx=0)

    def forward(self, ui, wi, l, n):
        '''
        ui: user indices
            (batch_size, n_users) LongTensor
        wi: word indices
            (batch_size, max_len) LongTensor
        l: document length
            (batch_size,) LongTensor
        '''
        if ui.dim() == 1:
            ui = ui.unsqueeze(1)
        batch_size = ui.size()[0]

        u = self.U(ui)
        w = self.W(wi)

        u_pos_sum = u.sum(1)
        u_neg_sum = self.U.weight[1:].sum(0, keepdim=True) - u_pos_sum
        u_pos_avg = u_pos_sum / n.float().unsqueeze(1)
        u_neg_avg = u_neg_sum / (self.n_users - n.float()).unsqueeze(1)
        w_avg = w.sum(1) / l.float().unsqueeze(1)

        s_pos = (u_pos_avg.unsqueeze(1) @ w_avg.unsqueeze(2)).view(-1)
        s_neg = (u_neg_avg.unsqueeze(1) @ w_avg.unsqueeze(2)).view(-1)

        '''
        bU = T.cat([T.autograd.Variable(T.zeros(1, 1)), self.bU], 1)
        bu_pos_sum = bU.expand(batch_size, self.n_users + 1).gather(ui, 1).sum(1)
        bu_neg_sum = bU.sum(1) - bu_pos_sum
        bu_pos_avg = bu_pos_sum / n.float()
        bu_neg_avg = bu_neg_sum / (self.n_users - n.float())
        '''
        return s_pos, s_neg

    def loss(self, ui, wi, wi_p, l, l_p, n, vi, vs, thres=1, rho=0.1):
        s, s_p = self.forward(ui, wi, l, n)
        s_p2, _ = self.forward(ui, wi_p, l_p, n)

        w = self.W(vi)
        w_norms = (self.W.weight ** 2).sum()
        if rho != 0:
            w_norms += rho * (w.norm(2, 2) ** order).sum()
        u_norms = (self.U.weight ** 2).sum()

        return ((thres + s_p - s).clamp(min=0).mean() +
                (thres + s_p2 - s).clamp(min=0).mean(),
                w_norms, u_norms)


class TFIDFRanker(BOWRanker):

    def forward(self, ui, wi, l, n, weight):
        '''
        ui, wi, l: same as BOWRanker
        weight: TF-IDF value
            (batch_size, max_len) FloatTensor
        '''
        if ui.dim() == 1:
            ui = ui.unsqueeze(1)
        batch_size = ui.size()[0]

        u = self.U(ui)
        w = self.W(wi)

        u_pos_sum = u.sum(1)
        u_neg_sum = self.U.weight.sum(0, keepdim=True) - u_pos_sum
        u_pos_avg = u_pos_sum / n.float().unsqueeze(1)
        u_neg_avg = u_neg_sum / (self.n_users - n.float()).unsqueeze(1)

        w_avg = T.bmm(F.softmax(weight * 10), w).squeeze() / l.float().unsqueeze(1)
        return ((u_pos_avg.unsqueeze(1) @ w_avg.unsqueeze(2)).view(-1),
                (u_neg_avg.unsqueeze(1) @ w_avg.unsqueeze(2)).view(-1))

    def loss(self, ui, wi, l, n, vi, vs, weight, thres=-1):
        s, s_p = self.forward(ui, wi, l, n, weight)

        w = self.W(vi)
        w_norms = (self.W.weight ** 2).sum()
        u_norms = (self.U.weight ** 2).sum()

        return (s_p - s).clamp(min=thres).mean() - thres
