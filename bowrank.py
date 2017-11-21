import torch as T
from torch import nn
import torch.nn.functional as F

class BOWRanker(nn.Module):
    def __init__(self, n_vocab, n_users, embed_size):
        nn.Module.__init__(self)
        self.U = nn.Embedding(n_users + 1, embed_size, padding_idx=0)
        self.W = nn.Embedding(n_vocab + 1, embed_size, padding_idx=0)

    def forward(self, ui, wi, l):
        '''
        ui: user indices
            (batch_size,) LongTensor
        wi: word indices
            (batch_size, max_len) LongTensor
        l: document length
            (batch_size,) LongTensor
        '''

        u = self.U(ui)
        w = self.W(wi)

        w_avg = w.sum(1) / l.float().unsqueeze(1)
        return (u.unsqueeze(1) @ w_avg.unsqueeze(2)).view(-1)

    def loss(self, ui, ui_p, wi, l, thres=-1):
        s = self.forward(ui, wi, l)
        s_p = self.forward(ui_p, wi, l)

        return (s_p - s).clamp(min=thres).mean() - thres


class TFIDFRanker(BOWRanker):

    def forward(self, ui, wi, l, weight):
        '''
        ui, wi, l: same as BOWRanker
        weight: TF-IDF value
            (batch_size, max_len) FloatTensor
        '''
        u = self.U(ui)
        w = self.W(wi)

        w_avg = T.bmm(F.softmax(weight * 10), w).squeeze() / l.float().unsqueeze(1)
        return (u.unsqueeze(1) @ w_avg.unsqueeze(2)).view(-1)

    def loss(self, ui, ui_p, wi, l, weight, thres=-1):
        s = self.forward(ui, wi, l, weight)
        s_p = self.forward(ui_p, wi, l, weight)

        return (s_p - s).clamp(min=thres).mean() - thres

