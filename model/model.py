import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def zero_t(dim_0, dim_1, type):
    return torch.zeros(dim_0, dim_1).type(type)


def _reverse(input_seq, sizes):
    xfs = []
    for x, s in zip(input_seq.transpose(0, 1), sizes):
        xf = torch.flip(x[:s], [0])
        xfs.append(xf)

    return pad_sequence(xfs)


class Attention(nn.Module):

    def __init__(self, mem_dim, cand_dim):
        super(Attention, self).__init__()

        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.transform = nn.Linear(cand_dim, mem_dim, bias=False)

    def forward(self, M, x):
        M_ = M.permute(1, 2, 0)
        x_ = self.transform(x).unsqueeze(1)
        alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]

        return attn_pool


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m, self.D_g, self.D_p, self.D_e = D_m, D_g, D_p, D_e

        self.g_cell = nn.GRUCell(D_m + D_p, D_g)
        self.p_cell = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)

        self.dropout = nn.Dropout(dropout)

        self.attention = Attention(D_g, D_m)

    def forward(self, utt, g_hist, q_prev, e_prev):
        g_prev = g_hist[-1] if len(g_hist) else zero_t(utt.size()[0], self.D_g, utt.type())
        g_ = self.g_cell(torch.cat([utt, q_prev], dim=1), g_prev)
        g_ = self.dropout(g_)

        c_ = self.attention(g_hist, utt) if len(g_hist) else g_prev

        q_ = self.p_cell(torch.cat([utt, c_], dim=1), q_prev)
        q_ = self.dropout(q_)

        e_prev = e_prev if len(e_prev) else zero_t(utt.size()[0], self.D_e, utt.type())
        e_ = self.e_cell(q_, e_prev)
        e_ = self.dropout(e_)

        return g_, q_, e_


class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.q_speakers = {}

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e, dropout)

    def forward(self, utt_batch):
        g_hist = torch.zeros(0).type(utt_batch.type())
        q_speak0 = torch.zeros(utt_batch.size()[1], self.D_p).type(utt_batch.type())
        q_speak1 = torch.zeros(utt_batch.size()[1], self.D_p).type(utt_batch.type())
        q_speakers = {
            0: q_speak0,
            1: q_speak1
        }
        e_ = torch.zeros(0).type(utt_batch.type())
        e = e_

        for iter, utt in enumerate(utt_batch):
            q_ = q_speakers[iter % 2]

            g_, q_, e_ = self.dialogue_cell(utt, g_hist, q_, e_)

            q_speakers[iter % 2] = q_
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            e = torch.cat([e, e_.unsqueeze(0)], 0)

        return e


class DialogueRNNModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h, n_classes=7, dropout_rec=0.5, dropout=0.5, bidirectional=True):
        super(DialogueRNNModel, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h

        self.bidirectional = bidirectional
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout_rec)

        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, dropout_rec)

        d_e_bid = 2 * D_e if bidirectional else D_e
        d_h_bid = 2 * D_h if bidirectional else D_h

        self.linear = nn.Linear(d_e_bid, d_h_bid)
        self.matchatt = Attention(d_e_bid, d_e_bid)
        self.smax_fc = nn.Linear(d_h_bid, n_classes)

    def forward(self, input_batch, sizes):
        emotions_f = self.dialog_rnn_f(input_batch)
        emotions_f = self.dropout_rec(emotions_f)

        emotions = emotions_f
        if self.bidirectional:
            reverse_batch = _reverse(input_batch, sizes)
            emotions_b = self.dialog_rnn_r(reverse_batch)
            emotions_b = self.dropout_rec(emotions_b)

            emotions_b = _reverse(emotions_b, sizes)
            emotions = torch.cat([emotions_f, emotions_b], dim=-1)

        hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)

        probabilities = F.log_softmax(self.smax_fc(hidden), 2)
        return probabilities


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5, bidirectional=True):
        super(LSTMModel, self).__init__()

        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=bidirectional, dropout=dropout)

        d_l_bid = 2 * D_e if bidirectional else D_e
        self.linear = nn.Linear(d_l_bid, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, input_batch, _):
        emotions, hidden = self.lstm(input_batch)

        hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)

        probabilities = F.log_softmax(self.smax_fc(hidden), 2)
        return probabilities
