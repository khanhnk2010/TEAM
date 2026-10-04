import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(-1, -2))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class Bottleneck_Perceptron_2_layer(torch.nn.Module):
    def __init__(self, in_dim):
        super(Bottleneck_Perceptron_2_layer, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.out_fc(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Discriminative_Pattern_Matching_without_sim(nn.Module):
    def __init__(self, agg_num, d_model=2048, n_head=8, d_k=256, d_v=256, attn_dropout=0.1, dropout=0.1, seq_len=8):
        super(Discriminative_Pattern_Matching_without_sim, self).__init__()

        self.agg_num = agg_num
        self.AP = nn.Parameter(torch.randn([1, agg_num, d_model]), requires_grad=True)
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(d_model, 0.1, max_len)

        self.ln = nn.LayerNorm(d_model)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=attn_dropout)

        self.refine = nn.Sequential(
            nn.Dropout(dropout)
        )
        self.mlp = Bottleneck_Perceptron_2_layer(d_model)

    def forward(self, x, other=None, sim_pos=None, sim_neg=None, return_weight=False):
        if other is None:
            AP = self.AP
            agg_num = self.agg_num
            tar_num, cls_num, t, c = x.shape

            x = x.reshape(-1, t, c)
            x = self.pe(x)

            output, attn_weight = self.attention(self.ln(AP), self.ln(x), self.ln(x))
            output = self.refine(output)

            output_pos = AP + output
            output_pos = output_pos + self.mlp(output_pos)
            output_pos = output_pos.reshape(tar_num, cls_num, agg_num, c)

            output_neg = AP - output
            output_neg = output_neg + self.mlp(output_neg)
            output_neg = output_neg.reshape(tar_num, cls_num, agg_num, c)

            if return_weight:
                return output_pos, output_neg, attn_weight

            return output_pos, output_neg

        else:
            AP = self.AP
            agg_num = self.agg_num
            tar_num, cls_num, t, c = x.shape

            x = x.reshape(-1, t, c)
            x = self.pe(x)
            output, attn_weight = self.attention(self.ln(AP), self.ln(x), self.ln(x))
            output = self.refine(output)
            output = output.reshape(tar_num, cls_num, 1, agg_num, c)

            other = other.reshape(-1, t, c)
            other = self.pe(other)
            output_other, attn_weight = self.attention(self.ln(AP), self.ln(other), self.ln(other))
            output_other = self.refine(output_other)
            output_other = output_other.reshape(tar_num, cls_num, cls_num - 1, agg_num, c)

            AP = AP.reshape(1, 1, 1, agg_num, c)

            output_disc_pos = AP + output - (output_other - output)
            output_disc_pos = output_disc_pos + self.mlp(output_disc_pos)

            output_disc_neg = AP - output + (output_other - output)
            output_disc_neg = output_disc_neg + self.mlp(output_disc_neg)

            return output_disc_pos, output_disc_neg


class Discriminative_Pattern_Matching(nn.Module):
    def __init__(self, agg_num, d_model=2048, n_head=8, d_k=256, d_v=256, attn_dropout=0.1, dropout=0.1, seq_len=8):
        super(Discriminative_Pattern_Matching, self).__init__()

        self.agg_num = agg_num
        self.AP = nn.Parameter(torch.randn([1, agg_num, d_model]), requires_grad=True)
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(d_model, 0.1, max_len)

        self.ln = nn.LayerNorm(d_model)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=attn_dropout)

        self.refine = nn.Sequential(
            nn.Dropout(dropout)
        )
        self.mlp = Bottleneck_Perceptron_2_layer(d_model)

    def forward(self, x, other=None, sim_pos=None, sim_neg=None, return_weight=False):
        if other is None:
            AP = self.AP
            agg_num = self.agg_num
            tar_num, cls_num, t, c = x.shape

            x = x.reshape(-1, t, c)
            x = self.pe(x)

            output, attn_weight = self.attention(self.ln(AP), self.ln(x), self.ln(x))
            output = self.refine(output)

            output_pos = AP + output
            output_pos = output_pos + self.mlp(output_pos)
            output_pos = output_pos.reshape(tar_num, cls_num, agg_num, c)

            output_neg = AP - output
            output_neg = output_neg + self.mlp(output_neg)
            output_neg = output_neg.reshape(tar_num, cls_num, agg_num, c)

            if return_weight:
                return output_pos, output_neg, attn_weight

            return output_pos, output_neg

        else:
            AP = self.AP
            agg_num = self.agg_num
            tar_num, cls_num, t, c = x.shape

            x = x.reshape(-1, t, c)
            x = self.pe(x)
            output, attn_weight = self.attention(self.ln(AP), self.ln(x), self.ln(x))
            output = self.refine(output)
            output = output.reshape(tar_num, cls_num, 1, agg_num, c)

            other = other.reshape(-1, t, c)
            other = self.pe(other)
            output_other, attn_weight = self.attention(self.ln(AP), self.ln(other), self.ln(other))
            output_other = self.refine(output_other)
            output_other = output_other.reshape(tar_num, cls_num, cls_num - 1, agg_num, c)

            AP = AP.reshape(1, 1, 1, agg_num, c)
            sim_pos, sim_neg = sim_pos.unsqueeze(dim=-1), sim_neg.unsqueeze(dim=-1)

            output_disc_pos = AP + output - sim_pos * (output_other - output)
            output_disc_pos = output_disc_pos + self.mlp(output_disc_pos)

            output_disc_neg = AP - output + sim_neg * (output_other - output)
            output_disc_neg = output_disc_neg + self.mlp(output_disc_neg)

            return output_disc_pos, output_disc_neg


