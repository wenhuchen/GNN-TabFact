import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from allennlp.nn import util
from transformers import BertModel, XLNetModel, XLNetForSequenceClassification, BertForSequenceClassification
from torch.nn import Identity


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=False)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r, inplace=False)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MHAtt(nn.Module):
    def __init__(self, head_num, hidden_size, dropout, hidden_size_head):
        super(MHAtt, self).__init__()
        self.head_num = head_num
        self.hidden_size = hidden_size
        self.hidden_size_head = hidden_size_head
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class SA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(SA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        output = self.mhatt(x, x, x, x_mask)
        dropout_output = self.dropout1(output)
        x = self.norm1(x + dropout_output)

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class GA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(GA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, y, y_mask, x_mask=None):
        if x_mask is None:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask))
        else:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask)) * x_mask.unsqueeze(-1)

        x = self.norm1(x + intermediate)
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class SAEncoder(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, layers):
        super(SAEncoder, self).__init__()
        self.encoders = nn.ModuleList([SA(hidden_size=hidden_size, head_num=head_num, ff_size=ff_size,
                                          dropout=dropout, hidden_size_head=hidden_size // head_num) for _ in range(layers)])

    def forward(self, x, x_mask=None):
        for layer in self.encoders:
            x = layer(x, x_mask)
        return x


class GAEncoder(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, layers):
        super(GAEncoder, self).__init__()
        self.encoders = nn.ModuleList([GA(hidden_size=hidden_size, head_num=head_num, ff_size=ff_size,
                                          dropout=dropout, hidden_size_head=hidden_size // head_num) for _ in range(layers)])

    def forward(self, x, y, y_mask, x_mask=None):
        for layer in self.encoders:
            x = layer(x, y, y_mask, x_mask)
        return x


class NumGNN(nn.Module):

    def __init__(self, node_dim, iteration_steps=1):
        super(NumGNN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)
        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, d_node, graph):
        d_node_len = d_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = 1 - diagmat

        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        for step in range(self.iteration_steps):
            d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)

            self_d_node_info = self._self_node_fc(d_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_left,
                0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)

            dd_node_info_right = self._dd_node_fc_right(d_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_right,
                0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)

            agg_d_node_info = (dd_node_info_left + dd_node_info_right) / d_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)

        return d_node


class SequenceSummary(nn.Module):
    r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, config):
        super(SequenceSummary, self).__init__()

        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, ..., seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class TableEncoder(nn.Module):

    def __init__(self, dim, head, model_type, layers=4, dropout=0.1):
        super(TableEncoder, self).__init__()
        self.BASE = BertModel.from_pretrained(model_type)
        self.ROW = SAEncoder(dim, head, 4 * dim, dropout, dim // head, 3)
        self.COL = NumGNN(dim)
        self.FUSION = GAEncoder(dim, head, 4 * dim, dropout, dim // head, 3)
        self.CLASSIFIER = nn.Linear(dim, 2)

    def forward(self, forward_type, *args):
        if forward_type == 'cell':
            return self.BASE(*args)
        elif forward_type == 'row':
            return self.ROW(*args)
        elif forward_type == 'col':
            return self.COL(*args)
        else:
            x = self.FUSION(*args)
            x = self.CLASSIFIER(x)
            return x


class Baseline(nn.Module):

    def __init__(self, dim, head, model_type, label_num, layers=4, dropout=0.1):
        super(Baseline, self).__init__()
        #self.BASE = BertModel.from_pretrained(model_type)
        #self.BASE = XLNetModel.from_pretrained(model_type)
        #self.sequence_summary = SequenceSummary(self.BASE.config)
        #self.FUSION = GAEncoder(dim, head, 4 * dim, dropout, dim // head, 3)
        #self.CLASSIFIER = nn.Linear(dim, label_num)
        if 'xlnet' in model_type:
            self.BASE = XLNetForSequenceClassification.from_pretrained(model_type, num_labels=label_num)
        else:
            self.BASE = BertForSequenceClassification.from_pretrained(model_type, num_labels=label_num)

    def forward(self, forward_type, **kwargs):
        if forward_type == 'cell':
            #outputs = self.BASE(**kwargs)[0]
            #pooled_output = self.sequence_summary(outputs)
            # return self.CLASSIFIER(pooled_output)
            return self.BASE(**kwargs)[0]
        else:
            raise NotImplementedError


class Baseline1(nn.Module):

    def __init__(self, dim, head, model_type, label_num, layers=4, dropout=0.1):
        super(Baseline1, self).__init__()
        self.BASE = BertModel.from_pretrained(model_type)
        self.CLASSIFIER = nn.Linear(dim, label_num)

    def forward(self, forward_type, *args, **kwargs):
        if forward_type == 'cell':
            outputs = self.BASE(**kwargs)[1]
            return self.CLASSIFIER(outputs)
        else:
            raise NotImplementedError


class Baseline2(nn.Module):

    def __init__(self, dim, head, model_type, label_num, layers=4, dropout=0.1):
        super(Baseline2, self).__init__()
        self.BASE = BertModel.from_pretrained(model_type)
        #self.BASE = XLNetModel.from_pretrained(model_type)
        #self.sequence_summary = SequenceSummary(self.BASE.config)
        #self.FUSION = GAEncoder(dim, head, 4 * dim, dropout, dim // head, 3)
        self.embedding = nn.Embedding(2, dim)
        self.GA = GAEncoder(dim, head, 4 * dim, dropout, dim // head, 3)
        self.SA = nn.Linear(dim, 1)
        self.CLASSIFIER = nn.Linear(dim, label_num)

    def forward(self, forward_type, **kwargs):
        if forward_type == 'cell':
            outputs = self.BASE(**kwargs)[1]
            #pooled_output = self.sequence_summary(outputs)
            return self.CLASSIFIER(outputs)
            # return self.BASE(**kwargs)[0]
        elif forward_type == 'row':
            outputs = self.BASE(**kwargs)
            return outputs
        elif forward_type == 'sa':
            # Guided Attention
            output = self.GA(**kwargs)
            scores = self.SA(output).squeeze()
            x_mask = (1 - kwargs['x_mask']).type(torch.bool)
            scores = scores.masked_fill(x_mask, 1e-9)
            att = torch.softmax(scores, -1)
            # Summarize over the outputs
            pooled_output = torch.sum(att.unsqueeze(-1) * output, 1)
            outputs = self.CLASSIFIER(pooled_output)
            return outputs
        else:
            raise NotImplementedError
