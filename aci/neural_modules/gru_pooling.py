import torch.nn as nn
import torch.nn.functional as F
import torch as th


class PoolingGRUAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size, pooling_types, device):
        super(PoolingGRUAgent, self).__init__()
        self.pooling_types = pooling_types

        # self.linear1 = nn.Linear(in_features=n_actions, out_features=hidden_size)
        self.rnn1 = nn.GRU(input_size=n_actions, hidden_size=hidden_size, batch_first=True)
        self.rnn2 = nn.GRU(
            input_size=hidden_size*(len(pooling_types) + 1), batch_first=True, 
            hidden_size=hidden_size)

        self.linear2 = nn.Linear(in_features=hidden_size, out_features=n_actions)

    def reset(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, inputs, mask):
        batch_size, seq_length, input_shape, n_actions = inputs.shape # input_shape: neighbors + self
        inputs_ = inputs.permute(0,2,1,3) # batch, input_shape, seq_length, n_actions
        inputs_ = inputs_.reshape(batch_size*input_shape,seq_length,n_actions) # batch*input_shape, seq_length, n_actions

        # lin1_out = F.relu(self.linear1(inputs_))
        rnn1_out, self.hidden = self.rnn1(inputs_, self.hidden1) # batch*input_shape, seq_length, hidden_size

        rnn1_out_ = rnn1_out.reshape(batch_size,input_shape,seq_length,rnn1_out.shape[-1]) # batch, input_shape, seq_length, hidden_size
        
        rnn1_out_self = rnn1_out_[:, 0]

        mask_ = ~mask.permute(0,2,1)[:, 1:].unsqueeze(-1)
        rnn1_out_others = rnn1_out_[:, 1:] * mask_

        pooled = [rnn1_out_self]

        if 'avg' in self.pooling_types:
            pooled.append(rnn1_out_others.sum(dim=1) / mask_.sum(dim=1)) # batch, seq_length, hidden_size
        if 'max' in self.pooling_types:
            pooled.append(rnn1_out_others.max(dim=1)[0]) # batch, seq_length, hidden_size
        if 'sum' in self.pooling_types:
            pooled.append(rnn1_out_others.mean(dim=1)) # batch, seq_length, hidden_size
        
        pooled_ = th.cat(pooled, axis=-1)

        rnn2_out, self.hidden2 = self.rnn2(pooled_, self.hidden2)
        q = self.linear2(rnn2_out)
        return q
