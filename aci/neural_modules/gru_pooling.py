import torch.nn as nn
import torch.nn.functional as F
import torch as th


class PoolingGRUAgent(nn.Module):
    def __init__(
            self, n_inputs, input_size, secrete_size, n_actions, hidden_size, pooling_types, 
            linear1, rnn1, linear2, rnn2, device):
        super(PoolingGRUAgent, self).__init__()
        self.pooling_types = pooling_types
        self.n_actions = n_actions

        next_in = input_size
        if linear1:
            self.linear1 = nn.Linear(in_features=next_in, out_features=hidden_size)
            next_in = hidden_size
        else:
            self.linear1 = None

        if rnn1:
            self.rnn1 = nn.GRU(input_size=next_in, hidden_size=hidden_size, batch_first=True)
            next_in = hidden_size
        else:
            self.rnn1 = None
        
        # pooling
        next_in = next_in*(len(pooling_types) + 1) + secrete_size

        # if 'attention' in pooling_types:
        #     assert num_heads is not None, 'For attention pooling num heads need to be defined.'
        #     self.attn = nn.MultiheadAttention(next_in, num_heads=num_heads)
        
        if linear2:
            self.linear2 = nn.Linear(in_features=next_in, out_features=hidden_size)
            next_in = hidden_size
        else:
            self.linear2 = None
        if rnn2:
            self.rnn2 = nn.GRU(
                input_size=next_in, batch_first=True, 
                hidden_size=hidden_size)
            next_in = hidden_size
        else:
            self.rnn2 = None

        self.linear3 = nn.Linear(in_features=next_in, out_features=self.n_actions)

    def reset(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, x, mask, secret=None):
        batch_size, seq_length, n_inputs, input_size = x.shape # n_inputs: neighbors + self      
        x = x.permute(0,2,1,3) # batch, n_inputs, seq_length, input_size
        mask = ~mask.permute(0,2,1).unsqueeze(-1) # batch, n_inputs, seq_length, 1

        if self.linear1:
            x = F.relu(self.linear1(x))

        if self.rnn1:
            x = x.reshape(batch_size*n_inputs,seq_length,-1) # batch*n_inputs, seq_length, input_size
            x, self.hidden1 = self.rnn1(x, self.hidden1) # batch*n_inputs, seq_length, hidden_size
            x = x.reshape(batch_size,n_inputs,seq_length,-1) # batch, n_inputs, seq_length, hidden_size

        x_self = x[:, 0]
        x_others = x[:, 1:] * mask[:, 1:]

        pooled = [x_self]
        if secret is not None:
            pooled.append(secret)

        if 'avg' in self.pooling_types:
            pooled.append(x_others.sum(dim=1) / mask.sum(dim=1)) # batch, seq_length, hidden_size
        if 'max' in self.pooling_types:
            pooled.append(x_others.max(dim=1)[0]) # batch, seq_length, hidden_size
        if 'sum' in self.pooling_types:
            pooled.append(x_others.mean(dim=1)) # batch, seq_length, hidden_size
        
        x = th.cat(pooled, axis=-1)

        if self.linear2:
            x = F.relu(self.linear2(x))
        if self.rnn2:
            x, self.hidden2 = self.rnn2(x, self.hidden2)
        q = self.linear3(x)
        return q
