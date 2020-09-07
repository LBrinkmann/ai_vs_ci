import torch.nn as nn
import torch.nn.functional as F
import torch as th


class PoolingGRUAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size, pooling_type, device):
        super(PoolingGRUAgent, self).__init__()
        self.pooling_type = pooling_type
        # self.linear1 = nn.Linear(in_features=n_actions, out_features=hidden_size)
        self.rnn1 = nn.GRU(input_size=n_actions, hidden_size=hidden_size, batch_first=True)
        self.rnn2 = nn.GRU(
            input_size=hidden_size*2 if pooling_type == 'both' else hidden_size, batch_first=True, 
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
        
        mask_ = mask.permute(0,2,1)
        
        rnn1_out_masked = rnn1_out_ * mask_.unsqueeze(-1)

        if self.pooling_type == 'avg':
            pooled = rnn1_out_masked.mean(dim=1) / rnn1_out_masked.sum(dim=1) # batch, seq_length, hidden_size
        elif self.pooling_type == 'max':
            pooled = rnn1_out_masked.max(dim=1) / rnn1_out_masked.sum(dim=1) # batch, seq_length, hidden_size
        elif self.pooling_type == 'both':
            pooled = th.concat([
                rnn1_out_masked.mean(dim=1) / rnn1_out_masked.sum(dim=1),
                rnn1_out_masked.max(dim=1) / rnn1_out_masked.sum(dim=1)
            ], axis=-1) # batch, seq_length, hidden_size * 2


        rnn2_out, self.hidden2 = self.rnn2(pooled, self.hidden2)
        q = self.linear2(rnn2_out)
        return q
