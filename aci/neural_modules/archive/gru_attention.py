import torch.nn as nn
import torch.nn.functional as F
import torch as th


class AttentionGRUAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size, num_heads, device):
        super(AttentionGRUAgent, self).__init__()
        # self.linear1 = nn.Linear(in_features=n_actions, out_features=hidden_size)
        self.rnn1 = nn.GRU(input_size=n_actions, hidden_size=hidden_size, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.rnn2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=n_actions)

    def reset(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, inputs, mask):
        batch_size, seq_length, input_shape, n_actions = inputs.shape
        inputs_ = inputs.permute(0,2,1,3) # batch, input_shape, seq_length, n_actions
        inputs_ = inputs_.reshape(batch_size*input_shape,seq_length,n_actions) # batch*input_shape, seq_length, n_actions

        # lin1_out = F.relu(self.linear1(inputs_))
        rnn1_out, self.hidden = self.rnn1(inputs_, self.hidden1)  # batch*input_shape, seq_length, n_hidden

        n_hidden = rnn1_out.shape[-1]

        rnn1_out_ = rnn1_out.reshape(batch_size,input_shape,seq_length,n_hidden)
        rnn1_out_ = rnn1_out_.permute(1,0,2,3)  # input_shape, batch_size, seq_length, n_hidden
        rnn1_out_ = rnn1_out_.reshape(input_shape,batch_size*seq_length,n_hidden)  # input_shape, batch * seq_length, n_hidden

        mask_ = mask.reshape(batch_size * seq_length, input_shape)

        attn_out, attn_out_w = self.attn(rnn1_out_[[0]], rnn1_out_, rnn1_out_, mask_)
        attn_out_ = attn_out.reshape(batch_size,seq_length,attn_out.shape[-1])

        rnn2_out, self.hidden2 = self.rnn2(attn_out_, self.hidden2)
        q = self.linear2(rnn2_out)
        return q
