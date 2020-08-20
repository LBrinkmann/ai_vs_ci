import torch as th
import torch.nn as nn


class OneHotTransformer(nn.Module):
    def __init__(self, batch_size, observation_shape, n_agents, n_actions, model):
        super(OneHotTransformer, self).__init__()
        self.model = model
        self.onehot = th.FloatTensor(batch_size, n_agents, *observation_shape, n_actions)

    def forward(self, x):
           # In your for loop
        self.onehot.zero_()
        self.onehot.scatter_(-1, x.unsqueeze(-1), 1)
        y = self.model(self.onehot)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)


class FloatTransformer(nn.Module):
    def __init__(self, model):
        super(FloatTransformer, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.type(th.float32)
        y = self.model(_x)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()


class FlatObservation(nn.Module):
    def __init__(self, model):
        super(FlatObservation, self).__init__()
        self.model = model

    def forward(self, x):
        """
        output in shape
        agents x batch x inputs
        """
        _x = x.reshape(x.shape[:2],-1)
        y = self.model(_x)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()



class FlatHistory(nn.Module):
    def __init__(self, model):
        super(FlatHistory, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.reshape(x.shape[0],*x.shape[2:])
        y = self.model(_x)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()


class SharedWeightModel(nn.Module):
    def __init__(self, model):
        super(SharedWeightModel, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.unsqueeze(1)
        _y = self.model(_x)
        y = _y.squeeze(1)
        return y

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)


class FakeBatch(nn.Module):
    def __init__(self, model):
        super(FakeBatch, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.unsqueeze(0)
        _y = self.model(_x)
        y = _y.squeeze(0)
        return y

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)