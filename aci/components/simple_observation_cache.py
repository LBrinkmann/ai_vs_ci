# import torch as th

class SimpleCache(object):
    def __init__(self, observation, size):
        if isinstance(observation, tuple):
            raise NotImplementedError("Not working currently")
            # self.memory = (
            #     o.repeat(size, *[1]*len(o.shape))
            #     for o in observation
            # )
            # self.multi = True
        else:
            self.memory = observation.repeat(size, *[1]*len(observation.shape)).transpose(0,1)
            self.multi = False

    def add(self, observation):
        if self.multi:
            # for m, o in zip(self.memory, observation):
            #     m[1:] = m[:-1].clone()
            #     m[0] = o
            raise NotImplementedError("Not working currently")
        else:
            self.memory[:, 1:] = self.memory[:,:-1].clone()
            self.memory[:, 0] = observation

    def get(self):
        return self.memory

    def add_get(self, observation):
        self.add(observation)
        return self.get()
