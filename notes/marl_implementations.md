## MAgend

https://github.com/geek-ai/MAgent

* library: tensorflow
* logging: print
* action space: discrete
* environment: MAgend
* Model
    * dqn
        * 2x conv
        * 2x dense
    * drqn
        * 2x conv
        * 2x dense
        * 1x GRU
    * a2c
        * 3x dense
        * commnet (inter agent communication)
        * 1x dense
* Learning
    * dqn
    * drqn
* Code comments
    * network and learning is coupled
    * reletive clean code, not very modular

## Pytorch MADDPG

https://github.com/xuehy/pytorch-maddpg

* library: pytorch
* model
    * critic
        * 4x dense
    * actor
        * 3x dense
* code:
    * okayish, relativly clean, not very complex!


## PyMarl

https://github.com/oxwhirl/pymarl

* library: pytorch
* logging: logger
* action space: discrete
* environment: Starcraft
* Model
    * dense
    * gru
    * dense
    * (mix)
* Code
    * very modular and decoupled
    * very clean code

## MADRL

https://github.com/sisl/MADRL

* Code
    * very modular, convoluted and coupled
    * multiple repositories

## A2C

https://github.com/cts198859/deeprl_network

* library: tensorflow
* logging: tensorbord
* action space: continous
* environment: trafic lights

## Mean Field Multi-Agent RL

https://github.com/mlii/mfrl

* library: tensorflow
* logging: print
* action space: discrete
* Model
    * mean field dql
        * 2x conv
        * 2x dense
        * mean field
        * 3x dense
* Code comments
    * similar to MAgent

## Multi-Agent Reinforcement Learning

https://github.com/rohan-sawhney/multi-agent-rl

* library: tensorflow
* logging: x
* action space: continuous
* environment: OpenAi continious
* Model
    * dqn
        * 3x dense
    * ddpg / maddpg
        * actor
            * 2x dense
        * critic
            * 2x dense
* Code comments
    * code okayish, not very modular


## More

https://github.com/cyoon1729/Multi-agent-reinforcement-learning
work in progess, clean decoupled code, slim

https://github.com/mohammadasghari/dqn-multi-agent-rl
not very convincing results