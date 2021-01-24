# Ephemeral Value Adjustments based on pfrl

![figure](./figure/eva.png)
- from https://arxiv.org/abs/1810.08163

I implemented an algorithm called Ephemeral Value Adjustments (EVA) using [pfrl](https://github.com/pfnet/pfrl).

[Japanese README](./README_JP.md)

## EVA
- [Fast deep reinforcement learning using online adjustments from the past](https://arxiv.org/abs/1810.08163)

## Requirement
- python >= 3.8
    - pfrl
    - OpenAI gym
        - atari-py >= 0.2.6 

## Usage
    python train.py [options]
### Options
- `--env` : Atari環境の環境名.
    - 例 : `--env PongNoFrameskip-v4`
- `--gpu` : ０だったらGPUを使います。もし0未満の数値を指定すればGPUは使いません。
- `--agent` : DQNとEVAを選べます