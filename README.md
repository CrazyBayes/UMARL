# UMARL: Unifying  Exploration and Exploitation  for Cooperative Multi-agent Reinforcement Learning
This official source code for UMARL is based on [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) and [PyMARL](https://github.com/oxwhirl/pymarl) framework. Our paper is submitted to "``IEEE Transactions on Neural Networks and Learning Systems``". 

# Prerequisites
## Install dependencies
See ``requirments.txt`` file for more information about how to install the dependencies.
## Environments 
### m-step Matrix Game
Please see [m-step Matrix Game](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f816dc0acface7498e10496222e9db10-Abstract.html) to know this Game.
### Level-based Foraging
[Level-based Foraging (LBF)](https://github.com/semitable/lb-foraging) is a mixed cooperative-competitive game, which focuses on the coordination of the agents involved. Agents navigate a grid world and collect food by cooperating with other agents if needed. Please refer to LBF (https://github.com/semitable/lb-foraging)  to install it.

### StarCraft Multi-Agent Challenge
[StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac) is designed for research in the field of collaborative multi-agent reinforcement learning (MARL). Please refer to SMAC (https://github.com/oxwhirl/smac) to install it. Note that the version of SMAC in our paper is ``SC2.4.6``.

### Google Research Football
[Google Research Football (GRF)](https://github.com/google-research/football) is created by the Google Brain team for research, a new reinforcement learning environment where agents are trained to play football in an advanced, physics-based 3D simulator. Please refer to GRF (https://github.com/google-research/football) to install it.

# How to run UMARL?
```python
python main.py --alg=qweight_vb --map=nstepmatrix  --cuda=True
```

```python
python main.py --alg=qweight_vb --map=foraging  --cuda=True
```
```python
python main.py --alg=qweight_vb --map=8m --cuda=True
```
```python
python main.py --alg=qweight_vb --map=academy_corner  --cuda=True
```

Directly run the ``main.py``, then the algorithm will start training on the setted environment ``foraging``/``8m``/``academy_corner``. Please note that the **qweight_vb** in this project refers to our proposed algorithm **UMARL**.

Or, if you just want to use this project for demonstration, you should set ```--evaluate=True --load_model=True```

# Videos from *GRF: academy_corner*
You can see the videos in the folder ``./videoFromacademy_corner/``.

Or,
#### UMARL
![UMARL](https://github.com/CrazyBayes/UMARL/assets/58516243/72715194-4745-425c-91bb-e22bbaed00ff)
#### QMIX
![QMIX](https://github.com/CrazyBayes/UMARL/assets/58516243/39a635ee-01a1-4b78-99e1-10bb685ea7d3)
# Acknowledgements
We want to express our gratitude to the authors of [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) and [PyMARL](https://github.com/oxwhirl/pymarl) framework for publishing the source codes. 
