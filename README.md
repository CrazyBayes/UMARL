# QWeight:
The official source code for QWeight based on [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) and [PyMARL](https://github.com/oxwhirl/pymarl) framework. Our paper is submitted to "``IJCAI2024``".

# Prerequisites
## Install dependencies
See ``requirments.txt`` file for more information about how to install the dependencies.
## Environments 
### Level-based Foraging
[Level-based Foraging (LBF)](https://github.com/semitable/lb-foraging) is a mixed cooperative-competitive game, which focuses on the coordination of the agents involved. Agents navigate a grid world and collect food by cooperating with other agents if needed. Please refer to LBF (https://github.com/semitable/lb-foraging)  to install it.

### StarCraft Multi-Agent Challenge
[StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac) is designed for research in the field of collaborative multi-agent reinforcement learning (MARL). Please refer to SMAC (https://github.com/oxwhirl/smac) to install it. Note that the version of SMAC in our paper is ``SC2.4.6``.

### Google Research Football
[Google Research Football (GRF)](https://github.com/google-research/football) is created by the Google Brain team for research, a new reinforcement learning environment where agents are trained to play football in an advanced, physics-based 3D simulator. Please refer to GRF (https://github.com/google-research/football) to install it.

# How to run QWeight?
```python
python main.py --alg=qweight_vb --map=foraging  --cuda=True
```
```python
python main.py --alg=qweight_vb --map=8m --cuda=True
```
```python
python main.py --alg=qweight_vb --map=academy_corner  --cuda=True
```

Directly run the ``main.py``, then the algorithm will start training on the setted environment ``foraging``/``8m``/``academy_corner``. Please note that the **qweight_vb** in this project refers to our proposed algorithm **QWeight**.

Or, if you just want to use this project for demonstration, you should set ```--evaluate=True --load_model=True```

# Acknowledgements
We want to express our gratitude to the authors of [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) and [PyMARL](https://github.com/oxwhirl/pymarl) framework for publishing the source codes. 
