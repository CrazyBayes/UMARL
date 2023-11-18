from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

from .lbforaging import ForagingEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "foraging": partial(env_fn, env=ForagingEnv)
}
