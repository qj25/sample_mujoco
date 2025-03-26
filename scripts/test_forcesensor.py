import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import mujoco
import pickle
import random
from time import time
from sample_mujoco.envs.forcesensor_env import TestForceEnv

# from sample_mujoco.utils.transform_utils import IDENTITY_QUATERNION

env1 = TestForceEnv()
ft_applied = np.array([1.,0.,0.,0., 1.0 ,0.])
for i in range(1000000):
    env1.step(ft_applied)
    if i == 100:
        ft_applied2 = ft_applied.copy()
        ft_applied2[4] -= 0.3
        env1.step(ft_applied)
    if i == 10000000000:
        env1.data.eq_active = False


"""
LESSON LEARNT:
    - equality messes up TORQUE senso --> doubles it.
    - better to apply force directly and sense force directly from object of interest.
    - in this case, place force sensor on the freejoint-ed body's site and take its negative.
"""