import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import mujoco
import pickle
import random
from time import time
from sample_mujoco.envs.robotonly_joint import TestJointEnv
from sample_mujoco.envs.robotonly_ik import TestIKEnv

# from sample_mujoco.utils.transform_utils import IDENTITY_QUATERNION

class manip_rope_seq:
    def __init__(
        self,
        plot_dat=False,
    ):
        # self.env = gym.make("sample_mujoco:Test-v0")
        # self.env = FlingLRRLRandEnv()
        self.env = TestIKEnv()

    def runabit(self, duration=100.):
        # self.env.move_to_pos(targ_pos=np.array([-0.3,0.3,0.5]), targ_quat=np.array([0.,1.,0.,0.]))
        # self.env.move_to_pos(targ_pos=np.array([0.3,0.3,0.5]))
        # self.env.move_to_pos()

        # move to start_qpos
        input(self.env.observations['eef_quat'])
        self.env.hold_pos(10.)
        start_qpos = np.array([-np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0])
        self.env.move_to_qpos(start_qpos)
        cur_pos = self.env.observations['eef_pos'].copy()
        new_pos = cur_pos + np.array([0., 0., -0.25])
        new_pos = np.array([0.5, 0.125, 0.25])
        self.env.move_to_pos(targ_pos=new_pos)
        input(self.env.observations['qpos'])
        # # move downwards until contact
        # fk_ee_pose = np.zeros(12)
        # self.env.ik_calc.forward(self.env.observations['qpos'].copy(), fk_ee_pose)
        # fk_eef_mat = np.zeros((4,4))
        # fk_eef_mat[:3,:] = np.asarray(fk_ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
        # fk_eef_mat[3,3] = 1.
        # input(fk_eef_mat)

        self.env.hold_pos(10.)
        # while self.env.data.time < duration:
        #     n_steps += 1
        #     if n_steps % 5 == 0:
        #         cur_qpos[0] += 0.01
        #     print(f"time = {self.env.data.time}")
        #     self.env.step()
        #     self.env.viewer.render()

        # self.env.viewer.close()

env1 = manip_rope_seq()
# env1.env.move_then_hold()
env1.env.move_to_pose(
    targ_pos=np.array([0.15,0.0,0.5]),
    targ_quat=np.array([0.,1.,0.,0.])
)
# for i in range(10000):
#     env1.env.step()
# env1.runabit()