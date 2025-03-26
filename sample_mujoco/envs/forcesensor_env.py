import os
import numpy as np

import mujoco
import mujoco_viewer

import gymnasium as gym
from gymnasium import utils
import pickle

import sample_mujoco.utils.transform_utils as T
# from sample_mujoco.controllers.pose_controller_ur5 import PoseController
from sample_mujoco.utils.mjc_utils import MjSimWrapper
from sample_mujoco.utils.xml_utils import XMLWrapper
import sample_mujoco.utils.mjc2_utils as mjc2
# from sample_mujoco.utils.ik_ur5.Ikfast_ur5 import Uik
# from sample_mujoco.controllers.joint_controller_ur5 import joint_sum
from sample_mujoco.utils.filters import ButterLowPass
from sample_mujoco.utils.mjc_utils import get_contact_force, get_sensor_force

class TestForceEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        do_render=True,
    ):
        utils.EzPickle.__init__(self)

        self.do_render = do_render
        
        xml = self._get_xmlstr()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.sim = MjSimWrapper(self.model, self.data)
        
        if self.do_render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.rend_rate = 1.0
        else:
            self.viewer = None

        # enable joint visualization option:
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        # for i in range(26):
            # print(f"id={i}:  type={mujoco.mju_type2Str(i)}")
            
        for i in range(6):
            print("joint names:")
            print(mjc2.obj_id2name(self.model,"joint",i))
        for i in range(12):
            print("actuator names:")
            print(mjc2.obj_id2name(self.model,"actuator",i))

        # other variables
        self.max_env_steps = 10000000
        self.env_steps = 0
        self.cur_time = 0
        self.dt = self.model.opt.timestep

        # filter
        fs = 1.0 / self.dt
        cutoff = 30
        self.lowpass_filter = ButterLowPass(cutoff, fs, order=5)

        # init obs
        self.observations = dict(
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            box_pos=np.zeros(3),
            box_quat=np.zeros(4),
            ft_world=np.zeros(6),
        )

        # gravity compensation
        self.model.opt.gravity[-1] = 0.0

        self.sim.forward()

        self._init_ids()

        self._get_observations()

        # # pickle stuff
        # self._init_pickletool()
        # self._save_initpickle()
        # self._load_initpickle()

    def _init_ids(self):
        # ref
        self.eef_site_name = "eef_site"
        self.eef_site_idx = mjc2.obj_name2id(
            self.model,
            "site",
            self.eef_site_name
        )
        self.eef_name = "eef_body"
        self.eef_id = mjc2.obj_name2id(
            self.model,"body",self.eef_name
        )

        self.box_site_name = "box_site"
        self.box_name = "box_body"
        self.box_id = mjc2.obj_name2id(
            self.model,"body",self.box_name
        )

    def _get_xmlstr(self):
        # load model
        # update rope model
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world_forcesensor.xml"
        )
        self.xml = XMLWrapper(world_base_path)

        asset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/overall.xml"
        )

        xml_string = self.xml.get_xml_string()

        with open(asset_path, "w+") as f:
            f.write(xml_string)

        return xml_string
    
    def step(self, action=np.zeros(6)):
        obj_id = 2
        # apply global torque on eef_body
        self.data.xfrc_applied[obj_id] += action
        if self.env_steps%100==0:
            print(f"ft_applied = {self.data.xfrc_applied[obj_id]}")
            print(f"cfrc_int = {self.data.cfrc_int.reshape(6,3)}")
            # print(f"cfrc_ext = {self.data.cfrc_ext.reshape(6,3)}")
            print(f"efc_force = {self.data.efc_force}")
            print(f"i_step = {self.env_steps}")

        self.sim.step()
        self.sim.forward()
        self.cur_time += self.dt

        if self.env_steps%self.rend_rate==0:
            if self.do_render:
                self.viewer.render()
                # self.viewer._paused = True

        self.env_steps += 1
    
        done = self.env_steps > self.max_env_steps
        
        return self._get_observations(), 0, done, False, 0
    
    def _get_observations(self):
        # get eef_pos and eef_quat
        self.observations['eef_pos'] = np.array(
            self.data.xpos[self.eef_id]
        ).copy()
        self.observations['eef_quat'] = np.array(
            self.data.xquat[self.eef_id]
        ).copy()

        self.observations['box_pos'] = np.array(
            self.data.xpos[self.box_id]
        ).copy()
        self.observations['box_quat'] = np.array(
            self.data.xquat[self.box_id]
        ).copy()

        # get ft_world
        box_ft = get_sensor_force(
            self.model,
            self.data,
            self.box_name,
            self.observations['box_pos'],
            self.observations['box_quat'],
        )
        box_ft_filtered = self.lowpass_filter(box_ft.reshape((-1, 6)))[0, :]
        f_world = box_ft_filtered[:3]
        t_world = box_ft_filtered[3:]
        f_world = box_ft[:3]
        t_world = box_ft[3:]
        self.observations['ft_world'] = np.concatenate((f_world,t_world))
        if self.env_steps%100==0:
            print("==========================================================")
            # print(f"force_{eef_id} = {self.data.sensordata.reshape((4,3))}")
            print(f"force = {self.observations['ft_world'].reshape((2,3))}")

        return None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.viewer == None:
            if self.do_render:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            if not self.do_render:
                self.viewer.close()
                self.viewer = None
        
        self.sim.forward()

        # # reset controller
        # self.controller.reset()

        # reset obs
        self.observations = dict(
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            box_pos=np.zeros(3),
            box_quat=np.zeros(4),
            ft_world=np.zeros(6),
        )

        # reset time
        self.cur_time = 0   #clock time of episode
        self.env_steps = 0

        # pickle
        # self._load_initpickle()

        return None
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| External Funcs ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def hold_pos(self, hold_time=2.):
        init_time = self.cur_time
        # self.print_collisions()
        while (self.cur_time-init_time) < hold_time:
            # print(f"self.cur_time = {self.cur_time}")
            self.step()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _init_pickletool(self):
        # self.rl2_picklepath = 'rl2_' + str(self.r_pieces) # 'rob3.pickle'
        # self.rl2_picklepath = self.rl2_picklepath + self.stiff_str + '.pickle'
        self.tm3_picklepath = 'tm3' # 'rob3.pickle'
        self.tm3_picklepath = self.tm3_picklepath + '.pickle'
        self.tm3_picklepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + self.tm3_picklepath
        )

    def _save_initpickle(self):
        ## initial movements
        start_qpos = np.array([-np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0])
        self.move_to_qpos(start_qpos)
        self.hold_pos(10.)
        # cur_pos = self.observations['eef_pos'].copy()
        # new_pos = cur_pos + np.array([0., 0., -0.5])
        # self.move_to_pos(targ_pos=new_pos)

        ## create pickle
        self.init_pickle = self.get_state()

        ## save pickle
        with open(self.tm3_picklepath, 'wb') as f:
            pickle.dump(self.init_pickle,f)
        input('Pickle saved!')

    def _load_initpickle(self):
        with open(self.tm3_picklepath, 'rb') as f:
            self.init_pickle = pickle.load(f)
        self.set_state(self.init_pickle)

    # def get_state(self):
    #     prev_ctrlstate = self.controller.get_ctrlstate()
    #     # (
    #     #     ropeend_pos,
    #     #     ropeend_quat,
    #     #     overall_rot,
    #     #     p_thetan
    #     # ) = self.der_sim.get_dersim()
    #     state = np.empty(
    #         mujoco.mj_stateSize(
    #             self.model,
    #             mujoco.mjtState.mjSTATE_PHYSICS
    #         )
    #     )
    #     mujoco.mj_getState(
    #         self.model, self.data, state,
    #         spec=mujoco.mjtState.mjSTATE_PHYSICS
    #     )
    #     return [
    #         np.concatenate((
    #             [0], # [self.cur_time],
    #             [0], # [self.env_steps],
    #             # ropeend_pos,
    #             # ropeend_quat,
    #             # [overall_rot],
    #             # [p_thetan],
    #         )),
    #         prev_ctrlstate,
    #         state
    #     ]
    
    # def set_state(self, p_state):
    #     self.cur_time = 0 # p_state[0][0]
    #     self.env_steps = 0 # p_state[0][1]
    #     # self.der_sim.set_dersim(
    #     #     p_state[0][2:5],
    #     #     p_state[0][5:9],
    #     #     p_state[0][9],
    #     #     p_state[0][10],
    #     # )
    #     mujoco.mj_setState(
    #         self.model, self.data, p_state[2],
    #         spec=mujoco.mjtState.mjSTATE_PHYSICS
    #     )
    #     self.controller.set_ctrlstate(p_state[1])
    #     self.sim.forward()
    #     # self.controller.reset()
    #     self.controller.update_state()
    #     self._get_observations()
    #     # self.der_sim._update_xvecs()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~

# TestEnv()