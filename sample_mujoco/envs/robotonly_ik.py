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
from sample_mujoco.utils.ik_utils import ik_denso
# from sample_mujoco.utils.ik_ur5.Ikfast_ur5 import Uik
# from sample_mujoco.controllers.joint_controller_ur5 import joint_sum



class TestIKEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        do_render=True,
    ):
        utils.EzPickle.__init__(self)

        self.do_render = do_render
        
        xml, arm_xml = self._get_xmlstr()

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

        # misc class data
        self.dt = self.model.opt.timestep

        # for i in range(26):
            # print(f"id={i}:  type={mujoco.mju_type2Str(i)}")
            
        for i in range(6):
            print("joint names:")
            print(mjc2.obj_id2name(self.model,"joint",i))
        for i in range(12):
            print("actuator names:")
            print(mjc2.obj_id2name(self.model,"actuator",i))

        # ref
        self.eef_site_name = "eef_site"
        self.eef_site_idx = mjc2.obj_name2id(
            self.model,
            "site",
            self.eef_site_name
        )
        actuator_names = []
        for i in range(6):
            actuator_names.append(mjc2.obj_id2name(self.model,"actuator",i))
        self._init_ids(actuator_names)
        grav_comp_act = []
        for i in range(6,12):
            grav_comp_act.append(mjc2.obj_id2name(self.model,"actuator",i))
        self.gcact_id = [
            mjc2.obj_name2id(self.model, "actuator", n)
            for n in grav_comp_act
        ]

        # other variables
        self.max_env_steps = 10000000
        self.env_steps = 0
        self.cur_time = 0
        self.dt = self.model.opt.timestep

        # init obs
        self.observations = dict(
            qpos=np.zeros(6),
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(6),
            ft_world=np.zeros(6),
        )

        # self.init_qpos = np.array(
            # [
                # -1.56900351e+00, 0.36831498e+00,  1.03298897e+00,
                # 0., 1.57133016e+00, -6.38394218e-08
            # ]
            # # [0.1,0.1,0.1,0.1,0.1,0.1]
        # )
        self.init_qpos = np.array([
            0.03361682,  0.56147072,  1.93752008,
            -0.27528426,  0.66659274, 1.8215281
        ])

        self.init_qvel = np.zeros(6)
        self.data.qpos[self.joint_qposids[:6]] = np.array(self.init_qpos)
        self.data.qvel[self.joint_dofids[:6]] = np.array(self.init_qvel)
        self.current_joint_positions = np.array([self.data.qpos[ji] for ji in self.joint_ids])        

        # gravity compensation
        self.model.opt.gravity[-1] = -9.81
        self.grav_comp()

        self.sim.forward()

        self._get_observations()

        self.ik_arm = ik_denso(init_qpos=self.init_qpos)

        # # pickle stuff
        # self._init_pickletool()
        # self._save_initpickle()
        # self._load_initpickle()

    def _init_ids(self, actuator_names):
        self.actuator_ids = [
            mjc2.obj_name2id(self.model, "actuator", n)
            for n in actuator_names
        ]
        self.actuator_names = actuator_names
        # print(self.actuator_ids)
        # print(actuator_names)
        # input()
        self.joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.joint_qposids = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofids = self.model.jnt_dofadr[self.joint_ids]

    def _get_xmlstr(self):
        # load model
        # update rope model
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world.xml"
        )
        robot_path = os.path.join(
            os.path.dirname(world_base_path),
            "densovs060/densovs060.xml"
        )
        self.xml = XMLWrapper(world_base_path)
        # derrope = XMLWrapper(rope_path)
        # anchorbox = XMLWrapper(box_path)
        robotarm = XMLWrapper(robot_path)
        # pandagripper = XMLWrapper(gripper_path)
        # miscobj = XMLWrapper(miscobj_path)

        self.xml.merge_multiple(
            robotarm, ["worldbody", "asset", "actuator", "extension"]
            # robotarm, ["worldbody", "asset", "actuator", "default", "keyframe", "sensor"]
        )

        asset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/overall.xml"
        )

        xml_string = self.xml.get_xml_string()

        with open(asset_path, "w+") as f:
            f.write(xml_string)

        # self.xml.merge_multiple(
        #     anchorbox, ["worldbody", "equality", "contact"]
        # )
        # self.xml.merge_multiple(
        #     derrope, ["worldbody"]
        # )
        # self.xml.merge_multiple(
        #     derrope, ["sensor"]
        # )
        # self.xml.merge_multiple(
        #     miscobj, ["worldbody"]
        # )
        
        return xml_string, robotarm
    
    def step(self, action=np.zeros(6)):
        self.grav_comp()

        self.data.ctrl[self.joint_ids] = action

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
        # get eef_vel
        ee_vel = mjc2.obj_getvel(
            self.model,
            self.data,
            "site",
            self.eef_site_idx
        )

        self.observations['eef_vel'] = ee_vel.copy()
        
        # get eef_pos and eef_quat
        eef_pos = np.array(
            self.data.site_xpos[self.eef_site_idx]
        )
        # print(mjc2.obj_id2name(self.model, "site", self.eef_site_idx))
        # input("here")
        self.observations['eef_pos'] = eef_pos
        eefmat = np.array(
            self.data.site_xmat[self.eef_site_idx].reshape((3, 3))
        )
        # print('here')
        # print(self.observations['eef_pos'])
        eef_quat = T.mat2quat(eefmat)
        # if np.linalg.norm(eef_quat+self.prev_quat) < 1e-3:
        #     self.quat_switch *= -1
        self.observations['eef_quat'] = eef_quat
        self.prev_quat = eef_quat
        self.eef_mat = eefmat

        self.observations["qpos"] = self.data.qpos[self.joint_qposids[:6]].copy()
        self.observations["qvel"] = self.data.qvel[self.joint_dofids[:6]].copy()

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
        
        self.data.qpos[self.joint_qposids[:6]] = np.array(self.init_qpos)
        self.data.qvel[self.joint_dofids[:6]] = np.array(self.init_qvel)

        self.sim.forward()

        # # reset controller
        # self.controller.reset()

        # reset obs
        self.observations = dict(
            qpos=np.zeros(6),
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(6),
            ft_world=np.zeros(6),
        )

        # reset time
        self.cur_time = 0   #clock time of episode
        self.env_steps = 0

        # pickle
        # self._load_initpickle()

        return None
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| External Funcs ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # def move_to_qpos(self, targ_pos):
    #     # move to joint pose
    #     # intialize variables
    #     done = False
    #     step_counter = 0
        
    #     while not done:
    #         step_counter += 1
    #         j_pos = self.observations['qpos']
    #         move_dir = targ_pos[:6] - j_pos
    #         move_dir = move_dir * 10
    #         move_step = move_dir.copy()
    #         self.step(move_step)
    #         done = T.check_proximity(
    #             targ_pos, self.observations['qpos'], d_tol=5e-3
    #         )
    #     print(step_counter)

    def hold_pos(self, hold_time=2.):
        init_time = self.cur_time
        # self.print_collisions()
        while (self.cur_time-init_time) < hold_time:
            # print(f"self.cur_time = {self.cur_time}")
            self.step()

    def move_then_hold(self):
        for i in range(10000):
            joint_add = np.zeros(6)
            joint_add[0] = 0.00002*self.env_steps
            joint_add[2] = -0.00003*self.env_steps
            j_desired = np.array(self.current_joint_positions+joint_add)
            self.step(action=j_desired)
        
        for i in range(10000):
            self.step(action=j_desired)
            
        final_qpos = np.array([self.data.qpos[ji] for ji in self.joint_ids])
        print(final_qpos)
        print(j_desired)
        print(j_desired-final_qpos)
        eef_pos = np.array(
            self.data.site_xpos[self.eef_site_idx]
        )
        eefmat = np.array(
            self.data.site_xmat[self.eef_site_idx].reshape((3, 3))
        )
        # print('here')
        # print(self.observations['eef_pos'])
        eef_quat = T.mat2quat(eefmat)
        print(eef_pos)
        print(T.quat2axisangle(eef_quat))

        self.viewer._paused = True
        self.step(action=j_desired)


    def grav_comp(self):
        self.base_torq = self.sim.data.qfrc_bias[self.joint_ids]
        self.sim.data.ctrl[self.gcact_id] = self.base_torq       

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| IK ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def move_to_qpos(
        self,
        targ_qpos,
        qpos_stepsize=0.005
    ):
        rob_qpos = self.observations['qpos'].copy()
        qpos_diff = targ_qpos-rob_qpos
        total_dqpos = np.linalg.norm(qpos_diff)
        total_steps = int(total_dqpos/qpos_stepsize+0.5)
        if total_steps > 0:
            qpos_step = qpos_diff / total_steps
            for i_step in range(total_steps+1):
                qpos_next = rob_qpos+qpos_step*i_step
                # print(self.observations['qpos'])
                # print(qpos_next)
                self.step(action=qpos_next)
        else:
            qpos_next = targ_qpos
        for i in range(10000):
            # print(qpos_next)
            # print(self.observations['qpos'])
            self.step(action=qpos_next)

        # mujoco.mj_forward(self.model, self.data)
        # print(repr(targ_qpos))
        # print(repr(qpos_next))
        input()
        self.step(action=qpos_next)

    def move_to_pose(
        self,
        targ_pos=None,
        targ_quat=None
    ):
        rob_pos = self.observations['eef_pos'].copy()
        rob_quat = self.observations['eef_quat'].copy()
        if targ_pos is None:
            targ_pos = rob_pos
        else:
            targ_pos = targ_pos
        if targ_quat is None:
            targ_quat = rob_quat # np.array([0, 1., 0, 0])
        else:
            targ_quat = targ_quat

        goal = np.concatenate((targ_pos,targ_quat))
        targ_qpos = self.ik_arm.calc_ik(goal, self.observations['qpos']).copy()
        if targ_qpos is None:
            input("targ pose out of reach!")
        self.move_to_qpos(targ_qpos=targ_qpos)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End IK ||~~~~~~~~~~~~~~~~~~~~~~~~~~
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