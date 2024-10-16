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



class TestJointEnv(gym.Env, utils.EzPickle):
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

        # for n in self.model.actuator_names[:]:
        #     print(n)

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

        self.init_qpos = np.array(
            [
                -1.56900351e+00, -1.36831498e+00,  2.03298897e+00,
                -2.22930662e+00, 1.57133016e+00, -6.38394218e-08
            ]
            # [0.1,0.1,0.1,0.1,0.1,0.1]
        )
        # self.init_qpos = np.array([-np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0])
        # self.init_qpos = np.zeros(6)
        self.init_qvel = np.zeros(6)
        self.data.qpos[self.joint_qposids[:6]] = np.array(self.init_qpos)
        self.data.qvel[self.joint_dofids[:6]] = np.array(self.init_qvel)
        self.current_joint_positions = np.array([self.data.qpos[ji] for ji in self.joint_ids])        
        # self.current_joint_positions = np.array([
        #     np.pi/2,
        #     np.pi/2,
        #     -np.pi/2,
        #     np.pi/2,
        #     np.pi/2,
        #     np.pi/2
        # ])
        self.pos_check = 4
        self.prev_qpos = self.data.qpos[self.pos_check]


        # gravity compensation
        self.model.opt.gravity[-1] = -9.81
        self.grav_comp()

        self.sim.forward()

        # self.controller = PoseController(
        #     self.model,
        #     self.data,
        #     actuator_names=actuator_names,
        #     eef_name="wrist_3_link",
        #     eef_site_name="eef_site",
        # )


        # mujoco.mj_resetData(self.model, self.data)
        self._get_observations()

        # self._init_ik(arm_xml=arm_xml)

        # # init gravity
        # self.model.opt.gravity[-1] = 0.

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

        for idx, joint_id in enumerate(self.joint_ids):
            self.data.ctrl[joint_id] = action[idx]
        # print([self.data.qpos[self.pos_check]]-self.prev_qpos)
        self.prev_qpos = self.data.qpos[self.pos_check]

        self.sim.step()
        self.sim.forward()
        self.cur_time += self.dt

        if self.env_steps%10==0:
            if self.do_render:
                self.viewer.render()
                # self.viewer._paused = True

        self.env_steps += 1
    
        done = self.env_steps > self.max_env_steps
        
        return self._get_observations(), 0, done, False, 0
    
    def _get_observations(self):
        # robot stuff
        # self.observations["eef_vel"] = self.controller.state["eef_vel"]
        # self.observations["eef_pos"] = self.controller.state["eef_pos"]
        # self.observations["eef_quat"] = self.controller.state["eef_quat"]
        # self.observations["qpos"] = self.controller.state["qpos"]
        # self.observations["ft_world"] = self.controller.state["ft_world"]

        # print(f"eef_pos = {self.observations['eef_pos']}")

        
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
            joint_add[self.pos_check] = -0.00003*self.env_steps
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
        # print(self._state['eef_pos'])
        eef_quat = T.mat2quat(eefmat)
        print(eef_pos)
        print(T.quat2axisangle(eef_quat))

        self.viewer._paused = True
        self.step(action=j_desired)


    def grav_comp(self):
        self.base_torq = self.sim.data.qfrc_bias[self.joint_ids]
        self.sim.data.ctrl[self.gcact_id] = self.base_torq       

    def move_to_pose(
        self,
        targ_pos=None,
        targ_quat=None,
        targ_qpos=None
    ):
        self.rob_pos = self.observations['eef_pos'].copy()
        self.rob_quat = self.observations['eef_quat'].copy()
        if targ_pos is None:
            self.targ_pos = self.rob_pos
        else:
            self.targ_pos = targ_pos
        if targ_quat is None:
            self.targ_quat = self.rob_quat # np.array([0, 1., 0, 0])
        else:
            self.targ_quat = targ_quat

        done = False
        while not done:
            move_dir = targ_pos - self.rob_pos # + reverse_move # +[0, -0.2, -5]
            move_ori = T.quat_error(self.rob_quat, self.targ_quat)
            # move_ori = np.array([0.01, 0., 0.])
            # move_dir *= 10.
            move_step = np.concatenate((move_dir, move_ori))
            # print(self.rob_pos)
            # input(move_step)
            # print(T.quat_slerp(self.rob_quat, self.targ_quat, 1))
            # print(f"targ_pos = {self.targ_pos}")
            # print(f"rob_pos = {self.rob_pos}")

            # print(f"targ_quat = {self.targ_quat}")
            # print(f"rob_quat = {self.rob_quat}")
            # print(f"move_ori = {move_ori}")
            # input()
            obs, rew, done, _, info = self.step(move_step)
            self.rob_qpos = self.observations['qpos'].copy()
            
            if (
                T.check_proximity(
                    self.rob_pos, self.targ_pos,
                    d_tol=5e-3
                )
                and T.check_proximity(
                    self.rob_quat, self.targ_quat,
                    check_mode='quat',
                    d_tol=5e-1
                )
            ):
                done = True
            done = False
            # a1 = T.check_proximity(
            #         self.rob_pos, self.targ_pos,
            #         d_tol=5e-3
            #     )
            # b1 = T.check_proximity(
            #         self.rob_quat, self.targ_quat,
            #         check_mode='quat',
            #         d_tol=5e-1
            #     )
            # print(self.cur_time)
            # print(f"a = {a1}")
            # print(f"b = {b1}")
            # print(done)
            self.rob_pos = self.observations['eef_pos'].copy()
            self.rob_quat = self.observations['eef_quat'].copy()

        return obs, 0, done, False, 0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| IK ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def _init_ik(self):
    #     self.ik_calc = Uik()
    #     ## find ikposediff:
    #     # = cureefpose - fkeefpose (for start_qpos)
    #     # put both in SE4 form then T.pose_difference to find base_mat
    #     self.controller.update_state()
    #     self._get_observations()
    #     cur_eef_mat = np.zeros((4,4))
    #     cur_eef_mat[:3,-1] = self.observations['eef_pos'].copy()
    #     cur_eef_mat[:3,:3] = T.quat2mat(self.observations['eef_quat'].copy())
    #     cur_eef_mat[3,3] = 1.

    #     fk_ee_pose = np.zeros(12)
    #     self.ik_calc.forward(self.observations['qpos'].copy(), fk_ee_pose)
    #     fk_eef_mat = np.zeros((4,4))
    #     fk_eef_mat[:3,:] = np.asarray(fk_ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
    #     fk_eef_mat[3,3] = 1.

    #     # print(cur_eef_mat)
    #     # print(fk_eef_mat)

    #     self.robbase_mat = np.zeros((4,4))
    #     # self.robbase_mat = T.pose_difference(cur_eef_mat, fk_eef_mat)
    #     ## NOTE: order of the dot product matters!
    #     self.robbase_mat = np.dot(
    #         cur_eef_mat,
    #         np.linalg.inv(fk_eef_mat)
    #     )
    #     # print(self.robbase_mat)
    #     # input()

    #     # basesite_id = mjc2.obj_name2id(
    #     #     self.model,
    #     #     "site",
    #     #     "base_site"
    #     # )
    #     # self.robbase_mat = np.zeros((4,4))
    #     # self.robbase_mat[:3,-1] = np.array(
    #     #     self.data.site_xpos[basesite_id]
    #     # )
    #     # self.robbase_mat[:3,:3] = np.array(
    #     #     self.data.site_xmat[basesite_id].reshape((3, 3)).copy()
    #     # )
    #     # self.robbase_mat[3,3] = 1.
    #     # input(self.robbase_mat)


    # def _get_ik(self, des_pose, prev_qpos):
    #     # self._init_ik()
    #     des_qpos = np.zeros(6)
    #     rel_mat = T.get_relmat(des_pose, self.robbase_mat)[:3,:]
    #     # rel_mat[:3,:3] = T.quat2mat(T.axisangle2quat(T.quat2axisangle(T.mat2quat(rel_mat[:3,:3]))))
    #     # input(np.dot(rel_mat[:3,:3],rel_mat[:3,:3].transpose()))
    #     rel_pose = np.zeros(7)
    #     rel_pose[:3] = rel_mat[:3,-1].copy()
    #     rel_pose[3:] = T.mat2quat(rel_mat[:3,:3].copy())
    #     rel_pose[3:] /= np.linalg.norm(rel_pose[3:])
    #     rel_pose[3:] /= np.linalg.norm(rel_pose[3:])
    #     print(np.linalg.norm(rel_pose[3:]))
    #     input(rel_pose)
    #     ik_succ = self.ik_calc.inverse(
    #         # des_pose,
    #         # rel_pose,
    #         rel_mat.reshape(-1).tolist(),
    #         prev_qpos,
    #         des_qpos
    #     )
    #     done = ik_succ
    #     if not done:
    #         a_pert_a = np.array([0., 0., 0.001])
    #         a_pert_b = np.array([0., 0., -0.001])
    #         new_rel_pose = rel_pose.copy()
    #         i_loop = 1
    #     while not done:
    #         new_rel_pose[3:] = T.quat_multiply(
    #             rel_pose[3:],
    #             T.axisangle2quat(a_pert_a*i_loop)
    #         )
    #         ik_succ = self.ik_calc.inverse(
    #             # des_pose,
    #             new_rel_pose.reshape(-1).tolist(),
    #             # rel_mat.reshape(-1),
    #             prev_qpos,
    #             des_qpos
    #         )
    #         if ik_succ:
    #             return des_qpos
    #         new_rel_pose[3:] = T.quat_multiply(
    #             rel_pose[3:],
    #             T.axisangle2quat(a_pert_b*i_loop)
    #         )
    #         print(new_rel_pose)
    #         ik_succ = self.ik_calc.inverse(
    #             # des_pose,
    #             new_rel_pose.reshape(-1).tolist(),
    #             # rel_mat.reshape(-1),
    #             prev_qpos,
    #             des_qpos
    #         )
    #         if ik_succ:
    #             return des_qpos
    #         i_loop += 1
    #         print(i_loop)
    #         if i_loop > 100:
    #             done = True
    #             print("Error: IK unsucessful!")
    #             return None

    #     return des_qpos
    
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

"""
[[ 1.80432751e-03 -9.99979560e-01 -6.13380804e-03  4.92719293e-01]
 [-9.99998236e-01 -1.80108827e-03 -5.33578529e-04  1.34835376e-01]
 [ 5.22520093e-04  6.13475997e-03 -9.99981046e-01  4.85597924e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 9.99998212e-01  1.80113071e-03  5.33586426e-04 -1.09985694e-01]
 [ 1.80437008e-03 -9.99979556e-01 -6.13383343e-03  4.87772733e-01]
 [ 5.22527727e-04  6.13478571e-03 -9.99981046e-01  4.29488271e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 2.30518624e-07 -9.99977864e-01 -6.65727793e-03  6.02096427e-01]
 [ 9.99984255e-01  3.75928764e-05 -5.61214931e-03  3.54359916e-01]
 [ 5.61227553e-03 -6.65717169e-03  9.99962091e-01 -5.36221347e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

"""