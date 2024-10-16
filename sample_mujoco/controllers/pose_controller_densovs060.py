import numpy as np
import mujoco

import sample_mujoco.utils.transform_utils as T
import sample_mujoco.utils.mjc2_utils as mjc2

class PoseController:
    def __init__(
        self,
        model,
        data,
        actuator_names,
        eef_site_name="eef_site",
        kp_pose=3.,
        damping_ratio=1.,
        control_freq=40.,
        ramp_ratio=0.8,
    ):
        self.model = model
        self.data = data
        self.nv = self.model.nv # 6 for a UR5 robot arm with 6R joints

        # time
        self.ramp_ratio = 0.8
        self.dynamic_timestep = self.model.opt.timestep
        self.control_freq = control_freq
        self.control_timestep = 1 / self.control_freq
        self.interpolate_steps = np.ceil(
            self.control_timestep / self.dynamic_timestep * ramp_ratio
        )

        # quat fix
        self.prev_quat = np.array([0., 1., 0., 0.])
        self.quat_switch = 1.

        # ref
        self.eef_site_name = eef_site_name
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

        self._state = dict(
            qpos=np.zeros(7),
            qvel=np.zeros(7),
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(6),
            # ft_world=np.zeros(6),
        )
        self._update_state()

        self.damping_ratio_pose = damping_ratio
        self._kp_pose = kp_pose
        self._kd_pose = 2 * np.sqrt(self._kp_pose) * self.damping_ratio_pose
        # self._kd_pose = 0.0
        self._pd = self._state['eef_pos']
        self._qd = self._state['eef_quat']

        self.action_lim_pose = [0.12, 0.015]

        self.grav_comp()


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

    def is_stepped(self):
        return self.steps > np.ceil(
            self.control_timestep / self.dynamic_timestep
        )

    def set_goal(self, action):
        self.steps = 0
        # pos
        self._p0 = self._pd
        self.action = self.scale_action(
            action[:3], out_max=self.action_lim_pose[0]
        )
        # quat
        self._q0 = self._qd
        ori_action = action[3:]
        scaled_ori_a = self.scale_action(
            ori_action, out_max=self.action_lim_pose[1]
        )
        scaled_quat_a = T.axisangle2quat(scaled_ori_a)
        self.goal_ori = T.quat_multiply(scaled_quat_a, self._q0)

    def set_joint_cmd(self, joint_act=np.zeros(6)):
        joint_act = self.scale_action(joint_act, out_max=0.01)
        self.grav_comp()
        if joint_act is not None:
            self.data.ctrl[self.actuator_ids] = self._state['qpos'] + joint_act
        
    def grav_comp(self):
        self.base_torq = self.data.qfrc_bias[self.joint_ids]
        self.data.ctrl[self.gcact_id] = self.base_torq   

    def compute_joint_cmd(self):
        self.steps += 1
        qpos_action = self.compute_pose2jointaction()
        self._update_state()
        return qpos_action

    def compute_pose2jointaction(self):
        # setting desired
        qd = T.quat_slerp(
            self._q0, self.goal_ori, self.steps / self.interpolate_steps
        )
        pd = (
            self._p0
            + self.action
            * self.steps
            / self.interpolate_steps
        )
        # ramp ratio
        if self.steps > self.interpolate_steps:
            pd = self._p0 + self.action
            qd = self.goal_ori
        
        # get Jacobian
        J_pos = np.zeros((3,self.nv))
        J_ori = np.zeros((3,self.nv))

        mujoco.mj_jacSite(
            self.model,
            self.data,
            J_pos,
            J_ori,
            self.eef_site_idx
        )
        J_full = np.vstack([J_pos, J_ori])[:, self.joint_ids]

        e_pos = pd - self._state['eef_pos']
        # quat
        eef_quat = self._state['eef_quat']
        e_ori = T.quat_error(eef_quat, qd)
        e_ori = np.zeros(3)
        # overall
        pose_error = np.concatenate((e_pos, e_ori))
        vd = np.zeros(6)    # desired velocity
        e_vel = vd - self._state['eef_vel']
        qpos_action = (
            np.dot(J_full.T, self._kp_pose * pose_error + self._kd_pose * e_vel)
        )
        # print(f"pd = {pd}")
        # print(f"eef_pos = {self._state['eef_pos']}")

        # print(f"e_pos = {e_pos}")
        # print(f"qpos_action = {qpos_action}")
        self._pd = pd
        self._qd = qd

        return qpos_action
    
    def reset(self):
        self.steps = 0
        self._update_state()
        self._pd = self._state['eef_pos']
        self._qd = self._state['eef_quat']

    def update_state(self):
        self._update_state()

    def _update_state(self):
        # get eef_vel
        ee_vel = mjc2.obj_getvel(
            self.model,
            self.data,
            "site",
            self.eef_site_idx
        )

        ee_ori_vel = ee_vel[:3].copy()
        ee_pos_vel = ee_vel[3:].copy()

        self._state['eef_vel'] = ee_vel.copy()
        
        # get eef_pos and eef_quat
        eef_pos = np.array(
            self.data.site_xpos[self.eef_site_idx]
        )
        # print(mjc2.obj_id2name(self.model, "site", self.eef_site_idx))
        # input("here")
        self._state['eef_pos'] = eef_pos
        eefmat = np.array(
            self.data.site_xmat[self.eef_site_idx].reshape((3, 3))
        )
        # print('here')
        # print(self._state['eef_pos'])
        eef_quat = T.mat2quat(eefmat)
        # if np.linalg.norm(eef_quat+self.prev_quat) < 1e-3:
        #     self.quat_switch *= -1
        self._state['eef_quat'] = eef_quat * self.quat_switch
        self.prev_quat = eef_quat
        self.eef_mat = eefmat

        self._state["qpos"] = self.data.qpos[self.joint_qposids[:6]].copy()
        self._state["qvel"] = self.data.qvel[self.joint_dofids[:6]].copy()

        # # get eef force
        # # input(self.data.sensordata)
        # eef_ft = get_sensor_force(
        #     self.model,
        #     self.data,
        #     # "b2",
        #     "right_hand",
        #     # self.eef_name,
        #     self._state['eef_pos'],
        #     # np.array([1.,0,0,0]),
        #     self._state['eef_quat'],
        # )
        # eef_ft_filtered = self.lowpass_filter(eef_ft.reshape((-1, 6)))[0, :]
        # force in world frame

        # f_world = eef_ft_filtered[:3]
        # t_world = eef_ft_filtered[3:]
        # f_world = self.data.get_sensor('force_ee')
        # self._state['ft_world'] = np.concatenate((f_world, t_world))

    @property
    def state(self):
        return self._state
    
    def scale_action(self, action, out_max = 0.015):
        """
        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max

        Args:
            action (Iterable): Actions to scale

        Returns:
            np.array: Re-scaled action
        """
        # print(f"out_max = {out_max}")
        len_input = len(action)
        input_max = np.array([1] * len_input)
        input_min = np.array([-1] * len_input)
        output_max = np.array([out_max] * len_input)
        output_min = np.array([-out_max] * len_input)

        action_scale = abs(output_max - output_min) / abs(
            input_max - input_min
        )
        action_output_transform = (output_max + output_min) / 2.0
        action_input_transform = (input_max + input_min) / 2.0
        action = np.clip(action, input_min, input_max)
        transformed_action = (
            action - action_input_transform
        ) * action_scale + action_output_transform

        return transformed_action

def joint_sum(a, b):
    return ((a+b)+np.pi)%(2.*np.pi)-np.pi