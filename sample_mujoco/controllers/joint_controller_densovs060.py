import numpy as np

class jointcontroller:
    def __init__(
        self,
        model,
        data,
        joint_ids,
        integrator_step=0.001
    ):
        self.model = model
        self.data = data
        self.joint_ids = joint_ids
        self.joint_qposids = self.model.jnt_qposadr[self.joint_ids]
        self.qpos_desired = self.data.qpos[self.joint_qposids[:6]].copy()
        self.qpos_curr = self.data.qpos[self.joint_qposids[:6]].copy()
        self.base_int_step = integrator_step
        self.prev_qpos = self.data.qpos[self.joint_qposids[:6]].copy()

    def move_to_qpos(self, j_cmd, ah_intstep=None):
        # if big diff in j_cmd, reset qpos_desired
        if np.linalg.norm(j_cmd-self.prev_qpos) > 0.1:
            self.qpos_desired = self.data.qpos[self.joint_qposids[:6]].copy()
        self.prev_qpos = j_cmd

        if ah_intstep is None:
            ah_intstep = self.base_int_step
        self.qpos_curr = self.data.qpos[self.joint_qposids[:6]].copy()
        qpos_err = joint_sum(j_cmd,-self.qpos_curr) * ah_intstep
        # self.qpos_desired = joint_sum(self.qpos_desired, qpos_err)
        self.qpos_desired = self.qpos_desired + qpos_err
        self.data.ctrl[self.joint_ids] = self.qpos_desired

def joint_sum(a, b):
    return ((a+b)+np.pi)%(2.*np.pi)-np.pi