import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time

# for denso
v_safety_multiplier = 1.0
a_multiplier = 2.5

v_lims_denso = np.array([
    425, 283.33, 309.35, 425, 327.01, 680
]) * np.pi/180 * v_safety_multiplier

a_lims_denso = np.array([
    10., 10., 10., 10., 10., 10.
]) * a_multiplier

class fling_optimize:
    def __init__(
        self,
        tperdiv=0.02,
        vlims=v_lims_denso,
        alims=a_lims_denso
    ):
        self.dof = len(vlims)
        self.tperdiv = tperdiv
        self.vlims = vlims
        self.alims = alims

    def optimize_path(self, way_pts):
        n_samp = len(way_pts)
        ss = np.linspace(0,1,n_samp)
        if len(way_pts[0]) != self.dof:
            print(f"Incorrect DOF ({self.dof}) for sample ({len(way_pts[0])})!")
            input()
        
        path = ta.SplineInterpolator(ss, way_pts)
        pc_vel = constraint.JointVelocityConstraint(self.vlims)
        pc_acc = constraint.JointAccelerationConstraint(self.alims)
        instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
        jnt_traj = instance.compute_trajectory()
        n_divs = int(jnt_traj.duration/self.tperdiv + 0.5)
        ts_sample = np.linspace(0,jnt_traj.duration,n_divs)
        qs_sample = jnt_traj(ts_sample)

        # self.plot_results(jnt_traj=jnt_traj,instance=instance)

        return np.concatenate((np.array([ts_sample]).T,qs_sample),axis=1)

    def plot_results(self, jnt_traj, instance):
        ts_sample = np.linspace(0, jnt_traj.duration, 100)
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)
        fig, axs = plt.subplots(3, 1, sharex=True)
        for i in range(len(qs_sample[0])):
            # plot the i-th joint trajectory
            axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
            axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
            axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
        axs[2].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (rad)")
        axs[1].set_ylabel("Velocity (rad/s)")
        axs[2].set_ylabel("Acceleration (rad/s2)")
        plt.show()

        instance.compute_feasible_sets()
        instance.inspect()