# sample_mujoco
Sample MuJoCo environment with Denso robot. Feel free to add more stuff and more robots or your own packages.

## IK
Inverse Kinematics follows pseudocode from ([Comparative Analysis of the Inverse Kinematics of a 6-DOF Manipulator](https://www.diva-portal.org/smash/get/diva2:1774792/FULLTEXT01.pdf)) implemented by [Alexis Fraudita](https://github.com/andyzeng/ikfastpy/tree/master).


## Requires
On Ubuntu, make sure you have the packages

liblapack-dev
libopenblas-dev
installed:

> sudo apt install liblapack-dev libopenblas-dev

Also requires:
-[MuJoCo](https://github.com/google-deepmind/mujoco), [Gymansium](https://github.com/Farama-Foundation/Gymnasium), [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer), [toppra](https://github.com/hungpham2511/toppra.git), [eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
