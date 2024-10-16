from setuptools import setup

setup(name='sample_mujoco',
      version='0.0.1',
      # packages=[
      #       package for package in find_packages() if package.startswith("rl_tut")
      # ]
      # python_requires='>3',
      install_requires=[], # And any other dependencies foo needs
      packages=['sample_mujoco'],
)

# install with command "pip install -e ."