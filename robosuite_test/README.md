Usefull [link](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

# Robosuite for OpenVLA Installation
In `test/robosuite` folder:
```bash
conda env create -f conda_environments/robosuite_1_0_1.yaml # Change Name in file: openvla_robosuite_1_0_1
# Install OpenVLA
pip install -r python_requirements/openvla_requiments.txt  # openvla-requirements
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/moojink/transformers-openvla-oft.git
cd tasks/training
pip install -e .
pip install pyquaternion

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt
cd ..
source install.sh
```

# Robosuite for TinyVLA Installation
In `test/robosuite_test` folder:
```bash
conda env create -f conda_environments/robosuite_1_0_1.yaml # Change Name in file: tinyvla_robosuite_1_0_1
pip install -r python_requirements/tinyvla_requirements.txt
pip install -e ../../.
pip install -e ../.

git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt 
cd ..
source install.sh
```

Install TinyVLA modules

```bash
cd ~/Multi-Task-LFD-Framework/repo/TinyVLA
cd policy_heads
pip install -e .
# install llava-pythia
cd ../llava-pythia
pip install -e . 
```