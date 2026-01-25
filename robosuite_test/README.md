Usefull [link](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

# Robosuite for OpenVLA Installation
In `test/robosuite` folder:
```bash
conda env create -f conda_environments/robosuite_1_0_1.yaml # Change Name in file: openvla_robosuite_1_0_1
# Install OpenVLA
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r python_requirements/openvla_requiments.txt  # openvla-requirements
pip install git+https://github.com/moojink/transformers-openvla-oft.git
cd tasks/training
pip install -e .
pip install pyquaternion
pip install 'Cython<3.0'

# OpenVLA folder
pip install -e .

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt
cd ..
source install.sh

# Cuda-12.8 version
pip install -r python_requirements/openvla_requirements.txt 
pip install torch==2.7.0+cu128  deepspeed==0.18.1   bitsandbytes==0.48.0  --extra-index-url https://download.pytorch.org/whl/cu128
pip install torchvision==0.22.0 
pip install flash-attn==2.5.5 --no-build-isolation
pip install pyquaternion

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