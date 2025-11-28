# VLA-Bench
This is the repository linked to the research project: *VLA-Bench: A Systematic Protocol for Evaluating Generalization of Vision-Language-Action Models*

# Install MUJOCO
Follow instruction reported [here](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html), for "Old bindings (≤ 2.1.1): mujoco-py".
We plan to update both mujoco and robosuite in next iterations


# Robosuite for OpenVLA Installation
In `robosuite_test` folder:
```bash
conda env create -f conda_environments/openvla_robosuite_1_0_1.yaml
# Install OpenVLA
pip install -r python_requirements/openvla_requiments.txt  # openvla-requirements
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/moojink/transformers-openvla-oft.git

git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt
cd ..
source install.sh
pip install 'Cython<3.0'
```

# Robosuite for TinyVLA Installation
In `robosuite_test` folder:
```bash
conda env create -f conda_environments/tinyvla_robosuite_1_0_1.yaml
pip install -r python_requirements/tinyvla_requirements.txt
# Install torch 
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 –index-url https://download.pytorch.org/whl/cu128
# Install utils
pip install -e ../.

git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt 
cd ..
source install.sh
pip install 'Cython<3.0'
```

Install TinyVLA modules

```bash
cd [PATH-TO-TinyVLA-Folder]
pip install -e .
cd policy_heads
pip install -e .
# install llava-pythia
cd ../llava-pythia
pip install -e . 
```

# How to run evaluation
**NOTE**
Before running evaluation you need to set:
* **task_suite_name** parameter in *run_robosuite_eval.sh*
* **config_path** parameter in *run_robosuite_eval.sh*, point to the configuration file of the model you wish to test. 
* In config .yml change **task_suite_name** and any parameter related to the checkpoint you wish to test

```bash
sbatch run_robosuite_eval.sh
```
