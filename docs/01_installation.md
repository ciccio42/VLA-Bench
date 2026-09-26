# ⚙️ Installation

[← Back to README](../README.md) · Next: [Evaluation protocol →](02_protocol.md)

VLA-Bench uses **two kinds of environments**:

- A **simulator env** (Python 3.9, robosuite 1.0.1, `mujoco-py`) that runs the rollout loop. OpenVLA-OFT and TinyVLA run inside it directly.
- One **server env per model** for policies whose dependencies clash with the simulator (LeRobot, mimic-video, Interleave-VLA). These talk to the simulator over HTTP.

---

## 0. MuJoCo 2.1 (`mujoco-py`)

Follow the *"Old bindings (≤ 2.1.1): mujoco-py"* section of [this guide](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html), then export:

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl            # headless rendering
export PYOPENGL_PLATFORM=egl
```

> [!NOTE]
> `mujoco-py` compiles a native extension the first time it is imported. On clusters, this build may only succeed on compute nodes, because login-node compilers can be incompatible.

Support for newer MuJoCo and robosuite versions is planned for a future release.

---

## 1. Patched robosuite

The benchmark uses a patched robosuite 1.0.1 that adds UR5e IK support. Download it from the [repository](https://github.com/ciccio42/robosuite.git) into `robosuite_test/robosuite/`, branch *ur5e_ik*. That folder is git-ignored. The install steps below then install it into each simulator env.

---

## 2a. Simulator env for OpenVLA-OFT

> Requires the [OpenVLA-OFT repository](https://github.com/ciccio42/openvla-oft.git).

From `robosuite_test/`:

```bash
conda env create -f conda_environments/conda_openvla_robosuite_1_0_1.yml
conda activate openvla_robosuite_1_0_1

pip install -r python_requirements/openvla_requirements.txt
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/moojink/transformers-openvla-oft.git
pip install -e ..                     # VLA-Benchmark utilities
pip install -e <PATH-TO-openvla-oft>  # OpenVLA-OFT itself

pip install -r robosuite/requirements.txt
pip install robosuite/.
source install.sh                     # installs tasks/ and tasks/training/ in editable mode
pip install pyquaternion 'Cython<3.0'
```

<details>
<summary>CUDA 12.8 variant</summary>

```bash
pip install -r python_requirements/openvla_requirements.txt
pip install torch==2.7.0+cu128 deepspeed==0.18.1 bitsandbytes==0.48.0 --extra-index-url https://download.pytorch.org/whl/cu128
pip install torchvision==0.22.0
pip install flash-attn==2.5.5 --no-build-isolation
pip install pyquaternion
```
</details>

---

## 2b. Simulator env for TinyVLA

> Requires the [TinyVLA repository](https://github.com/ciccio42/TinyVLA.git).

From `robosuite_test/`:

```bash
conda env create -f conda_environments/tinyvla_robosuite_1_0_1.yml
conda activate tinyvla_robosuite_1_0_1_provola

pip install -r python_requirements/tinyvla_requirements.txt
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ..                     # VLA-Benchmark utilities

pip install -r robosuite/requirements.txt
pip install robosuite/.
source install.sh
pip install 'Cython<3.0'

# TinyVLA modules
cd <PATH-TO-TinyVLA>
pip install -e .
pip install -e policy_heads
pip install -e llava-pythia
```

---

## 3. Server envs (LeRobot, mimic-video, Interleave-VLA)

This env is also the **client env** for the HTTP-served models (LeRobot, mimic-video, Interleave-VLA).

Install each model in its **own** environment by following its upstream instructions:

| Model | Server env needs | Server script started by the launcher |
|---|---|---|
| LeRobot (MolmoAct2, VLA-JEPA) | Python ≥ 3.12, `lerobot`, `flask` | `lerobot/run_eval_scripts/lerobot_policy_server.py` |
| mimic-video | Python 3.10, `cosmos_predict2`, `megatron-core` | `mimic-video/model/scripts/mimic_video_policy_server.py` |
| Interleave-VLA | torch 2.5.1, TensorFlow (`open-pi-zero`) | `open-pi-zero/scripts/interleave_vla_policy_server.py` |

The client side only needs `requests`, which is already present in the TinyVLA env. Edit the paths at the top of the matching `run_*_eval.sh` launcher so they point to your server env and checkpoints. See [Supported models](03_models.md) for details.

---

## 4. Optional: Cosmos-Reason2 via vLLM

This is only needed for the VLM-generated command perturbation (`use_cosmos_name: true`). Start a vLLM server that serves `nvidia/Cosmos-Reason2-8B`, then set `HOST_NAME` in [`robosuite_test/vllm_utils.py`](../robosuite_test/vllm_utils.py) to the node it runs on. The port goes in `model_cosmos_port` (default `8000`).
