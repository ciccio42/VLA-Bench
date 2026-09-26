# ➕ Adding a new model

[← Results & analysis](05_results.md) · [README](../README.md)

## Choose an integration style

- **In-process.** Use this if the model's dependencies install cleanly next to Python 3.9 + robosuite 1.0.1 + `mujoco-py`. Follow [`tinyvla.py`](../robosuite_test/models/tinyvla.py) and [`openvla.py`](../robosuite_test/models/openvla.py).
- **HTTP server** (recommended otherwise). Run a small server in the model's own env and write a thin client adapter. [`lerobot_policy.py`](../robosuite_test/models/lerobot_policy.py) with [`run_lerobot_eval.sh`](../robosuite_test/run_lerobot_eval.sh) is the simplest template.

## Steps

1. **Config.** Add a dataclass to [`models/configs.py`](../robosuite_test/models/configs.py):

   ```python
   @ModelConfig.register_subclass('my_vla')
   @dataclass
   class MyVLAConfig(ModelConfig):
       model_path: str = ""
       server_port: int = 8770
       chunk_size: int = 1
       task_suite_name: str = ''
       otd: bool = False
       use_cosmos_name: bool = False
       model_cosmos_name: str = "nvidia/Cosmos-Reason2-8B"
       model_cosmos_port: int = 8000
       dataset_path: str = ''
   ```

2. **Adapter.** Write `models/my_vla.py`. It returns a policy object with the same interface the rollout loop in [`test/pick_place.py`](../robosuite_test/test/pick_place.py) already calls on the existing adapters. Build the observation from the simulator's `obs` (`camera_front_image`, `robot0_eye_in_hand_image`, `eef_pos`, `eef_quat`, `joint_pos`) and apply `TASK_CROP` exactly as your training pipeline did.

3. **Dispatch.** Add a branch to [`run_robosuite_eval.py`](../robosuite_test/run_robosuite_eval.py):

   ```python
   elif cfg.model_family.lower() == "my_vla":
       from robosuite_test.models.my_vla import my_vla_policy
       policy = my_vla_policy(cfg.model_config)
   ```

4. **Image size.** Register the model's input resolution in `MODEL_IMAGE_SIZES` in [`robot_utils.py`](../robot_utils.py).

5. **YAML and launcher.** Add `models/my_vla_eval_config.yml` (copy an existing one and set `model_family` and `model_config.type`), plus an optional `run_my_vla_eval.sh`.

## Checklist before trusting the numbers

- [ ] Actions are **un-scaled** (`SCALE_FACTOR = 0.05`) and orientation goes through `R_EE_TO_GRIPPER`.
- [ ] Delta and absolute actions match the suite. Suites with `delta` in their name use delta actions.
- [ ] The gripper range and sign match the open/close threshold (`0.75`) in `test/pick_place.py`.
- [ ] Image crop, resize and camera order match training.
- [ ] A 1-trial smoke run shows the arm moving toward the target in the saved video.
