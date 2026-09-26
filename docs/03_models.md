# 🧠 Supported models

[← Evaluation protocol](02_protocol.md) · [README](../README.md) · Next: [Running evaluations →](04_running.md)

| Family (`model_family`) | Model | Integration | Adapter | Config |
|---|---|---|---|---|
| `openvla` | **OpenVLA-OFT** | in-process | [`openvla.py`](../robosuite_test/models/openvla.py) | [`openvla_eval_config.yml`](../robosuite_test/models/openvla_eval_config.yml) |
| `tinyvla` | **TinyVLA** | in-process | [`tinyvla.py`](../robosuite_test/models/tinyvla.py) | [`tinyvla_eval_config.yml`](../robosuite_test/models/tinyvla_eval_config.yml) |
| `lerobot` | **MolmoAct2**, **VLA-JEPA** | HTTP server | [`lerobot_policy.py`](../robosuite_test/models/lerobot_policy.py) | [`lerobot_molmoact2_eval_config.yml`](../robosuite_test/models/lerobot_molmoact2_eval_config.yml), [`lerobot_vla_jepa_eval_config.yml`](../robosuite_test/models/lerobot_vla_jepa_eval_config.yml) |
| `mimic_video` | **mimic-video** | HTTP server | [`mimic_video_policy.py`](../robosuite_test/models/mimic_video_policy.py) | [`mimic_video_eval_config.yml`](../robosuite_test/models/mimic_video_eval_config.yml) |
| `interleave_vla` | **Interleave-VLA** (Interleave-π0) | HTTP server | [`interleave_vla_policy.py`](../robosuite_test/models/interleave_vla_policy.py) | [`interleave_vla_eval_config.yml`](../robosuite_test/models/interleave_vla_eval_config.yml) |

All per-model options are dataclasses in [`models/configs.py`](../robosuite_test/models/configs.py), registered through a draccus `ChoiceRegistry`. The `type:` key under `model_config:` selects the dataclass.

---

## OpenVLA-OFT (`openvla`)

This is the fine-tuned OpenVLA-7B with the OFT recipe: parallel decoding, action chunking, and continuous actions. It runs in-process.

| Key | Default | Meaning |
|---|---|---|
| `model_path` | none | Local checkpoint dir **or** Hugging Face repo ID |
| `use_l1_regression` / `use_diffusion` | `true` / `false` | Type of action head |
| `use_film` | `false` | FiLM language conditioning |
| `num_images_in_input` | `2` | Front camera + wrist camera |
| `use_proprio`, `proprio_dim` | `true`, `6` | Proprioceptive input |
| `num_open_loop_steps` / `chunk_size` | `8` | Actions executed before the policy is queried again |
| `center_crop` | `true` | Use if the model was trained with random-crop augmentation |
| `load_in_8bit` / `load_in_4bit` | `false` | Quantized loading |

For publishing and loading checkpoints on the 🤗 Hub, see [`SAVE_ON_HUGGING.md`](../SAVE_ON_HUGGING.md).

## TinyVLA (`tinyvla`)

This is Llava-Pythia with a diffusion policy head and LoRA fine-tuning. It runs in-process in the `tinyvla_robosuite_1_0_1_provola` env.

| Key | Meaning |
|---|---|
| `model_path` | LoRA checkpoint dir |
| `model_base` | Base Llava-Pythia weights |
| `enable_lora` | Merge LoRA weights |
| `action_head` | e.g. `droid_diffusion` |

## LeRobot policies (`lerobot`)

MolmoAct2 and VLA-JEPA are fine-tuned in LeRobot on `ur5e_pick_place_delta_all`. The launcher starts `lerobot_policy_server.py` in the LeRobot env (Python ≥ 3.12), and the adapter sends front and wrist images plus proprio to `/predict`. The adapter then turns the predicted **delta** actions into absolute poses.

| Key | Default | Meaning |
|---|---|---|
| `model_path` | none | Checkpoint dir (used for logging and the output path) |
| `server_port` | `8765` | Use a different port per concurrent server |
| `chunk_size` | `1` | Actions executed per query |

Design notes and open issues are in [`LEROBOT_EVAL.md`](../robosuite_test/LEROBOT_EVAL.md).

## mimic-video (`mimic_video`)

This model pairs a Cosmos video2world backbone with a world2action DiT action head. The server loads both models, so the launcher requests **2 GPUs**. The client keeps a 5-frame image history at stride 2, which reproduces the 10 Hz conditioning window the model was trained with. Frames are letterboxed to 320×240.

| Key | Default |
|---|---|
| `server_port` | `8767` |
| `chunk_size` | `15` |

Each `info_<i>.json` also records the checkpoint iteration (`model_checkpoint`) that produced the rollout.

## Interleave-VLA (`interleave_vla`)

This is Interleave-π0, which takes instructions that interleave images and text. The client builds the front view, a target-object crop, and a bin-grounding image, all at 224×224, from simulator bounding boxes. It uses the same crop-and-resize transform as training.

| Key | Default | Meaning |
|---|---|---|
| `server_port` | `8766` | none |
| `chunk_size` | `3` | Steps executed open-loop (≤ horizon of 4) |
| `sim_camera_config_path` | `PickPlaceDistractor.yaml` | Camera pose used for bounding-box projection |
| `debug_save_images` / `debug_image_dir` | `false` | Dump the preprocessed inputs at every step |

---

## Shared keys (all families)

| Key | Meaning |
|---|---|
| `task_suite_name` | Kept in sync with the top-level suite |
| `otd` | Evaluate on held-out variations (see [protocol](02_protocol.md)) |
| `use_cosmos_name`, `model_cosmos_name`, `model_cosmos_port` | VLM-generated command perturbation |
| `dataset_path` | Source of the human videos for Cosmos captioning |

> [!IMPORTANT]
> **Action convention.** Actions in the dataset are stored scaled by `1/0.05`. Their orientation is expressed in a gripper frame rotated relative to robosuite's `eef_quat`. Every adapter applies `SCALE_FACTOR = 0.05` and `R_EE_TO_GRIPPER`. Skipping either one makes the arm fly off the table.
