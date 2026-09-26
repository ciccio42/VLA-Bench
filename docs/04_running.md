# 🚀 Running evaluations

[← Supported models](03_models.md) · [README](../README.md) · Next: [Results & analysis →](05_results.md)

Run everything from `robosuite_test/`. The entry point is [`run_robosuite_eval.py`](../robosuite_test/run_robosuite_eval.py), a draccus CLI over [`EvalConfig`](../robosuite_test/models/configs.py). Any YAML key can be overridden with `--key value` or `--model_config.key value`.

## Main `EvalConfig` options

| Option | Default | Meaning |
|---|---|---|
| `--config_path` | none | Model YAML in `models/` |
| `--model_family` | `openvla` | `openvla`, `tinyvla`, `lerobot`, `mimic_video`, `interleave_vla` |
| `--task_suite_name` | `ur5e_pick_place_rm_central_spawn` | See [protocol](02_protocol.md) |
| `--num_trials_per_task` | `10` | Rollouts per variation |
| `--run_number` | `0` | Offsets seeds; also part of the output folder name |
| `--change_spawn_regions` | `false` | Spatial OOD for the spawn-region suites |
| `--object_set` | `-1` | Visual perturbation |
| `--change_command` | `false` | Language perturbation |
| `--controller_path` | none | OSC controller JSON (`tasks/multi_task_robosuite_env/controllers/config/osc_pose.json`) |
| `--save` | `true` | Save `.pkl` + `.mp4` per episode |
| `--use_wandb` | `false` | Also log to Weights & Biases |
| `--debug` | `false` | Waits for a `debugpy` client on port 5678 |

## Plain Python

```bash
python run_robosuite_eval.py \
    --config_path models/openvla_eval_config.yml \
    --task_suite_name ur5e_pick_place_rm_12_13_14_15 \
    --model_config.otd true \
    --num_trials_per_task 10 \
    --run_number 0
```

## SLURM launchers

Each launcher exports the MuJoCo/EGL variables, selects the right env and, for server-based models, starts the policy server and waits for it before running the client.

> [!NOTE]
> Edit the `#SBATCH` account and partition lines, and the hard-coded env and checkpoint paths, at the top of each launcher to match your cluster.

**OpenVLA-OFT** (set `model_path` and the suite in `models/openvla_eval_config.yml` first)

```bash
sbatch run_robosuite_eval.sh <run_number> <change_spawn_regions:true|false>
```

**TinyVLA**

```bash
sbatch run_tinyvla_eval.sh <task_suite_name> <otd:true|false> <model_path> <model_base> [run_number] [num_trials_per_task]
```

**LeRobot (MolmoAct2 / VLA-JEPA)**

```bash
sbatch run_lerobot_eval.sh [config_yml] [policy_path] [port] [run_number] [num_trials_per_task] [task_suite_name] [otd] [object_set]
# shortcuts with preset arguments:
sbatch run_lerobot_molmoact2.sh
sbatch run_lerobot_vla_jepa.sh
```

**mimic-video**

```bash
sbatch run_mimic_video_eval.sh [config_yml] [checkpoint_dir] [port] [run_number] [num_trials_per_task]
```

**Interleave-VLA**

```bash
sbatch run_interleave_vla_eval.sh <config_yml> <checkpoint.pt> <train_config.yaml> <dataset_statistics.json> \
       [port] [run_number] [num_trials_per_task] [task_suite_name] [otd] [object_set]
```

### Environment-variable switches (server-based launchers)

| Variable | Default | Effect |
|---|---|---|
| `CHANGE_SPAWN_REGIONS` | `false` | Spatial OOD |
| `USE_COSMOS_NAME` | `false` | VLM-generated commands (needs a running vLLM server) |
| `MODEL_COSMOS_NAME` / `MODEL_COSMOS_PORT` | `nvidia/Cosmos-Reason2-8B` / `8000` | vLLM endpoint |
| `DATASET_PATH` | none | Human-video dataset for captioning |
| `CHUNK_SIZE` | `3` | *(Interleave-VLA)* open-loop steps per query |
| `NORMALIZATION_TYPE` | `auto` | *(Interleave-VLA)* action normalization; `bounds` for older checkpoints |
| `DEBUG_SAVE_IMAGES` / `DEBUG_IMAGE_DIR` | `false` | *(Interleave-VLA)* dump the model inputs |

## Sweeps

[`run_smoke_chain_with_telegram.sh`](../robosuite_test/run_smoke_chain_with_telegram.sh), [`submit_remaining_chain.sh`](../robosuite_test/submit_remaining_chain.sh) and [`notify_sweep_results.sh`](../robosuite_test/notify_sweep_results.sh) chain many evaluation jobs through SLURM dependencies. They can also post each job's final rates to Telegram, which requires your own bot credentials file.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Job exits with `corrupted size vs. prev_size` and a non-zero code | `mujoco-py` crashes at interpreter teardown. If the `info_*.json` files were written, the run is fine. |
| Re-run finishes instantly with the same numbers | Cached `info_<i>.json` files were replayed. Use a new `--run_number`. |
| `vLLM server is not reachable` | Start the Cosmos server and update `HOST_NAME` in `vllm_utils.py`. |
| `mujoco_py` build error on the login node | Run on a compute node (`srun` / `sbatch`). |
| Server launcher times out waiting for health | Check the server log. The server env or checkpoint path in the launcher is wrong. |
