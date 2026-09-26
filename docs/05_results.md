# 📊 Results & analysis

[← Running evaluations](04_running.md) · [README](../README.md) · Next: [Adding a model →](06_adding_a_model.md)

## Output layout

Rollouts are written **inside the checkpoint directory** (`model_config.model_path`). Each combination of perturbation settings gets its own folder:

```
<model_path>/rollout_pick_place_<run>_<change_spawn_regions>_obj_set_<k>_change_command_<b>_use_vllm_<b>_otd_<b>/
├── traj_<i>.pkl     # full trajectory: compressed observations, actions, rewards, raw sim state
├── traj_<i>.mp4     # rendered video with the instruction overlaid
└── info_<i>.json    # reached / picked / success (+ *_wrong), task_description, [model_checkpoint]
```

Text logs go to `robosuite_test/experiments/logs/EVAL-<suite>-<family>-<datetime>--<run_id_note>.txt`, and to W&B if `use_wandb: true`.

## Aggregating

```bash
python analyze_results.py --path <rollout_dir> [--otd True] [--obj_set 1] [--change_command True] [--vllm]
```

This prints reached, picked and success rates, broken down **by object color** and **by target bin**. The breakdown shows where a model fails, for example "never succeeds on the unseen red box" or "always places in bin 1".

## Videos & inspection

| Tool | Use |
|---|---|
| [`create_video.py`](../robosuite_test/create_video.py) / [`run_create_videos.sh`](../robosuite_test/run_create_videos.sh) | Re-render `.mp4`s from saved `.pkl` trajectories (needs a compute node) |
| [`read_pkl.py`](../robosuite_test/read_pkl.py) | Quick look inside a trajectory |
| [`trajectory_projection_utils.py`](../robosuite_test/trajectory_projection_utils.py), [`test_trajectory_projection.py`](../robosuite_test/test_trajectory_projection.py) | Project 3-D end-effector trajectories onto sim or real camera images |
