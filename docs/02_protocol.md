# 🧪 Evaluation protocol

[← Installation](01_installation.md) · [README](../README.md) · Next: [Supported models →](03_models.md)

## The world

A UR5e arm stands in front of a table with **4 colored boxes** and **4 bins**. The task family is `pick_place`, which has 16 variations, one for each object→bin pair:

|  | Bin 1 | Bin 2 | Bin 3 | Bin 4 |
|---|:-:|:-:|:-:|:-:|
| 🟩 green box  | 0  | 1  | 2  | 3  |
| 🟨 yellow box | 4  | 5  | 6  | 7  |
| 🟦 blue box   | 8  | 9  | 10 | 11 |
| 🟥 red box    | 12 | 13 | 14 | 15 |

The instruction always follows the template *"Pick the **\<color\>** box and place it into the **\<n-th\>** bin"*. The instructions for every task, variation and object set live in [`robosuite_test/command.json`](../robosuite_test/command.json).

## Metrics

| Metric | Meaning |
|---|---|
| 🎯 **Reached** | The gripper got within 4 cm of the *correct* object |
| ✋ **Picked** | The correct object was lifted off the table |
| ✅ **Success** | The correct object was placed in the *correct* bin |
| `reached_wrong` / `picked_wrong` | The same checks, but for a *distractor* object |

Because the metrics are nested and wrong-object events are logged separately, you can tell a model that **can't manipulate** apart from one that **grounds the instruction wrongly**.

## Task suites

Each suite corresponds to a training split. Evaluating on the held-out part isolates one kind of generalization.

| `task_suite_name` | What is held out at training time | How to probe it | Max steps |
|---|---|---|:-:|
| `ur5e_pick_place_delta_all` | Nothing (full in-distribution baseline) | none | 200 |
| `ur5e_pick_place_delta_removed_0_5_10_15` | Diagonal pairs `0, 5, 10, 15` (**compositional**) | `model_config.otd: true` | 130 |
| `ur5e_pick_place_rm_12_13_14_15` | Every red-box variation `12–15` (**unseen object**) | `model_config.otd: true` | 130 |
| `ur5e_pick_place_removed_spawn_regions` | One spawn region per object (**spatial**) | `--change_spawn_regions true` | 130 |
| `ur5e_pick_place_rm_one_spawn` | A single spawn region for all objects (**spatial**) | `--change_spawn_regions true` | 130 |
| `ur5e_pick_place_rm_central_spawn` | The central spawn region (**spatial**) | `--change_spawn_regions true` | 130 |
| `ur5e_pick_place` / `ur5e_pick_place_abs_pose` | Nothing (absolute-pose action variant) | none | 200 |

`otd` stands for *Out-of-Training-Distribution*.

- In the two variation-split suites, `otd: false` evaluates on the training variations and `otd: true` evaluates on the held-out ones.
- In the three spawn-region suites, `--change_spawn_regions false` samples objects only from training regions. `true` places them in the held-out region.

The splits are defined in `TASK_VARIATION_DICT` / `TASK_MAX_STEPS` ([`robot_utils.py`](../robot_utils.py)), and the spawn-region logic is in [`robosuite_test/test/pick_place.py`](../robosuite_test/test/pick_place.py).

## Extra perturbations

These can be combined with any suite:

| Flag | Axis | Effect |
|---|---|---|
| `--object_set N` | visual | Swaps in a different object set (new appearance, same task). `-1` keeps the default set. |
| `--change_command true` | language | Paraphrases the command (e.g. *red* → *orange*). |
| `model_config.use_cosmos_name: true` | language | Replaces the scripted command with one that **Cosmos-Reason2** generates from a *human demonstration video*, using the prompt in [`prompt/human_task_description_prompt.yaml`](../robosuite_test/prompt/human_task_description_prompt.yaml). |

## Reproducibility

- Episode `i` uses seed `seeds.txt[i mod N] + 10000 · run_number` ([`seeds/seeds.txt`](../robosuite_test/seeds/seeds.txt)).
- `num_trials_per_task` rollouts are run for **each** variation in the evaluated split.
- Before any rollout, `num_steps_wait` steps let the objects settle.
- Completed episodes (`info_<i>.json`) are skipped, so an interrupted run resumes where it stopped.
