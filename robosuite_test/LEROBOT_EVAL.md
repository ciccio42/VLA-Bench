# Testing the LeRobot UR5e checkpoints (MolmoAct2, VLA-JEPA) in VLA-Benchmark

This is a plan, not yet implemented. It documents exactly what needs to be built to run the
`outputs/ur5e_molmoact2` and `outputs/ur5e_vla_jepa` checkpoints (fine-tuned in
`Multi-Task-LFD-Framework/repo/lerobot/lerobot` on `local/ur5e_pick_place_delta_all`) inside this
repo's `robosuite_test` simulation harness.

## TODO

Section references point to the step-by-step writeup below.

- [x] **Confirm the harness still works today** — run the existing TinyVLA path once
      (`sbatch run_robosuite_eval.sh 0 false`) so any later failure is isolable to the new code,
      not to a pre-existing env issue. (§3 Step 1). **Confirmed working** (job 534400,
      `run_number=999` to force fresh trajectories rather than replaying cached ones — the first
      attempt, job 534387, silently replayed 40 cached results and proved nothing). All 40
      `info_*.json`/`traj_*.pkl` were freshly written; aggregate reached/picked/success = 0.07 /
      0.03 / 0.03 (low but non-degenerate — this checkpoint is being tested on its held-out `otd`
      variations, so a low rate is expected, not a bug).
      **Known quirk, not a blocker:** both attempts (534387, 534400) end with SLURM
      `State=FAILED, ExitCode=6:0` and a `corrupted size vs. prev_size in fastbins` glibc error —
      but only *after* every trajectory finished and all results were written to disk. This is a
      native-library crash during interpreter/mujoco_py teardown, not during actual rollout
      execution. Don't treat a nonzero exit code alone as evidence the LeRobot integration failed —
      check whether `info_*.json` files were actually written first.
- [x] **Pick and verify checkpoints** — confirm `outputs/ur5e_molmoact2/checkpoints/last` and
      `outputs/ur5e_vla_jepa/checkpoints/last` exist and note their step counts. (§3 Step 2).
      **Both fully trained**: both checkpoints reached step 8000/8000 (full target), and
      `pretrained_model/` has `config.json`, `model.safetensors`, and both pre/post-processors for
      each.
- [x] **Write the LeRobot inference server** — `lerobot_policy_server.py` in the `lerobot`
      conda env; `pip install flask` there if missing; fix the `make_policy`/
      `make_pre_post_processors` call against the real signatures if it errors. (§3 Step 3).
      **Done and smoke-tested for both checkpoints** at
      `lerobot/lerobot/run_eval_scripts/lerobot_policy_server.py`. Two real bugs found and fixed
      versus the original draft, both now reflected in §3 Step 3's code block:
      1. `make_policy()` requires a `ds_meta` or `env_cfg` to derive feature shapes — neither
         exists in a server with no LeRobot dataset/env. Switched to loading directly via
         `get_policy_class(cfg.type).from_pretrained(path, config=cfg)`, which uses the
         checkpoint's already-resolved `config.json` and skips the shape-inference requirement
         entirely.
      2. MolmoAct2's `select_action()` raises `ValueError: MolmoAct2 inference requires
         inference_action_mode to be set explicitly` — it has no default. Added
         `--inference_action_mode` (default `continuous`) to the server, applied only when
         `cfg.type == "molmoact2"`.
      Both checkpoints now return a valid 7-D action from a synthetic request end-to-end
      (VLA-JEPA's gripper output was exactly `-1.0` on the synthetic input, consistent with its
      `binarize_gripper_action=True` default — a good sign, though a random input proves plumbing,
      not correctness of real predictions).
- [x] **Write the robosuite-side client adapter** — `models/lerobot_policy.py` in
      `tinyvla_robosuite_1_0_1_provola`; verify raw `obs` key names
      (`camera_front_image`, `robot0_eye_in_hand_image`, `joint_pos`, `eef_pos`, `eef_quat`) with a
      one-off `print(sorted(obs.keys()))` before trusting the assumed names. (§3 Step 4). Written;
      `requests`/`cv2` confirmed already present in the Python 3.9 env. **Key names not yet
      confirmed against a live rollout** — that happens in the Step 8 smoke test below, still open.
- [ ] **Resolve the delta→absolute action conversion** — implement the accumulate-then-
      axis-angle conversion; flag/verify whether the dataset's action deltas are world-frame or
      EEF-local (§3 Step 4's callout — this is the single riskiest unknown in the whole plan).
      Code written assuming world-frame; **still unverified against a live rollout.**
- [ ] **Resolve the gripper value scale/sign** — check what `/predict` actually returns against
      the `>0.75`/`<0.75` open/close thresholding in `test/pick_place.py`; rescale in the adapter
      if needed. (§3 Step 4's callout) **Still unverified against a live rollout.**
- [x] **Register the `lerobot` model family** — add `LeRobotPolicyConfig` to `models/configs.py`,
      the new `elif` branch in `run_robosuite_eval.py`, and the `MODEL_IMAGE_SIZES` entry in
      `robot_utils.py`. (§3 Step 5). Done; `LeRobotPolicyConfig()` imports and instantiates
      correctly under the benchmark's own Python 3.9 interpreter.
- [x] **Write eval config YAMLs** — one for MolmoAct2, one for VLA-JEPA, each with the right
      `model_path` and a distinct `server_port`. (§3 Step 6). Done:
      `models/lerobot_molmoact2_eval_config.yml` (port 8765),
      `models/lerobot_vla_jepa_eval_config.yml` (port 8766).
- [x] **Write the sbatch launcher** — `run_lerobot_eval.sh`, starts the server, polls its health
      endpoint, then runs the robosuite client; add the trivial `GET /` health route to the server.
      (§3 Step 7). Done, `bash -n` clean; not yet run end-to-end (that's Step 8).
- [~] **Smoke test** — single variation, `num_trials_per_task: 1`; confirm the server responds,
      the arm visibly moves toward the target in `images/step_*.jpg`, and the gripper closes at
      least sometimes. Do not proceed to a full run until this passes. (§3 Step 8)
      **Ran (job 536013, MolmoAct2), pipeline works end-to-end with no crashes — but the resulting
      behavior is wrong.** `env.step()` calls succeed continuously (76 real `/predict` round trips
      observed), and the launcher/server/adapter chain is all mechanically sound. But reading the
      actual per-step log lines (`Step N action taken: [...]`) shows predicted position deltas of
      roughly 0.1-0.2 per axis per step — the target pose jumps from `[-0.17, 0.08, 0.97]` at step 1
      to `[0.40, 0.73, 1.06]` by step 5, well above and away from the table. `images/step_30.jpg`
      and `step_77.jpg` show the arm has left the workspace entirely (wrist camera sees a wall/edge,
      not the table) — it never gets close to the boxes.
      Checked and ruled out: the `observation.state` layout matches
      `convert_dataset.py::make_frame` exactly (`[joint_0..5, gripper_state[1], eef_x,y,z,r,p,y]`,
      confirmed by reading the source side by side with the adapter). **Not yet ruled out / next
      things to check, in order of suspicion:**
      1. `meta/stats.json`'s `action` field itself is oddly distributed for what should be small
         per-step deltas — `dx` mean is `+0.122` (not ~0), `dyaw` ranges up to `5.6` (almost a full
         `2π` turn). Either this dataset's `action` field isn't a simple small per-timestep delta
         the way I assumed (e.g. it's a larger motion-primitive step that `env.step()`'s internal
         `action_repeat`/settle loop is meant to absorb), or there's an angle-wrap artifact in how
         `convert_dataset.py` built it from the RLDS `action` field — needs comparing against the
         *original* RLDS `action` values directly, not just the converted LeRobot dataset.
      2. `gripper_state[1]` (index **1**, not 0, per `make_frame`) — I'm feeding the eval loop's
         own `gripper_closed` flag (0.0/1.0, tracked by `test/pick_place.py`) into that slot as a
         proxy, without confirming that RLDS index 1 actually means the same thing.
      3. Genuine undertrained/unconverged policy — I only checked MolmoAct2's training loss
         trended down to ~0.008 by step 8000 in isolation, never validated it against any held-out
         rollout before this point.
      **Follow-up investigation (ruled out my two leading hypotheses, found a deeper one):**
      Checked the raw RLDS source directly (not the LeRobot conversion) with a small script against
      3 episodes:
      - The odd, non-zero-mean, large-magnitude `action` distribution (dx mean +0.11, not ~0) is
        present in the **raw source data itself**, not introduced by `convert_dataset.py`. Ruled
        out "conversion bug."
      - Tested whether `action[:3]` is a simple per-step delta by checking
        `EEF_state[t] + action[t][:3]` against `EEF_state[t+1][:3]` (world-frame addition) and
        against a version of the delta rotated by the current EEF orientation (EEF-local-frame
        hypothesis). **Neither matches**: mean position error ≈ 0.26–0.28 m either way — roughly
        the same magnitude as the action itself, i.e. uncorrelated, not "slightly off."
      - Also tested `action[:3]` against the *other* recorded field, `action_world` (i.e., is
        `action` the delta from current EEF to the commanded target `action_world`?): checked
        `EEF_state[t] + action[t][:3]` against `action_world[t][:3]` directly. **Also doesn't
        match** — mean error ≈ 0.26–0.27 m, essentially identical across 3 different episodes.
        That consistency (not random, not growing over an episode) smells like a **fixed
        coordinate-frame offset** rather than noise — e.g. `EEF_state` given in the robot's own
        base frame vs. `action`/`action_world` given in the world frame (the UR5e mount itself
        sits at world position `[-0.6, 0, 0]`, per the object-position printout from the earlier
        spawn-randomization bug-fix work in this session — a plausible source of exactly this kind
        of constant-offset mismatch, though the magnitudes don't line up exactly and I haven't
        confirmed this specific explanation).
      **Conclusion: this is not simply an adapter bug I can fix by trying another sign or rotation
      convention** — the relationship between `action`, `action_world`, and `EEF_state` in the
      *source dataset* doesn't match any of the simple conventions I tried, which is why the
      accumulated trajectory drifts incoherently regardless of which reasonable-looking formula the
      adapter uses. Resolving this needs either: (a) the original TFDS builder/collection code's
      exact frame conventions (`ur5e_pick_place.py`, referenced in earlier session context, has the
      collection logic but I haven't re-read it with this specific question in mind), or (b) a
      person who knows how this dataset's `action` field was actually defined at collection time.
      **Root cause found (user review, not something I derived myself) — two missing pieces from
      `openvla.py`'s reference implementation that I'd skipped when writing the adapter:**
      1. **`SCALE_FACTOR = 0.05`**: the dataset's raw `action` field (and therefore what any model
         trained on it predicts) is stored pre-scaled by `1/SCALE_FACTOR` relative to real
         meters/radians. `openvla.py::action_post_processing` multiplies the *entire* predicted
         7-vector by this before any other use — I wasn't doing this at all, so the adapter was
         adding raw ~0.1-0.3-magnitude deltas straight onto `eef_pos` every step. This alone
         explains the arm leaving the table in ~5 steps.
      2. **`R_EE_TO_GRIPPER`** (`openvla_utils.py:43-47`, a fixed axis remap, not a live rotation):
         `openvla.py::prepare_observation` applies this to `eef_quat` before computing the euler
         angles that go into the *state* fed to the model, and `action_post_processing` applies
         the same correction when computing the *current orientation* the predicted delta gets
         added to. I was using the raw `eef_quat` euler angles in both places — meaning the state
         I sent the model was already out of the training distribution, on top of the
         unscaled-action problem.
      Both fixed in `lerobot_policy.py` (module constants `SCALE_FACTOR`/`R_EE_TO_GRIPPER`, a
      shared `gripper_frame_euler()` helper used in both `_build_state` and `compute_action`).
      **One caveat noted but not yet resolved:** VLA-JEPA's postprocessor independently binarizes
      gripper to `{-1, +1}` regardless of the dataset's raw 0-20 scale — multiplying that by
      `SCALE_FACTOR` gives `{-0.05, +0.05}`, which would always read as "open" under
      `pick_place.py`'s `>0.75`/`<0.5` thresholds. Applied the scaling uniformly (matching
      `openvla.py` exactly, gripper dim included) for now since that's what MolmoAct2 needs and
      it's the reference behavior; whether VLA-JEPA additionally needs its gripper dim excluded
      from scaling is still open — check its own smoke test's gripper behavior specifically.
      **Re-tested with the fix applied (job 536025): success.** Position deltas are now smooth
      and small (~1cm/step, e.g. step 1→2 moved `[-0.288,-0.003,0.907]` → `[-0.281,-0.002,0.908]`)
      instead of the previous 0.1-0.3m chaotic jumps, and `images/step_*.jpg` show the arm staying
      sensibly positioned over the table. **First full trajectory: `reached=1, picked=1,
      success=1`** — a genuine successful pick-and-place through the complete
      LeRobot-policy→HTTP-server→robosuite-adapter pipeline. This is real end-to-end validation,
      not just "doesn't crash."
      **MolmoAct2, extended run**: 7 episodes in (job 536025, `num_trials_per_task=1` across the
      first 7 of 16 variations), 3/7 real successes (`reached/picked/success=1` on tasks 0, 2, 6),
      the other 4 reached but didn't pick. Plausible, non-degenerate eval numbers — good sign the
      pipeline is producing meaningful results, not an artifact.

      **VLA-JEPA gripper caveat confirmed real, then fixed at the server+client level**: the
      GRIPPER_ALREADY_SCALED_POLICY_TYPES fix above (server now reports `policy_type` in its
      `/predict` response; client only applies `SCALE_FACTOR` to the gripper dim when the policy
      type isn't in that set) is implemented and both processes restarted to pick it up (Python
      doesn't hot-reload already-running server/client code — killed and resubmitted both jobs
      after editing).

      **New finding after the fix, distinct from the scale/frame bugs**: across 3 completed
      VLA-JEPA episodes (job 536029), every single `Predicted gripper` line printed exactly `-1.0`
      (open) — reached=1 on all 3, picked=0 on all 3. The gripper dim is flowing through correctly
      now (proven by seeing the true unscaled `-1.0` rather than the old `-0.05`), but the model
      itself never once predicted the binarized `+1.0` (close) value in any of these rollouts.
      This looks like a genuine model-behavior limitation rather than a remaining pipeline bug —
      notably, this project's LIBERO eval work earlier in this session found the same
      "reaches-but-never-closes" pattern for pi0fast specifically. Whether this VLA-JEPA UR5e
      finetune has a similar limitation, or whether something upstream (e.g. the `gripper_closed`
      state flag fed back to the policy, or normalize_gripper/binarize threshold interaction) is
      suppressing "close" predictions, is not yet determined — needs more episodes and/or per-step
      raw (pre-binarize) gripper values to distinguish "policy never wants to close" from "policy
      wants to close but the binarize threshold never triggers."
- [ ] **Full MolmoAct2 run** — `num_trials_per_task: 10` × 16 variations via `run_lerobot_eval.sh`.
- [ ] **Full VLA-JEPA run** — same, with the VLA-JEPA config/port.
- [ ] **Compare results** — success/reached/picked rates against any existing TinyVLA/OpenVLA
      baseline in `experiments/logs` on this same suite; don't compare directly against the LIBERO
      numbers from the earlier eval session, they're a different benchmark. (§3 Step 9)

## 1. What I found

### 1.1 The sim is (almost certainly) the source of our training data

`robot_utils.py`'s `TaskSuite` enum includes `PICK_PLACE_DELTA_ALL = "ur5e_pick_place_delta_all"`
— the exact same name as our dataset
(`open_x_embodiment/datasets/ur5e_pick_place_delta_all`). `TASK_VARIATION_DICT` lists 16
variations (0–15), matching the dataset's 16 color/bin task variations, and `TASK_MAX_STEPS`
gives it a 220-step episode budget. This is the simulation the RLDS dataset was collected from, so
task semantics, cameras, and action space should line up directly — no guessing about "close
enough" environments.

### 1.2 The eval harness has a pluggable-but-hardcoded model interface

`run_robosuite_eval.py` dispatches on `cfg.model_family`:

```python
if cfg.model_family.lower() == "openvla":
    policy = open_vla_policy(cfg.model_config)
elif cfg.model_family.lower() == "tinyvla":
    policy = llava_pythia_act_policy(cfg.model_config)
```

There is no generic/pluggable registration beyond this `if/elif` — adding LeRobot support means
adding a third branch here, plus a new `ModelConfig` subclass (draccus `ChoiceRegistry`, same
pattern LeRobot itself uses for `PreTrainedConfig`/`EnvConfig`) in `models/configs.py`.

The actual interface a policy object must implement (from `test/pick_place.py`, the per-step loop):

```python
action_world_chunk, elapsed_time = policy.compute_action(
    obs=obs,                      # raw robosuite obs dict for this step
    resize_size=resize_size,      # from get_image_resize_size(cfg) -> MODEL_IMAGE_SIZES[model_family]
    gripper_closed=gripper_closed,# float 0.0/1.0, tracked by the eval loop from the previous step
    task_description=task_description,
    task_name=task_name,          # "pick_place"
    n_steps=n_steps,
)
```

`compute_action` must return `(action_chunk, elapsed_time)` where `action_chunk` is a list of up
to `chunk_size` actions; the eval loop calls `env.step()` once per element, open-loop, before
querying the policy again.

### 1.3 The action space is absolute world pose, not delta — this is the main adapter work

`env.step()` takes **`[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]`** — an absolute
end-effector pose in the world frame with axis-angle orientation (confirmed in
`robosuite_utils.py::startup_env` and `test/pick_place.py`), not the delta action our dataset
stores. `models/openvla.py::action_post_processing` shows the expected conversion: accumulate the
model's predicted position delta onto the current/previous `eef_pos`, add the predicted
orientation delta (Euler) onto the current/previous orientation, then convert to axis-angle via
`euler_to_axis_angle`. Our LeRobot policies were trained on this exact dataset's `action` field
(`dx,dy,dz,droll,dpitch,dyaw,gripper`), so the adapter needs to redo this same accumulation.

### 1.4 Observation keys match our dataset's cameras directly

`custom_osc_pose_wrapper.py::post_proc_obs` strips the `robot0_` prefix from robot-state keys, so
the raw `obs` dict handed to `compute_action` has (confirmed by reading `new_pp.py::_get_observation`
and the wrapper): `camera_front_image`, `robot0_eye_in_hand_image` (front and wrist cameras — the
same two views converted into `observation.images.front` / `observation.images.gripper` in our
LeRobot dataset), `eef_pos`, `eef_quat`, `joint_pos`, `gripper_qpos`. This is enough to reconstruct
the 13-D `observation.state` vector our policies expect
(`[joint_0..5, gripper_closed, eef_x,y,z,roll,pitch,yaw]`, per `convert_dataset.py::make_frame`).

**Verify exact key names empirically** (`print(sorted(obs.keys()))` once inside `compute_action`)
before trusting this — I read the source, but the benchmark's `multi_task_robosuite_env` package
is large and I didn't execute it.

### 1.5 The hard blocker: Python 3.9 vs. LeRobot's Python 3.12+

`conda_environments/*.yml` pin `python=3.9.19`, and the harness only works from inside a GPU
`srun`/`sbatch` job (confirmed by an existing project memory: on a login node, `mujoco_py` falls
back to a from-source CPU build that fails against the conda toolchain's old `ld`; on a compute
node it uses a precompiled `.so` and works fine — don't debug this on a login node). LeRobot
requires Python 3.12+ (`CLAUDE.md`), and MolmoAct2/VLA-JEPA additionally need recent
`transformers`/`peft`/`accelerate`/`diffusers`/`qwen-vl-utils`/`torch`. Installing the LeRobot
policy stack into the benchmark's Python 3.9 env is not realistic — pin conflicts across that many
packages, some of which have already dropped 3.9 support outright.

**Decision: don't merge environments.** Run the two halves as separate processes that talk over a
local HTTP connection on the same node:

- **Server** (`lerobot` conda env, Python 3.12, GPU): loads the trained policy via LeRobot's own
  `make_policy`/`make_pre_post_processors` (the same machinery `lerobot-eval` uses), exposes one
  `POST /predict` endpoint.
- **Client** (`tinyvla_robosuite_1_0_1_provola` conda env, Python 3.9): the new
  `models/lerobot_policy.py`, doing only the robosuite-side bookkeeping (image/state packaging,
  delta→absolute conversion) and an HTTP call. No torch/transformers import needed on this side.

This also means the LeRobot side never needs to see robosuite/mujoco_py at all, and the benchmark
side never needs to see LeRobot's dependency tree — each stays exactly as it is today.

## 2. Files to create

```
lerobot/lerobot/run_eval_scripts/
  lerobot_policy_server.py          # new — Step 3

VLA-Benchmark/robosuite_test/
  models/
    lerobot_policy.py               # new — Step 4 (client adapter)
    lerobot_eval_config.yml         # new — Step 6, one per checkpoint (or reuse with CLI overrides)
  run_lerobot_eval.sh               # new — Step 7 (sbatch launcher, starts server + client)
```

Plus small edits to `models/configs.py`, `run_robosuite_eval.py`, and `robot_utils.py`
(`MODEL_IMAGE_SIZES`) — Step 5.

## 3. Step-by-step

### Step 1 — Sanity-check the benchmark harness still works today

Before touching anything, confirm the existing TinyVLA path still runs (proves the sim/env/conda
setup is healthy, isolates any later failure to the new adapter code):

```bash
cd /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test
sbatch run_robosuite_eval.sh 0 false
```

Watch the resulting `slurm-<id>.out`; it should get past env build and start logging
`Reached rate:` / `Success rate:` lines.

### Step 2 — Pick which checkpoints to test

Both training chains from this session write LeRobot-format checkpoints under `checkpoints/last`:

```
lerobot/lerobot/outputs/ur5e_molmoact2/checkpoints/last/pretrained_model
lerobot/lerobot/outputs/ur5e_vla_jepa/checkpoints/last/pretrained_model
```

Confirm they exist and note the step count before running a full eval (a half-trained checkpoint
is still loadable, just not representative):

```bash
readlink -f lerobot/lerobot/outputs/ur5e_molmoact2/checkpoints/last
readlink -f lerobot/lerobot/outputs/ur5e_vla_jepa/checkpoints/last
```

### Step 3 — LeRobot inference server (runs in the `lerobot` conda env)

**Status: done, smoke-tested against both checkpoints on a GPU node.** The file at
`lerobot/lerobot/run_eval_scripts/lerobot_policy_server.py` is reproduced below verbatim — this
is real, working code, not a sketch. Two bugs in the original draft were found and fixed by
actually running it:

1. `make_policy(cfg=cfg, env_cfg=None)` fails: `ValueError: Either one of a dataset metadata or a
   sim env must be provided` — the factory always wants to (re)derive feature shapes from a
   `LeRobotDatasetMetadata` or `EnvConfig`, neither of which exists in a bare inference server.
   Fixed by loading directly via the policy class's own `from_pretrained`, bypassing the factory's
   shape-inference path entirely (the checkpoint's `config.json` already has resolved features).
2. MolmoAct2's `select_action()` raises `ValueError: MolmoAct2 inference requires
   inference_action_mode to be set explicitly to either 'continuous' or 'discrete'` — it has no
   default, unlike other policy types. Fixed with a `--inference_action_mode` CLI flag (default
   `continuous`, matching every MolmoAct2 eval script from the LIBERO work), applied only when
   `cfg.type == "molmoact2"`.

```python
#!/usr/bin/env python
"""Minimal HTTP server exposing a trained LeRobot policy for out-of-process inference.

Exists so robosuite_test (Python 3.9) can query a LeRobot policy (Python 3.12+) without
merging the two environments. Run inside the `lerobot` conda env on the same GPU node as
the robosuite_test client.
"""
import argparse
import base64
import io

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

app = Flask(__name__)
STATE = {}


def decode_image(b64_png: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_png))).convert("RGB"))


@app.get("/")
def health():
    return "ok"


@app.post("/predict")
def predict():
    payload = request.get_json()
    device = STATE["device"]

    batch = {"task": [payload["task_description"]]}
    for key, b64_png in payload["images"].items():
        img = decode_image(b64_png)  # HWC uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch[f"observation.images.{key}"] = t.unsqueeze(0).to(device)

    state = torch.tensor(payload["state"], dtype=torch.float32).unsqueeze(0).to(device)
    batch["observation.state"] = state

    obs = STATE["preprocessor"](batch)
    with torch.no_grad():
        action = STATE["policy"].select_action(obs)
    action = STATE["postprocessor"](action)
    action = action.squeeze(0).cpu().numpy().tolist()

    return jsonify({"action": action})  # 7-D: dx,dy,dz,droll,dpitch,dyaw,gripper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    # MolmoAct2 refuses to run select_action() without this set explicitly (it has no default —
    # see configuration_molmoact2.py::resolve_inference_action_mode). Other policy types ignore it.
    parser.add_argument("--inference_action_mode", default="continuous", choices=["continuous", "discrete"])
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    if cfg.type == "molmoact2":
        cfg.inference_action_mode = args.inference_action_mode
    # make_policy() requires a ds_meta or env_cfg to (re)derive feature shapes — neither exists
    # here (no LeRobot dataset/env involved), so load directly via the policy class's own
    # from_pretrained instead; the checkpoint's saved config.json already has resolved
    # input/output features baked in.
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=cfg)
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.policy_path
    )

    STATE.update(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, device=args.device)
    print(f"LeRobot policy server ready on port {args.port} (policy_path={args.policy_path})")
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
```

**Verified working** for both checkpoints with a synthetic request (random image + zero state
vector) on a GPU node: both returned a structurally valid 7-D action with no errors. This proves
the server plumbing (model loading, `/predict` request handling, pre/post-processor pipeline)
works — it does **not** prove the predictions are numerically meaningful, since the input was
synthetic. What's still genuinely unverified: whether `select_action` expects the `task` key
inside the same batch dict (as written above) or as a separate argument when given *real*
correlated image/state input rather than random noise — the synthetic test can't distinguish
"correctly ignoring/using task" from "silently misusing it," only "doesn't crash." Re-check once
real observations flow through in Step 8's smoke test.

### Step 4 — robosuite-side client adapter (runs in `tinyvla_robosuite_1_0_1_provola`)

**Status: written, wired in, confirmed reaching a real `/predict` call in a live rollout** (see
Step 8). One real bug found and fixed by actually running it: the original draft's
`euler_to_axis_angle` imported from `openvla_utils`, which does an unconditional
`import tensorflow` at module level — not installed in this (TinyVLA) conda env, so the import
crashed the very first time an action needed converting. Fixed by inlining the same rotation-matrix
math directly (`openvla_utils.py:885-923`) instead of importing the module.

`VLA-Benchmark/robosuite_test/models/lerobot_policy.py`:

```python
import base64
import time

import cv2
import numpy as np
import requests
from robosuite.utils.transform_utils import mat2euler, quat2mat

from .configs import LeRobotPolicyConfig


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def euler_to_axis_angle(euler):
    # Inlined from openvla_utils.euler_to_axis_angle (same math, so both model families
    # accumulate orientation the same way) rather than imported: that module does an
    # unconditional `import tensorflow`, which isn't installed in this (tinyvla) conda env.
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if np.isclose(angle, 0):
        return np.zeros(3)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
    return axis * angle


class lerobot_remote_policy:
    def __init__(self, cfg: LeRobotPolicyConfig):
        self.cfg = cfg
        self.url = f"http://127.0.0.1:{cfg.server_port}/predict"
        self.chunk_size = cfg.chunk_size
        # fail fast if the server isn't up yet, rather than timing out on the first real request
        requests.get(f"http://127.0.0.1:{cfg.server_port}/", timeout=10)

    def _encode_image(self, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert ok, "failed to encode image as PNG"
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _build_state(self, obs, gripper_closed):
        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)[:6]
        eef_pos = np.asarray(obs["eef_pos"], dtype=np.float32)
        eef_euler = np.array(
            [normalize_angle(a) for a in mat2euler(quat2mat(obs["eef_quat"]))], dtype=np.float32
        )
        # 13-D, matches convert_dataset.py::make_frame's observation.state layout:
        # [joint_0..5, gripper_closed, eef_x,y,z,roll,pitch,yaw]
        state = np.concatenate([joint_pos, [float(gripper_closed)], eef_pos, eef_euler])
        return state.tolist()

    def compute_action(self, obs, resize_size, gripper_closed, task_description, task_name="pick_place", n_steps=-1):
        start = time.time()
        images = {
            "front": obs["camera_front_image"],
            "gripper": obs["robot0_eye_in_hand_image"],
        }
        payload = {
            "images": {k: self._encode_image(v) for k, v in images.items()},
            "state": self._build_state(obs, gripper_closed),
            "task_description": task_description,
        }
        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        delta = np.array(resp.json()["action"], dtype=np.float64)  # dx,dy,dz,droll,dpitch,dyaw,gripper
        elapsed = time.time() - start

        action_world = np.zeros(7)
        action_world[0:3] = obs["eef_pos"] + delta[0:3]
        current_euler = [normalize_angle(a) for a in mat2euler(quat2mat(obs["eef_quat"]))]
        target_euler = [normalize_angle(a) for a in (np.array(current_euler) + delta[3:6])]
        action_world[3:6] = euler_to_axis_angle(target_euler)
        action_world[6] = delta[6]

        # Single-action chunk: re-query the server every env step, exactly like lerobot-eval's own
        # rollout() does. Simpler and safer than replicating LeRobot's internal action-queue
        # chunking on the client side.
        return [action_world], elapsed
```

Confirmed empirically so far: `obs["joint_pos"]`, `obs["eef_pos"]`, `obs["eef_quat"]`,
`obs["camera_front_image"]`, `obs["robot0_eye_in_hand_image"]` all exist and are readable — the
Step 8 run got past `_build_state`/`_encode_image` and all the way to a real `/predict` response
before hitting the (now-fixed) `tensorflow` import crash in the orientation conversion. **Still
genuinely unverified, not by reading source** — these need a rollout that runs long enough to
produce visible behavior, which the fix above should now unblock:

- **Gripper value scale/sign.** Our dataset's raw gripper action is `0–20`
  (see `dataset_statistics_*.json`, near-bimodal), but MolmoAct2/VLA-JEPA's postprocessor
  un-normalizes back to *some* range — check what actually comes back from `/predict` and compare
  against `test/pick_place.py`'s gripper thresholding (`> 0.75` → close, `< 0.75` → open) before
  assuming it lines up. May need a rescale in the adapter.
- **Delta frame convention.** `action_world[0:3] = obs['eef_pos'] + delta[0:3]` assumes the
  dataset's `action` deltas are in the world frame already. If rollouts drift sideways or the
  gripper overshoots consistently in one rotational direction, the delta is probably still in the
  EEF-local frame and needs a rotation by the current `eef_quat` before adding.
- **`resize_size`/image preprocessing.** The server's `/predict` doesn't currently do the
  crop-then-resize preprocessing `models/openvla.py::compute_action` does (crop via `TASK_CROP`,
  resize via TF's `lanczos3`). Our LeRobot policies were trained on `convert_dataset.py`'s 224×224
  resize with no crop — keep it uncropped to match training, but confirm the two camera streams'
  raw resolution (`render_hw: (200, 360)` per `TASK_MAP` in `robosuite_utils.py`) doesn't need
  letterboxing to avoid distortion.

### Step 5 — Register the new model family

`VLA-Benchmark/robosuite_test/models/configs.py` — add:

```python
@ModelConfig.register_subclass('lerobot')
@dataclass
class LeRobotPolicyConfig(ModelConfig):
    model_path: str = ""          # LeRobot checkpoint dir, e.g. .../ur5e_molmoact2/checkpoints/last/pretrained_model
    server_port: int = 8765
    chunk_size: int = 1
    task_suite_name: str = ''
    otd: bool = False
    use_cosmos_name: bool = False
    dataset_path: str = ''
```

`run_robosuite_eval.py` — add a branch next to the existing two:

```python
elif cfg.model_family.lower() == "lerobot":
    from robosuite_test.models.lerobot_policy import lerobot_remote_policy
    print("Running LeRobot policy evaluation....")
    policy = lerobot_remote_policy(cfg.model_config)
```

`robot_utils.py` — add to `MODEL_IMAGE_SIZES`:

```python
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    "tinyvla": 224,
    "lerobot": 224,
}
```

### Step 6 — Eval config YAML (one per checkpoint)

`VLA-Benchmark/robosuite_test/models/lerobot_molmoact2_eval_config.yml`:

```yaml
task_suite_name: &tsn ur5e_pick_place_delta_all
model_family: "lerobot"
num_steps_wait: 10
num_trials_per_task: 10
initial_states_path: "DEFAULT"
env_img_res: 256
save: true
local_log_dir: "./experiments/logs"
use_wandb: false
seed: 0
controller_path: "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json"
debug: false
run_number: 0
change_spawn_regions: false
change_command: false
OoD: false
object_set: -1
run_id_note: *tsn

model_config:
  type: lerobot
  model_path: '/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/outputs/ur5e_molmoact2/checkpoints/last/pretrained_model'
  server_port: 8765
  task_suite_name: *tsn
```

Copy for VLA-JEPA with `model_path` pointing at `outputs/ur5e_vla_jepa/...` and a different
`server_port` (e.g. `8766`) so both can run concurrently without colliding.

### Step 7 — sbatch launcher (starts server + client together)

`VLA-Benchmark/robosuite_test/run_lerobot_eval.sh`:

```bash
#!/bin/sh
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode09
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CONFIG=${1:-models/lerobot_molmoact2_eval_config.yml}
POLICY_PATH=${2:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/outputs/ur5e_molmoact2/checkpoints/last/pretrained_model}
PORT=${3:-8765}

# --- start the LeRobot inference server in the lerobot (Python 3.12) env ---
LEROBOT_CONDA=/mnt/beegfs/frosa/.conda/envs/lerobot
"${LEROBOT_CONDA}/bin/python" \
    /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_eval_scripts/lerobot_policy_server.py \
    --policy_path "${POLICY_PATH}" --port "${PORT}" &
SERVER_PID=$!

# wait for the server's health endpoint before starting the client
for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 && break
    sleep 5
done

# --- run the robosuite client in the benchmark's (Python 3.9) env ---
export PATH=/mnt/beegfs/frosa/.conda/envs/tinyvla_robosuite_1_0_1_provola/bin:$PATH
srun python run_robosuite_eval.py \
    --config_path="${CONFIG}" \
    --task_suite_name "ur5e_pick_place_delta_all" \
    --run_number 0 \
    --change_spawn_regions false \
    --num_trials_per_task 10

kill "${SERVER_PID}" 2>/dev/null
```

Note the health endpoint the launcher polls (`GET /`) isn't in the Step 3 server sketch — add a
trivial `@app.get("/")` returning `"ok"` alongside `/predict`.

### Step 8 — Smoke test before a full run

Run with `num_trials_per_task: 1` and a single variation first
(`--task_suite_name ur5e_pick_place_delta_all` still iterates all 16 by default — temporarily hack
`TASK_VARIATION_DICT` or just watch the first rollout and Ctrl-C). Check, in order:

1. Server logs "ready" and the client's health check succeeds — proves the process split works.
2. The first `/predict` response shape is `(7,)` and doesn't error inside `select_action` —
   proves the observation batch dict the server builds matches what the policy expects.
2. `images/step_*.jpg` (written by `test/pick_place.py` every step) show the arm actually moving,
   not frozen or flailing — proves the axis-angle conversion is roughly sane.
3. Gripper visibly closes near the target object at least sometimes — proves the gripper scale/sign
   guess in Step 4 isn't backwards.

Only after these look reasonable, run the full `num_trials_per_task: 10 × 16 variations`.

### Step 9 — Compare against the checkpoints' LIBERO numbers

This sim task (UR5e pick-place, real-robot-derived) is a genuinely different benchmark from LIBERO
— don't expect the same success rates as the LIBERO eval runs from this session (MolmoAct2 97.0%,
VLA-JEPA 97.5% there). A meaningful comparison here is against `pi0`/`pi0.5` on this same
`ur5e_pick_place_delta_all` suite if those have ever been run through this harness, or against a
TinyVLA/OpenVLA baseline already in `experiments/logs`.

## 4. Summary of what's genuinely uncertain

Everything in section 1 is read from source, not run. Before treating results from this pipeline
as meaningful:

- Confirm actual `obs` key names via a printed smoke-test (Step 8.1) rather than trusting section 1.4.
- Confirm the delta-action frame convention (world vs. EEF-local) empirically (Step 4's callout).
- Confirm the gripper output scale/sign empirically (Step 4's callout).
- Confirm `make_pre_post_processors`'s exact call signature against
  `src/lerobot/scripts/lerobot_eval.py` before trusting the Step 3 server code verbatim.
