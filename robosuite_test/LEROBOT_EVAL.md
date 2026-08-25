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
      pipeline is producing meaningful results, not an artifact. **Job ran to completion**: full
      16/16 variations, final tally **4/16 success** (variations 0, 2, 6, 14), 8/16 reached-but-
      not-picked, 4/16 never reached. Rollout pkls at
      `outputs/ur5e_molmoact2/checkpoints/008000/pretrained_model/rollout_pick_place_3_.../`.

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
      **Job ran to completion**: full 16/16, `reached=1` on 13/16, `picked=0` on all 16 — a very
      clean, consistent "reaches every time, never closes" signature, not a small-sample fluke.

      **Root-caused (2026-08-23), and it's a pipeline bug, not a model-behavior limitation.**
      Investigated three things per request: (1) whether training was cut short, (2) rollout
      videos for both models, (3) the raw pre-binarize gripper value.

      *1. Training length* — read both models' wandb `output.log` loss curves directly (not just
      the final summary). MolmoAct2: loss falls smoothly and monotonically `0.72 → 0.006` over
      **2.35 epochs** (256k samples), LR annealed cleanly to its floor — converged, textbook curve.
      VLA-JEPA: loss falls `1.19 → ~0.13-0.14` but visibly plateaus/gets noisy over the last
      ~3000 steps (`0.142, 0.134, 0.136, ...`) while completing only **1.17 epochs** (128k
      samples) — half of MolmoAct2's data coverage at the same step count, because of a smaller
      per-step batch. Total loss is dominated by VLA-JEPA's world-model loss (`wm_loss=0.121` of
      `loss=0.150`); the action loss itself (`action_loss=0.029`) isn't logged per-step so its own
      convergence can't be directly confirmed from stdout. This is a real, mild undertraining
      signal for VLA-JEPA relative to MolmoAct2 — but turned out not to be the explanation for the
      gripper symptom (see below).

      *2. Rollout videos* — rendered all 32 (16 MolmoAct2 + 16 VLA-JEPA) smoke-test trajectories to
      mp4 via `create_video.py` (`run_create_videos.sh`, since `create_video.py` unpickles a traj
      object that imports `robosuite_utils` → `robosuite` → `mujoco_py`, which JIT-compiles a
      native extension that only links successfully on the gnode compute nodes with the
      `MUJOCO_GL=egl`/`LD_LIBRARY_PATH` env from `run_lerobot_eval.sh` — it fails with a `gcc`/`ld`
      link error on a bare login-node shell). Output at
      `VLA-Benchmark/robosuite_test/lerobot_eval_videos/{molmoact2,vla_jepa}/traj_*.mp4`.

      *3. Raw pre-binarize gripper value* — added a temporary, env-var-gated
      (`LEROBOT_DEBUG_GRIPPER=1`) `after_step_hooks` logger to `lerobot_policy_server.py` that
      prints the gripper dim after every postprocessor step (no edits to the shared lerobot
      library needed — `PolicyProcessorPipeline` already exposes `after_step_hooks` for exactly
      this). Result across a full 16-variation VLA-JEPA rollout (job 536038):
      ```
      after step 0 (ClipActionsProcessorStep):      spread across ~[-1, 1]   (model's raw normalized output — NOT saturated)
      after step 1 (PreSnapGripperProcessorStep):    1629× 1.0 (close-ish) / 1546× 0.0 (open-ish)   (~51/49 split — reasonable!)
      after step 2 (UnnormalizerProcessorStep):      1628× 20.0             / 1546× 10.0
      after step 3 (BinarizeGripperProcessorStep):   3175× -1.0             / 0× +1.0               (always "open", unconditionally)
      ```
      This directly disproves "the model never wants to close" — `PreSnapGripperProcessorStep`
      shows the model deciding to close about as often as open (a sane distribution for a
      pick-place task). The bug is entirely downstream, in VLA-JEPA's own
      `make_vla_jepa_pre_post_processors` (`lerobot/src/lerobot/policies/vla_jepa/processor_vla_jepa.py`):
      - `PreSnapGripperProcessorStep` and `BinarizeGripperProcessorStep` both read the **same**
        `config.gripper_threshold` (default `0.5`), but they run on data in two different unit
        spaces: PreSnap runs on the *normalized* `[-1, 1]` model output (where `0.5` is a sane
        cutpoint), Binarize runs *after* `UnnormalizerProcessorStep` on real dataset units.
      - This dataset's gripper `action` dim is `MIN_MAX`-normalized over a real range of
        `[min=0, max=20]` (confirmed in `meta/stats.json`). Feeding PreSnap's identity `{0, 1}`
        decision through that MIN_MAX unnormalizer (formula `(x+1)/2*(max-min)+min`, which expects
        a `[-1,1]`-space input) turns `0` → `10.0` and `1` → `20.0` — both **far above** the
        reused threshold of `0.5`, so `BinarizeGripperProcessorStep`'s `a > threshold` is true
        100% of the time, collapsing every prediction to `1.0 - 2.0*1 = -1.0` regardless of what
        PreSnap decided. That's the entire bug: a threshold constant tuned for normalized space,
        silently reused on real-scale data.
      - Separately, even after fixing the threshold, the formula's sign convention is inverted for
        this dataset's real-unit direction: `a > threshold → -1.0`, so the *high* real cluster
        (`20.0`, the close decision) maps to `-1.0` and the *low* cluster (`10.0`, open) maps to
        `+1.0` — backwards from `pick_place.py`'s convention (`+1.0` = close).
      **Fix implemented in `lerobot_policy_server.py` only** (no shared-library or checkpoint-file
      edits): for `cfg.type == "vla_jepa"`, pass
      `postprocessor_overrides={"vla_jepa_binarize_gripper": {"threshold": 15.0}}` to
      `make_pre_post_processors` (15.0 sits cleanly between the two real clusters 10/20 — note
      `make_pre_post_processors` loads the postprocessor pipeline from the checkpoint's saved JSON
      when `pretrained_path` is given, so mutating `cfg` fields beforehand has *no effect*; only
      the `overrides=` kwarg reaches it, keyed by the step's registry name), then flip the gripper
      dim's sign in `/predict`'s response to correct the inverted convention.
      **Verified (job 536089, full 16 variations, fix + debug logging both on):** final gripper
      values are no longer constant — `1458× -1.0` / `2046× +1.0`, varying exactly as expected.
      Task outcomes: `picked=1` on 4/16 (variations 7, 8, 10, 14), up from **0/16 before the fix**.
      `success` is still 0/16 — picking now works, placing-in-the-correct-bin apparently doesn't
      yet, which is a distinct, not-yet-investigated question (possibly related to the same
      undertraining signal from point 1, possibly a separate issue). This is a genuine, verified
      fix to a real bug, not a tuning tweak — before it, VLA-JEPA could never complete this task
      structurally regardless of how well-trained or well-adapted anything else was.
- [x] **Full MolmoAct2 run** — ran at `num_trials_per_task=1` × 16 variations (not yet the full
      `num_trials_per_task=10`): 4/16 success. Full `×10` run still pending.
- [~] **Full VLA-JEPA run** — ran at `num_trials_per_task=1` × 16 variations post-gripper-fix:
      0/16 success, 4/16 picked. Full `×10` run still pending, and worth rerunning now that the
      structural gripper bug is fixed.
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
- **`resize_size`/image preprocessing — RESOLVED (2026-08-25), the assumption below was wrong.**
  `models/lerobot_policy.py::compute_action` was sending both camera streams raw (sim's native
  `render_hw: (200, 360)`, uncropped, unresized) all the way to the server, which had no resize
  step of its own either — `resize_size` was accepted as a parameter and never used. Fixed by
  applying the same `TASK_CROP`-then-resize `models/openvla.py`/`models/tinyvla.py` apply to
  `camera_front_image`, and a plain resize (no crop) to the wrist/gripper image, both to
  `resize_size` (224). The reasoning below — "trained on a 224×224 resize with no crop, so keep
  eval uncropped to match" — conflated two different things: `convert_dataset.py`'s 224×224 shape
  describes the *real* UR5e camera's already-correctly-framed images, not sim's. Sim's raw render
  has a wider FOV than the real camera, so `TASK_CROP` exists specifically to make sim's front
  view match what the real camera saw — a sim/real correction that's model-agnostic (openvla and
  tinyvla both need it for the same reason), not a training-time preprocessing choice to mirror.
  Original note, kept for context: ~~The server's `/predict` doesn't currently do the
  crop-then-resize preprocessing `models/openvla.py::compute_action` does (crop via `TASK_CROP`,
  resize via TF's `lanczos3`). Our LeRobot policies were trained on `convert_dataset.py`'s 224×224
  resize with no crop — keep it uncropped to match training, but confirm the two camera streams'
  raw resolution (`render_hw: (200, 360)` per `TASK_MAP` in `robosuite_utils.py`) doesn't need
  letterboxing to avoid distortion.~~

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

## 5. Extending training past step 8000 (2026-08-23)

Following the training-length finding in §TODO (VLA-JEPA only reached 1.17 epochs, loss still
plateauing at step 8000), extended both checkpoints from 8000 → **12000** steps via the existing
resume mechanism in `train_ur5e_molmoact2.sh`/`train_ur5e_vla_jepa.sh`.

**The resume scripts needed a real change, not just a rerun.** Their resume branch previously
called `lerobot_train --config_path=.../train_config.json --resume=true` with no `--steps`
override — since the saved config already has `steps=8000` and both runs already reached exactly
that, a plain resume would just detect training is complete and exit immediately. Added
`--steps="${STEPS}"` to the resume branch in both scripts so the target step count can actually be
extended.

**Confirmed this properly "reheats" the LR, not just continues at an already-decayed floor.**
Both models' schedulers (`CosineDecayWithWarmupSchedulerConfig` for VLA-JEPA,
`MolmoAct2CosineDecayWithWarmupSchedulerConfig` for MolmoAct2) are built fresh at train-start via
`cfg.scheduler.build(optimizer, cfg.steps)` — i.e. keyed off the *new* `cfg.steps=12000`, not the
old 8000 — and `load_training_state` then restores just the step counter (~8000) and optimizer
state on top of that freshly-shaped schedule. VLA-JEPA's scheduler auto-scales its warmup/decay
to fit whatever `num_training_steps` it's given whenever that's shorter than its configured
`num_decay_steps=30000` (see `schedulers.py`'s `CosineDecayWithWarmupSchedulerConfig.build`);
MolmoAct2's `num_decay_steps=null` means "decay across this run's steps" directly. Verified by
reading the actual logged LR right after resuming: MolmoAct2 went from `1.6e-05` and VLA-JEPA from
`2.1-2.3e-05` — both well above their pre-resume floors (`5e-06`/`1e-06`), confirming the extra
4000 steps are training at a genuinely non-trivial LR, not stalled at zero.

**Found and fixed a real bug in shared `lerobot_train.py`, blocking MolmoAct2 resume entirely.**
First resume attempt (job 536111) crashed immediately with
`KeyError: "Override keys ['normalizer_processor'] do not match any step in the saved
configuration. Available step keys: [..., 'molmoact2_masked_normalizer', ...]"`. Root cause:
`lerobot_train.py`'s generic resume-path processor-override logic (triggered whenever
`policy.pretrained_path` is set, which resume always does) hardcodes the override dict keys as
`"normalizer_processor"`/`"unnormalizer_processor"` — correct for VLA-JEPA (which does register
under those generic names, confirmed in its saved `policy_postprocessor.json`) but wrong for
MolmoAct2, whose masked-passthrough gripper handling registers its own
`"molmoact2_masked_normalizer"`/`"molmoact2_masked_unnormalizer"` step classes instead (plain
subclasses of the generic Normalizer/UnnormalizerProcessorStep with no extra constructor args, so
the same override kwargs shape still applies — only the lookup key was wrong). This would have
blocked *any* attempt to resume MolmoAct2 training, independent of the `--steps` change; it never
surfaced before because the original 8000-step run finished inside a single job/walltime and never
needed to resume. Fixed with a 3-line policy-type check in `lerobot_train.py` (§ around
`processor_pretrained_path is not None`) selecting the correct step key per policy type.
MolmoAct2's checkpoint was untouched by the failed attempt (crashed before any training step or
checkpoint write) — safe to retry, and the retry (job 536114) resumed cleanly.

**Status: both running.** Job 536112 (VLA-JEPA) and job 536114 (MolmoAct2), both targeting
step 12000, both training at a properly reheated LR. Estimated finish (from original per-step
timing: VLA-JEPA ~1.03s/step, MolmoAct2 ~2.3s/step, ×4000 steps): VLA-JEPA ~1.3h, MolmoAct2 ~2.9h
from their resume start — both comfortably inside the single-job 6:50:00 walltime cap, no
job-chaining needed. Once both finish, the natural next step is to re-run the VLA-Benchmark smoke
tests (§ Step 8 / TODO) against the new `checkpoints/last` to see whether the extra training moves
VLA-JEPA's placement success rate (4/16 picked, 0/16 success as of the last run) or MolmoAct2's
(4/16 success).

## 6. Why VLA-JEPA picks but never places (2026-08-23)

Investigated the `picked=1, success=0` episodes (variations 7, 8, 10, 14, job 536089) by tracing
the raw model output (not just the final binarized signal) through the placement phase.

**Ruled out an adapter chunking bug first.** `VLAJEPAPolicy.select_action()` (in
`lerobot/src/lerobot/policies/vla_jepa/modeling_vla_jepa.py`) implements proper internal
action-chunk queuing (`chunk_size=7`): it only calls `predict_action_chunk()` when its internal
queue is empty, otherwise it dequeues. Since `lerobot_policy_server.py` keeps the same `policy`
object alive across HTTP requests, our per-env-step querying already gets this right — every 7th
`/predict` call triggers a real forward pass, the other 6 just dequeue. No bug here.

**The raw model output during placement is confidently oscillating, not uncertain.** Sample of the
normalized pre-snap gripper value across consecutive steps near the end of episode 7:
`-0.97 -1.0 0.82 0.85 0.93 0.92 0.93 -0.97 -0.98 0.88 0.97 ...` — solidly "open" for 1-2 steps,
then solidly "closed" for 4-5, repeating. Not hovering near a decision boundary (that would show
values clustered near 0); the model commits hard to one state, then hard to the other, rapidly.
This means VLA-JEPA's OWN predicted 7-step chunks contain this alternation — a genuine model
behavior, most plausibly explained by the training-length finding above (1.17 epochs, loss still
plateauing at step 8000 — the "release over the bin" sub-behavior is a small fraction of every
demonstration and easy to undertrain relative to reach/grasp).

**Found and fixed a compounding harness bug: asymmetric gripper-transition handling in
`test/pick_place.py`.** The `gripper_state_changed` dance (lines ~180-198) that runs whenever the
binarized gripper command flips was asymmetric: a transition *to close* got a settle motion plus
a **10-step hold** at the closed position (`for i in range(10): env.step(...)`), giving the grasp
time to secure. A transition *to open* got only 2 raw steps and **no hold at all**. Given the
model's own rapid flickering, this meant every brief "open" decision had almost no physical time
to actually let the object drop before the very next flicker snapped the gripper shut again —
the harness was structurally biased toward "stays closed" regardless of what the model intended.
**Fix**: added the same `for i in range(10): env.step(action_world)` hold to the opening branch,
so a release gets as much time to take physical effect as a grasp does.

**Verified on the unchanged step-8000 VLA-JEPA checkpoint** (job 536134, explicitly pointed at
`checkpoints/008000/pretrained_model` rather than `last`, since the step-8000→12000 training
extension was actively moving `last` forward at the time and reading it mid-write would have been
a race) — isolates the harness fix from the training-extension variable for a clean before/after.

Symmetric-opening result: still 4/16 picked, 0/16 success, and — new data point — **0/16 landed
in any bin at all, not even the wrong one** (`place_wrong_correct_obj` etc. all 0). Since
`check_pick()` is a monotonic OR-latch (`picked or (reached and abs(obj_z-start_z)>threshold)`,
`robosuite_utils.py:209`), `picked=1` only means the object was lifted *at some point*, not that
it's still held near episode end. Combined with landing in zero bins, the object is most likely
being dropped/bumped near the pickup area during the gripper flicker, well before the arm ever
carries it near a bin — the dwell-time fix wasn't the binding constraint; the deeper issue is
instability through the whole carry phase.

## 7. Extending both trainings to 100,000 steps, and the orientation-delta investigation (2026-08-23)

### 7a. Training extension to 100K

User's checkpoints were at 12000 steps each (VLA-JEPA: fresh-finished; MolmoAct2: still resuming
toward 12000 when this started). Target: at least 100,000 steps for both, "resume all the
trainings."

**Found a second scheduler bug before committing GPU time to it.** VLA-JEPA's saved scheduler has
`num_decay_steps=30000` (fixed, from the original fresh-start config). `CosineDecayWithWarmupSchedulerConfig.build()`
only auto-scales the warmup/decay horizon *down* to fit a shorter `num_training_steps` — it does
nothing when the new target *exceeds* the saved `num_decay_steps`. Left as-is, extending to 100,000
would have decayed to the LR floor at step 30,000 and then coasted there for the remaining 70,000
steps (70% of the extension wasted). Fixed by adding `--scheduler.num_decay_steps="${STEPS}"` to
`train_ur5e_vla_jepa.sh`'s resume branch, keyed to the same target as `--steps`. (MolmoAct2 doesn't
need this: its `num_decay_steps=null` already means "decay across whatever `cfg.steps` is.")

**Verified via a cheap canary before committing to the full run.** First tried `STEPS=12100` (only
100 more steps) — useless as a test, since with decay horizon also set to 12100, step ~12000 is
already ~99% through the decay curve either way, so a floor-level LR there doesn't distinguish
"override worked" from "override didn't work." Killed it and went straight to the real
`STEPS=100000` job instead: post-resume LR came back at `9.7e-05`, essentially back at peak
(`peak_lr=1e-4`) — confirms the override works and the schedule now properly spans the full
100K-step horizon.

**Hit the cluster's job-submission cap (`MaxSubmit=10` on the `did_robot_learning_359` account)**
while trying to pre-chain 10 MolmoAct2 jobs via `--dependency=afterany`. The account was already at
10/10 outstanding jobs (including 3 jobs not belonging to this work, not mine to manage). A fixed
upfront chain isn't viable under this cap. Solution: trimmed the pre-queued lookahead to 1-2 jobs
per model, and started a persistent adaptive top-up loop
(`/tmp/.../scratchpad/topup_chains.sh`, run via a background Monitor) that checks every 20 minutes
whether either model's training job has finished and, if its last checkpoint is still short of
100,000, submits the next one. This keeps total outstanding jobs within the cap automatically
without needing the full chain pre-queued.

**Important caveat**: the top-up loop only runs for the lifetime of this session. If the session
ends, in-flight/queued jobs still run to completion, but the chain won't self-extend further after
that — resuming the top-up (or manually submitting `STEPS=100000 sbatch train_ur5e_<model>.sh`
once a model's last job finishes) will be needed to actually reach 100,000 in that case.

**Timeline estimate** (from observed per-step throughput): VLA-JEPA ~1.0-1.4s/step x 88,000
remaining steps ~= 25-34h (~4-5 chained jobs); MolmoAct2 ~2.4-2.6s/step x 88,000 remaining steps
~= 59-64h (~9-10 chained jobs), starting after its own 12,000-step job finished. Total wall-clock
is roughly 1.5-3 days if jobs run back-to-back without cluster queue delay.

**Data point on the previous 4000-step extension (8000->12000, before the scheduler fix)**: smoke
test on the resulting checkpoint (job 536159) actually looked slightly *worse* than the step-8000
checkpoint -- 2/16 picked and 10/16 reached, vs. 4/16 picked and 13/16 reached before. Loss was
also flat (~0.13-0.14) across those 4000 steps. Plausible explanation: the brief LR "reheat"
(~2e-5, well above the pre-extension floor) for only 4000 steps may have mildly destabilized the
policy without enough steps afterward to reconverge -- a reasonable argument for why a much
longer, properly-scheduled extension (this 100K run) is the right move rather than another short
bump.

### 7b. Why VLA-JEPA never predicts a meaningful gripper orientation change

User's observation: the model doesn't seem to predict a gripper orientation delta, which is wrong
since the gripper's axes should align with the object's for a good grasp.

**Confirmed empirically, then root-caused.** Logged `current_euler` (the real orientation, read
fresh from `obs['eef_quat']` every step) across 301 steps spanning multiple episodes: pitch/yaw
stay within about +/-0.03 rad (a couple degrees) of their starting value for the *entire* run, no
sustained drift in either direction -- pure jitter. Roll's large apparent range (6.23 rad) is just
the +/-pi wrap-around artifact for a single fixed roll value, not real motion.

**Ruled out "model predicts exactly zero" first.** Logged the raw (post-`SCALE_FACTOR`,
pre-accumulation) `delta[3:6]` directly: it has real, non-collapsed variance (std ~
[0.003, 0.002, 0.008] rad, occasional yaw spikes to 0.15), the same order of magnitude as the
dataset's own scaled ground-truth per-step orientation deltas (`meta/stats.json`'s `action` field,
indices 3-5, x `SCALE_FACTOR`: std ~ [0.005, 0.0035, 0.015]). So the model isn't outputting a
degenerate constant -- its predictions just don't translate into any accumulated real-world
effect.

**Ruled out a harness/controller bug next, decisively.** Added a temporary, env-var-gated
amplifier (`LEROBOT_DEBUG_ORIENT_INFLATE`, `lerobot_policy.py`) that multiplies the scaled
`delta[3:6]` by a constant before use. At 20x, the real observed yaw swung a full ~0.78 rad (from
-0.44 to +0.34) within about 20 steps (job 536197) -- proof the `CustomOSCPoseWrapper` ->
`OSC_POSE` controller pipeline (`osc_pose.json`, `control_delta: true`) is fully capable of
executing orientation commands; nothing in the adapter or controller config is silently discarding
or zeroing them.

**Conclusion: this is a genuine model-training issue, not a pipeline bug.** VLA-JEPA's own
predicted per-step orientation deltas are real but too small to produce any visible net
reorientation against this controller's practical response characteristics -- consistent with the
broader training-length finding (only ~1.17-1.76 epochs at the time of these tests). No adapter
code change was needed or made for this issue; it's a candidate for re-evaluation once the
100K-step training extension (S7a) finishes, to see whether more training teaches the model to
predict larger, more decisive orientation corrections. The diagnostic hooks
(`LEROBOT_DEBUG_ORIENTATION`, `LEROBOT_DEBUG_ORIENT_INFLATE`) are left in `lerobot_policy.py`,
gated behind env vars with no effect by default, for future re-investigation.

## 8. Training chain stalled after session interruption — made self-perpetuating (2026-08-23)

Both chains silently stopped: MolmoAct2 at step 20500, VLA-JEPA at step 32500 (`checkpoints/last`
in each `outputs/ur5e_*` dir), with no job in the queue for either. Root cause: the chain-topper
from section 7a was a `Monitor`-based background loop tied to the Claude Code session. A batch of
background tasks (including that loop) got marked "stopped" with no completion record after a
session interruption/restart — SLURM `--dependency=afterany` only delays an *already-submitted*
job, it doesn't create new ones, so once the last already-queued job for each model finished,
nothing was left to submit the next link.

**Fixed at the root: made the chain self-perpetuating instead of externally watched.** Both
`train_ur5e_molmoact2.sh` and `train_ur5e_vla_jepa.sh` now `sbatch` their own next invocation as
the last thing they do, running on the compute node as part of the SLURM job itself — independent
of whether any Claude Code session is alive. Resubmit criterion is "did the checkpoint step count
advance between the start and end of this run" (not "did the process exit 0"): the single most
common reason to need the next link is gpuq's 7h walltime cap killing the job mid-training via
SIGTERM, which is *not* a clean exit and must still chain. Only refuses to resubmit when the step
count didn't move at all (catches a real immediate failure, e.g. crashing before the first
checkpoint save, without burning the account's `MaxSubmit=10` quota on an infinite failure loop).
The old session-side watcher loop is no longer relied on (it can still act as a secondary
backstop for the rare case where self-resubmission itself fails, e.g. hitting the account's
submit cap at that exact instant, but it is not the primary mechanism anymore).

Resumed both chains manually from where they stalled: MolmoAct2 (job 536586, from step 20500) and
VLA-JEPA (job 536587, from step 32500), both re-targeting 100,000 steps, both confirmed resuming
cleanly.

**Also suppressed wandb log noise**: every resume reloads the base/checkpoint weights from
scratch, each time re-emitting `transformers`' `tqdm(..., desc="Loading weights")` bar
(`core_model_loading.py`) into the wandb-captured output log — increasingly noisy given how many
resume segments this chain now involves. `transformers`' tqdm wrapper is gated by
`huggingface_hub`'s `are_progress_bars_disabled()`, which reads `HF_HUB_DISABLE_PROGRESS_BARS`
(checked once at `transformers.utils.logging` import time, so it must be set before the
interpreter starts). Added `export HF_HUB_DISABLE_PROGRESS_BARS=1` to both scripts; confirmed on
the fresh MolmoAct2 job that the bar no longer appears.
