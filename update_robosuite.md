# Update Robosuite to 1.5

This guide updates the benchmark stack to robosuite 1.5 while keeping the custom environment, controllers, and utils in `robosuite_test/tasks/multi_task_robosuite_env`.

## 1. Create the conda environment

```bash
conda create -n multi_task_robosuite_1_5 python=3.10 -y
conda activate multi_task_robosuite_1_5
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install MuJoCo and robosuite 1.5

```bash
python -m pip install "mujoco>=3.1"
python -m pip install "robosuite>=1.5"
```

## 3. Install the benchmark workspace in editable mode

```bash
cd /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test
python -m pip install -e ./tasks
python -m pip install -e .
```

## 4. Install the project-specific runtime dependencies

Use the dependency set that matches the model you are evaluating.

```bash
python -m pip install -r python_requirements/openvla_requiments.txt
python -m pip install -r python_requirements/tinyvla_requirements.txt
```

## 5. Keep the same task behavior

The custom task package under `tasks/multi_task_robosuite_env` should keep the same task names, camera setup, controller logic, and object placement settings. If any task stops matching the old behavior, check the corresponding file in:

- `robosuite_test/tasks/multi_task_robosuite_env/tasks`
- `robosuite_test/tasks/multi_task_robosuite_env/controllers`
- `robosuite_test/tasks/multi_task_robosuite_env/config`

## 6. Validate the update

Run a quick import and environment smoke test after installation.

```bash
python - <<'PY'
from multi_task_robosuite_env import get_env
print('multi_task_robosuite_env import ok')
PY
```

If you launch an evaluation script, make sure it points to the updated environment config and the robosuite 1.5-compatible controller settings.