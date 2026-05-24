import importlib
import debugpy
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import (
    get_expert_trajectory as place_expert,
)
from multi_task_robosuite_env.controllers.controllers.utils import _load_controller_config


DEFAULT_ENV_NAME = "Ur5e_PickPlaceDistractor"


def run_pick_place_smoke_test(env_name, ctrl_config, renderer=False, camera_obs=True, task_id=0, object_set=2, robot_type="UR5e",):
    
    controller_config = _load_controller_config(
        ctrl_config=ctrl_config,
        robot_type=robot_type,
        arms=("right",),
    )


    trajectory = place_expert(
        env_name,
        controller_type=controller_config,
        renderer=renderer,
        camera_obs=camera_obs,
        task=task_id,
        render_camera="camera_front",
        object_set=object_set,
    )

    print(f"Pick-place smoke test completed: env={env_name}, task={task_id}, steps={len(trajectory)}")
    return trajectory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, type=str, help="Pick-place environment name")
    parser.add_argument("--ctrl_config", default="IK_POSE", type=str, help="Controller configuration")
    parser.add_argument("--object_set", default=2, type=int, help="Pick-place object set")
    parser.add_argument("--task_id", default=0, type=int, help="Pick-place task id to test")
    parser.add_argument("--renderer", action="store_true", help="Display rendering GUI")
    parser.add_argument("--camera_obs", action="store_true", help="Enable camera observations")
    args = parser.parse_args()

    debugpy.listen(('0.0.0.0', 5678))
    debugpy.wait_for_client()
    print("Debugger attached, starting pick-place smoke test...")

    trajectory = run_pick_place_smoke_test(
        env_name=args.env_name,
        ctrl_config=args.ctrl_config,
        renderer=args.renderer,
        camera_obs=args.camera_obs,
        task_id=args.task_id,
        object_set=args.object_set,
    )

    print(f"Trajectory length: {len(trajectory)}")