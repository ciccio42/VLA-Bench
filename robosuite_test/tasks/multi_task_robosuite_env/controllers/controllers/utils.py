import os
import importlib


def _load_controller_config(ctrl_config, robot_type="UR5e", arms=("right",)):
    """
    Accepts:
      - a robosuite 1.5 composite dict,
      - a robosuite <=1.4 part-controller dict,
      - a part controller name such as OSC_POSE / IK_POSE,
      - a composite controller name such as BASIC,
      - a path to either a composite JSON or old part-controller JSON.
    Returns a robosuite 1.5-compatible composite controller config whenever possible.
    """
    import importlib
    import os

    if isinstance(ctrl_config, dict):
        # If it is already composite, keep it. If it is an old part config, refactor it.
        if "body_parts" in ctrl_config:
            return ctrl_config
        if ctrl_config.get("type") in {"IK_POSE", "OSC_POSE", "OSC_POSITION", "JOINT_POSITION", "JOINT_VELOCITY", "JOINT_TORQUE"}:
            try:
                from robosuite.controllers.composite.composite_controller_factory import (
                    refactor_composite_controller_config,
                )
                return refactor_composite_controller_config(
                    ctrl_config,
                    robot_type=robot_type,
                    arms=list(arms),
                )
            except Exception:
                return ctrl_config
        return ctrl_config

    controllers = importlib.import_module("robosuite.controllers")
    load_composite = getattr(controllers, "load_composite_controller_config", None)
    load_part = getattr(controllers, "load_part_controller_config", None)
    load_legacy = getattr(controllers, "load_controller_config", None)

    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )
    except Exception:
        refactor_composite_controller_config = None

    part_controller_names = {
        "IK_POSE",
        "OSC_POSE",
        "OSC_POSITION",
        "JOINT_POSITION",
        "JOINT_VELOCITY",
        "JOINT_TORQUE",
    }

    if isinstance(ctrl_config, str) and ctrl_config in part_controller_names:
        if load_part is None:
            if load_legacy is not None:
                return load_legacy(default_controller=ctrl_config)
            raise RuntimeError(f"No part controller loader found for {ctrl_config}")

        part_config = load_part(default_controller=ctrl_config)
        if refactor_composite_controller_config is not None:
            return refactor_composite_controller_config(
                part_config,
                robot_type=robot_type,
                arms=list(arms),
            )
        return part_config

    if isinstance(ctrl_config, str) and os.path.isfile(ctrl_config):
        if load_composite is not None:
            try:
                return load_composite(controller=ctrl_config)
            except Exception:
                pass

        if load_part is not None:
            part_config = load_part(custom_fpath=ctrl_config)
            if refactor_composite_controller_config is not None:
                return refactor_composite_controller_config(
                    part_config,
                    robot_type=robot_type,
                    arms=list(arms),
                )
            return part_config

        if load_legacy is not None:
            return load_legacy(custom_fpath=ctrl_config)

        raise RuntimeError(f"No controller loader found for file: {ctrl_config}")

    # Composite controller default, e.g. BASIC.
    if load_composite is not None:
        return load_composite(controller=ctrl_config)

    if load_legacy is not None:
        return load_legacy(default_controller=ctrl_config)

    raise RuntimeError(f"No controller loader found for controller: {ctrl_config}")