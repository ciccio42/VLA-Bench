"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
import numpy as np
import torch
from robosuite_test.models.configs import EvalConfig
import wandb
import logging
from typing import Any, Dict, List, Optional, Union

# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define task suite constants
from enum import Enum
import json

class TaskSuite(str, Enum):
    PICK_PLACE = "ur5e_pick_place"
    PICK_PLACE_ABS_POSE = "ur5e_pick_place_abs_pose"
    PICK_PLACE_DELTA_ALL = "ur5e_pick_place_delta_all"
    PICK_PLACE_DELTA_REMOVED_0_5_10_15 = "ur5e_pick_place_delta_removed_0_5_10_15"
    PICK_PLACE_REMOVED_SPAWN_REGIONS_DELTA_ALL = "ur5e_pick_place_removed_spawn_regions"
    PICK_PLACE_RM_ONE_SPAWN = "ur5e_pick_place_rm_one_spawn"
    PICK_PLACE_RM_12_13_14_15 = "ur5e_pick_place_rm_12_13_14_15"
    PICK_PLACE_RM_CENTRAL_SPAWN = "ur5e_pick_place_rm_central_spawn"
     
#
TASK_VARIATION_DICT = {
    "ur5e_pick_place": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # List of variations for the pick and place task
    "ur5e_pick_place_abs_pose": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "ur5e_pick_place_delta_all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "ur5e_pick_place_removed_spawn_regions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "ur5e_pick_place_delta_removed_0_5_10_15": [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14], #[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14], #[0, 5, 10, 15],  # Variations for the pick and place task with delta removed variations
    "ur5e_pick_place_rm_one_spawn": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Variations for the pick and place task with one spawn region removed
    "ur5e_pick_place_rm_12_13_14_15": [12, 13, 14, 15], #[12, 13, 14, 15],  # Variations for the pick and place task with 12, 13, 14, 15 spawn regions removed
    "ur5e_pick_place_rm_central_spawn": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Variations for the pick and place task with central spawn region removed
}

# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.PICK_PLACE: 200,
    TaskSuite.PICK_PLACE_ABS_POSE: 220, 
    TaskSuite.PICK_PLACE_DELTA_ALL: 130,
    TaskSuite.PICK_PLACE_DELTA_REMOVED_0_5_10_15: 130,
    TaskSuite.PICK_PLACE_REMOVED_SPAWN_REGIONS_DELTA_ALL: 130,
    TaskSuite.PICK_PLACE_RM_ONE_SPAWN: 130,
    TaskSuite.PICK_PLACE_RM_12_13_14_15: 130,
    TaskSuite.PICK_PLACE_RM_CENTRAL_SPAWN: 130
}

with open("command.json", "r") as f:
    COMMAND = json.load(f)


# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
    "tinyvla": 224,
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # tf.config.experimental.enable_op_determinism()
    # tf.random.set_seed(seed)

def setup_logging(cfg: EvalConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def get_image_resize_size(cfg: EvalConfig) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]

