import requests
import socket
from requests.exceptions import RequestException, Timeout
import os
import glob
import random
import torchvision.transforms.functional as F
from torchvision.io import read_video
import torchvision
import subprocess
import yaml
import os

HOST_NAME = "gnode09"
PATH_TO_BIN = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Video-Captioning/cosmos-reason2/.venv/bin/cosmos-reason2-inference"
    

def is_vllm_server_up(
    host: str,
    port: int = 8000,
    timeout: float = 3.0,
):
    """
    Check whether a vLLM server is up and reachable.

    Works for:
    - local server
    - remote node
    - SSH port-forwarded server

    Returns:
        (is_up: bool, message: str)
    """
    base_url = f"http://{host}:{port}"
    url = f"{base_url}/v1/models"

    # 1. Quick DNS / socket sanity check
    try:
        socket.gethostbyname(host)
    except socket.gaierror as e:
        return False, f"Host resolution failed: {e}"

    # 2. HTTP check
    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            return True, f"vLLM server is up. Models available: {model_ids}"

        return False, f"Server responded with status {response.status_code}"

    except Timeout:
        return False, "Connection timed out (server not reachable)"
    except RequestException as e:
        return False, f"Connection failed: {e}"


def pad_video_to_square(video_tensor):
    """Pads a [T, H, W, C] video tensor to a square [T, C, max, max]."""
    # Convert [T, H, W, C] -> [T, C, H, W] for torchvision
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    t, c, h, w = video_tensor.shape
    max_side = max(h, w)
    
    # Calculate padding to center the image
    pad_h = (max_side - h) // 2
    pad_w = (max_side - w) // 2
    
    # Padding: [left, top, right, bottom]
    padded_video = F.pad(video_tensor, [pad_w, pad_h, pad_w, pad_h], fill=0)
    return padded_video, pad_h, pad_w, max_side


def run_vllm_server(
    model_name: str = "nvidia/Cosmos-Reason2-8B",
    dataset_path: str = "/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset/",
    env_name: str = "pick_place",
    variation_id: int = 0,
    host: str = "gnode01",  # or HOST_NAME
    port: int = 8000,
    prompt_yaml: str = "./prompt/human_task_description_prompt.yaml"
):
    # read video files
    print(f"Calling vLLM server for model: {model_name} on {host}:{port}...")
    
    video_path = os.path.join(dataset_path, env_name, "human_dataset", f"task_{variation_id:02d}", "videos")
    video_files = glob.glob(os.path.join(video_path, "*front_image*.mp4"))
    video_files.sort(key=lambda x: int(x.split("/")[-1].split("traj")[-1].split("_")[0]))
    video_file = random.choice(video_files)
    print(f"Using video file: {video_file}")
    
    task_name = f"task_{variation_id:02d}"
    traj_name = os.path.basename(video_file).split(".mp4")[0]

    # Read and Pad Video
    video_frames, _, _ = read_video(video_file, pts_unit="sec")
    img_h, img_w = video_frames.shape[1:3]
    padded_video, offset_y, offset_x, square_size = pad_video_to_square(video_frames)
    print(f"Original Size: ({img_w}, {img_h}), Padded Size: ({padded_video.shape[3]}, {padded_video.shape[2]})")

    # Save padded video as temp file
    os.makedirs("temp_videos", exist_ok=True)
    temp_video_path = f"temp_padded_video_{task_name}_{traj_name}.mp4"
    out_tmp_video_path = os.path.join("temp_videos", temp_video_path)
    torchvision.io.write_video(out_tmp_video_path, padded_video.permute(0, 2, 3, 1), fps=30)
    
    # Prepare command
    cmd = [
        PATH_TO_BIN,
        "online",
        "--host", host,
        "--port", str(port),
        "-i", prompt_yaml,
        "--no-reasoning",
        "--videos", out_tmp_video_path,
        "--fps", "4",
        "--max-tokens", "64",
        "--temperature", "0.2"
    ]
    
    print(f"Running vLLM mockup command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print stdout/stderr
    print("=== STDOUT ===")
    print(process.stdout)
    print("=== STDERR ===")
    print(process.stderr)
    if process.returncode != 0:
        raise RuntimeError(f"Cosmos inference failed with code {process.returncode}")
    
    # Parse output for task description
    task_description = None
    lines = process.stdout.splitlines()
    
    for indx, line in enumerate(lines):
        if 'Assistant:' in line:
            task_description = lines[indx + 1].strip()
            if "1" in task_description:
                task_description = task_description.replace("1", "first")
            if "2" in task_description:
                task_description = task_description.replace("2", "second")
            if "3" in task_description:
                task_description = task_description.replace("3", "third")
            if "4" in task_description:
                task_description = task_description.replace("4", "fourth")
            if 'four' in task_description and 'fourth' not in task_description:
                task_description = task_description.replace('four', 'fourth')
            if 'three' in task_description:
                task_description = task_description.replace('three', 'third')
            if 'compartment box' in task_description:
                task_description = task_description.replace('compartment box', 'box')
            break
        
    task_description = task_description.replace('.', '')
    print(f"Generated Task Description: {task_description}")
    return task_description



# def run_vllm_mockup():
#     host = "gnode09"
#     port = 8000

#     # IMPORTANT: must be server-local path
#     video_path = "sample.mp4"   # relative to allowed-local-media-path
#     prompt_yaml = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Video-Captioning/cosmos-reason2/prompts/caption.yaml"

#     # Prepare command
#     # cosmos-reason2-inference online \
#     # --port 8000 \
#     # -i $PROMPT_PATH \
#     # --reasoning \
#     # --videos $VIDEO_PATH \
#     # --fps 4
#     #"--reasoning",
#     cmd = [
#         "cosmos-reason2-inference", "online",
#         "--host", host,
#         "--port", str(port),
#         "-i", prompt_yaml,
#         "--no-reasoning",
#         "--videos", video_path,
#         "--fps", "4",
#         "--max-tokens", "64",
#         "--temperature", "0.0"
#     ]
    
#     print(f"Running vLLM mockup command: {' '.join(cmd)}")
#     process = subprocess.run(cmd, capture_output=True, text=True)

#     # Print stdout/stderr
#     print("=== STDOUT ===")
#     print(process.stdout)
#     print("=== STDERR ===")
#     print(process.stderr)

#     if process.returncode != 0:
#         raise RuntimeError(f"Cosmos inference failed with code {process.returncode}")