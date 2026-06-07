# SAVE_ON_HUGGING

This guide explains how to load VLA checkpoints from the Hugging Face Hub in this benchmark.

## What the loader expects

The benchmark code checks whether `model_path` points to a Hugging Face repository ID. If it does, it loads the model directly from the Hub instead of a local checkpoint directory.

For OpenVLA-style models, the main loading path uses:

* `AutoModelForVision2Seq.from_pretrained(..., trust_remote_code=True)`
* `AutoProcessor.from_pretrained(..., trust_remote_code=True)`
* `hf_hub_download(...)` for auxiliary checkpoint files such as dataset statistics and action-head components

## Recommended Hugging Face repository layout

For a model repo on Hugging Face, keep these files available when the benchmark needs them:

* `config.json`
* model weights files created by `save_pretrained`
* `dataset_statistics.json`
* `action_head--*.pt` if the model uses an action head stored separately
* `proprio_projector--*.pt` if the model uses proprioception components stored separately

If the model was trained with custom OpenVLA classes, make sure `config.json` contains the proper `auto_map` entries so `transformers` can resolve the custom config and model classes.

## How loading works in the code

When `model_path` is a local directory, the loader:

1. Registers the OpenVLA classes with the Hugging Face auto classes.
2. Updates `config.json` to set the correct `auto_map` values.
3. Syncs local logic files when needed.

When `model_path` is a Hugging Face repo ID, the loader:

1. Skips local file syncing.
2. Loads the model and processor directly from the Hub.
3. Downloads extra files such as `dataset_statistics.json` with `hf_hub_download`.

## Example: loading a checkpoint from Hugging Face

Set `model_path` to the repository ID in your eval config:

```yaml
model_config:
  type: openvla
  model_path: moojink/openvla-7b-oft-finetuned-libero-spatial
  use_l1_regression: true
  use_diffusion: false
  use_film: false
  num_images_in_input: 2
  use_proprio: true
```

Then the benchmark can load it with the standard Hugging Face API:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

repo_id = "moojink/openvla-7b-oft-finetuned-libero-spatial"

model = AutoModelForVision2Seq.from_pretrained(
    repo_id,
    torch_dtype="bfloat16",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
```

## Example: uploading a checkpoint to Hugging Face

After training locally, save the checkpoint in a directory that contains the full model state and config files, then push that directory to the Hub.

```python
from huggingface_hub import login

login()

# After calling save_pretrained on the model and processor,
# push the checkpoint directory to your repository.
```

If your model uses extra artifacts outside the standard `save_pretrained` output, upload them too:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="dataset_statistics.json",
    path_in_repo="dataset_statistics.json",
    repo_id="your-namespace/your-vla-repo",
)
```

## Notes for this benchmark

* `model_path` can be either a local directory or a Hugging Face repo ID.
* If the model is on the Hub, the benchmark assumes the published files are the source of truth.
* If you change the local modeling code and want those changes reflected, use a local checkpoint path instead of a Hub repo ID.
* For OpenVLA checkpoints that store action heads or proprio projectors separately, make sure those filenames are present in the repository and match what the loader expects.

## Your checkpoint: exact upload and load commands

Use this local checkpoint directory:

```bash
CKPT_DIR="/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/open_vla/openvla-7b+libero_object_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio--135000_chkpt"
```

This folder already includes the required files for this benchmark, including:

* `config.json`
* `model-*.safetensors` + `model.safetensors.index.json`
* `dataset_statistics.json`
* `action_head--135000_checkpoint.pt`
* `proprio_projector--135000_checkpoint.pt`

### 1) Create a Hugging Face repo and upload everything

Pick a repository name, for example `YOUR_USERNAME/openvla-libero-object-135000`.

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login
huggingface-cli repo create YOUR_USERNAME/openvla-libero-object-135000 --type model
huggingface-cli upload-large-folder YOUR_USERNAME/openvla-libero-object-135000 "$CKPT_DIR" --repo-type model
```

### 2) Point benchmark config to the Hub repo

```yaml
model_config:
  type: openvla
  model_path: YOUR_USERNAME/openvla-libero-object-135000
  use_l1_regression: true
  use_diffusion: false
  use_film: false
  num_images_in_input: 2
  use_proprio: true
```

### 3) Quick sanity check after upload

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from huggingface_hub import hf_hub_download

repo_id = "YOUR_USERNAME/openvla-libero-object-135000"

model = AutoModelForVision2Seq.from_pretrained(
    repo_id,
    trust_remote_code=True,
    torch_dtype="bfloat16",
)
processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)

hf_hub_download(repo_id=repo_id, filename="dataset_statistics.json")
hf_hub_download(repo_id=repo_id, filename="action_head--135000_checkpoint.pt")
hf_hub_download(repo_id=repo_id, filename="proprio_projector--135000_checkpoint.pt")
```

If all three `hf_hub_download` calls succeed, your model is correctly published for this benchmark.
