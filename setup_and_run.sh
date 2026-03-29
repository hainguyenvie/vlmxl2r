#!/bin/bash
# Setup and run CLIP open-set recognition on ImageNet
# Usage: bash setup_and_run.sh

set -e

echo "=== [1/4] Setup Python venv ==="
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install torch torchvision openai-clip numpy scipy scikit-learn matplotlib tqdm \
    pyarrow huggingface_hub

echo "=== [2/4] Login HuggingFace (need token for ILSVRC/imagenet-1k) ==="
.venv/bin/python -c "from huggingface_hub import login; login()"

echo "=== [3/4] Download ImageNet val (14 parquet files ~6GB) ==="
mkdir -p /tmp/imagenet_hf/data
.venv/bin/python - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os

repo = "ILSVRC/imagenet-1k"
out = "/tmp/imagenet_hf/data"
os.makedirs(out, exist_ok=True)

for i in range(14):
    fname = f"data/validation-{str(i).zfill(5)}-of-00014.parquet"
    dst = os.path.join(out, os.path.basename(fname))
    if os.path.exists(dst):
        print(f"{i+1}/14 skip {os.path.basename(fname)}", flush=True)
        continue
    print(f"{i+1}/14 downloading {os.path.basename(fname)}...", flush=True)
    hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset",
                    local_dir=out, local_dir_use_symlinks=False)
    print(f"{i+1}/14 done", flush=True)

print("Download complete!")
PYEOF

echo "=== [4/4] Run CLIP inference ==="
.venv/bin/python classification/run_clip_imagenet.py

echo "=== Evaluate open-set performance ==="
.venv/bin/python classification/scripts/evaluate.py \
    --file pred_files/clip/imagenet/standard_cosine_clip.npy \
    --auroc

echo "=== DONE ==="
