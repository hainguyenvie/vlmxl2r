"""
Reproduce CLIP results on GTSRB (Table 1 in paper).
Downloads dataset automatically, runs inference, saves pred files.
"""
import os
import sys
import numpy as np
import torch
import torchvision
import clip
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data.gtsrb_labels import gtsrb_classes as class_list

DATASET = 'gtsrb'
DATA_DIR = 'data/GTSRB'
PRED_DIR = f'pred_files/clip/{DATASET}'
os.makedirs(PRED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP
print("Loading CLIP ViT-B/32...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Load GTSRB test set (auto-download)
print("Loading GTSRB test set...")
all_images = torchvision.datasets.GTSRB(
    DATA_DIR, split='test', transform=clip_preprocess, download=True
)
print(f"  {len(all_images)} test images, {len(class_list)} classes")

# Encode text queries
token_text = [f'a photo of a {x}.' for x in class_list]
with torch.no_grad():
    clip_text = clip.tokenize(token_text).to(device)
    text_features = clip_model.encode_text(clip_text)

# ── Standard test (no negative embeddings) ──────────────────────────────────
gt_path = f'{PRED_DIR}/standard_gt_clip.npy'
cos_path = f'{PRED_DIR}/standard_cosine_clip.npy'

print("\nRunning standard inference...")
loader = torch.utils.data.DataLoader(all_images, batch_size=64, num_workers=4)

all_cosine = []
gt_labels = []

for images, target in tqdm(loader):
    images = images.to(device)
    with torch.no_grad():
        logits, _ = clip_model(images, clip_text)
    all_cosine += logits.cpu().tolist()
    gt_labels += target.tolist()

gt_labels = np.array(gt_labels)
all_cosine = np.array(all_cosine)

np.save(gt_path, gt_labels)
np.save(cos_path, all_cosine)
print(f"Saved: {cos_path}")
print(f"Saved: {gt_path}")
print(f"Logit matrix shape: {all_cosine.shape}")

# Quick sanity check
preds = np.argmax(all_cosine, axis=1)
acc = (preds == gt_labels).mean() * 100
print(f"\nClosed-set accuracy (top-1): {acc:.1f}%")
print("\nDone. Now run:")
print(f"  python scripts/evaluate.py --file {cos_path}")
