"""
Reproduce CLIP results on ImageNet (Table 1 in paper).
Reads directly from HuggingFace parquet files — no need to extract 50k images.
"""
import os
import sys
import glob
import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
import clip

sys.path.insert(0, os.path.dirname(__file__))
from data.imagenet_labels import imagenet_classes as class_list

PARQUET_DIR = '/tmp/imagenet_hf/data'
PRED_DIR = 'pred_files/clip/imagenet'
os.makedirs(PRED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class ImageNetParquetDataset(Dataset):
    def __init__(self, parquet_dir, transform=None):
        self.transform = transform
        self.records = []
        files = sorted(glob.glob(f'{parquet_dir}/validation-*.parquet'))
        print(f"Loading {len(files)} parquet files...")
        for f in tqdm(files):
            table = pq.read_table(f, columns=['image', 'label'])
            d = table.to_pydict()
            for img_dict, label in zip(d['image'], d['label']):
                self.records.append((img_dict['bytes'], int(label)))
        print(f"Total images: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_bytes, label = self.records[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# Load CLIP
print("Loading CLIP ViT-B/32...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Load dataset
dataset = ImageNetParquetDataset(PARQUET_DIR, transform=clip_preprocess)

# Encode text queries
print(f"Encoding {len(class_list)} class text queries...")
token_text = [f'a photo of a {x}.' for x in class_list]
with torch.no_grad():
    clip_text = clip.tokenize(token_text).to(device)

# ── Standard test ────────────────────────────────────────────────────────────
gt_path = f'{PRED_DIR}/standard_gt_clip.npy'
cos_path = f'{PRED_DIR}/standard_cosine_clip.npy'

print("\nRunning standard inference...")
loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

all_cosine = []
gt_labels = []

for images, targets in tqdm(loader):
    images = images.to(device)
    with torch.no_grad():
        logits, _ = clip_model(images, clip_text)
    all_cosine += logits.cpu().tolist()
    gt_labels += targets.tolist()

gt_labels = np.array(gt_labels)
all_cosine = np.array(all_cosine)

np.save(gt_path, gt_labels)
np.save(cos_path, all_cosine)
print(f"Saved: {cos_path}  shape={all_cosine.shape}")
print(f"Saved: {gt_path}")

preds = np.argmax(all_cosine, axis=1)
acc = (preds == gt_labels).mean() * 100
print(f"\nClosed-set accuracy (top-1): {acc:.1f}%")
print("\nNow run:")
print(f"  python scripts/evaluate.py --file {cos_path} --auroc")
