import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, MaskBlock,
    ColSplitter, PILImage, PILMask, Resize, DataLoaders
)

# Top-level helper functions for fast.ai DataBlock to support Windows multiprocessing pickling

def _get_img_path(row):
    return os.path.join(row['data_dir'], row['image_path'])

def _get_cls_label(row):
    return row['label']

def _get_bbox(row):
    # fast.ai BBox format: [[xmin, ymin, xmax, ymax]]
    return [[row['xmin'], row['ymin'], row['xmax'], row['ymax']]]

def _get_bbox_lbl(row):
    return [row['label']]

def _get_mask_path(row):
    return os.path.join(row['data_dir'], row['mask_path'])


def get_data_df(data_dir):
    """Loads the dataset CSV metadata and adds data_dir column."""
    csv_path = os.path.join(data_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}. Run prepare_dataset.py first.")
    df = pd.read_csv(csv_path)
    df['data_dir'] = data_dir
    df['is_valid'] = df['split'].isin(['val', 'test'])
    return df

# ---------------------------------------------------------
# 1. Classification DataLoader (Stage 1: f_cls)
# ---------------------------------------------------------
def get_classification_dataloaders(data_dir, bs=16, img_size=224, num_workers=0):
    """
    Returns fast.ai DataLoaders for disease/anomaly classification (e.g. polyp vs normal).
    """
    df = get_data_df(data_dir)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=_get_img_path,
        get_y=_get_cls_label,
        splitter=ColSplitter('is_valid'),
        item_tfms=Resize(img_size)
    )
    
    dls = dblock.dataloaders(df, bs=bs, num_workers=num_workers)
    return dls

# ---------------------------------------------------------
# 2. Object Detection BBox DataLoader (Stage 2: f_det)
#    Uses a plain PyTorch Dataset to avoid fast.ai's
#    clip_remove_empty() version incompatibility.
# ---------------------------------------------------------
class _BBoxDataset(Dataset):
    """Returns (image_tensor, bbox_tensor) where bbox is [xmin,ymin,xmax,ymax] normalised 0-1."""
    def __init__(self, df, img_size=224):
        self.df       = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = PILImage.create(row['data_dir'] + os.sep + row['image_path'])
        img = img.resize((self.img_size, self.img_size))
        img_t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Normalise bbox to [0, 1]
        xmin = float(row.get('xmin', 0)) / self.img_size
        ymin = float(row.get('ymin', 0)) / self.img_size
        xmax = float(row.get('xmax', self.img_size)) / self.img_size
        ymax = float(row.get('ymax', self.img_size)) / self.img_size
        bbox_t = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        return img_t, bbox_t


def get_detection_dataloaders(data_dir, bs=16, img_size=224, num_workers=0):
    """
    Returns fast.ai-compatible DataLoaders for bbox regression (4 normalised floats).
    Bypasses BBoxBlock / clip_remove_empty entirely.
    """
    df = get_data_df(data_dir)
    df_pos = df[df['has_polyp'] == 1].copy()

    train_ds = _BBoxDataset(df_pos[~df_pos['is_valid']], img_size)
    valid_ds = _BBoxDataset(df_pos[df_pos['is_valid']],  img_size)

    from torch.utils.data import DataLoader as _DL
    train_dl = _DL(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers)
    valid_dl = _DL(valid_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    return DataLoaders(train_dl, valid_dl)

# ---------------------------------------------------------
# 3. Semantic Segmentation Mask DataLoader (Stage 3: f_seg)
#    Uses a plain PyTorch Dataset so we can explicitly remap
#    mask pixel values 0/255 → class indices 0/1, avoiding the
#    "Target 255 is out of bounds" CrossEntropyLoss error.
# ---------------------------------------------------------
class _SegDataset(Dataset):
    """Returns (image_tensor [C,H,W] float32, mask_tensor [H,W] long) with mask in {0,1}."""
    def __init__(self, df, img_size=224):
        self.df       = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image
        img = PILImage.create(row['data_dir'] + os.sep + row['image_path'])
        img = img.resize((self.img_size, self.img_size))
        img_t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Mask — remap 0/255 → 0/1 class indices
        mask_rel = str(row.get('mask_path', ''))
        mask_full = row['data_dir'] + os.sep + mask_rel if mask_rel else None
        if mask_full and os.path.exists(mask_full):
            mask_img = PILMask.create(mask_full).resize((self.img_size, self.img_size))
            mask_np  = np.array(mask_img)
        else:
            # Synthetic fallback: full-positive or full-negative mask
            has_polyp = int(row.get('has_polyp', 1))
            mask_np = np.full((self.img_size, self.img_size),
                              255 if has_polyp else 0, dtype=np.uint8)

        # Threshold: any value > 128 → class 1 (polyp), else 0 (background)
        mask_binary = (mask_np > 128).astype(np.int64)
        mask_t = torch.tensor(mask_binary, dtype=torch.long)
        return img_t, mask_t


def get_segmentation_dataloaders(data_dir, bs=16, img_size=224, num_workers=0):
    """
    Returns fast.ai-compatible DataLoaders for pixel-level binary segmentation.
    Bypasses MaskBlock to avoid the 0/255 → class-index bug.
    """
    df = get_data_df(data_dir)

    train_ds = _SegDataset(df[~df['is_valid']], img_size)
    valid_ds = _SegDataset(df[df['is_valid']],  img_size)

    from torch.utils.data import DataLoader as _DL
    train_dl = _DL(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers)
    valid_dl = _DL(valid_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    return DataLoaders(train_dl, valid_dl)
