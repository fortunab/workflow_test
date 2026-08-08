import os
import pandas as pd
import numpy as np
import torch
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, MaskBlock, BBoxBlock, BBoxLblBlock,
    ColSplitter, PILImage, PILMask, TensorBBox, Resize, aug_transforms
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
# 2. Object Detection Bounding Box DataLoader (Stage 2: f_det)
# ---------------------------------------------------------
def get_detection_dataloaders(data_dir, bs=16, img_size=224, num_workers=0):
    """
    Returns fast.ai DataLoaders for bounding box localization [ymin, xmin, ymax, xmax].
    """
    df = get_data_df(data_dir)
    # Filter only positive cases for bbox training
    df_det = df[df['has_polyp'] == 1].copy().reset_index(drop=True)

    dblock = DataBlock(
        blocks=(ImageBlock, BBoxBlock),
        get_x=_get_img_path,
        get_y=_get_bbox,
        splitter=ColSplitter('is_valid'),
        item_tfms=Resize(img_size)
    )

    dls = dblock.dataloaders(df_det, bs=bs, num_workers=num_workers)
    return dls

# ---------------------------------------------------------
# 3. Semantic Segmentation Mask DataLoader (Stage 3: f_seg)
# ---------------------------------------------------------
def get_segmentation_dataloaders(data_dir, bs=16, img_size=224, num_workers=0):
    """
    Returns fast.ai DataLoaders for pixel-level binary mask segmentation (U-Net / SAM style).
    """
    df = get_data_df(data_dir)

    # Codes: 0 = background, 1 = polyp
    codes = np.array(["background", "polyp"], dtype=str)

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=codes)),
        get_x=_get_img_path,
        get_y=_get_mask_path,
        splitter=ColSplitter('is_valid'),
        item_tfms=Resize(img_size)
    )

    dls = dblock.dataloaders(df, bs=bs, num_workers=num_workers)
    return dls
