# tests/test_pipeline.py
"""
Pytest smoke/integration tests for the single-slice pipeline.
Uses synthetic images so no DICOM I/O is required.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Try to import the pipeline module from src/ or project root
MODULE_NAME = "simple_tissue_segmentation"
if os.path.isdir("src") and os.path.exists(os.path.join("src", MODULE_NAME + ".py")):
    sys.path.insert(0, os.path.abspath("src"))
else:
    # assume module in repo root
    sys.path.insert(0, os.path.abspath("."))

try:
    mod = __import__(MODULE_NAME)
except Exception as e:
    raise ImportError(f"Could not import {MODULE_NAME}. Make sure the file exists in src/ or project root.") from e

# Bring functions into local namespace
skull_fn = getattr(mod, "skull_strip_t1_reconstruction", None)
gmm_fn = getattr(mod, "gmm_segment", None)
load_dicom = getattr(mod, "load_dicom", None)

if skull_fn is None or gmm_fn is None:
    raise ImportError("Required functions 'skull_strip_t1_reconstruction' or 'gmm_segment' not found in module.")

# ---------- helper: synthetic brain-like image ----------
def synthetic_brain(shape=(128,128), radius=40, csf_width=6, seed=123):
    rng = np.random.RandomState(seed)
    cy, cx = shape[0]//2, shape[1]//2
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    img = np.zeros(shape, dtype=float)
    img[dist <= radius] = 0.6                    # brain interior
    img[dist < (radius*0.25)] = 0.15            # ventricles darker
    # bright rim for skull/CSF contrast (thin)
    rim_mask = (dist > radius - csf_width) & (dist <= radius + csf_width)
    img[rim_mask] = 0.95
    # add small noise
    img += rng.normal(0, 0.02, size=shape)
    img = np.clip(img, 0.0, 1.0)
    return img

# ---------- tests ----------
def test_skull_strip_returns_mask_and_nonempty():
    img = synthetic_brain()
    # call with debug False to avoid plotting during tests
    mask, dbg = skull_fn(img, debug=False, gaussian_blur=(5,5), closing_disk=6, seed_fraction=0.45)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == img.shape
    assert mask.dtype == bool or mask.dtype == np.bool_ or mask.dtype == np.uint8
    # Must contain some True voxels
    assert mask.sum() > 100

def test_gmm_segment_labels_in_range():
    img = synthetic_brain()
    mask, _ = skull_fn(img, debug=False)
    labels, gm = gmm_fn(img, mask, n_components=3)
    assert labels.shape == img.shape
    # labels should be 0..3 (0 background, 1..3 tissues)
    unique = np.unique(labels)
    assert unique.min() >= 0
    assert unique.max() <= 3

def test_gmm_consistency_across_runs():
    # GMM should run deterministically with same random_state inside function
    img = synthetic_brain(seed=42)
    mask, _ = skull_fn(img, debug=False)
    labels1, gm1 = gmm_fn(img, mask, n_components=3)
    labels2, gm2 = gmm_fn(img, mask, n_components=3)
    # labels arrays should be identical (same seed in pipeline's GaussianMixture)
    assert np.array_equal(labels1, labels2)

def test_pipeline_handles_empty_mask_gracefully():
    # if mask all False, gmm should raise or be handled; ensure code raises ValueError as implemented
    img = synthetic_brain()
    empty_mask = np.zeros_like(img, dtype=bool)
    try:
        _ = gmm_fn(img, empty_mask, n_components=3)
        raised = False
    except ValueError:
        raised = True
    assert raised, "gmm_segment should raise ValueError on empty mask"

def test_compute_percentages_sum_leq_total():
    # ensure tissue counts do not exceed mask voxels
    img = synthetic_brain()
    mask, _ = skull_fn(img, debug=False)
    labels, gm = gmm_fn(img, mask, n_components=3)
    total = int(np.sum(mask))
    csf = int((labels==1).sum())
    gmcount = int((labels==2).sum())
    wm = int((labels==3).sum())
    assert csf + gmcount + wm <= total
