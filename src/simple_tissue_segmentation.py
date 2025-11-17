"""
simple_brain_segmentation_t1_fixed.py

Robust T1 single-slice DICOM brain segmentation.
Primary skull-strip approach:
 - Otsu -> largest connected component -> fill holes
 - Compute distance transform -> use an inner seed (fraction of max distance)
 - Morphological reconstruction (dilation) of seed under filled mask
This keeps ventricles and reduces cortical erosion compared to naive erosion.

Outputs:
 - Debug panels showing intermediate masks (if DEBUG=True)
 - GMM segmentation of CSF/GM/WM, percentages, and plots.

Dependencies:
 pip install numpy matplotlib pydicom opencv-python scikit-learn scikit-image scipy
"""

import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.morphology import reconstruction


# For demo, default to example_data/sample_anonymized.dcm
DICOM_PATH = os.environ.get("DICOM_PATH", os.path.join(os.path.dirname(__file__), "..", "example_data", "anonymous12.dcm"))
DEBUG = True


def main():
    if not os.path.exists(DICOM_PATH):
        print("[ERROR] DICOM not found:", DICOM_PATH)
        return
    try:
        img = load_dicom(DICOM_PATH)
        mask, dbg = skull_strip_t1_reconstruction(img, debug=DEBUG,
                                                  gaussian_blur=(5,5),
                                                  closing_disk=10,
                                                  seed_fraction=0.4)
        if mask.sum() == 0:
            print("[FATAL] Final mask is empty after skull strip. Aborting.")
            return
        labels, gm = gmm_segment(img, mask)
        compute_tissue_percentages(labels, mask)
        plot_segmentation(img, mask, labels)
        plot_gmm_histogram(img, mask, gm)
        print("[DONE]")
    except Exception as e:
        print("[FATAL] pipeline failed:", e)
        traceback.print_exc()

# -------------- Utilities --------------
def load_dicom(path):
    """Load single-slice DICOM, robust percentile clip and normalize to [0,1]."""
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # clip extremes then scale
        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip(arr, p1, p99)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        if DEBUG:
            print(f"[INFO] Loaded {path} shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f}")
        return arr
    except Exception as e:
        print("[ERROR] load_dicom failed:", e)
        traceback.print_exc()
        raise

# -------------- Skull stripping (distance-seed + reconstruction) --------------
def skull_strip_t1_reconstruction(img, debug=DEBUG,
                                  gaussian_blur=(5,5),
                                  closing_disk=6,
                                  seed_fraction=0.45):
    """
    Robust T1 skull stripping that preserves ventricles and avoids over-eroding cortex.

    Steps:
      - Blur -> Otsu threshold
      - Keep largest connected component -> fill holes (filled_mask)
      - Distance transform on filled_mask; seed = distance > (seed_fraction * max_dist)
      - Use morphological reconstruction (dilation) of seed under filled_mask -> final_mask
      - If seed is empty, fall back to a light erosion-based seed
    """
    try:
        # prepare image as uint8 for some ops
        img_u8 = (img * 255).astype(np.uint8)

        # 1) blur
        blur = cv2.GaussianBlur(img_u8, gaussian_blur, 0)

        # 2) Otsu threshold (on blurred image)
        _, otsu_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_bool = (otsu_bin > 0)

        # 3) Keep the largest connected component of otsu (remove background islands)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu_bin.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            largest_cc = otsu_bool.astype(np.uint8)
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = int(np.argmax(areas)) + 1
            largest_cc = (labels == largest_idx).astype(np.uint8)

        # 4) Fill holes to produce a solid brain+ventricles interior
        filled_mask = binary_fill_holes(largest_cc > 0).astype(np.uint8)  # 0/1

        # 5) Optionally small closing to smooth tiny gaps (keep small disk)
        if closing_disk > 0:
            # using OpenCV closing for speed
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*closing_disk+1, 2*closing_disk+1))
            filled_mask = cv2.morphologyEx((filled_mask*255).astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
            filled_mask = filled_mask.astype(np.uint8)

        # 6) Distance transform on filled_mask (distance to background)
        dist = distance_transform_edt(filled_mask > 0)
        maxd = dist.max() if dist.size else 0.0

        # 7) Seed definition: take voxels with distance > seed_fraction*maxd
        seed_thr = seed_fraction * (maxd if maxd>0 else 1.0)
        seed = (dist > seed_thr).astype(np.uint8)

        # If seed is empty (rare), fallback to light erosion of filled_mask
        if seed.sum() == 0:
            e_k = 3
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*e_k+1, 2*e_k+1))
            eroded = cv2.erode((filled_mask*255).astype(np.uint8), k, iterations=1) > 0
            seed = eroded.astype(np.uint8)
            if DEBUG:
                print("[WARN] seed was empty; used erosion fallback seed.")

        # 8) Reconstruction (dilate seed under filled_mask) -> restores interior without rim
        # recon expects arrays where seed <= mask
        seed_uint8 = (seed > 0).astype(np.uint8)
        mask_uint8 = (filled_mask > 0).astype(np.uint8)
        # reconstruction from skimage: dilation until it reaches mask
        rec = reconstruction(seed_uint8, mask_uint8, method='dilation')
        final_mask = (rec > 0).astype(np.uint8)

        # 9) Safety: if final mask is empty, fallback to filled_mask
        if final_mask.sum() == 0:
            if DEBUG:
                print("[WARN] final_mask empty after reconstruction; falling back to filled_mask")
            final_mask = filled_mask.copy()

        # 10) Convert to boolean mask
        final_bool = final_mask.astype(bool)

        # Debug images
        dbg = {
            'img_u8': img_u8,
            'blur': blur,
            'otsu_bin': otsu_bin,
            'largest_cc': (largest_cc*255).astype(np.uint8),
            'filled_mask': (filled_mask*255).astype(np.uint8),
            'dist': dist,
            'seed': (seed*255).astype(np.uint8),
            'reconstructed': (rec.astype(np.uint8)*255),
            'final': (final_bool.astype(np.uint8)*255)
        }

        if debug:
            _debug_show_reconstruction_steps(dbg)

        return final_bool, dbg

    except Exception as e:
        print("[ERROR] skull_strip_t1_reconstruction failed:", e)
        traceback.print_exc()
        raise

def _debug_show_reconstruction_steps(dbg):
    """Show the debug images for skull stripping steps."""
    try:
        keys = ['img_u8','blur','otsu_bin','largest_cc','filled_mask','seed','reconstructed','final']
        n = len(keys)
        cols = 4
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(4*cols, 3*rows))
        for i,k in enumerate(keys,1):
            plt.subplot(rows, cols, i)
            im = dbg[k]
            # dist is float -> display with colormap if present
            if k == 'dist':
                plt.imshow(np.rot90(im), cmap='viridis')
            else:
                plt.imshow(np.rot90(im), cmap='gray')
            plt.title(k)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("[WARN] debug plot failed:", e)
        traceback.print_exc()

# -------------- GMM segmentation --------------
def gmm_segment(img, mask, n_components=3):
    """
    GaussianMixture segmentation on masked voxels.
    Returns remapped labels: 0 background, 1 CSF, 2 GM, 3 WM
    """
    try:
        vox = img[mask].reshape(-1,1)
        if vox.size == 0:
            raise ValueError("No brain voxels for GMM.")
        z = (vox - vox.mean()) / (vox.std() + 1e-8)
        gm = GaussianMixture(n_components=n_components, random_state=0)
        gm.fit(z)
        assigned = gm.predict(z)
        labels = np.zeros(mask.shape, dtype=np.uint8)
        flat_idx = np.where(mask.ravel())[0]
        lf = labels.ravel()
        lf[flat_idx] = assigned + 1
        labels = lf.reshape(mask.shape)
        # compute means (original intensity space)
        comp_means = []
        for k in range(n_components):
            vals = vox[assigned == k]
            comp_means.append(vals.mean() if vals.size>0 else np.inf)
        order = np.argsort(np.array(comp_means))
        mapping = np.zeros(n_components, dtype=int)
        mapping[order[0]] = 1
        mapping[order[1]] = 2
        mapping[order[2]] = 3
        remapped = np.zeros_like(labels)
        for comp_idx in range(n_components):
            remapped[labels == (comp_idx+1)] = mapping[comp_idx]
        return remapped, gm
    except Exception as e:
        print("[ERROR] gmm_segment failed:", e)
        traceback.print_exc()
        raise

# -------------- Utilities for outputs --------------
def compute_tissue_percentages(labels, mask):
    try:
        total = int(np.sum(mask))
        if total == 0:
            print("[WARN] empty mask -> cannot compute percentages")
            return
        print("\n[Tissue composition] (percent of brain mask voxels):")
        for i,name in enumerate(["CSF","Gray Matter","White Matter"], start=1):
            cnt = int((labels==i).sum())
            perc = 100.0 * cnt / total
            print(f" - {name:12s}: {perc:6.2f}% ({cnt} voxels)")
    except Exception as e:
        print("[ERROR] compute_tissue_percentages failed:", e)
        traceback.print_exc()

def plot_segmentation(img, mask, labels):
    try:
        disp_img = np.rot90((img*255).astype(np.uint8))
        disp_mask = np.rot90((mask.astype(np.uint8))*255)
        disp_labels = np.rot90(labels)
        fig,axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(disp_img,cmap='gray'); axs[0].set_title("Original")
        axs[1].imshow(disp_mask,cmap='gray'); axs[1].set_title("Brain Mask")
        axs[2].imshow(disp_img,cmap='gray')
        axs[2].imshow(np.ma.masked_where(disp_labels==0, disp_labels), alpha=0.6, cmap='jet')
        axs[2].set_title("Segmentation (1=CSF,2=GM,3=WM)")
        for a in axs: a.axis('off')
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("[ERROR] plot_segmentation failed:", e)
        traceback.print_exc()

def plot_gmm_histogram(img, mask, gm):
    try:
        vox = img[mask].reshape(-1,1)
        z = (vox - vox.mean()) / (vox.std() + 1e-8)
        x = np.linspace(z.min(), z.max(), 400).reshape(-1,1)
        logp = gm.score_samples(x)
        resp = gm.predict_proba(x)
        pdf = np.exp(logp)
        pdf_ind = resp * pdf[:, np.newaxis]
        plt.figure(figsize=(7,4))
        plt.hist(z, bins=60, density=True, alpha=0.5, color='gray', label='data (z)')
        plt.plot(x, pdf, '-k', label='mixture')
        for i, (c,name) in enumerate(zip(['r','g','b'], ['CSF','GM','WM'])):
            plt.plot(x, pdf_ind[:,i], color=c, lw=2, label=name)
        plt.legend(); plt.title("GMM components (z-scored)"); plt.xlabel("z"); plt.ylabel("density")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("[ERROR] plot_gmm_histogram failed:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
