# Brain MRI Tissue Segmentation (CS50P final project)

Segmentation of brain tissue in DICOM T1 axial slices MRI.
https://www.youtube.com/watch?v=CsjcGFDXO14.

A compact Python pipeline that demonstrates basic MRI preprocessing and tissue classification on single T1-weighted axial slices (2D).  
The goal is educational: to learn how DICOM images are read, how skull-stripping can be approached for T1 images, and how a simple Gaussian-Mixture model can separate CSF / gray matter / white matter.

## Features
- Load a single DICOM slice and robustly normalize intensities.
- Custom T1-optimized skull-stripping (Otsu-based + reconstruction).
- 3-component Gaussian Mixture Model (GMM) tissue segmentation (CSF / GM / WM).
- Quick QC visualizations: intermediate masks, final overlay, and GMM histograms, that can be turned off.
- Small, dependency-friendly: standard Python scientific stack.
- Outputs an image of segmented tissues and % CSF/GM/WM of total tissue in terminal

## Requirements
Tested with Python 3.10 and the packages below. Install with:
```bash
python -m pip install -r requirements.txt
