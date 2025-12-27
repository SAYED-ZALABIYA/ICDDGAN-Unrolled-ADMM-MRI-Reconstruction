# ICDDGAN-Unrolled-ADMM-MRI-Reconstruction
Physics-guided accelerated MRI reconstruction using ICDDGAN initialization and unrolled ADMM refinement.


# ICDDGAN + Unrolled ADMM for Accelerated MRI Reconstruction

This repository implements a physics-guided MRI reconstruction framework that combines:

- **ICDDGAN-based diffusion initializer**
- **Unrolled ADMM refinement**
- **Exact k-space data consistency**
- **Adversarial and diffusion-aware training**

The method reconstructs fully-sampled MRI images from undersampled multi-coil k-space measurements.

---

## Pipeline Overview
1. Undersampled k-space input (fastMRI)
2. Physics-guided ICDDGAN initialization
3. Unrolled ADMM refinement with learned proximal operators
4. Final physically consistent MRI reconstruction

---

## Dataset
The model is trained and evaluated on **fastMRI prostate (multi-coil)**.

> Due to licensing, the dataset is **not included** in this repository.

### Download
```bash
bash scripts/download_fastmri.sh

---

## License
All rights reserved.  
This code is proprietary and may not be used, modified, or distributed without
explicit permission from the author.
