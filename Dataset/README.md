## Finding the right dataset is crucial because MRI reconstruction models are highly sensitive to the physics of the data (coils, noise, and k-space trajectory).

## 1. Primary Choice: fastMRI Prostate (NYU/Meta)
This is the "gold standard" for open-source raw MRI data. It contains raw k-space measurements and multi-coil data, which is exactly what your A_forward and A_adjoint functions need.

What's inside: Over 300 patients, including T2-weighted and Diffusion-weighted images.

Access:
Go to the fastMRI Dataset page.
You need to fill out a quick Data Use Agreement (DUA). It’s usually approved automatically for research/educational purposes.
Once approved, they will provide a download link (often via an S3 bucket or a direct portal).

## 2. Alternative: Calgary-Campinas (Brain MRI)
If you have trouble getting the NYU data or want to test your model on a different anatomy (brain), this is an excellent second choice.

What's inside: Raw k-space for 12-coil and 32-coil brain scans.

Access:
Calgary-Campinas Public Dataset.
Adjustment: You would need to change your target_hw (usually 256x256) and set coils_max to 12 or 32 in the config.

## 3. Developer Tools (The "fastmri" Library)

Meta (Facebook AI Research) released a helper library to handle this data. I highly recommend installing it alongside your code:

### Download
```bash
pip install fastmri


# It includes standard utilities for cropping, masking, and SSIM math that is specifically tuned for MRI range values (addressing the normalization concern I mentioned earlier).
How to use the data once downloaded:
Once you have the .h5 files:
Organize them into multicoil_train/ and multicoil_val/ folders.

# Update the DATA_ROOT variable in your code:
python
DATA_ROOT = "/your/local/path/to/fastMRI_prostate"
Ensure your kspace_max and coils_max in the config match the data (Prostate is typically 300–600 in resolution and up to 15–30 coils).
Pro-tip: When starting, don't use the full dataset. Just download 2-3 .h5 files and try to run 1 epoch to ensure your GPU doesn't run out of memory (OOM), as raw k-space data is very heavy.
