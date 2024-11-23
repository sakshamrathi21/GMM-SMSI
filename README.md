# Accurate Image Segmentation using Gaussian Mixture Model with Saliency Map

## Introduction

This project implements a Gaussian Mixture Model (GMM) for image segmentation enhanced with a **Saliency Map** to incorporate spatial information. Unlike conventional GMMs, this approach leverages saliency to capture visually significant regions, yielding improved segmentation.

---

## Features

1. **Saliency Map Calculation**: 
   - Extracts gradient-rich regions to emphasize outlines and salient features.
   - Customizable to exaggerate boundaries for detailed segmentation.

2. **Improved Likelihood Function**:
   - Integrates spatial neighborhood information via saliency-weighted conditional probabilities.
   - Supports parameter optimization using the Expectation-Maximization (EM) algorithm.

3. **Experimental Results**:
   - Enhanced segmentation for images with tiny objects in large backgrounds.
   - Improved preservation of minute details in complex images.
   - Effective segmentation of medical and natural images.

---

## Workflow

### Step 1: Saliency Map Extraction
1. Convert the image into the spectral domain using FFT to extract amplitude and phase spectra.
2. Compute the log spectrum and residual values.
3. Generate the saliency map.

### Step 2: GMM with Saliency-Weighted Spatial Information
1. Initialize parameters using k-means.
2. Apply the EM algorithm with the saliency map to refine the parameters.
3. Classify image pixels based on posterior probabilities.

---

## Dataset

The experiments were conducted on images from the **Berkeley Image Dataset** and select medical images. Results demonstrated improved segmentation compared to conventional GMM and GMM with MRF.

---

## Results

### Key Experiments
- **Tiny Object in Large Background**: Enhanced outline detection and feature preservation.
- **Minute Details**: Better handling of subtle features like eyebrows, clothing details, and overlapping regions.
- **Medical Images**: Clear segmentation of brain regions and other anatomical details.

---

## Conclusion

The project successfully validates that incorporating saliency maps into GMM enhances segmentation performance. While results closely match those in the reference research paper, certain discrepancies highlight scope for future improvement.

---

## References

1. Research Paper: [Springer Article](https://link.springer.com/article/10.1007/s10044-017-0672-1)
2. Code Reference: [GitHub Repository](https://github.com/SrikanthAmudala/GaussainDistribution)
