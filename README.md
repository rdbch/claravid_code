# ClaraVid
[![Project Page](https://img.shields.io/badge/Project%20Page-ClaraVid-blue?style=flat&logo=github)](https://rdbch.github.io/claravid/) 
[![Hugging Face](https://img.shields.io/badge/HuggingFace-ClaraVid-FFD21F?style=flat&logo=huggingface)](https://huggingface.co/datasets/radubeche/claravid)
[![arXiv Preprint](https://img.shields.io/badge/arXiv-2503.17856-b31b1b?style=flat&logo=arXiv&logoColor=white)](https://arxiv.org/abs/2503.17856)

Official repo for: *ClaraVid: A Holistic Scene Reconstruction Benchmark From Aerial Perspective With Delentropy-Based Complexity Profiling* - Accepted ICCV 2025
 

**ClaraVid** is a synthetic dataset built for semantic and geometric neural reconstruction from low altitude UAV/aerial imagery. 
It contains *16,917 multimodal frames* collected across 8 UAV missions over 5 diverse environments: urban, urban high, rural, highway, and nature.

**Delentropic Scene Profile (DSP)** is a metric for estimating scene complexity from images, 
tailored for structured UAV mapping scenarios. DSP helps predict reconstruction difficulty.

## Channel Log / TODOs
- [x] Release dataset SDK
- [ ] Release DSP code (closer to conference)
- [ ] Release pip package
- [ ] Release dataset splits
- [ ] Set-up script
- [x] All data uploaded

# Installation
Easiest way to install this package is to use pip:
```bash
pip install claravid
```

Alternatively you can clone the repository and install it manually:
```bash
git clone https://github.com/rdbch/claravid_code.git
pip install -e . 

```

# Bibtex
If you found this work useful, please cite us as:

```
@misc{beche2025claravid,
  title={ClaraVid: A Holistic Scene Reconstruction Benchmark From Aerial Perspective With Delentropy-Based Complexity Profiling},
  author={Beche, Radu and Nedevschi, Sergiu},
  journal={arXiv preprint arXiv:2503.17856},
  year={2025}
}
```
