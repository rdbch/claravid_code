# ClaraVid

[![Project Page](https://img.shields.io/badge/Project%20Page-ClaraVid-blue?style=flat&logo=github)](https://rdbch.github.io/claravid/) 
[![Hugging Face](https://img.shields.io/badge/HuggingFace-ClaraVid-FFD21F?style=flat&logo=huggingface)](https://huggingface.co/datasets/radubeche/claravid)
[![arXiv Preprint](https://img.shields.io/badge/arXiv-2503.17856-b31b1b?style=flat&logo=arXiv&logoColor=white)](https://arxiv.org/abs/2503.17856)
> If you find this useful, please consider giving us a star ⭐

Official repo for: *ClaraVid: A Holistic Scene Reconstruction Benchmark From Aerial Perspective With Delentropy-Based Complexity Profiling* - Accepted ICCV 2025



![ClaraVid Overview](https://rdbch.github.io/claravid/images/overview.jpg)

**ClaraVid** is a synthetic dataset built for semantic and geometric neural reconstruction from low altitude UAV/aerial imagery. 
It contains *16,917 multimodal frames* collected across 8 UAV missions over 5 diverse environments: urban, urban high, rural, highway, and nature.

**Delentropic Scene Profile (DSP)** is a metric for estimating scene complexity from images, 
tailored for structured UAV mapping scenarios. DSP helps predict reconstruction difficulty.

## Channel Log / TODOs
- [x] All data uploaded
- [x] Release dataset SDK
- [x] Release pip package
- [ ] Release dataset splits
- [ ] Add Nerfstudio support
- [ ] Dataset download script
- [ ] Release DSP code (closer to conference)

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

# Examples
We provide 2 examples for this code base:
### Dataset interface
In [examples/demo.ipynb](examples/demo.ipynb) we provide an example for loading and exploring a scene and configuring the various flight parameters and modalities: 
```python
from claravid import ClaravidDataset

dataset = ClaravidDataset(
    root=Path('/path/to/claravid'),
    missions=['highway_1', ],     # see ClaravidMissions
    altitude=['low', ],           # see ClaravidAltitude
    direction=['v', 'h'],         # see ClaravidGridDirection
    fields=['rgb', 'pan_seg', 'depth', ...],
)
data = dataset[0]
{"rgb":..., "pan_seg":..., "depth":..., ...}
```
### 3D Manipulation
In [examples/pcl_project.py](examples/pcl_project.py) we provide an example for loading the scene PCL and projecting it to back to a certain frame. 
This serves as an example on how to handle extrinsics, 3D un/projection and manipulating scene pointclouds.

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
