"""
Author: Radu Beche (Technical University of Cluj-Napoca, UTCN)
Licensed under the MIT License.
"""

__all__ = ['ComplexityProfile', 'ComplexityProfileConfig',
           'compute_delentropy', 'compute_shannon_pixel_entropy', 'compute_glcm_entropy']

import cv2
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal
from pathlib import Path
from scipy.stats import beta
from dataclasses import dataclass
from multiprocessing import Pool
from tqdm.auto import tqdm
from skimage.feature import graycomatrix

sns.set_theme(style='ticks')

# ######################################################################################################################
#                                                  COMPLEXITY FUNCTIONS
# ######################################################################################################################
def compute_delentropy(img: np.ndarray) -> float:
    """
    Compute delentropy from a grayscale image.
    Args:
        img: grayscale image as a 2D numpy array.

    Returns:
        image delentropy value
    """
    # params
    num_bins = 512 - 1
    dbin = 0.5
    diff_range = 255.5
    range = diff_range + dbin
    full_range = False
    assert len(img.shape) == 2, "Input image must be grayscale"

    # if img.dtype == np.uint8, negative gradients will be clipped, use np.int16 for full range
    img_compute = img if full_range else img.astype(np.int16)

    fx = cv2.Sobel(img_compute, ddepth=-1, dx=1, dy=0, ksize=3,).astype(int)
    fy = cv2.Sobel(img_compute, ddepth=-1, dx=0, dy=1, ksize=3).astype(int)

    # compute joint histogram
    del_density, xedges, yedges = np.histogram2d(x=fx.flatten(),
                                                y=fy.flatten(),
                                                bins=num_bins,
                                                range=[[-range, range], [-range, range]])

    # deldensity
    del_density = del_density / np.sum(del_density)
    del_density = del_density.T
    nonzero = del_density > 0

    # delentropy
    H = -0.5 * np.sum(del_density[nonzero] * np.log2(del_density[nonzero]))
    return H


def compute_glcm_entropy(img: np.ndarray) -> float:
    """
    Compute the gray-level co-occurrence matrix (GLCM) entropy for a grayscale image.
    Adapted from:
        https://github.com/scikit-image/scikit-image/blob/959c3b8500cb212dabf3e0ed594a8169d44a113a/src/skimage/feature/texture.py#L170

    Args:
        img: grayscale image as a 2D numpy array.

    Returns:
        image glcm entropy value
    """
    distances = [1]
    angles = [0] # used in paper
    # angles = [0, np.pi/2, np.pi, 3*np.pi/2] # 0, 90, 180, 270 degrees
    levels = 256
    assert len(img.shape) == 2, "Input image must be grayscale"

    # build GLCM
    glcm = graycomatrix(img,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)

    # normalize
    P = glcm.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    # compute entropy
    ln = -np.log2(P, where=(P != 0), out=np.zeros_like(P))
    H = np.sum(P * ln, axis=(0, 1))  # entropy for each (d, theta) pair
    H = H.mean()  # aggregate over angles/distances

    return H

def compute_shannon_pixel_entropy(img: np.ndarray) -> float:
    """
    Compute the Shannon entropy of pixel intensities in a grayscale image.

    Args:
        img: grayscale image as a 2D numpy array.

    Returns:
        image pixel entropy value
    """
    bins = 256
    assert len(img.shape) == 2, "Input image must be grayscale"

    hist, _ = np.histogram(img.flatten(), bins=bins, range=(0, 256))

    # normalize
    p = hist.astype(float) / hist.sum()
    p_nonzero = p[p > 0]

    # compute entropy
    H = -np.sum(p_nonzero * np.log2(p_nonzero))

    return H

@dataclass
class ComplexityFunctions:
    delent: callable = compute_delentropy
    """Compute complexity scene profile using delentropy - Delentropic Scene Profile (DSP)"""
    pixent: callable = compute_shannon_pixel_entropy
    """Compute scene complexity profile using Shannon pixel entropy"""
    glcment: callable = compute_glcm_entropy
    """Compute scene complexity profile  GLCM entropy"""

@dataclass
class ComplexityFunctionsName:
    delent: str = "Delentropy"
    pixent: str = "Pixel Entropy"
    glcment: str = "GLCM Entropy"


# ######################################################################################################################
#                                            SCENE COMPLEXITY PROFILE
# ######################################################################################################################
@dataclass
class ComplexityProfileConfig:

    use_blur: bool = True
    """Whether to apply Gaussian blur to the image before computing complexity."""
    blur_ksize: int = 3
    """Kernel size for Gaussian blur. Must be an odd integer."""
    num_workers: int = 8
    """Number of parallel workers for processing image complexities."""
    save_plot: bool = True
    """Whether to save a plot of the scene complexity profile."""
    plot_path: Path = Path("complexity_profile.png")
    """Path to save the complexity profile plot."""
    save_output: bool = True
    """Whether to save the computed complexities to a JSON file."""
    output_path: Path = Path("complexity_results.json")
    """Path to save the complexity results JSON file."""
    complexity_func: Literal['delent', 'pixent', 'glcment'] = 'glcment'
    """Complexity measure to use: 'delent' for delentropy, 'pixent' for Shannon pixel entropy, 'glcm' for GLCM entropy."""


class ComplexityProfile:
    def __init__(self, config: ComplexityProfileConfig):
        self.config: ComplexityProfileConfig = config
        self._complexity_func:callable = getattr(ComplexityFunctions, config.complexity_func)

    def process(self, image_path_list: list[Path]) -> None:
        """
        Process a list of images to compute their complexities and generate a complexity profile.
        Args:
            image_path_list: list of image file paths.

        Returns:
            -
        """
        with Pool(processes=self.config.num_workers) as pool:
            rows = [r for r in tqdm(pool.imap(self.process_image, image_path_list), total=len(image_path_list))]
            # rows = pool.map(self.process_image, image_path_list) # used for debug
            rows = np.array(list(rows))

        beta_form = self.beta_fit(np.array(rows))

        if self.config.save_output:
            self.save_complexities(image_path_list, rows, self.config.output_path)

        if self.config.save_plot:
            self.plot(rows, beta_form)


    def process_image(self, image_path: Path) -> float:
        """
        Load an image and compute its complexity.
        Args:
            image_path: path to the image file

        Returns:
            complexity value as a float
        """

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Could not load image at {image_path}")
            return np.nan

        if self.config.use_blur:
            img = cv2.GaussianBlur(img, (self.config.blur_ksize, self.config.blur_ksize), 1.0)
        complexity = self._complexity_func(img)

        return complexity

    def beta_fit(self, c_vals: np.ndarray = None) -> tuple[float, float, float, float]:
        """
        Fit a Beta distribution to the complexity values.
        Args:
            complexity_vals: complexity values

        Returns:
            parameters of the fitted beta distribution (a, b, c_min, c_max)
        """
        c_min = c_vals.min()
        c_max = c_vals.max()

        scaled_c_vals = (c_vals - c_min) / (c_max - c_min)
        EPS = 1e-6
        scaled = np.clip(scaled_c_vals, EPS, 1 - EPS)
        a_fit, b_fit, a, b = beta.fit(scaled, floc=0, fscale=1, method='MM')
        print(f"Fitted Beta parameters: a={a_fit}, b={b_fit}", a, b)

        return a_fit, b_fit, c_min, c_max

    def plot(self, c_vals: np.ndarray, beta_form: tuple[float, float, float, float]) -> None:
        """
        Plot the complexity distribution and fitted Beta distribution.

        Args:
            c_vals: complexity values
            beta_form: fitted beta distribution parameters (a, b, c_min, c_max)

        Returns:

        """
        a_fit, b_fit, local_min, local_max = beta_form

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # plot hist
        sns.histplot(c_vals, stat="density", ax=ax)

        # plot analytical beta
        x_grid = np.linspace(local_min, local_max, 200)
        x_scaled = (x_grid - local_min) / (local_max - local_min)
        pdf = beta.pdf(x_scaled, a_fit, b_fit) / (local_max - local_min)

        s = f"α={a_fit:2.2f} \nβ={b_fit:2.2f} \nμ={c_vals.mean():2.2f} \nσ={(c_vals.std()):2.2f}"
        ax.plot(x_grid, pdf, 'r--', label=s)

        # plot mu
        ax.axvline(c_vals.mean(), color='red')

        ax.grid(True, alpha=0.5, linestyle='--', color='silver')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.legend()
        fig.suptitle(f"Scene Complexity Profile - {self.config.complexity_func}")
        plt.tight_layout()

        save_path = self.config.plot_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.05, transparent=False)


    def save_complexities(self, image_path_list: list[Path], c_vals: np.array, output_path: Path = None) -> None:
        """
        Save the computed complexities to a JSON file.
        Args:
            image_path_list: list of image file paths
            c_vals: complexity values
            output_path:

        Returns:
            -
        """
        assert len(image_path_list) == len(c_vals), (f"Image paths and complexities length mismatch."
                                                           f" {len(image_path_list)} vs {len(c_vals)}")
        res = []
        for p, c in zip(image_path_list, c_vals):
            res.append({"image_path": str(p), "complexity": c})

        res_dict = {
            "metric": self.config.complexity_func,
            "num_images": len(res),
            "use_blur": self.config.use_blur,
            "blur_ksize": self.config.blur_ksize,
            "results": res
        }

        output_path = output_path if output_path is not None else self.config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(res_dict, f, indent=4)

        assert output_path.exists(), f"Failed to save results to {output_path}"
        print(f"Saved complexity results to {output_path}")
