"""
Author: Radu Beche (Technical University of Cluj-Napoca, UTCN)
Licensed under the MIT License.
"""

# script for computing complexity profile for a set of images
import argparse
from pathlib import Path
from claravid import scene_complexity_profile as xsp


def collect_image_paths(input_path: Path, pattern: str = "*.png") -> list[Path]:
    """
    Collect image paths from a directory or a text file.

    Args:
        input_path: Path to a directory or a text file containing one path per line.
        pattern: Glob pattern for image selection if input is a directory. Defaults to '*.png'.

    Returns:
        List of image paths
    """
    input_path = Path(input_path)

    # directory parsing
    if input_path.is_dir():
        files = list(input_path.glob(pattern))
        return [f.resolve() for f in files if f.is_file()]

    # file parsing
    elif input_path.is_file():
        # Read lines from file (one path per line)
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return [Path(line).resolve() for line in lines]

    raise ValueError(f"Invalid path: {input_path} (not a file or directory)")

# ######################################################################################################################
#                                                   MAIN
# ######################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute complexity profile for a set of images")
    parser.add_argument('--input', type=Path, required=True, help="Input directory containing images")
    parser.add_argument('--pattern', type=str, default='"*.jpg"', help="Glob pattern to match images")
    parser.add_argument('--complexity_func', type=str, default='delent', choices=['delent', 'pixent', 'glcment'], help="Complexity function to use")

    parser.add_argument('--save_output', type=bool, default=True, help="Whether to save the complexity results")
    parser.add_argument('--output_path', type=Path, default="complexity_results.json",  help="Path to save the complexity results")
    parser.add_argument('--save_plot', type=bool, default=True, help="Whether to save the complexity profile plot")
    parser.add_argument('--plot_path', type=Path, default="complexity_profile.png", help="Path to save the complexity profile plot")

    parser.add_argument('--use_blur', type=bool, default=True, help="Apply Gaussian blur before computing complexity")
    parser.add_argument('--blur_ksize', type=int, default=3, help="Kernel size for blur.")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of workers. Uses multiprocessing")

    args = parser.parse_args()

    image_list = collect_image_paths(args.input, args.pattern)
    print("Number of images found:", len(image_list))

    # merge params
    cfg = xsp.ComplexityProfileConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        elif k not in ["input", "pattern"]:
            print(f"Warning: Unknown config parameter '{k}'")

    # compute complexity profile
    cp = xsp.ComplexityProfile(cfg)
    print("Computing complexity profile...")
    cp.process(image_list)

    print("Done")