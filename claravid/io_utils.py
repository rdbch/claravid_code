import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Literal


# ######################################################################################################################
#                                                       IMAGE I/O
# ######################################################################################################################
def _write_img(img: np.ndarray, path: Path, name: str = "Image") -> None:
    """Write RGB image to disk"""
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(img).save(path)
    assert path.exists(), f"{name.title()} could not be written - {path}"


def _read_img(path: Path, name: str = "Image", mode: Literal["RGB", "L"] = "RGB") -> np.ndarray:
    """Read RGB image from file"""
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"{name.title()} does not exist - {path} "

    rgb = Image.open(path).convert(mode)
    rgb = np.array(rgb)
    return rgb


def read_image(path: Path) -> np.ndarray:
    return _read_img(path, name="Image")


def write_image(img: np.ndarray, path: Path) -> None:
    _write_img(img, path, name="Image")


def read_image_gray(path: Path) -> np.ndarray:
    return _read_img(path, name="Image", mode="L")


def write_image_gray(img: np.ndarray, path: Path) -> None:
    _write_img(img, path, name="Image")


# ######################################################################################################################
#                                                    EXTRINSICS I/O
# ######################################################################################################################
def _read_json(path: Path, name='Json') -> Dict[str, Any]:
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"{name.title()} does not exist - {path} "
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def _write_json(data: Any, path: Path, name: str = "Json") -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=4)
    assert path.exists(), f"{name.title()} could not be written - {path}"


def read_extrinsics(path: Path) -> np.ndarray:
    """Read 4x4 extrinsics matrix from json."""
    ext = _read_json(path, name="Extrinsics")
    ext = np.array(ext).astype(np.float32).reshape(4, 4)
    return ext


def serialize_extrinsics(extrinsics: np.ndarray) -> List[List[float]]:
    """Serialize 4x4 extrinsics matrix to list."""
    return extrinsics.tolist()


def write_extrinsics(extrinsics: np.ndarray, path: Path) -> None:
    """Write 4x4 extrinsics matrix to json file."""
    ext = serialize_extrinsics(extrinsics)
    _write_json(ext, path, name="Ext")


# ######################################################################################################################
#                                                       DEPTH I/O
# ######################################################################################################################
def write_depth(depth: np.ndarray, path: Path, norm_max: float = 1000.0) -> None:
    """Write depth image to file. Maximum value is 1000.0 [m]"""

    carla_depth = serialize_depth(depth, norm_max)
    _write_img(carla_depth, path, name="Depth")


def serialize_depth(depth: np.ndarray, norm_max: float = 1000.0) -> np.ndarray:
    """Serialize depth image to Carla format. Maximum value is 1000.0 [m]"""
    _CFACT = 256 * 256 * 256 - 1

    depth = ((depth / norm_max) * _CFACT).astype(int).squeeze()
    rgb = np.zeros(shape=(depth.shape[0], depth.shape[1], 3))

    rgb[..., 2] = (depth & 0b111111110000000000000000) >> 16
    rgb[..., 1] = (depth & 0b1111111100000000) >> 8
    rgb[..., 0] = depth & 0b11111111

    return rgb


def read_depth(path: Path, norm_max: float = 1000.0) -> np.ndarray:
    """Read depth image from file and convert to meters. Maximum value is 1000.0 [m]"""
    rgb = _read_img(path, name="Depth")
    rgb = rgb.astype(np.float32)
    _CFACT = 256 * 256 * 256 - 1
    d = (rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 256 * 256) / _CFACT
    d = d * norm_max
    return d


# ######################################################################################################################
#                                                   PAN SEG I/O
# ######################################################################################################################
def read_pan_seg(path: Path) -> np.ndarray:
    """Read panoptic segmentation image from file. channels 0,1 - instance ids, channel 2 - semantic class id"""
    rgb = _read_img(path, name="Pan Seg").astype(np.int32)

    pan_seg = np.zeros((rgb.shape[0], rgb.shape[1], 2), dtype=np.int32)
    pan_seg[..., 0] = rgb[..., 2] * 256 + rgb[..., 1]  # instance
    pan_seg[..., 1] = rgb[..., 0]  # semantics

    return pan_seg


def serialize_pan_seg(pan_seg: np.ndarray) -> np.ndarray:
    """Serialize panoptic segmentation image to RGB format. The first 2 channels are instance ids and the 3rd channel is
     the class id of the semantic class.
    """
    rgb = np.zeros((pan_seg.shape[0], pan_seg.shape[1], 3), dtype=np.uint8)

    instance_ids = pan_seg[..., 0]  # instance
    class_ids = pan_seg[..., 1]  # semantics

    rgb[..., 2] = (instance_ids // 256).astype(np.uint8)
    rgb[..., 1] = (instance_ids % 256).astype(np.uint8)
    rgb[..., 0] = class_ids.astype(np.uint8)

    return rgb


def write_pan_seg(pan_seg: np.ndarray, path: Path) -> None:
    """Write panoptic segmentation image to file."""
    rgb = serialize_pan_seg(pan_seg)
    _write_img(rgb, path, name="Pan Seg")
