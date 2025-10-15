"""
Author: Radu Beche (Technical University of Cluj-Napoca, UTCN)
Licensed under the MIT License.
"""

import numpy as np
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Literal, Union
from . import io_utils


# ######################################################################################################################
#                                                       DATA TYPES
# ######################################################################################################################
@dataclass
class ClaravidMissions:
    """ClaraVid has 8 flight missions over 5 different environments."""
    rural_1: str = '001_rural_1'
    """Rural environment. Crop fields, electrical infrastructure, water, remote industrial area."""
    rural_2: str = '002_rural_2'
    """Rural environment. Rural houses, logistic hub, crop fields, lake."""
    urban_1: str = '003_urban_1'
    """Urban environment. Suburban area, medium height buildings, gas pump, parking lot mall."""
    highway_1: str = '004_highway_1'
    """Highway environment. Butterfly connection, solar panels, wind turbines, water, boat."""
    highway_2: str = '005_highway_2'
    """Highway environment. Freeway, intersection, hidro-plane, boats, tents, campfire, cargo area."""
    urban_2: str = '006_urban_2'
    """Urban environment. Residential area. Construction. Pedestrians."""
    nature_1: str = '007_nature_1'
    """Nature environment. Forest, lakes, tents, comm towers."""
    urban_high_1: str = '008_urban_dense_1'
    """Urban Dense environment. High buildings, mall, pedestrians, playgrounds."""


@dataclass
class ClaravidAltitude:
    """ClaraVid collects simultaneous data at 3 different altitudes and pitch angles."""
    low: str = '45deg_low'
    """Low altitude, 45-55m."""
    mid: str = '55deg_mid'
    """Mid altitude, 55-65m."""
    high: str = '90deg_high'
    """High altitude, 65-75m."""


@dataclass
class ClaravidGridDirection:
    """ClaraVid collects data in a regular grid patters having both  horizontal and vertical passes."""
    h: str = 'h'
    v: str = 'v'


@dataclass
class ClaravidDataTypes:
    """ClaraVid offers multiple data modalities. Visual data is collected at a resolution of 4032x3024 @ FoV 75 deg."""
    rgb: str = 'left_rgb'
    """RGB image"""
    depth: str = 'depth'
    """Metric depth image. Range 0-1000.0 [m]."""
    seg_color: str = 'semantic_colormap'
    """Color semantic segmentation image. Used for debugging."""
    pan_seg: str = 'panoptic_seg'
    """Panoptic segmentation image. Channel 0 - instance, Chanel 1 - semantic class id"""
    dynamic_mask: str = 'dynamic_mask'
    """Dynamic mask image. Used to filter out dynamic objects. 255 - static, 0 - dynamic."""
    extrinsics: str = 'extrinsics'
    """Extrinsics matrix in global coordinates. 4x4 matrix."""
    intrinsics: str = 'intrinsics'
    """Intrinsics matrix for simple pinhole camera model. 3x3 matrix."""


@dataclass
class ClaravidDataExt:
    """File extensions for ClaraVid data modalities."""
    rgb: str = 'jpg'
    depth: str = 'png'
    seg_color: str = 'png'
    pan_seg: str = 'png'
    dynamic_mask: str = 'png'
    extrinsics: str = 'json'
    intrinsics: str = None


@dataclass
class Label:
    """ClaraVid has a vocabulary of 18 semantic classes, some of which have instances."""
    name: str
    class_id: int
    has_instances: bool
    color: tuple[int, int, int]


# ######################################################################################################################
#                                                       CLARAVID DATASET
# ######################################################################################################################
class ClaravidDataset:
    SEMANTIC_PALLETE = [
        # name, class_id, has_instances, train_color
        Label('Building', 0, True, (70, 70, 70)),
        Label('Fence', 1, False, (100, 40, 40)),
        Label('Wire', 2, False, (55, 90, 80)), # electrical wires between poles
        Label('Pedestrian', 3, True, (220, 20, 60)),
        Label('Pole', 4, False, (153, 153, 153)),
        Label('RoadLine', 5, False, (157, 234, 50)),
        Label('Road', 6, False, (128, 64, 128)),
        Label('SideWalk', 7, False, (244, 35, 232)),
        Label('Vegetation', 8, False, (107, 142, 35)),  # trees
        Label('Vehicles', 9, True, (0, 0, 142)),
        Label('TrafficSign', 10, False, (220, 220, 0)),
        Label('GreenEnergy', 13, False, (230, 150, 140)),  # solar panels & wind turbines
        Label('GuardRail', 14, False, (180, 165, 180)),
        Label('Urban props', 15, False, (110, 190, 160)),
        Label('Water', 16, False, (45, 60, 150)),
        Label('Terrain', 17, False, (145, 170, 100)),
        Label('Tent', 18, False, (0, 60, 100)),
        Label('Constr props', 19, False, (170, 120, 50)),
        Label('Bridge', 22, False, (150, 100, 100)),
        Label('Unlabeled', 255, False, (0, 0, 0)),
    ]

    K = np.array([[2627.3023516478706, 0, 2016.0],
                  [0, 2627.3023516478706, 1512.0],
                  [0, 0, 1.0]], dtype=np.float32)

    def __init__(self,
                 root: Path,
                 missions: list[str] = None,
                 altitude: list[Literal["low", "mid", "high"]] = None,
                 direction: list[Literal["h", "v"]] = None,
                 fields: list[str] = None,
                 paths_func: callable = None
                 ) -> None:
        """
        Code interface for ClaraVid dataset. Allows to select missions, fields, altitude and direction.

        Args:
            root: root directory of the ClaraVid dataset.
            missions: list of missions to load, see ClaravidMissions.
            altitude: list of altitudes to load, see ClaravidAltitude.
            direction: list of directions to load, see ClaravidGridDirection.
            fields: list of fields to load, see ClaravidDataTypes.
            paths_func: method to get paths to the dataset files.
        """
        super().__init__()

        self.root = Path(root) if not isinstance(root, Path) else root

        # mission selection
        self.missions = missions if missions else [m.name for m in dataclass_fields(ClaravidMissions)]

        # flight parameters params
        self.fields = fields if fields else [f.name for f in dataclass_fields(ClaravidDataTypes)]
        self.altitude = altitude if altitude else [a.name for a in dataclass_fields(ClaravidAltitude)]
        self.direction = direction if direction else [d.name for d in dataclass_fields(ClaravidGridDirection)]

        # get paths
        self.paths_func = paths_func if paths_func else ClaravidDataset.get_mission_paths
        self.path_list = self.paths_func(self.root,
                                         missions=self.missions,
                                         fields=self.fields,
                                         viewpoint_altitude=self.altitude,
                                         viewpoint_direction=self.direction,
                                         )

    @staticmethod
    def get_mission_paths(
            root: Path,
            missions: list[str],
            fields: list[str],
            viewpoint_altitude: list[Literal["low", "mid", "high"]],
            viewpoint_direction: list[Literal["h", "v"]],
    ) -> list[dict[str, Path]]:
        """
        Get paths to all files in the dataset based on missions, fields, altitude and direction.
        Typically you will override this method to customize the dataset or load just parts of it.
        Args:
            root: directory with the ClaraVid dataset.
            missions: list of missions to load, see ClaravidMissions.
            fields: list of fields to load, see ClaravidDataTypes.
            viewpoint_altitude:list of altitudes to load, see ClaravidAltitude.
            viewpoint_direction: list of altitude and direction to load, see ClaravidGridDirection.

        Returns:
            list of dictionaries with paths to the dataset files.
        """
        path_list = []

        # deal with missions
        for mission in missions:
            mssn = getattr(ClaravidMissions, mission)

            # deal with viewpoints
            for altitude in viewpoint_altitude:
                alt = getattr(ClaravidAltitude, altitude)
                for direction in viewpoint_direction:
                    drct = getattr(ClaravidGridDirection, direction)

                    # select a base file and get all samples based on that"
                    fld = getattr(ClaravidDataTypes, fields[0])
                    ext = getattr(ClaravidDataExt, fields[0])
                    anchor_dir = root / mssn / fld / f'{alt}_{drct}'

                    # deal with all modalities
                    for anchor_file in sorted(anchor_dir.glob(f'*.{ext}')):
                        sample = {}
                        for field in fields:
                            fld = getattr(ClaravidDataTypes, field)
                            ext = getattr(ClaravidDataExt, field)
                            if ext is None:
                                # 'intrinsics' has no files
                                continue

                            sample[field] = root / mssn / fld / f'{alt}_{drct}' / f'{anchor_file.stem}.{ext}'
                            # assert sample[field].exists(), f"File {sample[field]} does not exist"
                        path_list.append(sample)

        return path_list

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx) -> dict[str, np.ndarray]:
        items = {}

        for name in self.fields:
            items[name] = getattr(self, f'read_{name}')(idx)

        items['frame_idx'] = idx
        return items

    def read_rgb(self, idx: int) -> np.ndarray:
        """Read RGB image. Orig 4032x3024."""
        path = self.path_list[idx]['rgb']
        rgb = io_utils.read_image(path)
        return rgb

    def read_depth(self, idx: int) -> np.ndarray:
        """Read metric depth. Range 0-1000.0 [m]."""
        path = self.path_list[idx]['depth']
        depth = io_utils.read_depth(path)
        return depth

    def read_seg_color(self, idx: int) -> np.ndarray:
        """Read color semantic segmentation. Typically used for debugging."""
        path = self.path_list[idx]['seg_color']
        seg_color = io_utils.read_image(path)
        return seg_color

    def read_pan_seg(self, idx: int) -> np.ndarray:
        """Read panoptic segmentation image from file. Channel 0 - instance, Chanel 1 - semantic class id"""
        path = self.path_list[idx]['pan_seg']
        pan_seg = io_utils.read_pan_seg(path)
        return pan_seg

    def read_dynamic_mask(self, idx: int) -> np.ndarray:
        """Read dynamic mask image. 255 - static, 0 - dynamic."""
        path = self.path_list[idx]['dynamic_mask']
        dynamic_mask = io_utils.read_image_gray(path)
        return dynamic_mask

    def read_extrinsics(self, idx: int) -> np.ndarray:
        """Read extrinsics matrix from file."""
        path = self.path_list[idx]['extrinsics']
        extrinsics = io_utils.read_extrinsics(path)
        return extrinsics

    def read_intrinsics(self, *args, **kwargs) -> np.ndarray:
        """Intrinsics matrix for simple pinhole. 4032x3024 @ HFov 75 deg."""
        return self.K.copy()

    def read_all_extrinsics(self) -> np.ndarray:
        """Read all extrinsics."""
        assert len(self.missions) == 1, "This method is only for single mission datasets."
        extrinsics_list = []
        for item in self.path_list:
            path = item['extrinsics']
            extrinsics = io_utils.read_extrinsics(path)
            extrinsics_list.append(extrinsics)
        return np.array(extrinsics_list)

    @staticmethod
    def read_pcl(pcl_path: Union[str, Path]):
        """Open3D import is slow and also adds it as dependency."""
        import open3d as o3d
        pcl = o3d.io.read_point_cloud(str(pcl_path))
        return pcl
