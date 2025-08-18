"""Example of projecting the global point cloud into images."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from claravid import ClaravidDataset

# ######################################################################################################################
#                                               UTILITY METHODS
# ######################################################################################################################
def cam_from_world(points_w: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Convert pts from world coordinates to camera coordinates.
    Args:
        points_w: point cloud in world coordinates - (N, 3)
        c2w: matrix camera-to-world - (4, 4)

    Returns:

    """
    w2c = np.linalg.inv(c2w)
    pts_h = np.hstack([points_w, np.ones((points_w.shape[0], 1), np.float32)])
    pts_c = (w2c @ pts_h.T).T[:, :3]
    return pts_c


def project(pts: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective projection. Returns pixel coordinates (N×2) and a mask of valid points.
    Args:
        pts: point cloud in camera coordinates - (N, 3)
        K: intrinsic camera matrix - (3, 3)

    Returns:
        uv: pixel coordinates (N, 2)
        valid: boolean mask of valid points of original pcl(N,)
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    d = -pts[:, 2] # undo the z-flip  (depth > 0 in front)
    valid = d > 0  # discard points behind the camera

    u = fx * pts[valid, 0] / d[valid] + cx
    v = -fy * pts[valid, 1] / d[valid] + cy
    uv = np.stack([u, v], axis=1).astype(np.int32)

    return uv, valid


def pcl_project(rgb:np.array,
                c2w:np.array,
                K:np.array,
                pts:np.array,
                pts_colors:np.array,
                d_factor:int=1) -> tuple[np.array, np.ndarray, np.ndarray]:
    """
        Utility function to project a point cloud into an image. Useful for showcasing pose and intrinsics usage.
        The function does not account for occlusions or depth ordering, it simply paints the point cloud colors.
    Args:
        rgb: color image - (H, W, 3)
        c2w: camera-to-world matrix - (4, 4)
        K: intrinsic camera matrix - (3, 3)
        pts: point cloud in world coordinates - (N, 3)
        pts_colors: colors of the point cloud - (N, 3)
        d_factor: downscale factor for the image and intrinsic matrix (default: 1.0)

    Returns:
        rgb - color image with projected point cloud colors - (H', W', 3)
        uv - pixel coordinates of the projected points - (N, 2)
        pts_colors - colors of the projected points - (N, 3)
    """
    w, h = rgb.shape[1] // d_factor, rgb.shape[0] // d_factor
    K[:2] /= d_factor

    # bring global pcl to frame coordinates
    pts_cam = cam_from_world(pts, c2w)

    # project pcl into image
    uv, valid_in_front = project(pts_cam, K)
    valid_in_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    # only keep pixels in front of camera and inside the image
    uv = uv[valid_in_img]
    pts_colors = pts_colors[valid_in_front]
    pts_colors = pts_colors[valid_in_img]

    # drawing
    rgb = Image.fromarray(rgb).resize((w, h), Image.Resampling.LANCZOS)
    rgb = np.array(rgb, dtype=np.uint8).copy()
    rgb[uv[:, 1], uv[:, 0]] = pts_colors
    return rgb, uv, pts_colors


# ######################################################################################################################
#                                                     MAIN
# ######################################################################################################################
if __name__ == '__main__':

    # Config params
    DOWNSCALE = 8
    MISSION = 'urban_high_1' # see ClaravidMissions
    ROOT = Path("/path/to/claravid/root") # <===== FILL HERE
    PCL_PATH = Path('path/to/pcl/global_fused_100_cm.ply')

    # init data
    dataset = ClaravidDataset(
        root=ROOT,
        missions=[MISSION, ],  # see ClaravidMissions
        altitude=['low', ],  # see ClaravidAltitude
        direction=['v', 'h'],  # see ClaravidGridDirection
        fields=['rgb', 'extrinsics', 'intrinsics'],  # see ClaravidFields
    )

    # load and prepare data
    pcl = dataset.read_pcl(PCL_PATH)  # get point cloud for the first mission
    print("Point loaded: ", pcl)
    pts = np.array(pcl.points, dtype=np.float32)
    pts_colors = (np.array(pcl.colors) * 255).astype(np.uint8)

    data = dataset[0]

    # get projection
    rgb, uv, pts_colors = pcl_project(data['rgb'],
                                      data['extrinsics'],
                                      data['intrinsics'],
                                      pts,
                                      pts_colors,
                                      DOWNSCALE
                                      )
    plt.imshow(rgb)
    plt.show()
