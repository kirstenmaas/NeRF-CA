import tigre
from tigre.utilities.geometry import Geometry
import numpy as np
import json
import torch
import matplotlib.pyplot as plt

from proj_helpers import source_matrix
from general_helpers import normalize, maintain_weighted_img_dict

class ConeGeometry(Geometry):
    """
    Cone beam CT geometry.
    """

    def __init__(self, data, scale_factor = 1e-2 ):

        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"] * scale_factor  # Distance Source Detector      (m)
        self.DSO = data["DSO"] * scale_factor  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"]) * scale_factor  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])[::-1]  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"]) * scale_factor  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"]) * scale_factor  # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data["offDetector"][0], data["offDetector"][1], 0]) * scale_factor  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]

def get_near_far(geo, adjust=0):
    """
    Compute the near and far threshold.; from https://github.com/Ruyi-Zha/naf_cbct/blob/main/src/dataset/tigre.py
    """
    dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
    dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
    dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
    dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
    dist_max = np.max([dist1, dist2, dist3, dist4])
    near = np.max([0, geo.DSO - dist_max - adjust])
    far = np.min([geo.DSO * 2, geo.DSO + dist_max + adjust])

    return near, far

def forward_project_geo(vol, geo, angles):
    if geo.nVoxel[0] != vol.shape[0]:
        vol = np.transpose(vol, (2, 1, 0)).copy()

    projections = tigre.Ax(vol, geo, angles, projection_type='interpolated')[0]
    return projections

def store_general_geo(data_geo, near_thresh, far_thresh, max_pixel_value, train_folder_name, scale_factor=1e-2):
    data_geo['near_thresh'] = near_thresh
    data_geo['far_thresh'] = far_thresh
    data_geo['max_pixel_value'] = np.log(max_pixel_value)

    # scale the values that need scaling
    data_geo["DSD"] *= scale_factor
    data_geo["DSO"] *= scale_factor
    data_geo["dDetector"] = (np.array(data_geo["dDetector"]) * scale_factor).tolist()
    data_geo['nVoxel'] = data_geo['nVoxel'].tolist()
    data_geo["dVoxel"] = (np.array(data_geo["dVoxel"]).astype('float') * scale_factor).tolist()
    data_geo["offOrigin"] = (np.array(data_geo["offOrigin"]) * scale_factor).tolist()
    data_geo["offDetector"] = (np.array(data_geo["offDetector"]) * scale_factor).tolist()

    with open(f'{train_folder_name}general.json', 'w') as fp:
        json.dump(data_geo, fp)

def get_ray_values_tigre(theta, phi, larm, geo, device):
    src_pt = np.array([0, 0, -geo.DSO])

    src_matrix = source_matrix(src_pt, theta, phi, larm)
    pose = torch.from_numpy(src_matrix).to(device).float()

    img_width, img_height = geo.nDetector

    # do point & ray sampling
    ii, jj = torch.meshgrid(
        torch.linspace(0, img_width-1, img_width).to(device),
        torch.linspace(0, img_height-1, img_height).to(device),
        indexing='xy'
    )

    uu = (ii.t() + 0.5 - img_width / 2) * geo.dDetector[0] + geo.offDetector[0]
    vv = (jj.t() + 0.5 - img_height / 2) * geo.dDetector[1] + geo.offDetector[1]
    dirs = torch.stack([uu / geo.DSD, vv / geo.DSD, torch.ones_like(uu)], -1)

    ray_directions = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
    ray_origins = pose[:3, -1].expand(ray_directions.shape)

    return ray_origins, ray_directions, src_matrix, ii, jj

def obtain_img_and_store_tigre(image_id, vol, geo, theta, phi, larm, hrt_phase, int_hrt_phase, resp_phase, max_pixel_value, images_weighted_dict, train_folder_name):
    view_point_key = f'{theta}-{phi}'
    image_id_str = f'image-hrt={int_hrt_phase}-resp={int(resp_phase)}-angles={view_point_key}'

    # CA angles
    angles = np.deg2rad(np.vstack([-theta, phi, 0]).T)

    # absorption image
    img = forward_project_geo(vol, geo, angles)

    # correct for detector geometry
    img = np.flipud(img)

    # transmission
    img_t = max_pixel_value * np.exp(-img)

    # absorption
    log_img_t = np.log(img_t)

    # min max normalize img
    norm_img, img_min, img_max = normalize(log_img_t)

    # save img as plt and npy
    plt.imsave(f'{train_folder_name}{image_id_str}.png', norm_img, cmap='gray')
    np.save(f'{train_folder_name}{image_id_str}.npy', norm_img)

    # maintain the weighted dict for variance
    images_weighted_dict = maintain_weighted_img_dict(img, view_point_key, images_weighted_dict)

    # keep view point keys in regular file
    data_frame = store_train_img(image_id, image_id_str, img_min, img_max, view_point_key, resp_phase, int_hrt_phase, hrt_phase, theta, phi, larm, train_folder_name)

    return data_frame, images_weighted_dict

def get_xcat_properties_tigre(data_size, vol_dimensions):
    if data_size == 200:
        geo_data = {
            "DSD": 2500,
            "DSO": 450, #4,
            "nDetector": [200, 200],
            "dDetector": [1, 1],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.25,0.25,0.25],
            "offOrigin": [10,-25,25],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    elif data_size == 50:
        geo_data = {
            "DSD": 2500,
            "DSO": 450, #4,
            "nDetector": [50, 50],
            "dDetector": [4, 4],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.25,0.25,0.25],
            "offOrigin": [10,-25,25],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    else:
        print('UNKNOWN DATA SIZE')
        return
    return geo_data

def get_ccta_properties_tigre(data_size, vol_dimensions):
    if data_size == 200:
        geo_data = {
            "DSD": 2000,
            "DSO": 600,
            "nDetector": [200, 200],
            "dDetector": [1,1],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.9, 0.9, 0.9],
            "offOrigin": [0, 0, 0],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    elif data_size == 50:
        geo_data = {
            "DSD": 2000,
            "DSO": 600,
            "nDetector": [50, 50],
            "dDetector": [4, 4],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.9, 0.9, 0.9],
            "offOrigin": [0, 0, 0],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    else:
        print('UNKNOWN DATA SIZE')
        return
    return geo_data

def store_train_img(image_id, image_id_str, img_min, img_max, view_point_key, resp_phase, int_hrt_phase, hrt_phase, theta, phi, larm, train_folder_name):
    data_frame = {}

    data_frame['image_id_str'] = image_id_str
    data_frame['image_id'] = image_id
    data_frame['file_path'] = f'{train_folder_name}{image_id_str}.npy'
    data_frame['img_min_max'] = [img_min.astype('float64'), img_max.astype('float64')]
    data_frame['weighted_file_path'] = f'{train_folder_name}image-{view_point_key}-var.npy'
    data_frame['resp_phase'] = resp_phase
    data_frame['heart_phase'] = int_hrt_phase # range = [0,9]
    data_frame['org_heart_phase'] = int(hrt_phase)
    data_frame['theta'] = float(theta)
    data_frame['phi'] = float(phi)
    data_frame['larm'] = float(larm)

    return data_frame