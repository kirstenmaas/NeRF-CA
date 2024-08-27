import numpy as np
import torch

def x_rotation_matrix(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0], 
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def y_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def z_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation_matrix(vec):
    m = np.identity(4)
    m[:3, 3] = vec[:3]
    return m

def get_rotation_matrix(theta, phi, larm):
    R1 = x_rotation_matrix(-np.pi/2)
    R2 = x_rotation_matrix(np.deg2rad(phi))
    R3 = z_rotation_matrix(np.pi/2)
    R4 = z_rotation_matrix(np.deg2rad(theta))   

    R = np.dot(np.dot(R4, np.dot(R3, R2)), R1)
    return R

def source_matrix(source_pt, theta, phi, larm=0, translation=[0,0,0], type='rotation'):
    rot = get_rotation_matrix(theta, phi, larm)
    worldtocam = rot.dot(translation_matrix(source_pt))

    return worldtocam

def get_ray_values(theta, phi, larm, src_pt, img_width, img_height, focal_length, device, translation=np.array([0,0,0])):
    # obtain rotation matrix based on angles
    src_matrix = source_matrix(src_pt, theta, phi, larm, translation)
    tform_cam2world = torch.from_numpy(src_matrix).to(device)

    # do point & ray sampling
    ii, jj = torch.meshgrid(
        torch.arange(0, img_width).to(tform_cam2world),
        torch.arange(0, img_height).to(tform_cam2world),
        indexing='xy'
    )

    directions = torch.stack([(ii - img_width / 2) / focal_length,
                        -(jj - img_height / 2) / focal_length,
                        -torch.ones_like(ii)
                        ], dim=-1)

    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1).to(device)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape).to(device) 
    return ray_origins, ray_directions, src_matrix, ii, jj

def get_depth_values(near_thresh, far_thresh, depth_samples_per_ray, device, stratified=True):
    t_vals = torch.linspace(0., 1., depth_samples_per_ray)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)

    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)

    # stratified samples in those intervals
    if stratified:
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(device)
    return depth_values

def ray_tracing(interpolator, angles, ray_origins, ray_directions, depth_values, img_width, img_height, ii, jj, batch_size, device, proj_folder_name, type='ct', max_pixel_value=8.670397):
    img = torch.ones((int(np.ceil(img_width)), int(np.ceil(img_height))))

    if batch_size > img_width or batch_size > img_height:
        batch_size = min(img_width, img_height)

    # loop over image (because it has high memory consumption)
    for i_index in range(0, ii.shape[0], batch_size):
        for j_index in range(0, jj.shape[0], batch_size):

            query_points = ray_origins[..., None, :][i_index:i_index+batch_size, j_index:j_index+batch_size] + ray_directions[..., None, :][i_index:i_index+batch_size, j_index:j_index+batch_size] * depth_values[..., :, None]

            one_e_10 = torch.tensor([1e-10], dtype=depth_values.dtype, device=depth_values.device)
            dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

            ray_points = query_points.cpu().numpy().reshape((-1, 3))

            interp_vals = torch.from_numpy(interpolator(ray_points)).to(device).reshape(query_points.shape[:-1])

            # x-ray image
            if type == 'ct':
                norm_dists = dists
                weights = interp_vals * norm_dists
                base_intensity = max_pixel_value #based on XCAT phantom CT generator
                # transmission
                img_val = base_intensity * torch.exp(-torch.sum(weights, dim=-1))
            elif type == 'mip':
                img_val = torch.max(interp_vals, dim=-1).values

            img[i_index:i_index+batch_size,j_index:j_index+batch_size] = img_val

    return img