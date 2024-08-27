import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import json

from scipy.interpolate import RegularGridInterpolator


from general_helpers import setup_experiment_type, load_vol_grid, obtain_weighted_imgs
from proj_helpers import get_depth_values, ray_tracing
from tigre_helpers import ConeGeometry, get_near_far, store_general_geo, get_xcat_properties_tigre, get_ccta_properties_tigre, obtain_img_and_store_tigre, get_ray_values_tigre
from vis_helpers import visualize_geometry_tigre

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def datatoray(data_args):
    MAX_PIXEL_VALUE = 8.670397
    SCALE_FACTOR = 1e-2
    
    image_id = 0
    vol_file_name = 'full_volume_tigre.npy'
    weighted_method = 'var' # apply pixel variance to weigh the pixels during training

    train_folder_name = f"data/{data_args.data_name}/{data_args.data_size}/"
    if not os.path.exists(train_folder_name):
        os.makedirs(train_folder_name)

    if data_args.data_name == 'XCAT-V1':
        data_store_path = f'D:/NeRFCA/XCAT'
        vol_dimensions = np.array([512, 512, 401])
        data_geo = get_xcat_properties_tigre(data_args.data_size, vol_dimensions)
    elif data_args.data_name == 'MAGIX-LCA':
        data_store_path = f'D:/NeRFCA/MAGIX'
        vol_dimensions = np.load(f'{data_store_path}/{0}/volume-shape.npy')
        data_geo = get_ccta_properties_tigre(data_args.data_size, vol_dimensions)
    
    geo = ConeGeometry(data_geo)
    near_thresh, far_thresh = get_near_far(geo)

    store_general_geo(data_geo, near_thresh, far_thresh, MAX_PIXEL_VALUE, train_folder_name, scale_factor=SCALE_FACTOR)

    train_file_name, render_train_imgs, data_per_view, test_file_name, render_test_imgs, data_per_view_test, phase_volume_lst = setup_experiment_type(data_args, train_folder_name)

    temp_store_src_matrices = {}

    images_weighted_dict = {}
    view_point_keys = []
    weighted_pixel_values = []
    for i, phase_volume in enumerate(phase_volume_lst):
        # find the volume with the same hrt and resp phase
        hrt_phase = phase_volume['hrt_phase']
        int_hrt_phase = int(hrt_phase*10) #convert from decimal to int
        resp_phase = phase_volume['resp_phase']
        train_view_points = phase_volume['train_viewpoints']
        test_view_points = phase_volume['test_viewpoints']

        case_folder_name = f'{data_store_path}/{int_hrt_phase}'
        
        print(f'going through volume {int_hrt_phase} with heart phase {hrt_phase}')

        vol_npy = np.load(f'{case_folder_name}/{vol_file_name}')
        
        if data_args.only_prepare_data or data_args.visualize_geometry:
            volume_scale_factor_geo = geo.dVoxel[0]
            manual_translation_geo = geo.offOrigin[::-1]

            near_thresh_geo, far_thresh_geo = get_near_far(geo)
            depth_values_geo = get_depth_values(near_thresh_geo, far_thresh_geo, data_args.depth_samples_per_ray_coarse, device)

            vol_grid_geo, vol_np_geo = load_vol_grid(vol_file_name, vol_dimensions, case_folder_name)
            vol_np_geo = vol_np_geo.reshape(vol_dimensions)

            # bring to [0, 0, 0]
            final_grid_geo = vol_grid_geo.translate(-np.array(vol_grid_geo.center), inplace=False)
            # scale
            final_grid_geo = final_grid_geo.scale([volume_scale_factor_geo, volume_scale_factor_geo, volume_scale_factor_geo], inplace=False)
            # translate according to scaled offset
            final_grid_geo = final_grid_geo.translate(manual_translation_geo, inplace=False)

            points = final_grid_geo.points
            points = points.round(decimals=3)
            points_x = np.unique(points[:,0])
            points_y = np.unique(points[:,1])
            points_z = np.unique(points[:,2])

            fill_value = 0
            interpolator_geo = RegularGridInterpolator((points_x, points_y, points_z), vol_np_geo, method='linear', bounds_error=False, fill_value=fill_value)

            if i == 0 and data_args.visualize_geometry:
                visualize_geometry_tigre(train_view_points, final_grid_geo, geo, depth_values_geo, interpolator_geo, device)

        print('ray tracing...')
        for j, viewpoint in enumerate(train_view_points):
            theta, phi = viewpoint
            view_point_key = f'{theta}-{phi}'
            view_point_keys.append(view_point_key)

            print(f'{view_point_key} - {j+1}/{len(train_view_points)}')
            image_id_str = f'image-hrt={int_hrt_phase}-resp={int(resp_phase)}-angles={view_point_key}'
            
            data_frame, images_weighted_dict = obtain_img_and_store_tigre(image_id, vol_npy, geo, theta, phi, 0., hrt_phase, int_hrt_phase, resp_phase, MAX_PIXEL_VALUE, images_weighted_dict, train_folder_name)

            ray_origins, ray_directions, src_matrix, ii, jj = get_ray_values_tigre(-theta, phi, 0., geo, device)
            
            temp_store_src_matrices[view_point_key] = src_matrix.tolist()

            image_id += 1

            curr_frames = data_per_view['frames']
            curr_frames.append(data_frame)
            data_per_view['frames'] = curr_frames
        
        # we want our test image to be in the same phase as the first train image
        for j, test_view_point in enumerate(test_view_points): #assumes that there is only one test view
            theta, phi = test_view_point
            
            view_point_key = f'{theta}-{phi}'
            view_point_keys.append(view_point_key)

            image_id_str = f'image-hrt={int_hrt_phase}-resp={int(resp_phase)}-angles={view_point_key}'

            data_frame, images_weighted_dict = obtain_img_and_store_tigre(image_id, vol_npy, geo, theta, phi, 0., hrt_phase, int_hrt_phase, resp_phase, MAX_PIXEL_VALUE, images_weighted_dict, train_folder_name)
            print(image_id_str)

            if data_args.only_prepare_data:
                # generate MIP images
                ray_origins, ray_directions, src_matrix, ii, jj = get_ray_values_tigre(-theta, phi, 0., geo, device)
                img = ray_tracing(interpolator_geo, [theta, phi, 0], ray_origins, ray_directions, depth_values_geo, data_args.data_size, data_args.data_size, ii, jj, 50, device, train_folder_name, type='mip')
                
                # min max normalize img
                img = img.cpu().numpy()
                # flip and rotate to match tigre output
                img = np.fliplr(np.rot90(img, k=3))
                # norm_img, _, _ = normalize(img)

                # save img as plt and npy
                image_id_str = f'image-hrt={int_hrt_phase}-resp={int(resp_phase)}-angles={view_point_key}'
                plt.imsave(f'{train_folder_name}{image_id_str}-mip.png', img, cmap='gray')
                np.save(f'{train_folder_name}{image_id_str}-mip.npy', img)

            image_id += 1

            curr_frames = data_per_view_test['frames']
            curr_frames.append(data_frame)
            data_per_view_test['frames'] = curr_frames

        with open(test_file_name, 'w') as fp:
            json.dump(data_per_view_test, fp)

        if not os.path.exists(f'{train_folder_name}/evaluate.json'):
            with open(f'{train_folder_name}/evaluate.json', 'w') as fp:
                json.dump(data_per_view_test, fp)

    with open(train_file_name, 'w') as fp:
        json.dump(data_per_view, fp)

    # only overwrite when we use all timesteps
    if data_args.data_time_range_end - data_args.data_time_range_start == 10:
        _, weighted_pixel_values = obtain_weighted_imgs(view_point_keys, images_weighted_dict, weighted_pixel_values, weighted_method, data_args.data_size, data_args.data_size, train_folder_name)

    with open('srcmatrices.json', 'w') as fp:
        json.dump(temp_store_src_matrices, fp)

if __name__ == "__main__":
    data_args = {
        'only_prepare_data': False,
        'data_name': 'MAGIX-LCA',
        'data_size': 200,
        'use_experiment_name': False,
        'data_time_range_start': 0,
        'data_time_range_end': 1,
        'data_limited_range': float(60),
        'data_step_size': float(15),
        'visualize_geometry': False,
        'depth_samples_per_ray_coarse': 1000,
        'data_limited_range_test': 180,
        'data_numb_angles': int(4),
        'data_step_size_test': None, #30
    }

    data_args = argparse.Namespace(**data_args)

    datatoray(data_args)