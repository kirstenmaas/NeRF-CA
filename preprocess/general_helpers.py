import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import json
import os

def maintain_weighted_img_dict(img, view_point_key, images_weighted_dict):
    if view_point_key not in images_weighted_dict:
        images_weighted_dict[view_point_key] = [img]
    else:
        new_images_weighted = images_weighted_dict[view_point_key]
        new_images_weighted.append(img)

        images_weighted_dict[view_point_key] = new_images_weighted
    return images_weighted_dict

def obtain_weighted_imgs(view_point_keys, images_weighted_dict, weighted_pixel_values, weighted_method, img_width, img_height, train_folder_name):
    images_weighted = []
    view_point_vars = {}
    for view_point_key in view_point_keys:
        if not view_point_key in view_point_vars:
            view_point_imgs = np.array(images_weighted_dict[view_point_key])
            view_point_imgs = view_point_imgs.reshape((-1, img_width*img_height))
            
            view_point_var_pix = np.ones((img_width, img_height))
            if weighted_method == 'var' and len(view_point_imgs) > 1:
                # transmission to absorption
                view_point_imgs = np.exp(-view_point_imgs)
                view_point_var = np.var(view_point_imgs, axis=0)
                view_point_var_pix = view_point_var.reshape((img_width, img_height))
                view_point_var_pix = (view_point_var_pix - np.min(view_point_var_pix)) / (np.max(view_point_var_pix) - np.min(view_point_var_pix)+1e-10)

            # view_point_var_pix[view_point_var_pix < 0.03] = 0
            # view_point_var_pix[view_point_var_pix >= 0.03] = 1
            plt.imsave(f'{train_folder_name}image-{view_point_key}-var.png', view_point_var_pix, cmap='Reds')

            view_point_vars[view_point_key] = view_point_var_pix + 1 #+1 to make sure that the weight of a value is at least 1 (range of [1,2])
            np.save(f'{train_folder_name}image-{view_point_key}-var.npy', view_point_var_pix+1)

            images_weighted.append(view_point_vars[view_point_key].tolist())
            weighted_pixel_values.append(view_point_vars[view_point_key].flatten())
    plt.close()
    
    return images_weighted, weighted_pixel_values

def load_vol_grid(vol_file_name, dimensions, case_folder_name):
    vol_np = np.load(f'{case_folder_name}/{vol_file_name}')
    vol_vtk = np_to_vtk(vol_np, dimensions)

    return vol_vtk, vol_np

def np_to_vtk(np_vol, dimensions):
    xs = np.linspace(0, dimensions[0], dimensions[0])
    ys = np.linspace(0, dimensions[1], dimensions[1])
    zs = np.linspace(0, dimensions[2], dimensions[2])
    mesh_grid = np.array(np.meshgrid(xs, ys, zs))

    grid = pv.StructuredGrid(mesh_grid[0], mesh_grid[1], mesh_grid[2])
    grid.point_data['scalars'] = np_vol.flatten()

    return grid

def normalize(img):
    img_max = np.max(img)
    img_min = np.min(img)
    norm_img = (img - img_min) / (img_max - img_min)
    return norm_img, img_min, img_max

def setup_experiment_type(data_args, train_folder_name):
    if data_args.use_experiment_name:
        train_file_name = f"{train_folder_name}train-{data_args.experiment_name}.json"
        test_file_name = f"{train_folder_name}test-{data_args.experiment_name}.json"

        # TODO: load the dict with the experiment details
        # stores the desired views with phases for this experiment
        experiment_info_path = f"preprocess/xcat/{data_args.experiment_name}.json"
        with open(experiment_info_path) as f:
            phase_volume_lst = json.load(f)
    else:
        time_steps = np.arange(data_args.data_time_range_start, data_args.data_time_range_end) / 10

        if data_args.data_limited_range_test and data_args.data_step_size_test:
            limited_range_test = data_args.data_limited_range_test
            step_size_test = data_args.data_step_size_test

            test_theta_angles = np.arange(-limited_range_test, limited_range_test+1, step_size_test)
            test_phi_angles = np.arange(-limited_range_test, limited_range_test+1, step_size_test)
            test_angles = np.array(np.meshgrid(test_theta_angles, test_phi_angles, indexing='ij')).reshape((2, -1)).T
            test_angles = np.insert(test_angles, 0, [0, -90], axis=0)
        else:
            # LAO = +theta, RAO = -theta
            # CRA = +phi, CAU = -phi
            # preset validation angles based on clinical views
            test_angles = np.array([[-5, 40], [-5, -40], [90, 0], [-30, 0]])

        limited_range = data_args.data_limited_range
        step_size = data_args.data_step_size #15
        numb_angles = data_args.data_numb_angles

        # define based on limited range and step size
        if step_size <= limited_range:
            # theta_angles = np.linspace(0, 180, 70)
            # # pdb.set_trace()
            # phi_angles = [0]

            theta_angles = np.arange(-limited_range, limited_range+1, step_size)
            phi_angles = np.arange(-limited_range, limited_range+1, step_size)

            # all possible combinations of theta and phi angles
            # [[x0, y0], [x0, y1], [x0, y2], ...]
            set_angle_comb = np.array(np.meshgrid(theta_angles, phi_angles, indexing='ij')).reshape((2, -1)).T

            # take angles out if they are too close to preset validation angles
            close_thresh = 15
            angle_comb = []
            for train_angle in set_angle_comb:
                far_away = True
                for test_angle in test_angles:
                    diff_angle = np.abs(np.array(train_angle) - np.array(test_angle))
                    if np.sum(diff_angle) <= close_thresh:
                        far_away = False
                if far_away:
                    angle_comb.append(train_angle)
            angle_comb = np.array(angle_comb)
            
            if angle_comb.shape[0] != set_angle_comb.shape[0]:
                print('Removed some to avoid too close to validation angles!')

            # if exactly 4 projections == take preset views
            # otherwise, evenly distribute in limited range
            if angle_comb.shape[0] == 4:
                four_angles = np.array([[-30, 30], [-30, -30], [60, -30], [60, 30]])
                angle_comb = four_angles
        elif numb_angles != None:
            if numb_angles == 4:
                predf_angles = np.array([[-30, 30], [-30, -30], [60, -30], [60, 30]])
            elif numb_angles == 3:
                predf_angles = np.array([[-30, -30], [60, -30], [60, 30]])
            elif numb_angles == 2:
                predf_angles = np.array([[-30, -30], [60, 30]])
            angle_comb = predf_angles
        else:
            return Exception()

        train_file_name = f"{train_folder_name}train-{float(limited_range)}-{float(step_size)}-{data_args.data_time_range_start}-{data_args.data_time_range_end}.json"
        test_file_name = f"{train_folder_name}test-{float(limited_range)}-{float(step_size)}-{data_args.data_time_range_start}-{data_args.data_time_range_end}.json"
        
        # construct the phase volume lst from all combinations of theta and phi angles for all timesteps
        phase_volume_lst = []
        for i, time_step in enumerate(time_steps):
            phase_obj = {
                "hrt_phase": time_step,
                "resp_phase": 0, # for xcat
                "train_viewpoints": angle_comb, 
                "test_viewpoints": [],
            }

            # the first hrt phase has a test view
            phase_obj['test_viewpoints'] = test_angles
            
            phase_volume_lst.append(phase_obj)

    render_train_imgs = not os.path.isfile(train_file_name)
    render_test_imgs = not os.path.isfile(test_file_name)

    data_per_view = {}
    data_per_view['frames'] = []

    data_per_view_test = {}
    data_per_view_test['frames'] = []

    return train_file_name, render_train_imgs, data_per_view, test_file_name, render_test_imgs, data_per_view_test, phase_volume_lst