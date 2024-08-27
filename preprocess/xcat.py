import numpy as np
import subprocess
import json
import os
import pyvista as pv
import time

from preprocess.general_helpers import load_vol_grid

def prepare_and_run_xcat(hrt_phase, resp_phase, xcat_path='D:/XCAT_V1_PC', hrt_str='hrt_start_phase_index', resp_str='resp_start_phase_index', lca_val=0.15, save_vtk=False):
    override_param_strs = np.array([hrt_str, resp_str])
    override_vals = np.array([hrt_phase, resp_phase])

    # this id should be determined based on the ids already in the .json file
    phase_info_path = f'{xcat_path}/phases.json'
    with open(phase_info_path) as f:
        phases_lst = json.load(f)

    curr_phase_obj, run_id = find_hrt_resp_phase_id(phases_lst, hrt_phase, resp_phase)
    run_file_path = f'{xcat_path}/{run_id}'

    print(curr_phase_obj)
    
    # phase obj does not exist yet, retrieve it
    if len(curr_phase_obj.keys()) == 0 or True:

        # create directory
        if not os.path.exists(run_file_path):
            os.makedirs(run_file_path)

        # override the hrt and resp phase in the general.samp.par file
        param_file_names = ['volume', 'noarteries']
        for file_name in param_file_names:
            param_file_path = f'{xcat_path}/{file_name}.samp.par'
            save_param_file_path = f'{run_file_path}/{file_name}.samp.par'

            volume_file_name = f'{run_id}/{file_name}'
            if not (os.path.exists(save_param_file_path) or os.path.exists(f'{xcat_path}/{volume_file_name}')):
                override_xcat_param_file(param_file_path, save_param_file_path, override_vals, override_param_strs)

                # move to xcat path (as otherwise the software doesn't recognize the config files)
                os.chdir(xcat_path)
                print(f'running xcat for hrt_phase={hrt_phase} and resp_phase={resp_phase} for {file_name} file')
                try:
                    start_time = time.time()
                    run_xcat('', f'{volume_file_name}.samp.par', volume_file_name)
                    print(f'time to generate {time.time() - start_time} s')
                except Exception as e:
                    print('exception')

        if not os.path.exists(f'{run_file_path}/lca.npy') or True:
            print('generating lca.npy...')
            full_vol = load_xcat_bin_file(f'{run_file_path}/volume_atn_1.bin')

            no_artery_vol = load_xcat_bin_file(f'{run_file_path}/noarteries_atn_1.bin')
            artery_vol = extract_artery_vol(full_vol, no_artery_vol)
            extract_lca_from_artery_vol(full_vol, artery_vol, run_file_path, lca_val=lca_val, save_vtk=save_vtk)

    lca_center = extract_bounding_box(run_file_path)

    new_phase_obj = {
        "id": run_id,
        "hrt_phase": hrt_phase,
        "resp_phase": resp_phase,
        'bounding_box_center': lca_center,
    }

    if len(curr_phase_obj.keys()) > 0:
        phases_lst[run_id] = new_phase_obj
    else:
        phases_lst.append(new_phase_obj)

    with open(phase_info_path, 'w') as f:
        json.dump(phases_lst, f)

def find_hrt_resp_phase_id(phases_lst, hrt_phase, resp_phase):
    # check if hrt & resp phase already in dict, otherwise return id + 1
    check_phases = [phase_obj for phase_obj in phases_lst if np.around(phase_obj['hrt_phase'], 4) == np.around(hrt_phase, 4) and np.around(phase_obj['resp_phase'], 4) == np.around(resp_phase, 4)]

    if len(check_phases) == 1:
        curr_phase_obj = check_phases[0]
        curr_id = curr_phase_obj['id']
    # return a new id equaling the length of current lst
    elif len(check_phases) == 0:
        curr_phase_obj = {}
        curr_id = len(phases_lst)

    return curr_phase_obj, curr_id

def run_xcat(xcat_path, par_file_name, save_file_name):
    run_line = f'dxcat1_bin.exe {par_file_name} {save_file_name}'
    p1 = subprocess.Popen(run_line, stdout=subprocess.PIPE, shell=True)
    (output, err) = p1.communicate()
    p_status = p1.wait()

def override_xcat_param_file(param_file_path, save_param_file_path, override_vals, override_param_strs):
    with open(param_file_path) as file:
        general_lines = np.array([line.rstrip() for line in file])

        for i, override_str in enumerate(override_param_strs):
            override_val = override_vals[i]
            line_idx = np.argwhere(np.char.find(general_lines, override_str) > -1).flatten()[0]
            line = general_lines[line_idx]

            general_lines[line_idx] = str(override_val) + line.split('\t')[1]
        
        with open(save_param_file_path, 'w') as f:
            for line in general_lines:
                f.write(f"{line}\n")

def load_xcat_bin_file(file_name, dimensions=[512, 512, 401]):
    volume = np.fromfile(file_name, dtype=np.float32).reshape(*dimensions)
    return volume

def extract_artery_vol(full_vol, no_artery_vol):
    artery_vol = np.abs(full_vol - no_artery_vol)
    return artery_vol

def extract_lca_from_artery_vol(full_vol, artery_vol, vol_id_path, lca_val=0.15, dimensions=[512, 512, 401], bounds=np.array([[0, 280], [250, 500], [0, 260]]), save_vtk=False):
    xs = np.linspace(0, dimensions[0], dimensions[0])
    ys = np.linspace(0, dimensions[1], dimensions[1])
    zs = np.linspace(0, dimensions[2], dimensions[2])
    mesh_grid = np.array(np.meshgrid(xs, ys, zs))

    grid = pv.StructuredGrid(mesh_grid[0], mesh_grid[1], mesh_grid[2])
    points = grid.points
    
    # within bounds
    ids = np.arange(0, points.shape[0])
    ids_3d = ids.reshape(grid.dimensions)
    for dim in range(points.shape[1]):
        ids = np.intersect1d(ids, np.argwhere((points[:, dim] >= bounds[dim, 0]) & (points[:,dim] <= bounds[dim, 1])).flatten())

    # has a value in the difference volume
    ids = np.intersect1d(ids, np.argwhere(artery_vol.flatten() > 0))

    # find values in 3d grid ids
    lca_ids = np.argwhere(np.isin(ids_3d, ids))

    # update the full volume with the new lca value
    full_vol = full_vol.reshape(artery_vol.shape)
    full_vol[lca_ids[:,0], lca_ids[:,1], lca_ids[:,2]] = lca_val
    with open(f"{vol_id_path}/full_volume.npy", 'wb') as f:
        np.save(f, full_vol.flatten())

    # set the lca vol
    lca_vol = np.zeros(artery_vol.shape)
    lca_vol[lca_ids[:,0], lca_ids[:,1], lca_ids[:,2]] = lca_val
    with open(f"{vol_id_path}/lca.npy", 'wb') as f:
        np.save(f, lca_vol.flatten())

    # save in vtk format already
    if save_vtk:
        grid.point_data['scalars'] = full_vol.flatten()
        grid.save(f"{vol_id_path}/full_volume.vtk")

        grid.point_data['scalars'] = lca_vol.flatten()
        grid.save(f"{vol_id_path}/lca.vtk")

def extract_bounding_box(case_folder_name, dimensions=[512, 512, 401], vol_file_name='lca.npy'):
    lca_grid, _ = load_vol_grid(vol_file_name, dimensions, case_folder_name)
    points = lca_grid.points.reshape((dimensions[0], dimensions[1], dimensions[2], 3))

    lca_vol = np.load(f'{case_folder_name}/{vol_file_name}').reshape(dimensions)

    occ_pts = np.where(lca_vol > 0)
    occ_x = occ_pts[0]
    occ_y = occ_pts[1]
    occ_z = occ_pts[2]

    x_bounds = [np.inf, -np.inf]
    y_bounds = [np.inf, -np.inf]
    z_bounds = [np.inf, -np.inf]
    for i in range(occ_x.shape[0]):
        x = occ_x[i]
        y = occ_y[i]
        z = occ_z[i]
        point_x, point_y, point_z = points[x,y,z]

        if lca_vol[x,y,z] > 0:
            x_bounds[0] = min(point_x, x_bounds[0])
            y_bounds[0] = min(point_y, y_bounds[0])
            z_bounds[0] = min(point_z, z_bounds[0])
            x_bounds[1] = max(point_x, x_bounds[1])
            y_bounds[1] = max(point_y, y_bounds[1])
            z_bounds[1] = max(point_z, z_bounds[1])

    x_center = (x_bounds[0] + x_bounds[1]) / 2
    y_center = (y_bounds[0] + y_bounds[1]) / 2
    z_center = (z_bounds[0] + z_bounds[1]) / 2
    center = [x_center, y_center, z_center]
    return center