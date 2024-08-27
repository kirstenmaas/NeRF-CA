import torch
import numpy as np
import sys
import wandb
import yaml
import json
import traceback
import time

sys.path.append('.')
torch.cuda.empty_cache()

from model_helpers import *
from data_helpers import *

from model.CPPN import CPPN
from preprocess.datatoray import datatoray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = config_parser(config_file='temporal/3d.txt', sweep_file='temporal/sweep-3d.yaml')
    run_args = parser.parse_args()
    debug_mode = run_args.debug_mode

    if debug_mode:
        print('DEBUG MODE ON')
        try:
            train()
        except Exception as e:
            print(traceback.print_exc(), file=sys.stderr)
            exit(1)
    else:
        train()

def generate_data(run_args):
    datatoray(run_args)

def train():
    parser = config_parser(config_file='temporal/3d.txt', sweep_file='temporal/sweep-3d.yaml')
    run_args = parser.parse_args()

    exp_name = initialize_wandb()
    run_args = overwrite_args_wandb(run_args, wandb.config)
    wandb.log(vars(run_args))

    store_folder_name = f'cases/{run_args.data_name}/'
    log_dir = initialize_save_folder(store_folder_name, exp_name)

    data_folder_name = f'data/{run_args.data_name}/{run_args.data_size}/'
    general_file_name = f'{data_folder_name}general.json'

    if run_args.use_experiment_name:
        train_file_name = f'{data_folder_name}train-{run_args.experiment_name}.json'
        test_file_name = f'{data_folder_name}test-{run_args.experiment_name}.json'
    else:
        train_file_name = f'{data_folder_name}train-{float(run_args.data_limited_range)}-{float(run_args.data_step_size)}-{run_args.data_time_range_start}-{run_args.data_time_range_end}.json'
        test_file_name = f'{data_folder_name}test-{float(run_args.data_limited_range)}-{float(run_args.data_step_size)}-{run_args.data_time_range_start}-{run_args.data_time_range_end}.json'

    # if not (os.path.isfile(general_file_name) and os.path.isfile(train_file_name) and os.path.isfile(test_file_name)):
    run_args.data_time_range_end = run_args.data_time_range_start + 1

    train_file_name = f'{data_folder_name}train-{float(run_args.data_limited_range)}-{float(run_args.data_step_size)}-{run_args.data_time_range_start}-{run_args.data_time_range_end}.json'
    test_file_name = f'{data_folder_name}test-{float(run_args.data_limited_range)}-{float(run_args.data_step_size)}-{run_args.data_time_range_start}-{run_args.data_time_range_end}.json'
    
    generate_data(run_args)

    with open(general_file_name) as f:
        data_info = json.load(f)

    with open(train_file_name) as f:
        train_data = json.load(f)['frames']

    with open(test_file_name) as f:
        test_data = json.load(f)['frames']

    train_img_indices = [row['image_id_str'] for row in train_data]
    test_img_indices = [row['image_id_str'] for row in test_data]
    all_img_indices = np.concatenate((train_img_indices, test_img_indices))

    img_width, img_height = data_info['nDetector']
    near_thresh = data_info['near_thresh']
    far_thresh = data_info['far_thresh']
    max_pixel_value = data_info['max_pixel_value']

    rays_train, _ = prepare_data_for_loader_tigre(train_data, data_info, img_width, img_height, run_args.depth_samples_per_ray_coarse, run_args.weighted_loss_max, device, use_weighting=(run_args.var_sample_perc > 0))

    # depth values based on near-far-number of samples
    depth_values = create_depth_values(near_thresh, far_thresh, run_args.depth_samples_per_ray_coarse, device)

    # sample the variance rays more frequently, define them based on quartile
    var_ray_ids = np.argwhere(rays_train[:,-1,0] > 1. + run_args.var_sample_thre/100.).flatten()
    all_ray_ids = np.arange(0, rays_train.shape[0])
    non_var_ray_ids = np.setxor1d(var_ray_ids, all_ray_ids)
    
    if run_args.var_sample_perc > 0:
        nb_var_rays = 0
        if run_args.var_sample_perc > 0:
            nb_var_rays = int((run_args.var_sample_perc / 100.) * run_args.img_sample_size)
        
        nb_non_var_rays = run_args.img_sample_size - nb_var_rays

    # prepare the set intensities for training
    train_initial_intensities = torch.Tensor([max_pixel_value for _ in np.arange(0, run_args.img_sample_size)])

    # load the test data
    test_rays = np.stack([get_ray_values_tigre(row['theta'], row['phi'], row['larm'], data_info, device) for row in test_data], 0) #[N_img, ro+rd, W, H, 3]
    test_images = np.stack([denormalize_image(np.load(row['file_path']), img_width, img_height, row['img_min_max']) for row in test_data], 0) #[N_img, W, H]
    test_images = np.repeat(test_images[:, :, :, None], 3, axis=-1) #[N_img, W, H, 3]
    test_phases = np.array([row['heart_phase'] for row in test_data]) #[N_img]
    test_phases = np.tile(test_phases[:, None, None, None], (img_width, img_height, run_args.depth_samples_per_ray_coarse)) #[N_img, W, H, N_samples]

    # assumes that there is only one test image
    test_proj_id = test_img_indices[0]
    test_origins, test_directions = torch.Tensor(test_rays[0]) #[W, H, 3]
    test_origins = test_origins.reshape((-1, 3)).to(device) #[W*H, 3]
    test_directions = test_directions.reshape((-1, 3)).to(device) #[W*H, 3]
    test_weighted_pixs = torch.ones((img_width, img_height)).to(device) #[W, H]
    test_initial_intensities = torch.Tensor([max_pixel_value for _ in np.arange(0, test_origins.shape[0])]).to(device)

    test_phase = torch.Tensor(test_phases[0].reshape((-1, 1))).to(device) #[W*H*N_samples, -1]
    test_img = torch.Tensor(test_images[0,:,:,0]).to(device) #[W*H]
    norm_test_img = (test_img - torch.min(test_img)) / (torch.max(test_img) - torch.min(test_img))

    test_img_ids = np.array([test_proj_id])
    test_img_ids = np.tile(test_img_ids[:, None, None], (img_width, img_height, run_args.depth_samples_per_ray_coarse)) #[N_img, W, H, N_samples]
    test_img_ids = np.reshape(test_img_ids, [-1, 1]) #[W*H]

    test_depth_values = randomize_depth(depth_values, device)
    test_query_points = test_origins[..., None, :] + test_directions[..., None, :] * test_depth_values[..., :, None]
    test_query_points = test_query_points.reshape((-1, 3)).float()

    static_fourier_gaussian = None
    if run_args.static_pos_enc == 'fourier':
        static_fourier_gaussian = torch.randn([run_args.num_input_channels * run_args.static_pos_enc_basis])

    nerf_params = {
        'num_early_layers': run_args.static_num_early_layers,
        'num_late_layers': run_args.static_num_late_layers,
        'num_filters': run_args.static_num_filters,
        'num_input_channels': run_args.num_input_channels,
        'num_output_channels': run_args.num_output_channels,
        'use_bias': True,
        'pos_enc': run_args.static_pos_enc,
        'pos_enc_basis': run_args.static_pos_enc_basis,
        'pos_enc_window_start': run_args.static_pos_enc_window_start,
        'fourier_sigma': run_args.static_pos_enc_fourier_sigma,
        'fourier_gaussian': static_fourier_gaussian,
        'act_func': 'relu',
        'device': device,
    }

    # setup model
    nerf_params = dict(nerf_params)
    
    static_model = CPPN(nerf_params)
    static_model.to(device)

    model_parameters = []
    for name, param in static_model.named_parameters():
        model_parameters.append(param)

    optimizer = torch.optim.Adam([
        { 'params': model_parameters, 'lr': run_args.lr },
    ], lr=run_args.lr)

    # linearly decay the learning rate
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=run_args.lr_end_factor, total_iters=run_args.lr_decay_steps)

    # linearly increase the windowing for positional encoding
    if run_args.static_pos_enc == 'windowed':
        window_update_iter_static = int(run_args.static_pos_enc_window_decay_steps / (run_args.static_pos_enc_basis-run_args.static_pos_enc_window_start))
        static_model.update_windowed_alpha(run_args.static_pos_enc_window_start)
        print('window iter step static', window_update_iter_static)

    # weighted training
    mse_criterion = weighted_MSELoss()

    # log all the hyperparameters in a txt file
    if run_args.config is not None:
        f = f'{log_dir}config.json'
        with open(f, 'w') as file:
            file.write(json.dumps(vars(run_args)))

    print('start training...')
    for n_iter in range(run_args.n_iters+1):
        static_model.train()

        start_time = time.time()

        # update the windowing for positional encoding
        if n_iter > 0 and run_args.static_pos_enc == 'nerfies_windowed':
            static_model.update_windowed_alpha(n_iter, run_args.static_pos_enc_window_decay_steps)

        # update the windowing for frequency positional encoding
        if run_args.static_pos_enc == 'free_windowed':
            static_model.update_freq_mask_alpha(n_iter, run_args.static_pos_enc_window_decay_steps)
        
        # sample a certain percentage from the "variance" rays (determined likely to be dynamic)
        if run_args.var_sample_perc > 0:
            random_ray_ids = np.random.choice(non_var_ray_ids, size=(nb_non_var_rays))

            if run_args.var_sample_perc > 0:
                batch_var_ray_ids = np.random.choice(var_ray_ids, size=(nb_var_rays))
                random_ray_ids = np.concatenate((random_ray_ids, batch_var_ray_ids))

            np.random.shuffle(random_ray_ids)
        # sample a random amount of ids within the range of the rays
        else:
            random_ray_ids = np.random.randint(low=0, high=rays_train.shape[0], size=(run_args.img_sample_size))
        
        batch_rays = torch.from_numpy(rays_train[random_ray_ids]).to(device) #(sample_size, or+dr+pix+weighted, 3)
        
        batch_origins = batch_rays[:,0,:] #(sample_size, 3)
        batch_directions = batch_rays[:,1,:] #(sample_size, 3)
        batch_pix_vals = batch_rays[:,2,:] #(sample_size)
        weighted_pixs = batch_rays[:,3,:] #(sample_size)
        
        batch_pix_vals = batch_pix_vals[:,0] #[sample_size]
        weighted_pixs = weighted_pixs[:,0] #[sample_size]

        batch_initial_intensities = train_initial_intensities.to(device)[:batch_rays.shape[0]]  #[sample_size]
        
        pix_pred_vals, static_sigma, dists = obtain_train_predictions_static(static_model, batch_origins, batch_directions, batch_initial_intensities, depth_values, run_args.output_activation, run_args.batch_size, device)

        # weighted pixel loss
        pixel_loss = mse_criterion(pix_pred_vals, batch_pix_vals, weighted_pixs).mean()
        occl_loss = torch.sum(compute_occl_loss(static_sigma, dists, run_args.occl_reg_perc))

        loss = pixel_loss + run_args.occl_weight_start*occl_loss

        psnr = -10. * torch.log10(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if n_iter % run_args.log_every == 0:
            with torch.no_grad():
                log_dict = {
                    "train_loss": loss.detach().cpu(), 
                    "train_psnr": psnr,
                    "train_pixel_loss": pixel_loss,
                    "train_occl_loss": occl_loss,
                    'train_time': (time.time() - start_time),
                }

                if 'windowed' in run_args.static_pos_enc:
                    log_dict['train_static_windowed'] = static_model.windowed_alpha

                wandb.log(log_dict)

        if n_iter % run_args.display_every == 0:
            static_model.eval()

            with torch.no_grad():
                static_radiance_field_flattened = get_predictions_static(static_model, test_query_points, run_args.batch_size)

                unflattened_shape = (img_width*img_height, run_args.depth_samples_per_ray_coarse, static_model.num_output_channels)
                static_batch_pred_vals = torch.reshape(static_radiance_field_flattened, unflattened_shape)

                test_pix_pred_vals, test_static_sigma, test_dists = render_volume_density(static_batch_pred_vals, test_initial_intensities, test_directions, test_depth_values, run_args.output_activation)

                coarse_test_pred_img = test_pix_pred_vals.reshape((img_width, img_height))
                test_pixel_loss = mse_criterion(coarse_test_pred_img, test_img, test_weighted_pixs).mean()
                test_occl_loss = torch.sum(compute_occl_loss(test_static_sigma, test_dists, run_args.occl_reg_perc))

                test_loss = test_pixel_loss + run_args.occl_weight_start*test_occl_loss

                test_psnr = -10. * torch.log10(test_loss)

                # log in weights and biases
                wandb.log({
                    "test_loss": test_loss, 
                    "test_psnr": test_psnr,
                    "test_occl_loss": test_occl_loss,
                    "test_pixel_loss": test_pixel_loss,
                })

                norm_pred_test_img = (coarse_test_pred_img - torch.min(coarse_test_pred_img)) / (torch.max(coarse_test_pred_img) - torch.min(coarse_test_pred_img))

                wandb.log({
                    'prediction': wandb.Image(norm_pred_test_img),
                    'original': wandb.Image(norm_test_img),
                    'difference': wandb.Image(torch.abs(norm_pred_test_img-norm_test_img)),
                })

                print("Iteration:", n_iter)
                print("Loss coarse:", test_loss.item())
                print("PSNR coarse:", test_psnr.item())

                if n_iter % run_args.save_every == 0:
                    try:
                        static_model.save(f'{log_dir}staticmodel.pth', {})
                    except Exception as e:
                        print(e)
                        print('error saving model')
                        break
    
    # compute_scores_static(static_model, run_args.data_name, run_args.data_size, data_info, test_data, log_dir, device)

if __name__ == "__main__":
    wandb.login()

    parser = config_parser(config_file='temporal/3d.txt', sweep_file='temporal/sweep-3d.yaml')
    run_args = parser.parse_args()

    project_name = '4D-LIMITED'

    if run_args.only_prepare_data:
        print('NOT TRAINING, JUST PREPROCESSING DATA')
        run_args.data_time_range_end = 1
        generate_data(run_args)
    else:
        if run_args.use_wandb:
            with open(run_args.wandb_sweep_yaml, 'r') as f:
                sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)

            sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
            wandb.agent(sweep_id, function=main)
        else:
            main()