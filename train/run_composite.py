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

torch.set_printoptions(precision=10)

from model_helpers import *
from data_helpers import *
from model.Temporal import Temporal
from model.CPPN import CPPN
from preprocess.datatoray import datatoray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = config_parser()
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
    parser = config_parser()
    run_args = parser.parse_args()

    exp_name = initialize_wandb('-composite')
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
    generate_data(run_args)

    with open(general_file_name) as f:
        data_info = json.load(f)

    with open(train_file_name) as f:
        train_data = json.load(f)['frames']

    with open(test_file_name) as f:
        test_data = json.load(f)['frames']

        # always only use one test image
        if len(test_data) > 0:
            test_data = [test_data[0]]

    train_img_indices = [row['image_id_str'] for row in train_data]
    test_img_indices = [row['image_id_str'] for row in test_data]
    all_img_indices = np.concatenate((train_img_indices, test_img_indices))

    # TIGRE CODE
    img_width, img_height = data_info['nDetector']
    near_thresh = data_info['near_thresh']
    far_thresh = data_info['far_thresh']
    max_pixel_value = data_info['max_pixel_value']
    
    rays_train, phases_train = prepare_data_for_loader_tigre(train_data, data_info, img_width, img_height, run_args.depth_samples_per_ray_coarse, run_args.weighted_loss_max, device)

    # depth values based on near-far-number of samples
    depth_values_coarse = create_depth_values(near_thresh, far_thresh, run_args.depth_samples_per_ray_coarse, device)
    if run_args.depth_samples_per_ray_fine > 0:
        depth_values_fine = create_depth_values(near_thresh, far_thresh, run_args.depth_samples_per_ray_fine, device)

    # sample the variance rays more frequently, define them based on percentage of variance
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
    weighted_pixs_ones = torch.ones(run_args.img_sample_size).to(device)

    # load the test data
    test_rays = np.stack([get_ray_values_tigre(row['theta'], row['phi'], row['larm'], data_info, device) for row in test_data], 0) #[N_img, ro+rd, W, H, 3]
    test_images = np.stack([denormalize_image(np.load(row['file_path']), img_width, img_height, row['img_min_max']) for row in test_data], 0) #[N_img, W, H]
    test_images = np.repeat(test_images[:, :, :, None], 3, axis=-1) #[N_img, W, H, 3]
    test_phases = np.array([row['heart_phase'] for row in test_data]) #[N_img]
    test_phases = np.tile(test_phases[:, None, None], (img_width, img_height)) #[N_img, W, H]

    # assumes that there is only one test image
    test_proj_id = test_img_indices[0]
    test_origins, test_directions = torch.Tensor(test_rays[0]) #[W, H, 3]
    test_origins = test_origins.reshape((-1, 3)).to(device) #[W*H, 3]
    test_directions = test_directions.reshape((-1, 3)).to(device) #[W*H, 3]
    test_weighted_pixs = torch.ones((img_width, img_height)).to(device) #[W, H]
    test_initial_intensities = torch.Tensor([max_pixel_value for _ in np.arange(0, test_origins.shape[0])]).to(device)

    test_phase = torch.Tensor(test_phases[0].reshape((-1, 1))).to(device) #[W*H, 1]
    test_img = torch.Tensor(test_images[0,:,:,0]).to(device) #[W*H]
    norm_test_img = (test_img - torch.min(test_img)) / (torch.max(test_img) - torch.min(test_img))

    test_img_ids = np.array([test_proj_id])
    test_img_ids = np.tile(test_img_ids[:, None, None], (img_width, img_height, run_args.depth_samples_per_ray_coarse)) #[N_img, W, H, N_samples]
    test_img_ids = np.reshape(test_img_ids, [-1, 1]) #[W*H]

    test_depth_values_coarse = randomize_depth(depth_values_coarse, device)
    if run_args.depth_samples_per_ray_fine > 0:
        test_depth_values_fine = randomize_depth(depth_values_fine, device)
    

    static_fourier_gaussian = None
    if run_args.temp_pos_enc == 'fourier':
        static_fourier_gaussian = torch.randn([run_args.num_input_channels * run_args.temp_pos_enc_basis])

    temp_fourier_gaussian = None
    if run_args.temp_pos_enc == 'fourier':
        temp_fourier_gaussian = torch.randn([run_args.num_input_channels * run_args.temp_pos_enc_basis])

    temp_params = {
        'num_early_layers': run_args.temp_num_early_layers,
        'num_late_layers': run_args.temp_num_late_layers,
        'num_filters': run_args.temp_num_filters,
        'num_input_channels': run_args.num_input_channels,
        'num_input_times': 1,
        'num_output_channels': run_args.num_output_channels,
        'use_bias': True,
        'use_time_latents': run_args.use_time_latents,
        'num_time_dim': run_args.num_time_dim,
        'pos_enc': run_args.temp_pos_enc,
        'pos_enc_window_start': run_args.temp_pos_enc_window_start,
        'pos_enc_basis': run_args.temp_pos_enc_basis,
        'fourier_sigma': run_args.temp_pos_enc_fourier_sigma,
        'fourier_gaussian': temp_fourier_gaussian,
        'act_func': 'relu',
        'device': device,
    }

    static_params = {
        'num_early_layers': run_args.static_num_early_layers,
        'num_late_layers': run_args.static_num_late_layers,
        'num_filters': run_args.static_num_filters,
        'num_input_channels': run_args.num_input_channels,
        'num_output_channels': run_args.num_output_channels,
        'use_bias': True,
        'pos_enc': run_args.static_pos_enc,
        'pos_enc_window_start': run_args.static_pos_enc_window_start,
        'pos_enc_basis': run_args.static_pos_enc_basis,
        'fourier_sigma': run_args.static_pos_enc_fourier_sigma,
        'fourier_gaussian': static_fourier_gaussian,
        'act_func': 'relu',
        'device': device,
    }

    # setup model
    temp_params = dict(temp_params)
    static_params = dict(static_params)

    temp_model_coarse = Temporal(temp_params)
    temp_model_coarse.to(device)
    
    static_model_coarse = CPPN(static_params)
    static_model_coarse.to(device)

    optimizer_params = list(temp_model_coarse.parameters()) + list(static_model_coarse.parameters())

    temp_model_fine = None
    static_model_fine = None
    if run_args.depth_samples_per_ray_fine > 0:
        temp_params_fine = temp_params.copy()
        temp_params_fine['num_filters'] = run_args.temp_num_filters_fine
        temp_model_fine = Temporal(temp_params_fine)
        temp_model_fine.to(device)

        static_params_fine = static_params.copy()
        static_params_fine['num_filters'] = run_args.static_num_filters_fine
        static_model_fine = CPPN(static_params_fine)
        static_model_fine.to(device)

        optimizer_params += list(temp_model_fine.parameters()) + list(static_model_fine.parameters())

    optimizer = torch.optim.Adam([
        { 'params': optimizer_params, 
         'lr': run_args.lr },
    ], lr=run_args.lr)

    # linearly decay the learning rate
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=run_args.lr_end_factor, total_iters=run_args.lr_decay_steps)

    # weighted training
    mse_criterion = weighted_MSELoss()

    # log all the hyperparameters in a txt file
    if run_args.config is not None:
        f = f'{log_dir}config.json'
        with open(f, 'w') as file:
            file.write(json.dumps(vars(run_args)))

    print('start training...')
    for n_iter in range(run_args.n_iters+1):
        static_model_coarse.train()
        temp_model_coarse.train()

        start_time = time.time()

        if run_args.static_pos_enc == 'nerfies_windowed':
           static_model_coarse.update_windowed_alpha(n_iter, run_args.static_pos_enc_window_decay_steps)
        if run_args.temp_pos_enc == 'nerfies_windowed':
           temp_model_coarse.update_windowed_alpha(n_iter, run_args.temp_pos_enc_window_decay_steps)

        if run_args.static_pos_enc == 'free_windowed':
            static_model_coarse.update_freq_mask_alpha(n_iter, run_args.static_pos_enc_window_decay_steps)

            if run_args.depth_samples_per_ray_fine > 0:
                static_model_fine.update_freq_mask_alpha(n_iter, run_args.static_pos_enc_window_decay_steps)
        if run_args.temp_pos_enc == 'free_windowed':
            temp_model_coarse.update_freq_mask_alpha(n_iter, run_args.temp_pos_enc_window_decay_steps)

            if run_args.depth_samples_per_ray_fine > 0:
                temp_model_fine.update_freq_mask_alpha(n_iter, run_args.temp_pos_enc_window_decay_steps)
        
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
        batch_phases = torch.from_numpy(phases_train[random_ray_ids]).to(device) #(sample_size,depth_samples)

        batch_phases_samples = batch_phases[:,None].repeat(1, run_args.depth_samples_per_ray_coarse)
        
        batch_origins = batch_rays[:,0,:] #(sample_size, 3)
        batch_directions = batch_rays[:,1,:] #(sample_size, 3)
        batch_pix_vals = batch_rays[:,2,:] #(sample_size)
        weighted_pixs = batch_rays[:,3,:] #(sample_size)
        
        batch_pix_vals = batch_pix_vals[:,0] #[sample_size]
        weighted_pixs = weighted_pixs[:,0] #[sample_size]

        # hyperparameter decay
        favor_s_weight = linear_param_decay(n_iter, run_args.favor_s_weight_start, run_args.favor_s_weight_end, run_args.hyperparam_decay_steps, delay_steps=run_args.favor_s_weight_delay_steps)
        dynamic_entro_weight = linear_param_decay(n_iter, run_args.dynamic_entro_weight_start, run_args.dynamic_entro_weight_end, run_args.hyperparam_decay_steps)
        occl_weight = linear_param_decay(n_iter, run_args.occl_weight_start, run_args.occl_weight_end, run_args.hyperparam_decay_steps, delay_steps=run_args.favor_s_weight_delay_steps)
        l1_weight = linear_param_decay(n_iter, run_args.l1_weight_start, run_args.l1_weight_end, run_args.hyperparam_decay_steps)

        batch_initial_intensities = train_initial_intensities.to(device)[:batch_rays.shape[0]]  #[sample_size]
            
        pix_pred_vals_coarse, static_sigma_coarse, temp_sigma_coarse, dists_coarse, \
            pix_pred_vals_fine, static_sigma_fine, temp_sigma_fine, dists_fine = obtain_train_predictions_iter(static_model_coarse, temp_model_coarse, static_model_fine, temp_model_fine, batch_origins, batch_directions, batch_phases_samples, batch_initial_intensities, depth_values_coarse, run_args.output_activation, run_args.batch_size, run_args.depth_samples_per_ray_fine, device)

        # weighted pixel loss (TEMPORARY SET TO 1! == no weighted loss)
        pixel_loss_coarse = mse_criterion(pix_pred_vals_coarse, batch_pix_vals, weighted_pixs).mean()

        # regularization losses
        blendw, sigma_s_max, sigma_d_max, favor_s_loss, static_entropy_loss, static_entropy_sum, \
            dynamic_entropy_loss, dynamic_entropy_sum, dynamic_occl_loss, static_l1_loss, static_l2_loss = compute_losses(static_sigma_coarse, temp_sigma_coarse, dists_coarse, weighted_pixs, run_args)
        loss = pixel_loss_coarse + favor_s_weight*favor_s_loss + dynamic_entro_weight * dynamic_entropy_loss + occl_weight * dynamic_occl_loss + l1_weight * static_l2_loss + l1_weight * static_l1_loss

        pixel_loss_fine = None
        if run_args.depth_samples_per_ray_fine > 0:
            # weighted pixel loss
            pixel_loss_fine = mse_criterion(pix_pred_vals_fine, batch_pix_vals, weighted_pixs_ones).mean()
            # regularization losses
            blendw, sigma_s_max, sigma_d_max, favor_s_loss, static_entropy_loss, static_entropy_sum, \
                dynamic_entropy_loss, dynamic_entropy_sum, dynamic_occl_loss, static_l1_loss, static_l2_loss = compute_losses(static_sigma_fine, temp_sigma_fine, dists_fine, weighted_pixs, run_args)
            loss += pixel_loss_fine + favor_s_weight*favor_s_loss + dynamic_entro_weight * dynamic_entropy_loss + occl_weight * dynamic_occl_loss + l1_weight * static_l2_loss + l1_weight * static_l1_loss     

        psnr = -10. * torch.log10(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (dynamic_entropy_loss < 1e-15 or favor_s_loss < 1e-15) and n_iter >= run_args.static_pos_enc_window_decay_steps:
            print(f'Early stop: dynamic entropy loss: {dynamic_entropy_loss} and favor_loss: {favor_s_loss}')
            break

        if n_iter % run_args.log_every == 0:
            with torch.no_grad():
                log_dict = {
                    "train_loss": loss, 
                    "train_psnr": psnr,
                    "train_pixel_loss_coarse": pixel_loss_coarse,
                    "train_pixel_loss_fine": pixel_loss_fine,
                    "train_blendw": blendw,
                    'train_sigma_s_max': sigma_s_max,
                    'train_sigma_d_max': sigma_d_max,
                    "train_favor_s_loss": favor_s_loss,
                    "train_s_entropy_loss": static_entropy_loss,
                    'train_d_entropy_loss': dynamic_entropy_loss,
                    'train_s_entropy_sum': static_entropy_sum,
                    'train_d_entropy_sum': dynamic_entropy_sum,
                    'train_d_occl_loss': dynamic_occl_loss,
                    'train_s_l1': static_l1_loss,
                    'train_s_l2': static_l2_loss,
                    'favor_s_weight': favor_s_weight,
                    'dynamic_entro_weight': dynamic_entro_weight,
                    'occl_weight': occl_weight,
                    'l1_weight': l1_weight,
                    'train_time': (time.time() - start_time),
                }

                if 'windowed' in run_args.static_pos_enc:
                    log_dict['train_static_windowed'] = static_model_coarse.windowed_alpha
                if 'windowed' in run_args.temp_pos_enc:
                    log_dict['train_temp_windowed'] = temp_model_coarse.windowed_alpha

                wandb.log(log_dict)

        if n_iter % run_args.display_every == 0:
            static_model_coarse.eval()
            temp_model_coarse.eval()

            with torch.no_grad():
                test_query_points = test_origins[..., None, :] + test_directions[..., None, :] * test_depth_values_coarse[..., :, None]
                test_query_points = test_query_points.reshape((-1, 3)).float()

                batch_test_phase = test_phase.repeat(run_args.depth_samples_per_ray_coarse, 1).flatten()
                static_radiance_field_flattened, temp_radiance_field_flattened = get_predictions_composite(static_model_coarse, temp_model_coarse, test_query_points, batch_test_phase, run_args.batch_size)

                unflattened_shape = (img_width*img_height, test_depth_values_coarse.shape[0], static_model_coarse.num_output_channels)
                static_batch_pred_vals_coarse = torch.reshape(static_radiance_field_flattened, unflattened_shape)
                temp_batch_pred_vals_coarse = torch.reshape(temp_radiance_field_flattened, unflattened_shape)

                test_pix_pred_vals_coarse, static_sigma_coarse, temp_sigma_coarse, dists_coarse = render_volume_density_composite(static_batch_pred_vals_coarse, temp_batch_pred_vals_coarse, test_initial_intensities, test_directions, test_depth_values_coarse, run_args.output_activation)

                pred_img_coarse = test_pix_pred_vals_coarse.reshape((img_width, img_height))
                test_pixel_loss_coarse = mse_criterion(pred_img_coarse, test_img, test_weighted_pixs).mean()

                blendw, sigma_s_max, sigma_d_max, favor_s_loss, static_entropy_loss, static_entropy_sum, \
                    dynamic_entropy_loss, dynamic_entropy_sum, dynamic_occl_loss, static_l1_loss, static_l2_loss = compute_losses(static_sigma_coarse, temp_sigma_coarse, dists_coarse, test_weighted_pixs.flatten(), run_args)
                test_loss = test_pixel_loss_coarse + favor_s_weight*favor_s_loss + dynamic_entro_weight * dynamic_entropy_loss + occl_weight * dynamic_occl_loss + l1_weight * static_l2_loss + l1_weight * static_l1_loss

                test_pixel_loss_fine = None
                if run_args.depth_samples_per_ray_fine > 0:
                    test_query_points = test_origins[..., None, :] + test_directions[..., None, :] * test_depth_values_fine[..., :, None]
                    test_query_points = test_query_points.reshape((-1, 3)).float()

                    batch_test_phase = test_phase.repeat(run_args.depth_samples_per_ray_fine, 1).flatten()
                    static_radiance_field_flattened, temp_radiance_field_flattened = get_predictions_composite(static_model_fine, temp_model_fine, test_query_points, batch_test_phase, run_args.batch_size)

                    unflattened_shape = (img_width*img_height, test_depth_values_fine.shape[0], static_model_fine.num_output_channels)
                    static_batch_pred_vals_fine = torch.reshape(static_radiance_field_flattened, unflattened_shape)
                    temp_batch_pred_vals_fine = torch.reshape(temp_radiance_field_flattened, unflattened_shape)

                    test_pix_pred_vals_fine, static_sigma_fine, temp_sigma_fine, dists_fine = render_volume_density_composite(static_batch_pred_vals_fine, temp_batch_pred_vals_fine, test_initial_intensities, test_directions, test_depth_values_fine, run_args.output_activation)

                    pred_img_fine = test_pix_pred_vals_fine.reshape((img_width, img_height))
                    test_pixel_loss_fine = mse_criterion(pred_img_fine, test_img, test_weighted_pixs).mean()

                    blendw, sigma_s_max, sigma_d_max, favor_s_loss, static_entropy_loss, static_entropy_sum, \
                        dynamic_entropy_loss, dynamic_entropy_sum, dynamic_occl_loss, static_l1_loss, static_l2_loss = compute_losses(static_sigma_fine, temp_sigma_fine, dists_fine, test_weighted_pixs.flatten(), run_args)
                    test_loss += test_pixel_loss_fine + favor_s_weight*favor_s_loss + dynamic_entro_weight * dynamic_entropy_loss + occl_weight * dynamic_occl_loss + l1_weight * static_l2_loss + l1_weight * static_l1_loss

                test_psnr = -10. * torch.log10(test_loss)

                # log in weights and biases
                wandb.log({
                    "test_loss": test_loss, 
                    "test_psnr": test_psnr,
                    "test_pixel_loss_coarse": test_pixel_loss_coarse,
                    "test_pixel_loss_fine": test_pixel_loss_fine,
                    "test_favor_s_loss": favor_s_loss,
                    "test_blendw": blendw,
                    "test_s_entropy_loss": static_entropy_loss,
                    "test_d_entropy_loss": dynamic_entropy_loss,
                })

                pred_img_coarse = (pred_img_coarse - torch.min(pred_img_coarse)) / (torch.max(pred_img_coarse) - torch.min(pred_img_coarse))
                
                temp_test_pix_pred_vals, _, _ = render_volume_density(temp_batch_pred_vals_coarse, test_initial_intensities, test_directions, test_depth_values_coarse, run_args.output_activation)
                pred_img_coarse_temp = temp_test_pix_pred_vals.float().reshape((img_width, img_height))
                pred_img_coarse_temp = (pred_img_coarse_temp - torch.min(pred_img_coarse_temp)) / (torch.max(pred_img_coarse_temp) - torch.min(pred_img_coarse_temp))

                static_test_pix_pred_vals, _, _ = render_volume_density(static_batch_pred_vals_coarse, test_initial_intensities, test_directions, test_depth_values_coarse, run_args.output_activation)
                pred_img_coarse_static = static_test_pix_pred_vals.float().reshape((img_width, img_height))
                pred_img_coarse_static = (pred_img_coarse_static - torch.min(pred_img_coarse_static)) / (torch.max(pred_img_coarse_static) - torch.min(pred_img_coarse_static))

                image_log = {
                    'prediction_coarse': wandb.Image(pred_img_coarse),
                    'original_coarse': wandb.Image(norm_test_img),
                    'difference_coarse': wandb.Image(torch.abs(pred_img_coarse-norm_test_img)),
                    'dynamic_coarse': wandb.Image(pred_img_coarse_temp),
                    'static_coarse': wandb.Image(pred_img_coarse_static),
                }

                if run_args.depth_samples_per_ray_fine > 0:
                    pred_img_fine = (pred_img_fine - torch.min(pred_img_fine)) / (torch.max(pred_img_fine) - torch.min(pred_img_fine))
                
                    temp_test_pix_pred_vals, _, _ = render_volume_density(temp_batch_pred_vals_fine, test_initial_intensities, test_directions, test_depth_values_fine, run_args.output_activation)
                    pred_img_fine_temp = temp_test_pix_pred_vals.float().reshape((img_width, img_height))
                    pred_img_fine_temp = (pred_img_fine_temp - torch.min(pred_img_fine_temp)) / (torch.max(pred_img_fine_temp) - torch.min(pred_img_fine_temp))

                    static_test_pix_pred_vals, _, _ = render_volume_density(static_batch_pred_vals_fine, test_initial_intensities, test_directions, test_depth_values_fine, run_args.output_activation)
                    pred_img_fine_static = static_test_pix_pred_vals.float().reshape((img_width, img_height))
                    pred_img_fine_static = (pred_img_fine_static - torch.min(pred_img_fine_static)) / (torch.max(pred_img_fine_static) - torch.min(pred_img_fine_static))

                    image_log_fine = {
                        'prediction_fine': wandb.Image(pred_img_fine),
                        'original_fine': wandb.Image(norm_test_img),
                        'difference_fine': wandb.Image(torch.abs(pred_img_fine-norm_test_img)),
                        'dynamic_fine': wandb.Image(pred_img_fine_temp),
                        'static_fine': wandb.Image(pred_img_fine_static)
                    }

                    image_log = image_log | image_log_fine

                wandb.log(image_log)

                print("Iteration:", n_iter)
                print("Loss coarse:", test_loss.item())
                print("PSNR coarse:", test_psnr.item())

                if n_iter % run_args.save_every == 0:
                    try:
                        temp_model_coarse.save(f'{log_dir}tempmodel-coarse.pth', {})
                        static_model_coarse.save(f'{log_dir}staticmodel-coarse.pth', {})

                        if run_args.depth_samples_per_ray_fine > 0:
                            temp_model_fine.save(f'{log_dir}tempmodel-fine.pth', {})
                            static_model_fine.save(f'{log_dir}staticmodel-fine.pth', {})
                    except Exception as e:
                        print(e)
                        print('error saving model')
                        break

if __name__ == "__main__":
    wandb.login()

    parser = config_parser()
    run_args = parser.parse_args()

    project_name = '4D-LIMITED'

    if run_args.only_prepare_data:
        print('NOT TRAINING, JUST PREPROCESSING DATA')
        generate_data(run_args)
    else:
        if run_args.use_wandb:
            with open(run_args.wandb_sweep_yaml, 'r') as f:
                sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)

            sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
            wandb.agent(sweep_id, function=main)
        else:
            main()