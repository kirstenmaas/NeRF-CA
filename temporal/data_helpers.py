import numpy as np
import torch
from datetime import datetime
import wandb
import os
import copy

from proj_helpers import get_ray_values_tigre

def config_parser(config_file='temporal/composite.txt', sweep_file='temporal/sweep-composite.yaml'):

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path', default=config_file)
    parser.add_argument('--wandb_sweep_yaml', type=str, default=sweep_file)
    parser.add_argument('--use_wandb', default=True, type=lambda x: (str(x).lower() == 'true'))

    # general run info
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--x_ray_type", type=str, default='roadmap')
    parser.add_argument('--take_mask', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data_size', type=int)

    # data args
    parser.add_argument('--use_experiment_name', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument('--data_limited_range', type=float)
    parser.add_argument('--data_step_size', type=float)
    parser.add_argument('--data_numb_angles', type=int, default=None)
    parser.add_argument('--data_time_range_start', type=int)
    parser.add_argument('--data_time_range_end', type=int)
    parser.add_argument('--data_limited_range_test', type=int, default=None)
    parser.add_argument('--data_step_size_test', type=float, default=None)

    parser.add_argument('--only_prepare_data', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--debug_mode', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--visualize_geometry', default=False, type=lambda x: (str(x).lower() == 'true'))

    # run info
    parser.add_argument('--n_iters', type=int)
    parser.add_argument('--display_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32768)

    # models
    parser.add_argument('--num_input_channels', type=int, default=3)
    parser.add_argument('--num_output_channels', type=int, default=1)
    parser.add_argument('--temp_num_early_layers', type=int, default=4)
    parser.add_argument('--temp_num_late_layers', type=int, default=0)
    parser.add_argument('--temp_num_filters', type=int, default=32)
    parser.add_argument('--temp_num_filters_fine', type=int, default=32)
    parser.add_argument('--static_num_early_layers', type=int, default=4)
    parser.add_argument('--static_num_late_layers', type=int, default=0)
    parser.add_argument('--static_num_filters', type=int, default=32)
    parser.add_argument('--static_num_filters_fine', type=int, default=32)
    parser.add_argument('--output_activation', type=str, default='Softplus')
    
    # parameters nerf
    parser.add_argument('--depth_samples_per_ray_coarse', type=int)
    parser.add_argument('--depth_samples_per_ray_fine', type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_end_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_steps', type=int, default=100000)

    parser.add_argument('--sample_mode', type=str, default='pixel')
    parser.add_argument('--sample_weights_name', type=str, default=None)
    parser.add_argument('--img_sample_size', type=int, default=64**2)
    parser.add_argument('--var_sample_perc', type=float, default=0.)
    parser.add_argument('--var_sample_thre', type=float, default=0.)
    parser.add_argument('--raw_noise_std', type=float, default=0)

    # positional encoding temp
    parser.add_argument('--temp_pos_enc', type=str)
    parser.add_argument('--temp_pos_enc_basis', type=int)
    parser.add_argument('--temp_pos_enc_fourier_sigma', type=int)
    parser.add_argument('--temp_pos_enc_window_start', type=int, default=0)
    parser.add_argument('--temp_pos_enc_window_decay_steps', type=int)

    parser.add_argument('--static_pos_enc', type=str)
    parser.add_argument('--static_pos_enc_basis', type=int)
    parser.add_argument('--static_pos_enc_fourier_sigma', type=int)
    parser.add_argument('--static_pos_enc_window_start', type=int, default=0)
    parser.add_argument('--static_pos_enc_window_decay_steps', type=int)

    # positional encoding windowing
    parser.add_argument('--window_weight_start', type=int, default=0)
    parser.add_argument('--window_weight_end', type=int, default=10)
    parser.add_argument('--window_decay_steps', type=int, default=100000)

    # time latents
    parser.add_argument("--use_time_latents", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--num_time_dim", type=int)

    # D^2 NeRF parameters
    parser.add_argument('--skewness_val', type=float, default=1.)

    parser.add_argument('--favor_s_weight_start', type=float)
    parser.add_argument('--favor_s_weight_end', type=float)
    parser.add_argument('--favor_s_weight_delay_steps', type=int)

    parser.add_argument('--dynamic_entro_weight_start', type=float)
    parser.add_argument('--dynamic_entro_weight_end', type=float)

    parser.add_argument('--occl_weight_start', type=float)
    parser.add_argument('--occl_weight_end', type=float)

    parser.add_argument('--l1_weight_start', type=float)
    parser.add_argument('--l1_weight_end', type=float)

    parser.add_argument('--hyperparam_decay_steps', type=int)
    
    
    parser.add_argument('--entro_mask_thre', type=float)
    parser.add_argument('--entro_use_weighting', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--entro_weighted_thresh', type=float, default=0.)

    parser.add_argument('--occl_reg_perc', type=float)

    parser.add_argument('--weighted_loss_max', type=float)
    parser.add_argument('--weighted_loss', type=str, default='None')

    parser.add_argument('--favor_s_opt', type=str)
    parser.add_argument('--favor_s_opt_weight', type=float)

    return parser

def denormalize_image(image, img_width, img_height, img_min_max):
    # transpose image!
    image = image.reshape((img_width, img_height)).T

    # image is normalized
    if int(np.min(image)) == 0 and int(np.max(image)) == 1:
        denormalized_image = image * (img_min_max[1] - img_min_max[0]) + img_min_max[0]
    else:
        denormalized_image = image
    
    return denormalized_image

def prepare_data_for_loader_tigre(data, geo_info, img_width, img_height, depth_samples_per_ray, weighted_loss_max, device, use_weighting=True):
    # load the train data
    rays = np.stack([get_ray_values_tigre(row['theta'], row['phi'], row['larm'], geo_info, device) for row in data], 0) #[N_img, ro+rd, W, H, 3]

    images = np.stack([denormalize_image(np.load(row['file_path']), img_width, img_height, row['img_min_max']) for row in data], 0) #[N_img, W, H]

    images = np.repeat(images[:, None, :, :, None], 3, axis=-1) #[N_img, 1, W, H, 3]

    weighted_images = np.ones((images.shape[0], img_width, img_height))
    if use_weighting:
        weighted_images = np.stack([np.load(row['weighted_file_path']).reshape((img_width, img_height)).T for row in data], 0) #[N_img, W, H]
    
    # define the strength of the weighted images in computing the mse loss, go from [1,2] to [1, weighted_loss_max+1]
    weighted_images = (weighted_images - 1) * weighted_loss_max + 1
    weighted_images = np.repeat(weighted_images[:, None, :, :, None], rays.shape[-1], axis=-1) #[N_img, 1, W, H, 3]

    heart_phases = np.array([row['heart_phase'] for row in data]) #[N_img]
    heart_phases = np.tile(heart_phases[:, None, None], (img_width, img_height)) #[N_img, W, H]
    phases_train = np.reshape(heart_phases, [-1]) #[N_img*W*H]

    rays_train = np.concatenate([rays, images, weighted_images], 1) #[N_img, ro+rd+img+wimg, W, H, 3]
    rays_train = np.transpose(rays_train, [0,2,3,1,4]) #[N_img, W, H, ro+rd+img+wimg, 3]
    rays_train = np.reshape(rays_train, [-1, rays_train.shape[-2], rays_train.shape[-1]]) #[N_img*W*H, ro+rd+img+img_w, 3]

    return rays_train, phases_train

def create_depth_values(near_thresh, far_thresh, depth_samples_per_ray_coarse, device):
    t_vals = torch.linspace(0., 1., depth_samples_per_ray_coarse)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)
    depth_values = z_vals.to(device)
    return depth_values

def initialize_wandb(extra_chars=''):
    exp_name = datetime.now().strftime("%Y-%m-%d-%H%M") + extra_chars
    wandb.init(
        notes=exp_name,
        # mode='offline'
    )
    return exp_name

def initialize_save_folder(folder_name, exp_name):
    log_dir = folder_name + 'runs/' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir

def overwrite_args_wandb(run_args, wandb_args):
    # we want to overwrite the args based on the sweep args
    new_args = copy.deepcopy(run_args)
    for key in wandb_args.keys():
        setattr(new_args, key, wandb_args[key])
    
    return new_args