import torch

def randomize_depth(z_vals, device):
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape).to(device)
    z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(device)

    return depth_values
    
def get_minibatches(inputs, chunksize=1024*8):
  r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
  """
  return [[inputs[i:i + chunksize]] for i in range(0, inputs.shape[0], chunksize)]

def get_minibatches_time(inputs, time_inputs, chunksize=1024*8):
  r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
  """
  return [[inputs[i:i + chunksize], time_inputs[i:i+chunksize]] for i in range(0, inputs.shape[0], chunksize)]

def get_predictions_static(static_model, flattened_query_points, chunksize):
  batches = get_minibatches(flattened_query_points, chunksize=chunksize)

  static_preds = []
  for i, batch in enumerate(batches):
    query_points = batch[0]

    static_point_pred = static_model(query_points)
    static_preds.append(static_point_pred)
  
  static_radiance_field_flattened = torch.cat(static_preds, dim=0)
  return static_radiance_field_flattened

def get_predictions_composite(static_model, temp_model, flattened_query_points, flattened_time_points, chunksize, use_nerf_acc=False):
  batches = get_minibatches_time(flattened_query_points, flattened_time_points, chunksize=chunksize)

  static_preds = []
  temp_preds = []
  static_point_pred = None
  for i, batch in enumerate(batches):
      query_points, time_points = batch

      if not use_nerf_acc:
        static_point_pred = static_model(query_points)
      
      temp_point_pred = temp_model.forward_composite(query_points, time_points)

      static_preds.append(static_point_pred)
      temp_preds.append(temp_point_pred)

  static_radiance_field_flattened = torch.cat(static_preds, dim=0)
  temp_radiance_field_flattened = torch.cat(temp_preds, dim=0)

  return static_radiance_field_flattened, temp_radiance_field_flattened

def get_activation_func(output_activation):
  activation_func = torch.nn.Sigmoid()
  if output_activation == 'softplus':
    activation_func = torch.nn.Softplus()
  elif output_activation == 'clamp':
    activation_func = lambda x : torch.nn.functional.hardtanh(torch.nn.Softplus()(x), min_val=0., max_val=1.)
  
  return activation_func

def render_volume_density_composite(static_radiance_field, temp_radiance_field, initial_intensities, ray_directions, depth_values, output_activation='softplus', scale_value=1e-2):
  one_e_10 = torch.tensor([1e-10], dtype=ray_directions.dtype, device=ray_directions.device)
  dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

  activation_func = get_activation_func(output_activation)
  static_sigma = activation_func(static_radiance_field[..., -1]) * scale_value
  temp_sigma = activation_func(temp_radiance_field[...,-1]) * scale_value

  weights = (static_sigma + temp_sigma) * dists
  # int_map = torch.sum(weights, dim=-1)
  int_map = initial_intensities - torch.sum(weights, dim=-1)

  return int_map, static_sigma, temp_sigma, dists

def render_volume_density(radiance_field, initial_intensities, ray_directions, depth_values, output_activation='softplus', scale_value=1e-2):
  one_e_10 = torch.tensor([1e-10], dtype=ray_directions.dtype, device=ray_directions.device)
  dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
  
  activation_func = get_activation_func(output_activation)
  sigma_a = activation_func(radiance_field[...,-1]) 
  weights = sigma_a * dists * scale_value

  # ASSUMES THAT map prediction and initial ints are in logarithmic space
  int_map = initial_intensities - torch.sum(weights, dim=-1)

  return int_map, sigma_a, dists

def obtain_train_predictions_static(static_model, batch_origins, batch_directions, batch_initial_intensities, depth_values, output_activation, batch_size, device):
    batch_depth_values = randomize_depth(depth_values, device) #[depth_samples]
    batch_query_points = batch_origins[..., None, :] + batch_directions[..., None, :] * batch_depth_values[..., :, None] #[sample_size, depth_samples, 3]
    
    n_query_points = batch_query_points.reshape((-1, 3)).float() #[sample_size*depth_samples, 3]
    query_points = n_query_points.to(device)

    static_radiance_field_flattened = get_predictions_static(static_model, query_points, batch_size)

    unflattened_shape = list(batch_query_points.shape[:-1]) + [static_model.num_output_channels]
    static_batch_pred_vals = torch.reshape(static_radiance_field_flattened, unflattened_shape)

    pix_pred_vals, static_sigma, dists = render_volume_density(static_batch_pred_vals, batch_initial_intensities, batch_directions, batch_depth_values, output_activation)

    return pix_pred_vals, static_sigma, dists

def obtain_train_predictions_iter(static_model_coarse, temp_model_coarse, static_model_fine, temp_model_fine, batch_origins, batch_directions, batch_phases, 
                                  batch_initial_intensities, depth_values, output_activation, batch_size, depth_samples_per_ray_fine, device):
    batch_depth_values = randomize_depth(depth_values, device) #[depth_samples]
    batch_query_points = batch_origins[..., None, :] + batch_directions[..., None, :] * batch_depth_values[..., :, None] #[sample_size, depth_samples, 3]
    
    n_query_points = batch_query_points.reshape((-1, 3)).float() #[sample_size*depth_samples, 3]
    query_points = n_query_points.to(device)
    batch_phases_coarse = batch_phases.flatten().int()

    static_radiance_field_flattened, temp_radiance_field_flattened = get_predictions_composite(static_model_coarse, temp_model_coarse, query_points, batch_phases_coarse, batch_size)
    unflattened_shape = list(batch_query_points.shape[:-1]) + [temp_model_coarse.num_output_channels]
    static_batch_pred_vals = torch.reshape(static_radiance_field_flattened, unflattened_shape)
    temp_batch_pred_vals = torch.reshape(temp_radiance_field_flattened, unflattened_shape)

    pix_pred_vals_coarse, static_sigma_coarse, temp_sigma_coarse, dists_coarse = render_volume_density_composite(static_batch_pred_vals, temp_batch_pred_vals, batch_initial_intensities, batch_directions, batch_depth_values, output_activation)

    # fine sampling
    pix_pred_vals_fine, static_sigma_fine, temp_sigma_fine, dists_fine = None, None, None, None
    if depth_samples_per_ray_fine > 0:
        depth_samples_per_ray_total = batch_depth_values.shape[0] + depth_samples_per_ray_fine
        
        # static
        eps = torch.ones_like(static_sigma_coarse[:, :1]) * 1e-10
        weights = torch.cat([eps, torch.abs((static_sigma_coarse[:, 1:] + temp_sigma_coarse[:, 1:]) - (static_sigma_coarse[:, :-1] + temp_sigma_coarse[:, :-1]))], dim=-1)
        weights = weights / torch.max(weights)

        batch_depth_values = batch_depth_values[None, :].repeat(pix_pred_vals_coarse.shape[0], 1) #[img_sample_size, depth_vals_coarse]

        depth_vals_mid = .5 * (batch_depth_values[...,1:] + batch_depth_values[...,:-1])
        pdf_depth_values = sample_pdf(depth_vals_mid, weights[..., 1:-1], depth_samples_per_ray_fine, device)
        depth_vals_fine, _ = torch.sort(torch.cat([pdf_depth_values, batch_depth_values.detach()], -1), -1)
        fine_query_points = batch_origins[..., None, :] + batch_directions[..., None, :] * depth_vals_fine[..., :, None]
        fine_query_points = fine_query_points.reshape((-1, 3)).float()

        # remove the pixel value depth [img_sample_size, depth_samples_per_ray ] -> [depth_samples_per_ray]
        depth_vals_fine = depth_vals_fine[0, :]

        # reshape the batch phases somehow
        fine_batch_phases = batch_phases[:, 0, None].repeat(1, depth_samples_per_ray_total).flatten()
        static_radiance_field_flattened, temp_radiance_field_flattened = get_predictions_composite(static_model_fine, temp_model_fine, fine_query_points, fine_batch_phases, batch_size)
        unflattened_shape = [pix_pred_vals_coarse.shape[0], depth_samples_per_ray_total, temp_model_coarse.num_output_channels]
        static_batch_pred_vals = torch.reshape(static_radiance_field_flattened, unflattened_shape)
        temp_batch_pred_vals = torch.reshape(temp_radiance_field_flattened, unflattened_shape)
        pix_pred_vals_fine, static_sigma_fine, temp_sigma_fine, dists_fine = render_volume_density_composite(static_batch_pred_vals, temp_batch_pred_vals, batch_initial_intensities, batch_directions, depth_vals_fine, output_activation)

    return pix_pred_vals_coarse, static_sigma_coarse, temp_sigma_coarse, dists_coarse, pix_pred_vals_fine, static_sigma_fine, temp_sigma_fine, dists_fine

def sample_pdf(bins, weights, N_samples, device):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], dim=-1) # (batch, len(bins))

    # Take uniform samples
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights)

    # Use inverse inverse transform sampling to sample the depths
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def compute_ratio(sigma_s, sigma_d, favor_s_opt=None, sigma_s_max=None, sigma_d_max=None, weight_max=0.05):
  sigma_sc = sigma_s
  sigma_dc = sigma_d

  with torch.no_grad():
    sigma_s_max = torch.max(sigma_s)
    sigma_d_max = torch.max(sigma_d)

  blendw = sigma_dc / (sigma_sc+sigma_dc+1e-10)
  return blendw, sigma_s_max, sigma_d_max

def compute_blendw_loss(blendw, clip_threshold=1e-19, skewness=1):
   blendw = torch.clip(blendw**skewness, min=clip_threshold, max=1-clip_threshold)
   rev_blendw = torch.clip(1-blendw, min=clip_threshold)
   entropy = torch.mean(-(blendw * torch.log(blendw) + rev_blendw * torch.log(rev_blendw)), dim=-1)
   return entropy.mean()

def compute_sigma_s_ray_loss(sigma_s, dists, mask_threshold=0.1, clip_threshold=1e-19, use_weighting=False, weighted_pixs=[], weighted_thresh=0.25):
  sigma_dist = sigma_s * dists
  sigma_s_sum = torch.sum(sigma_dist, dim=-1, keepdim=True)

  mask = torch.where(sigma_s_sum < mask_threshold, 0., 1.).flatten().int()
  
  # for the pixels part of the high variance mask (calculated based on motion)
  # we calculate the entropy w.r.t. a different threshold
  # this is an approach to avoid them nearing zero in entropy
  if len(weighted_pixs) > 0 and use_weighting:
    weighted_mask = torch.zeros(mask.shape).to(mask.device).int()
    # weighted pixs value is from [1, ...]
    weighted_mask[:weighted_pixs.shape[0]] = torch.where(weighted_pixs > 1 + weighted_thresh, 1., 0.).int()
    mask = torch.bitwise_or(weighted_mask, mask)

  ray_p = sigma_dist / torch.clip(sigma_s_sum, min=clip_threshold)

  entropy = mask * -torch.sum(ray_p * torch.log(ray_p + 1e-10), dim=-1)
  return entropy.mean(), sigma_s_sum.mean()

def compute_occl_loss(sigma_s, dists, reg_perc=0.1, use_back=False):
  '''
  Calculates the occlusion near the front (and back) of the cameras;
  '''

  # defines a threshold based on the total distance, e.g. 10% of the ray start
  cum_dists = torch.cumsum(dists, dim=0).unsqueeze(dim=0).repeat((sigma_s.shape[0], 1))
  
  # reg perc defines the percentage of the ray that we use to calculate the occlusion from
  dists_range_front = reg_perc * cum_dists[-1, -1]
  dists_range_back = (1-reg_perc) * cum_dists[-1, -1]

  mask_front = torch.where(cum_dists < dists_range_front, 1., 0.).int()

  mask_back = torch.ones(mask_front.shape).to(cum_dists.device)
  if use_back:
    mask_back = torch.where(cum_dists > dists_range_back, 1., 0.)
  mask_back = mask_back.int()

  mask = torch.bitwise_or(mask_front, mask_back)

  loss = torch.sum(sigma_s * dists * mask, dim=-1)
  return loss.mean()

def compute_losses(static_sigma, temp_sigma, dists, weighted_pixs, run_args):
  #https://github.com/ChikaYan/d2nerf/blob/main/hypernerf/training.py#L178
  blendw, sigma_s_max, sigma_d_max = compute_ratio(static_sigma, temp_sigma, run_args.favor_s_opt)
  favor_s_loss = compute_blendw_loss(blendw, skewness=run_args.skewness_val)
  static_entropy_loss, static_entropy_sum = compute_sigma_s_ray_loss(static_sigma, dists, mask_threshold=run_args.entro_mask_thre)
  dynamic_entropy_loss, dynamic_entropy_sum = compute_sigma_s_ray_loss(temp_sigma, dists, mask_threshold=run_args.entro_mask_thre, use_weighting=run_args.entro_use_weighting, weighted_pixs=weighted_pixs, weighted_thresh=run_args.entro_weighted_thresh)
  dynamic_occl_loss = compute_occl_loss(temp_sigma, dists, run_args.occl_reg_perc)

  static_l1_loss = torch.sum(static_sigma*dists, dim=-1).sum()
  static_l2_loss = torch.sum((static_sigma*dists)**2, dim=-1).sum()

  return blendw.mean(), sigma_s_max, sigma_d_max, favor_s_loss, static_entropy_loss, static_entropy_sum, \
    dynamic_entropy_loss, dynamic_entropy_sum, dynamic_occl_loss, static_l1_loss, static_l2_loss

def linear_param_decay(curr_iter, start_weight, end_weight, steps, delay_steps=0):
  if curr_iter < delay_steps:
     return 0
  
  alpha = min((curr_iter - delay_steps) / steps, 1.0)
  return (1.0 - alpha) * start_weight + alpha * end_weight

def exp_param_decay(curr_iter, start_weight, end_weight, steps, delay_steps=0):
   if curr_iter < delay_steps:
      return 0
   
   if start_weight == end_weight:
      return start_weight
   
   base = end_weight / start_weight
   exponent = curr_iter / (steps - 1)
   if curr_iter >= steps:
      return end_weight
   return start_weight * base**exponent

class weighted_MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gts, weights):
        return ((preds - gts)**2) * weights