import torch
import torch.nn as nn
import numpy as np

class Temporal(nn.Module):
    def __init__(self, model_definition: dict) -> None:
        super().__init__()
        self.version = "v0.00"
        self.model_definition = model_definition
        # self.model_definition_nerf = model_definition_nerf
        self.device = model_definition['device']

        # getting the parameters
        self.num_early_layers = model_definition['num_early_layers']
        self.num_late_layers = model_definition['num_late_layers']
        self.num_filters = model_definition['num_filters']
        self.num_input_channels = model_definition['num_input_channels'] # x,y,z
        self.num_input_times = model_definition['num_input_times'] #t
        self.num_output_channels = model_definition['num_output_channels']
        self.use_bias = model_definition['use_bias']
        self.use_time_latents = model_definition['use_time_latents']
        self.act_func = model_definition['act_func']

        if self.use_time_latents:
            self.num_time_dim = model_definition['num_time_dim']
            self.fixed_frame_ids = torch.arange(0, 10) #for CAVAREV
            # self.fixed_frame_ids = np.setdiff1d(self.unique_frame_ids, self.trainable_frame_ids) #for angio
            self.time_latents = nn.Parameter(torch.rand((self.fixed_frame_ids.shape[0], self.num_time_dim)))

        self.first_act_func = nn.ReLU()
        self.act_func = nn.ReLU()
        
        self.use_pos_enc = model_definition['pos_enc']

        self.first_act_func = nn.ReLU()
        self.act_func = nn.ReLU()

        self.input_features_pts = self.num_input_channels
        self.input_features_time = self.num_input_times

        if self.use_pos_enc != 'none':
            self.pos_enc_basis = model_definition['pos_enc_basis']
            self.pos_enc_window_start = model_definition['pos_enc_window_start']
            self.input_features_pts = self.num_input_channels + self.num_input_channels * 2 * self.pos_enc_basis

            if self.use_pos_enc == 'fourier':
                self.input_features_pts = self.num_input_channels * 2 * self.pos_enc_basis
                self.fourier_sigma = model_definition['fourier_sigma']
                self.fourier_coefficients = (model_definition['fourier_gaussian'] * self.fourier_sigma).to(self.device)

            self.input_features_time = self.num_input_times
            self.windowed_alpha = 0
        
        self.input_features = self.input_features_pts + self.input_features_time
        if self.use_time_latents:
            self.input_features = self.input_features_pts + self.num_time_dim

        # model understanding API
        self.store_activations = False
        self.activation_dictionary = {}

        self.create_time_net()

    def create_time_net(self):
        input_features = self.input_features
        num_filters = self.num_filters
        use_bias = self.use_bias
        num_output_channels = self.num_output_channels

        # creating the learnable blocks
        early_pts_layers = []
        # input layer
        early_pts_layers += self.__create_layer(input_features, num_filters,
                                           use_bias, activation=self.first_act_func)
        # hidden layers: early
        for _ in range(self.num_early_layers):
            early_pts_layers += self.__create_layer(num_filters, num_filters,
                                               use_bias, activation=self.act_func)

        self.early_pts_layers = nn.ModuleList(early_pts_layers)

        # skip connection
        if self.num_late_layers > 0:
            self.skip_connection = self.__create_layer(num_filters + input_features, num_filters,
                                                use_bias, activation=self.act_func)

            late_pts_layers = []
            for _ in range(self.num_late_layers - 1):
                late_pts_layers += self.__create_layer(num_filters, num_filters,
                                                use_bias, activation=self.act_func)

            self.late_pts_layers = nn.ModuleList(late_pts_layers)
        # output layer
        self.output_linear = self.__create_layer(num_filters, num_output_channels,
                                        use_bias, activation=None)

    @staticmethod
    def __create_layer(num_in_filters: int, num_out_filters: int,
                       use_bias: bool, activation=nn.ReLU(), dropout=0.5) -> nn.Sequential:
        block = []
        block.append(nn.Linear(num_in_filters, num_out_filters, bias=use_bias)) # Dense layer
        if activation:
            block.append(activation)
            # block.append(nn.Dropout(dropout))
        block = nn.Sequential(*block)

        return block

    def activations(self, store_activations: bool) -> None:
        """
        Configure the model to retain or discard the activations during the forward pass

        Args:
            activations (bool): keep/discard the activations during inference
        """

        self.store_activations = store_activations

        if not store_activations:
            self.activation_dictionary = {}

    def query_time(self, xs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        input_pts = xs
        time_pts = ts

        pos_enc = self.use_pos_enc

        pts_encoded = input_pts
        if pos_enc != 'none':
            pts_encoded = self.pos_enc(input_pts, self.pos_enc_basis)

        time_encoded = time_pts
        values = torch.cat([pts_encoded, time_encoded], dim=-1)
        for _, pts_layer in enumerate(self.early_pts_layers):
            values = pts_layer(values)

        if self.num_late_layers > 0:
            values = self.skip_connection(torch.cat([pts_encoded, time_encoded, values], dim=-1))

            for _, pts_layer in enumerate(self.late_pts_layers):
                values = pts_layer(values)
        else:
            outputs = self.output_linear(values)

        return outputs
    
    #def forward_composite(self, x: torch.Tensor, ts: torch.Tensor, ts_ids: torch.Tensor, ts_cycles: torch.Tensor) -> torch.Tensor:
    def forward_composite(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        input_pts = x
        time_pts = ts

        if self.use_time_latents:
            # to integers
            ts_int = time_pts.flatten()

            # get latent vectors based on ids (for when no interpolation is needed)
            learned_time_pts = self.time_latents[ts_int.long()]

        outputs = self.query_time(input_pts, learned_time_pts)

        return outputs
    
    def interpolate_time_latents(self, learned_time_pts, ts_int, ts_interp, max_frame_nb):
        unique_frame_ids = torch.Tensor(self.fixed_frame_ids).to(self.device)
        ts_int_repeat = ts_int.unsqueeze(-1).repeat(1, max_frame_nb+1).double()
        unique_frame_ids_repeat = unique_frame_ids.unsqueeze(0).repeat(ts_int_repeat.shape[0], 1)
        
        # reduce the integer times to maintain within the cycle time
        cyclic_ts_int_repeat = ts_int_repeat % (max_frame_nb+1)

        # get the nearest integer time for the time (which may not be integer because of learned offset)
        # we need to find two ids (floor & ceil), the first one should be correct from here
        ts_closest, ts_closest_id = torch.sort(torch.abs(cyclic_ts_int_repeat - unique_frame_ids_repeat), dim=-1)

        # check if we get numbers beyond the cycle
        ts_cycl = torch.argwhere(ts_closest[:,1] > 1).flatten()
        # ts_cycl = torch.argwhere(ts_closest[:,1] > 1).flatten()
        # print('ts_cycl', ts_cycl)

        # cyclic trainable frame ids (we restart at 0)
        # TODO: not sure if this logic is 100% correct for all cases
        if len(ts_cycl) > 0:
            # if value is minus we take the "1" of the cycle, otherwise we take the "0" of the cycle
            # the second id (for the ceil func) is incorrect when we go outside of the cycle, so we find it based on the remainder
            ts_int_repeat[ts_cycl] = ts_int_repeat[ts_cycl].double() % ( max_frame_nb + 1)

            _, cyclic_ts_closest_id = torch.sort(torch.abs(ts_int_repeat[ts_cycl] - max_frame_nb - 1 - unique_frame_ids_repeat[ts_cycl]), dim=-1, descending=False)
            ts_closest_id[ts_cycl, 1] = cyclic_ts_closest_id[:, 0]

        # get the final floor and ceiling pts
        ts_closest_pts = unique_frame_ids[ts_closest_id[ts_interp, :2]]
        # print('ts_closest_pts', ts_closest_pts)

        # interpolate
        start_pts = self.time_latents[ts_closest_pts[:,0].long()]
        end_pts = self.time_latents[ts_closest_pts[:,1].long()]

        weights = (ts_int[ts_interp].flatten() % (max_frame_nb+1) - ts_closest_pts[:,0]).unsqueeze(-1).repeat(1, start_pts.shape[1])
        # print('weights', weights)
        interpolated = torch.lerp(start_pts, end_pts, weights)

        # update only interpolated pts
        learned_time_pts[ts_interp] = interpolated

        return learned_time_pts

    def pos_enc(self, values, pos_enc_basis):
        input_values = values
        if pos_enc_basis > 0:
            if self.use_pos_enc == 'fourier':
                
                basis_values = torch.cat(pos_enc_basis * [input_values], dim=-1)
                value = 2 * np.pi * basis_values * self.fourier_coefficients
                fin_values = torch.cat([torch.sin(value), torch.cos(value)], dim=-1)
            else:
                batch_shape = values.shape[:-1]
                scales = 2.0 ** torch.arange(0, pos_enc_basis).to(self.device)
                xb = values[..., None, :] * scales[:, None]
                four_feat = torch.sin(torch.stack([xb, xb + 0.5 * torch.pi], axis=-2))

                if self.use_pos_enc == 'nerfies_windowed':
                    window = self.windowed_pos_enc(pos_enc_basis)
                    four_feat = window[..., None, None] * four_feat
                elif self.use_pos_enc == 'free_windowed':
                    window = self.freq_mask_alpha.to(self.device)
                    four_feat = window[..., None, None] * four_feat
                
                four_feat = four_feat.reshape((*batch_shape, -1))
                fin_values = torch.cat([input_values, four_feat], dim=-1)
        else: fin_values = input_values
        return fin_values

    def windowed_pos_enc(self, pos_enc_basis):
        alpha = self.windowed_alpha

        bands = torch.arange(0, pos_enc_basis).to(self.device)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))

    def update_windowed_alpha(self, current_iter, max_iter):
        self.windowed_alpha = (self.pos_enc_basis * current_iter) / max_iter

    def update_freq_mask_alpha(self, current_iter, max_iter):
        # based on https://github.com/Jiawei-Yang/FreeNeRF/blob/main/internal/math.py#L277
        pos_enc_basis = self.pos_enc_basis
        if current_iter < max_iter:
            freq_mask = np.zeros(pos_enc_basis)
            ptr = (pos_enc_basis * current_iter) / max_iter + self.pos_enc_window_start
            # ptr = ptr if ptr < pos_enc_basis / 3 else pos_enc_basis / 3
            int_ptr = int(ptr)

            freq_mask[: int_ptr + 1] = 1.0  # assign the integer part
            freq_mask[int_ptr : int_ptr + 1] = (ptr - int_ptr)  # assign the fractional part

            self.freq_mask_alpha = torch.clip(torch.from_numpy(freq_mask), 1e-8, 1-1e-8).float() # for numerical stability
            self.windowed_alpha = ptr
        else:
            self.freq_mask_alpha = torch.ones(pos_enc_basis).float()
            self.windowed_alpha = pos_enc_basis + 1
    
    def save(self, filename: str, training_information: dict) -> None:
        """
        Save the CPPN model
        

        Args:
            filename (str): path filepath on which the model will be saved
            training_information (dict): dictionary containing information on the training
        """
        save_parameters = {
            'version': self.version,
            'parameters': self.model_definition,
            # 'nerf_parameters': self.model_definition_nerf,
            'training_information': training_information,
            'model': self.state_dict(),
        }

        if 'nerfies_windowed' in self.use_pos_enc:
            save_parameters['windowed_alpha'] = self.windowed_alpha
        
        if 'free_windowed' in self.use_pos_enc:
            save_parameters['freq_mask_alpha'] = self.freq_mask_alpha

        torch.save(
            save_parameters,
            f=filename)