import torch
import torch.nn as nn
import numpy as np

class CPPN(nn.Module):
    """
    A CPPN model, mapping a number of input coordinates to a multidimensional output (e.g. color)
    """
    def __init__(self, model_definition: dict) -> None:
        """
        Args:
            model_definition: dictionary containing all the needed parameters
                - num_layers: number of hidden layers
                - num_filters: number of filters in the hidden blocks
                - num_input_channels: number of expected input channels
                - num_output_channels: number of expected output channels
                - use_bias: whether biases are used
                - pos_enc: which positional encoding to apply: 'none', 'fourier', 'windowed'
                - pos_enc_basis: basis for positional encoding (L)
                - num_img: number of images for training (translation/rotation)
        """
        super().__init__()
        self.version = "v0.00"
        self.model_definition = model_definition
        self.device = model_definition['device']

        # getting the parameters
        self.num_early_layers = model_definition['num_early_layers']
        self.num_late_layers = model_definition['num_late_layers']
        self.num_filters = model_definition['num_filters']
        self.num_input_channels = model_definition['num_input_channels'] # x,y,z
        self.num_output_channels = model_definition['num_output_channels']
        self.use_bias = model_definition['use_bias']
        self.use_pos_enc = model_definition['pos_enc']
        self.act_func = model_definition['act_func']

        self.input_features = self.num_input_channels
        num_filters = self.num_filters
        use_bias = self.use_bias
        num_output_channels = self.num_output_channels

        self.first_act_func = nn.ReLU()
        self.act_func = nn.ReLU()

        if self.use_pos_enc != 'none':
            self.pos_enc_basis = model_definition['pos_enc_basis']
            self.pos_enc_window_start = model_definition['pos_enc_window_start']
            self.input_features = self.num_input_channels + self.num_input_channels * 2 * self.pos_enc_basis
                
            if self.use_pos_enc == 'fourier':
                self.input_features = self.num_input_channels * 2 * self.pos_enc_basis
                self.fourier_sigma = model_definition['fourier_sigma']
                self.fourier_coefficients = (model_definition['fourier_gaussian'] * self.fourier_sigma).to(self.device)

        # creating the learnable blocks
        early_pts_layers = []
        # input layer
        early_pts_layers += self.__create_layer(self.input_features, num_filters,
                                           use_bias, activation=self.first_act_func)
        # hidden layers: early
        for _ in range(self.num_early_layers):
            early_pts_layers += self.__create_layer(num_filters, num_filters,
                                               use_bias, activation=self.act_func)

        self.early_pts_layers = nn.ModuleList(early_pts_layers)

        # skip connection
        if self.num_late_layers > 0:
            self.skip_connection = self.__create_layer(num_filters + self.input_features, num_filters,
                                                use_bias, activation=self.act_func)

            late_pts_layers = []
            for _ in range(self.num_late_layers - 1):
                late_pts_layers += self.__create_layer(num_filters, num_filters,
                                                use_bias, activation=self.act_func)

            self.late_pts_layers = nn.ModuleList(late_pts_layers)
        # output layer
        self.output_linear = self.__create_layer(num_filters, num_output_channels,
                                        use_bias, activation=None)
        
        # model understanding API
        self.store_activations = False
        self.activation_dictionary = {}

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_pts = x
        
        values = input_pts

        # positional encoding
        pos_enc = self.use_pos_enc
        pts_encoded = input_pts
        if pos_enc != 'none':
            pts_encoded = self.pos_enc(input_pts, self.pos_enc_basis, 'pts')

        values = pts_encoded
        for _, pts_layer in enumerate(self.early_pts_layers):
            values = pts_layer(values)

        if self.num_late_layers > 0:
            values = self.skip_connection(torch.cat([pts_encoded, values], dim=-1))

            for _, pts_layer in enumerate(self.late_pts_layers):
                values = pts_layer(values)
        
        outputs = self.output_linear(values)

        return outputs

    def pos_enc(self, values, pos_enc_basis, type):
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
                    window = self.windowed_pos_enc(pos_enc_basis, type)
                    four_feat = window[..., None, None] * four_feat
                elif self.use_pos_enc == 'free_windowed':
                    window = self.freq_mask_alpha.to(self.device)
                    four_feat = window[..., None, None] * four_feat
            
                four_feat = four_feat.reshape((*batch_shape, -1))
                fin_values = torch.cat([input_values, four_feat], dim=-1)
        else: fin_values = input_values
        return fin_values

    def windowed_pos_enc(self, pos_enc_basis, type):
        alpha = self.windowed_alpha

        bands = torch.arange(0, pos_enc_basis).to(self.device)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))

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

    def update_windowed_alpha(self, current_iter, max_iter):
        self.windowed_alpha = (self.pos_enc_basis * current_iter) / max_iter

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
                'training_information': training_information,
                'model': self.state_dict(), # overwrites itself during initialization, so save it in this way
            }
        
        if 'nerfies_windowed' in self.use_pos_enc:
            save_parameters['windowed_alpha'] = self.windowed_alpha
        
        if 'free_windowed' in self.use_pos_enc:
            save_parameters['freq_mask_alpha'] = self.freq_mask_alpha

        torch.save(
            save_parameters,
            f=filename)