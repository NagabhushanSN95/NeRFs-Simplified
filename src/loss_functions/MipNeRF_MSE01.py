# Shree KRISHNAya Namaha
# MSE loss function
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

from pathlib import Path

import torch

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MipNeRF_MSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_loss_needed = ('num_samples_coarse' in self.configs['model']) and \
                                  (self.configs['model']['num_samples_coarse'] > 0)
        self.fine_loss_needed = ('num_samples_fine' in self.configs['model']) and \
                                (self.configs['model']['num_samples_fine'] > 0)
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, training: bool = True):
        target_rgb = input_dict['target_rgb']
        total_loss = 0
        loss_maps = {}

        if self.coarse_loss_needed:
            pred_rgb_coarse = output_dict['rgb_coarse']
            mse_coarse = self.compute_mse(pred_rgb_coarse, target_rgb, training)
            total_loss += (mse_coarse['loss_value'] * self.loss_configs['weight_coarse'])
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, mse_coarse['loss_maps'], suffix='coarse')

        if self.fine_loss_needed:
            pred_rgb_fine = output_dict['rgb_fine']
            mse_fine = self.compute_mse(pred_rgb_fine, target_rgb, training)
            total_loss += (mse_fine['loss_value'] * self.loss_configs['weight_fine'])
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, mse_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if not training:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def compute_mse(pred_value, true_value, training: bool):
        error = pred_value - true_value
        mse = torch.mean(torch.square(error), dim=1)
        mean_mse = torch.mean(mse)
        loss_dict = {
            'loss_value': mean_mse,
        }
        if not training:
            loss_dict['loss_maps'] = {
                this_filename: mse
            }
        return loss_dict
