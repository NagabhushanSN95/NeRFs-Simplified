# Shree KRISHNAya Namaha
# Utility functions for losses
# Author: Nagabhushan S N
# Last Modified: 23/09/2022


def update_loss_map_dict(old_dict: dict, new_dict: dict, suffix: str):
    for key in new_dict.keys():
        old_dict[f'{key}_{suffix}'] = new_dict[key]
    return old_dict
