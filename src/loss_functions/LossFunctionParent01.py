# Shree KRISHNAya Namaha
# Abstract parent class
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

import abc


class LossFunctionParent:
    @abc.abstractmethod
    def compute_loss(self, input_dict: dict, output_dict: dict, training: bool = True):
        pass
