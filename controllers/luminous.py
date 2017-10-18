# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import math

import pandas as pd

class Luminous:

    def __init__(self, base_luminous_csv_path):
        self._base = base_luminous_csv_path
        self._base_df = pd.read_csv(
                base_luminous_csv_path,
                header=None,
                )
        self.base_wavelength_list = self._base_df.loc[:, 0].values
        self.base_luminous_list = self._base_df.loc[:, 1].values
        _list_num = len(self.base_luminous_list)
        self.gradient_dict = {
                self.base_wavelength_list[i]: \
                        (self.base_luminous_list[i+1] - self.base_luminous_list[i], self.base_luminous_list[i])
                for i in range(_list_num - 1)
                }

    def _gradient(self, target_wavelength):
        base_wavelength = math.floor(target_wavelength)
        try:
            gradient = self.gradient_dict[base_wavelength][0]
        except KeyError:
            gradient = 0.0
        return gradient

    def _luminous(self, wavelength, intensity):
        try:
            return (self._gradient(wavelength) * (wavelength - math.floor(wavelength)) \
                    + self.gradient_dict[math.floor(wavelength)][1]) * intensity
        except KeyError:
            return 0.0

    def calc(self, target_wavelength_list, target_intensity):
        result =  [self._luminous(wavelength, intensity) 
                for wavelength, intensity in zip(target_wavelength_list, target_intensity)]
        return result







if __name__ == '__main__':
    base_luminous_csv_path = 'storage/luminous_efficiency_list.csv'
    luminous_list = Luminous(base_luminous_csv_path)

