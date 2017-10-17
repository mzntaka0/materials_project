# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import math

import pandas as pd

class LuminousList:

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
                self.base_wavelength_list[i]: self.base_luminous_list[i+1] - self.base_luminous_list[i]
                for i in range(_list_num - 1)
                }

    def _gradient(self, target_wavelength):
        base_wavelength = math.floor(target_wavelength)
        return self.gradient_dict[base_wavelength]

    def calc(self, target_wavelength_list):
        return [_gradient(wavelength) * (wavelength - math.floor(wavelength))
            for wavelength in target_wavelength_list]






if __name__ == '__main__':
    base_luminous_csv_path = 'storage/luminous_efficiency_list.csv'
    luminous_list = LuminousList(base_luminous_csv_path)

