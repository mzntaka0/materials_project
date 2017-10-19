# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import numpy as np


def _calc_solid_angle(distance, area=1):
    return area / float(distance**2)


def lumination(distance, total_luminous_flux):
    try:
        return total_luminous_flux * _calc_solid_angle(distance) / 4 * np.pi
    except ZeroDivisionError:
        return 0.0


if __name__ == '__main__':
    pass

