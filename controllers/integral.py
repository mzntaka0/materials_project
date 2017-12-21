# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

def trapezoidal(x_list, y_list):
    S = 0.0
    for i in range(len(y_list) - 1):
        S += 0.5 * abs(y_list[i+1] + y_list[i]) * abs(x_list[i+1] - x_list[i])
    return S


if __name__ == '__main__':
    x = np.linspace(0, 4*np.pi, 5000)
    y = np.sin(x)

    plt.scatter(x, y)
    plt.show()

    S = trapezoidal(x, y)
    print(S)



