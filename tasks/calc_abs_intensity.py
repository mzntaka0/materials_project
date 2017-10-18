# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from controllers.graph2coords import Graph2Coords
from controllers.luminous import Luminous
from controllers.integral import trapezoidal




if __name__ == '__main__':
    img_path = 'storage/arbunit_graph.jpg'
    base_luminous_csv_path = 'storage/luminous_efficiency_list.csv'

    graph2coords = Graph2Coords(img_path)
    graph2coords.run()

    luminous = Luminous(base_luminous_csv_path)
    plt.scatter(luminous.base_wavelength_list, luminous.base_luminous_list)
    plt.show()
    calculated_luminous_list = luminous.calc(graph2coords.wavelength_list, graph2coords.intensity_list)
    plt.scatter(graph2coords.wavelength_list, calculated_luminous_list)
    plt.show()

    integrated_value = trapezoidal(
            graph2coords.wavelength_list, 
            calculated_luminous_list
            )
    print(integrated_value)

