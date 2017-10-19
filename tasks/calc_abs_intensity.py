# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.constants import Planck, c

from controllers.graph2coords import Graph2Coords
from controllers.luminous import Luminous
from controllers.integral import trapezoidal
from controllers.light import lumination




class ConversionEfficiency:

    def __init__(self, wavelength_list, photon_num_list):
        self.wavelength_list = wavelength_list
        self.photon_num_list = photon_num_list
        self.power_in = self.power_in()

    def power_in(self):
        return sum([photon_num * Planck * (c / wavelength)
                for wavelength, photon_num in zip(self.wavelength_list, self.photon_num_list)])

    def power_out(self, bandgap_frequency):
        return lambda bandgap_frequency: sum([photon_num * Planck * bandgap_frequency for wavelength, photon_num in self.wavelength_list, self.photon_num_list if c / wavelength >= bandgap_frequency])

    def ideal(self, bandgap_frequency):
        return 100 * self.power_out(bandgap_frequency) / self.power_in



if __name__ == '__main__':
    img_path = 'storage/arbunit_graph.jpg'
    base_luminous_csv_path = 'storage/luminous_efficiency_list.csv'
    I = 0.36
    V = 34.8
    ELECTRIC_POWER = I * V
    LUMINOUS_FLUX = 2190  # R70 Rank

    graph2coords = Graph2Coords(img_path)
    graph2coords.run()

    luminous = Luminous(base_luminous_csv_path)
    print(luminous.max)
    plt.scatter(luminous.base_wavelength_list, luminous.base_luminous_list)
    plt.show()
    calculated_luminous_list = luminous.calc(graph2coords.wavelength_list, graph2coords.intensity_list)
    plt.scatter(graph2coords.wavelength_list, calculated_luminous_list)
    plt.show()

    total_luminous_flux = trapezoidal(
            graph2coords.wavelength_list, 
            calculated_luminous_list
            )
    print(total_luminous_flux)
    distance_list = np.linspace(0, 2.0, 100)
    lumination_list = list(map(lambda d: lumination(d, total_luminous_flux), distance_list))
    _lumination_list = list(map(lambda d: lumination(d, LUMINOUS_FLUX), distance_list))
    print(lumination_list)
    print(_lumination_list)
    plt.scatter(distance_list, lumination_list)
    plt.show()
    maximum_luminous_efficiency = LUMINOUS_FLUX / total_luminous_flux
    print(maximum_luminous_efficiency)
