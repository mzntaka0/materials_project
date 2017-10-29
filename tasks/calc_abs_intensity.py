# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bpdb import set_trace
from scipy import integrate
from scipy.constants import Planck, c, k, e

from controllers.graph2coords import Graph2Coords
from controllers.luminous import Luminous
from controllers.integral import trapezoidal



class ConversionEfficiency:
    J = 1.60217646e-19

    def __init__(self, wavelength_list, photon_num_list):
        self.wavelength_list = wavelength_list * 1e-9
        self.photon_num_list = photon_num_list
        self.Pin = self.Pin()
        self.band_gap_list = np.arange(0.01, 3.01, 0.01)
        #self.band_gap_frequency_list = self.band_gap_list / Planck
        #self.band_gap_wavelength_list = Planck * c / (self.band_gap_list * self.J)
        #print('band_gap_frequency_list: {}'.format(self.band_gap_frequency_list))
        #print('band_gap_wavelength_list: {}'.format(self.band_gap_wavelength_list))
        #self.fc = 1
        #self.Tc = 1.0  # assume
        #self.Ts = 1.0  # assume
        #self.Vc = k * self.Tc / e
        #self.xc = self.Tc / self.Ts

    def conclude(self):
        return u() * nu() * FF()
    

    ##############################
    # 1. u
    ##############################
    def u(self):
        return [100 * self.Pout(band_gap) / self.Pin for band_gap in self.band_gap_list]

    # factor of u
    def Pin(self):
        return np.dot(self.photon_num_list, Planck * c / self.wavelength_list)

    # factor of u
    def Pout(self, band_gap):
        wavelengthEg = Planck * c / (band_gap * self.J)
        return band_gap * self.J * np.where(self.wavelength_list <= wavelengthEg, self.photon_num_list, 0).sum()


    ##############################
    # 2. nu
    ##############################
    def nu(self, band_gap):
        xg = self._xg(band_gap)
        return Vop(xg) / Vg

    # factor of nu
    def Vop(self, band_gap):
        xg = self._xg(band_gap)
        wavelengthEg = None
        nug = None

        return self.Vc * np.log((self.fc * num_of_photon(wavelengthEg)) / 2*np.pi*Pflux(nug, xg))

    # factor of nu(Vop)
    def num_of_photon(self, wavelengthEg):
        pass

    # factor of nu(Vop)
    def Pflux(self, nug, xg):
        return ((2*k**3*self.Tc**3)/h**3*c**2) * _nu_denom(xg, self.xc)

    # factor of nu(Pflux)
    def _nu_denom(self, xg):
        return integrate.quad(lambda x: x**2/(np.exp(x) - 1), xg/self.xc, np.inf)

    def _xg(self, band_gap):
        return bandgap / (k * self.Ts)


    ##############################
    # 3. FF
    ##############################
    # ganna use
    def FF_V(self, Vmax, Vop):
        return ((Vmax/self.Vc)**2) / ((1 + (Vmax/self.Vc) - np.exp(Vmax/self.Vc)) * (Vop / self.Vc))

    def FF(self):
        return (I(Vmax) * Vmax) / (Ish * Vop)

    def FF_zm(self, zm):
        return zm**2 / ((1 + zm - np.exp(-zm)) * (zm + np.log(1 + zm)))






if __name__ == '__main__':
    img_path = 'storage/arbunit_graph.jpg'
    base_luminous_csv_path = 'storage/luminous_efficiency_list.csv'
    J = 1.60217646e-19
    LUMINOUS_FLUX = 2190  # [lm] R70 Rank
    Km = 683  # [lm/W]
    r = 2.0  # [m] the radius of lighted surface
    alpha = 0.0
    imshow = False
    
    # get coordinates from graph image
    graph2coords = Graph2Coords(img_path)
    graph2coords.run()

    # integrate each value(luminous*) of wavelength
    luminous = Luminous(base_luminous_csv_path)
    if imshow:
        plt.scatter(luminous.base_wavelength_list, luminous.base_luminous_list)
        plt.show()

    calculated_luminous_list = luminous.calc(graph2coords.wavelength_list, graph2coords.intensity_list)
    if imshow:
        plt.scatter(graph2coords.wavelength_list, calculated_luminous_list)
        plt.show()

    print("number of graph point: {}".format(len(calculated_luminous_list)))


    total_luminous_flux = Km * trapezoidal(  # [lm/W]
            graph2coords.wavelength_list, 
            calculated_luminous_list
            )

    print(total_luminous_flux)
    alpha = LUMINOUS_FLUX / (float(total_luminous_flux) * np.pi * r**2)
    print("alpha: {}".format(alpha))


    graph2coords.num_photons_list = \
            (graph2coords.intensity_list * alpha) / (Planck * c / graph2coords.wavelength_list)
    graph2coords.show_graph(graph2coords.wavelength_list, graph2coords.num_photons_list)

    conversion_efficiency = ConversionEfficiency(graph2coords.wavelength_list, graph2coords.num_photons_list)
    print('Pin: {}'.format(conversion_efficiency.Pin))
    ideal = conversion_efficiency.u()
    plt.scatter(conversion_efficiency.band_gap_list, ideal)
    plt.show()

