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
        self.Tc = 300  # [K]
        self.Vc = k * self.Tc / e
        self.fc = 1.0
        self.Fs = np.array([self.num_of_photon(band_gap) for band_gap in self.band_gap_list])
        self.Ish = e * self.Fs
        plt.scatter(self.band_gap_list, self.Ish)
        plt.title('Ish - bandgap')
        plt.xlabel('Band gap [eV]')
        plt.ylabel('Short currency [A]')
        plt.show()
        #self.band_gap_frequency_list = self.band_gap_list / Planck
        #print('band_gap_frequency_list: {}'.format(self.band_gap_frequency_list))
        #print('band_gap_wavelength_list: {}'.format(self.band_gap_wavelength_list))
        #self.Ts = 1.0  # assume
        #self.xc = self.Tc / self.Ts



    ##############################
    # 1. u
    ##############################
    def u(self):
        return np.array([100 * self.Pout(band_gap) / self.Pin for band_gap in self.band_gap_list])

    def Pin(self):
        return trapezoidal(self.wavelength_list, self.photon_num_list * (Planck * c / self.wavelength_list))

    def Pout(self, band_gap):
        wavelengthEg = Planck * c / (band_gap * self.J)
        return band_gap * self.J * trapezoidal(self.wavelength_list, np.where(self.wavelength_list <= wavelengthEg, self.photon_num_list, 0))


    ##############################
    # 2. nu
    ##############################
    def nu(self, band_gap):
        Vg = band_gap * self.J / e
        return self.Vop(band_gap * self.J) / Vg

    def Vop(self, band_gap):
        return self.Vc * np.log((self.fc * self.num_of_photon(band_gap)) / self.Fc0(band_gap))

    def num_of_photon(self, band_gap):
        wavelengthEg = Planck * c / (band_gap * self.J)
        return trapezoidal(self.wavelength_list, np.where(self.wavelength_list <= wavelengthEg, self.photon_num_list, 0))

    def Fc0(self, band_gap):
        return 2*np.pi*self.Pflux(band_gap)

    def Pflux(self, band_gap):
        """determined by the temperature of solar cell"""
        nug = (band_gap * self.J) / Planck
        nu_list = c / self.wavelength_list
        target_nu_list = np.where(nu_list > nug, nu_list, 0.0)
        pflux = np.vectorize(self._integrated_f)(target_nu_list)
        return trapezoidal(nu_list, pflux)

    def _integrated_f(self, nu):
        if nu == 0.0:
            return 0.0
        else:
            return (2/c**2)*((nu**2)/(np.exp(Planck*nu/(k*self.Tc)) - 1.0))
        
        #return trapezoidal(nu_list, np.where(nu_list > nug, self._integrated_f(nu), 0))
        #print('num_of_zero: {}'.format(len(np.where(target_nu_list)[0])))
        #power = Planck*target_nu_list / (k*self.Tc)
        #denom = np.exp(power) - 1
        #print('sum_denom: {}'.format(np.sum(denom)))
        #numer = target_nu_list**2
        #print('sum_numer: {}'.format(np.sum(numer)))
        #integrated = numer/denom
        #integrated[np.isnan(integrated)] = 0.0
        #return (2/c**2) * trapezoidal(nu_list, integrated)



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


def show(y):
    try:
        plt.scatter(np.arange(len(y)), y)
        plt.show()
    except TypeError:
        print('type not correct. pass')
        return


if __name__ == '__main__':
    img_path = 'storage/arbunit_graph.jpg'
    base_luminous_csv_path = 'storage/luminous_efficiency_list.csv'
    J = 1.60217646e-19
    LUMINOUS_FLUX = 2190  # [lm] R70 Rank
    Km = 683  # [lm/W]
    r = 2.0  # [m] the radius of lighted surface
    alpha = 0.0
    imshow = True
    
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
    print('max efficiency: {}'.format(conversion_efficiency.band_gap_list[np.argmax(ideal)]))
    plt.scatter(conversion_efficiency.band_gap_list, ideal)
    plt.xlabel('band gap [eV]')
    plt.ylabel('efficiency [%]')
    plt.show()
    nu = np.vectorize(conversion_efficiency.nu)(conversion_efficiency.band_gap_list)
    plt.scatter(conversion_efficiency.band_gap_list, nu*ideal)
    plt.show()
