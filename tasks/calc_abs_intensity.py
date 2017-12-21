# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import optimize, log
try:
    from bpdb import set_trace
except ImportError:
    from pdb import set_trace
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
        print('wavelength_list: {}'.format(len(self.wavelength_list)))
        self.Pin = self.Pin()
        self.band_gap_list = np.linspace(0.01, 3.01, 1800)
        self.Tc = 300  # [K]
        self.Vc = k * self.Tc / e
        self.fc = 1.0


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
        """
        nu = Vop / Vg = Vc*log(fc*num_of_photon(band_gap)/2*pi*Pflux(band_gap))
        """
        Vg = band_gap * self.J / e
        Vop = self._Vop(band_gap)
        return Vop / Vg

    def Vop(self, band_gap):
        return self.Vc * np.log((self.fc * self.num_of_photon(band_gap)) / (2.0 * np.pi * self.Pflux(band_gap)))

    def _Vop(self, band_gap):
        return self.Vc * np.log(self.fc * (self.num_of_photon(band_gap) / self.Fc0(band_gap)) + 1)
        

    def Fc0(self, band_gap):
        return self.Pflux(band_gap) * 2.0 * np.pi


    def num_of_photon(self, band_gap):
        wavelengthEg = Planck * c / (band_gap * self.J)  # self.J point
        return trapezoidal(self.wavelength_list, np.where(self.wavelength_list <= wavelengthEg, self.photon_num_list, 0))
        #return trapezoidal(self.wavelength_list, np.where(self.wavelength_list <= wavelengthEg, self.photon_num_list, 0))

    def Pflux(self, band_gap, T=300):
        """determined by the temperature50 of solar cell"""
        nug = band_gap * self.J / Planck  # self.J point
        y = lambda x: (2.0/c**2)*(x**2)/(np.exp((Planck*x)/(k*self.Tc)) - 1.0)
        S = integrate.quad(y, nug, 1e16) 
        return S[0]

    def _integrated_func(self, nu, T=300):
        if nu == 0.0:
            return 0.0
        else:
            return (2/(c**2))*(nu**2)/(np.exp((Planck*nu)/(k*self.Tc)) - 1.0)
        
    def _integrated(self, nu, T=300):
        return (2/(c**2))*(nu**2)/(np.exp((Planck*nu)/(k*self.Tc)) - 1.0)

    ##############################
    # 3. FF
    ##############################
    # ganna use
    def FF_V(self, Vmax, Vop):
        return ((Vmax/self.Vc)**2) / ((1 + (Vmax/self.Vc) - np.exp(Vmax/self.Vc)) * (Vop / self.Vc))

    def FF(self):
        return (I(Vmax) * Vmax) / (self.Ish * self.Vop)

    def FF_zm(self, band_gap):
        zm = self.solve_zm(band_gap)
        return zm**2 / ((1 + zm - np.exp(-zm)) * (zm + np.log(1 + zm)))

    def solve_zm(self, band_gap):
        return optimize.fsolve(self.formula_zm, 0, args=(band_gap))

    def formula_zm(self, zm, band_gap):
        return zm + np.log(1 + zm) - (self._Vop(band_gap) / self.Vc)




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

    if imshow:
        graph2coords.show_graph(graph2coords.wavelength_list, graph2coords.num_photons_list)

        plt.scatter(graph2coords.wavelength_list, graph2coords.intensity_list*alpha)
        plt.title('peak spectral intensity')
        plt.xlabel('wavelength[nm]')
        plt.ylabel('peak spectral intensity[W*m^-2*nm^1]')
        plt.show()

    conversion_efficiency = ConversionEfficiency(graph2coords.wavelength_list, graph2coords.num_photons_list)
    print('Pin: {}'.format(conversion_efficiency.Pin))
    ideal = conversion_efficiency.u()
    print('max efficiency: {}'.format(conversion_efficiency.band_gap_list[np.argmax(ideal)]))
    if imshow:
        plt.scatter(conversion_efficiency.band_gap_list, ideal)
        plt.xlabel('band gap [eV]')
        plt.ylabel('efficiency [%]')
        plt.show()
    nu = np.vectorize(conversion_efficiency.nu)(conversion_efficiency.band_gap_list)
    FF = np.vectorize(conversion_efficiency.FF_zm)(conversion_efficiency.band_gap_list)
    ita = nu*ideal*FF
    ita_max = np.argmax(ita)
    print('Eg: {}, ita: {}'.format(conversion_efficiency.band_gap_list[ita_max], ita[ita_max]))
    plt.scatter(conversion_efficiency.band_gap_list, ita)
    plt.title('conversion efficiency considering with detailed balance + FF')
    plt.xlabel('Band gap [eV]')
    plt.ylabel('conversion efficiency [%]')
    plt.savefig('conversion efficiency')
    plt.show()
    ############################################
    # until here OK
    ############################################
