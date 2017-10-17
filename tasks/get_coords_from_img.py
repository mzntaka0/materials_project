# -*- coding: utf-8 -*-
"""
"""
import os
import sys

from bpdb import set_trace
import cv2
import numpy as np
import matplotlib.pyplot as plt

from luminous import LuminousList


class Graph2CoordsConverter:

    def __init__(self, img_path, **kwargs):
        self.params = dict()
        self.params['LEFT'] = 350 
        self.params['RIGHT'] = 800.0
        self.params['BOTTOM'] = 0.0
        self.params['TOP'] = 1.0
        self.params['binary_threshord'] = 100

        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self._img2gray()
        self._img2binary()

        if kwargs:
            for arg_name, val in kwargs.items():
                self.params[arg_name] = val

    def _img2gray(self):    
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    def _img2binary(self):
        self.img = np.where(self.img <= self.params['binary_threshord'], 0, 255)

    def _reverse_y_coord(self, y):
        return self.img.shape[0] - y

    def _convert_x(self, x):
        return self.params['LEFT'] + x * (self.params['RIGHT'] - self.params['LEFT']) / self.img.shape[1]

    def _convert_y(self, y):
        return self.params['BOTTOM'] + self._reverse_y_coord(y) * (self.params['TOP']  - self.params['BOTTOM'] ) / self.img.shape[0]

    def _get_x_coords(self):
        return np.arange(self.img.shape[1])

    def _get_y_coords(self):
        y_coord_list = list()
        for i in range(self.img.shape[1]):
            y_coord = np.where(self.img[:, i]==0)[0].mean()
            if np.isnan(y_coord):
                print('please interpolate the graph pixel which doesnt have coord (e.g.modify with PhotoShop)')
                sys.exit(0)
            y_coord_list.append(y_coord)
        return np.array(y_coord_list)
        

    def run(self, save=False, imshow=True):
        origin_x_list = self._get_x_coords()
        origin_y_list = self._get_y_coords()

        modified_x_list = list(map(lambda x: self._convert_x(x), origin_x_list))
        modified_y_list = list(map(lambda y: self._convert_y(y), origin_y_list))

        if imshow:
            plt.scatter(modified_x_list, modified_y_list)
            plt.show()






if __name__ == '__main__':
    img_path = 'storage/arbunit_graph.jpg'

    converter = Graph2CoordsConverter(img_path)
    converter.run()
