#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:39:28 2022

@author: alessandroaustoni
"""
import numpy as np

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx