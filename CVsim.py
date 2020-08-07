# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:05:46 2020

@author: bleiji

This is the file that can be used to import CVsim into jupyter notebook

"""

# the current version of the simulation
version = '1.0'

# the directory of the files
direc = 'CVsim_v'+version

import sys
sys.path.append('G:\\Measurements\\Yorick\\Scripting\\Python\\Functions\\'+direc)
from main import CV_sim as CVsim



