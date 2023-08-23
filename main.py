import numpy as np 
import pandas as pd
import random
import copy
import os
import sys
from ECVRP import TwoECVrp, scatter_search_vns, output_csv_plot

# Visualization
import matplotlib.pyplot as plt 
Data_Folder = 'excel_params'
Plot_Folder = 'Visualization'
for file in os.listdir(Data_Folder):
    #path = os.path.join(Data_Folder, file) 
    max_iteration = 10
    neighborhood_size = 2
    output_csv_plot(file, max_iteration, neighborhood_size)
    
