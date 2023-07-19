import numpy as np 
import pandas as pd
import random
import copy
import os
import sys
from ECVRP import TwoECVrp, scatter_search_vns, output_csv

Data_Folder = 'data'

for file in os.listdir(Data_Folder):
    #path = os.path.join(Data_Folder, file) 
    max_iteration = 100
    num_solution = 100
    neighborhood_size = 2
    output_csv(file, max_iteration, num_solution, neighborhood_size)
    
