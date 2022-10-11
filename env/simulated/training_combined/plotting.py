# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:33:35 2022

@author: User
"""

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

folder_path = 'experiments'

test_no_transf = 'new670_320_1'
file_no_transf = os.path.join(folder_path, 'logs/', test_no_transf)

test_Tscan = 'new670_320_transf_scan_10'
file_Tscan =  os.path.join(folder_path, 'logs/', test_Tscan)

test_Tstages = 'new670_320_trasnf_stages_10'
file_Tstages =  os.path.join(folder_path, 'logs/', test_Tstages)

plot_results([file_no_transf, file_Tscan, file_Tstages], 25000, results_plotter.X_TIMESTEPS, "Learning starts from 10 timesteps")
plt.legend()
plt.axhline(y = -3, color = 'black',linestyle = 'dashed', label='Convergence region' )  
#plt.show()
plt.savefig('changes_exp1.png', dmi=300)

