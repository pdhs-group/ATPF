"""
Test ATPF simulation for BATCH experiment
"""

# General Imports
import matplotlib.pyplot as plt
import numpy as np
import os ,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import importlib

# Import ATPFSolver class and some utils
from atpf import ATPFSolver
from atpf.utils.atpf_utils import visualize_E_t

# Create a class instance based on config_file
cf_file = 'atpf_config_batch.py'
a1 = ATPFSolver(cf_file=cf_file)

# You can still change some parameters here if you want (or in the config.py file beforehand)
a1.vg = [30]
a1.scl_A = [100]
a1.R_max = 1e-6

a1.v_bot_tot = 5e-4
a1.v_top_tot = 1e-4

## The following settings DISABLE flotation
# a1.fr_case = 'const'
# a1.fr0 = 0

# If you change anything, be sure to re-initialize the calculation
a1.init_calc()

# Solve the ATPF experiment and "extract" some value from the instance
a1.solve(t_eval=np.linspace(0,a1.t_max,100))
E = a1.E
w = a1.w

# Visualize reslt
plt.close('all')
visualize_E_t(a1.t, E, w[0,-1,:])
