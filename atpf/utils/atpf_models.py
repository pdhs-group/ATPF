"""
ATPF Model Python package
--- MODELS / CORRELATIONS ---
"""

## IMPORT
import os ,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"..",".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import mod.my_plotter as mp
from scipy.optimize import minimize 
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression

from atpf.utils import my_plotter as mp
from atpf import ATPFSolver

mp.init_plot(scl_a4=1, page_lnewdth_cm=15.9, fnt='Arial', mrksze=7, 
             fontsize = 11, labelfontsize=11, tickfontsize=9, 
             aspect_ratio=[15.9, 6])

# Non-linear regression for bubble size 
def regression_bubble(raw_file='data/raw/v_d32_twill.txt'):
    # Append full path
    full_path = os.path.join(os.path.dirname( __file__ ),"..","..",raw_file)
    exp_name = raw_file.split('_')[-1].split('.')[0]
    
    # Import the data from .txt
    df = pd.read_csv(full_path, delimiter=';', skiprows=3, 
                     header=None)
    
    v = df.iloc[:,0].astype(float).values
    d = df.iloc[:,1].astype(float).values
    
    # Regression base on cost function
    res = minimize(fun=cost_bubble, x0=[1,1,1], args=(v,d))
    P = res.x
    
    # Calculate model for plot
    v_mod = np.linspace(5,max(v),1000)
    d_mod = P[0]*v_mod+P[1]*np.log(v_mod)+P[2]
    
    # Create plot
    mp.init_plot(scl_a4=1, page_lnewdth_cm=15.9, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[15.9, 6])
    
    fig, ax = plt.subplots() 
    ax.scatter(v, d, edgecolor='k', color=mp.green, zorder=3)
    ax.plot(v_mod, d_mod, linestyle='-.', color=mp.green, zorder=2)
    
    # Customize axes
    ax.set_xlabel(r'$\dot{V}_{\mathrm{gas}}$ / mL$\,$min$^{-1}$')
    ax.set_ylabel(r'$\overline{d}$ / $\mu$m')
    ax.text(0.98, 0.05, r"$\mathrm{RMSE}"+f"={cost_bubble(P,v,d):.3f}$", 
            transform=ax.transAxes, verticalalignment='bottom', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    ax.text(0.02, 0.95, exp_name, 
            transform=ax.transAxes, verticalalignment='top', 
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    ax.grid(True)


    plt.tight_layout()
    
    # Generate export strings
    fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                               'data/gas_correlations/gas_v_d_'+exp_name+'.pdf')
    file_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                                'data/gas_correlations/gas_v_d_'+exp_name+'.npy')
    
    # Export figure
    plt.savefig(fig_exp_pth)
    
    # Export model
    np.save(file_exp_pth,
            {'Model':'d=P[0]*v+P[1]*np.log(v)+P[2]',
             'P':P, 'vg_min':min(v), 'vg_max':max(v)})
    
# Cost function for bubble regression:
def cost_bubble(P,v_exp,d_exp):
    a = P[0]
    b = P[1]
    c = P[2]
    
    d_mod=a*v_exp+b*np.log(v_exp)+c
    
    return np.sqrt(np.mean((d_mod-d_exp)**2))

# Adsorption kintetic:
def adsorption_kinetik(t,y,k_a,k_d,R_max,A,V):
    # t in s
    # k_a in mol/m³s
    # k_d in 1/s
    # R_max in mol/m2
    # A in m²
    # V in m³
    c = y[0]    # Bulk concentration
    R = y[1]    # Surface loading
    dRdt = k_a*c*(R_max-R)-k_d*R
    dcdt = -dRdt*A/V
    return [dcdt,dRdt]

# Calculate flotation rate based on gas volume flow
    
# Solve kinetik
def adsorption_test():
    t_max = 10.25       # s adsorption time
    t_vec = np.linspace(0,t_max,1000)
    k_a = 1e-1
    k_d = 5e-5
    R_max = 1e-9
    A = 1.5e-1        # m²
    V = 1.67e-4     # m³
    R_0 = 0
    c_0 = 1.64      # mol/m³
    
    res = solve_ivp(adsorption_kinetik, [0,t_max], 
                    [c_0,R_0], t_eval=t_vec, args=(k_a,k_d,R_max,A,V))
    
    # Flotation rate:
    # Calculate the change in concentration
    dcdt = (c_0-res.y[0][-1])/t_max
    theta = 100*res.y[1][-1]/R_max
    
    # fr is defined as dc/dtc
    fr = dcdt/c_0
    
    print(f'The resulting flotation rate is {fr:.3e} [1/s]')    
    print(f'The bubble surface is {theta:.3f} % saturated')      
    print(f'The final concentration is {res.y[0][-1]:.3f} mol/m**3')  
    
    # Create plot
    fig, ax1 = plt.subplots() 
    ax2 = ax1.twinx() 
    ax1.plot(res.t, res.y[0], color=mp.green, zorder=3)    
    ax2.plot(res.t, res.y[1], color=mp.red, zorder=3)
    
    # Customize axes
    ax1.set_ylabel(r'$c_l$ / mol$\,$m$^{-3}$')
    ax1.set_xlabel(r'$t$ / s')
    ax2.set_ylabel(r'$\Gamma$ / mol$\,$m$^{-2}$')
    ax1.set_ylim([0,2*c_0])
    ax1.grid(True)
    
if __name__ == '__main__':
    plt.close('all')
    
    # Regression for twill and metal
    regression_bubble(raw_file='data/raw/v_d32_twill.txt')
    regression_bubble(raw_file='data/raw/v_d32_metal.txt')
    

    
    
    
    
    
    

