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
from scipy.optimize import minimize 
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from atpf.utils import my_plotter as mp
from atpf import ATPFSolver

mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Latin Modern Roman', mrksze=7, 
             fontsize = 11, labelfontsize=11, tickfontsize=9, 
             aspect_ratio=[13.7, 6])

# Non-linear regression for bubble size 
def regression_bubble(raw_file='data/raw/v_d32_twill.txt', text=True):
    # Append full path
    full_path = os.path.join(os.path.dirname( __file__ ),"..","..", raw_file)
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
    mp.init_plot(scl_a4=2, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7])
    
    fig, ax = plt.subplots() 
    ax.scatter(v, d, edgecolor='k', color=mp.green, zorder=3)
    ax.plot(v_mod, d_mod, linestyle='-.', color=mp.green, zorder=2)
    
    # Customize axes
    ax.set_xlabel(r'$\dot{V}_{\mathrm{g}}$ / $\mathrm{mL\,min^{-1}}$')
    ax.set_ylabel(r'$d_\mathrm{b}$ / $\mathrm{\mu m}$')
    ax.text(0.98, 0.05, r"$\mathrm{RMSE}"+f"={cost_bubble(P,v,d):.1f}"+r"\,\mu\mathrm{m}$", 
            transform=ax.transAxes, verticalalignment='bottom', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    if text:
        ax.text(0.02, 0.95, exp_name, 
                transform=ax.transAxes, verticalalignment='top', 
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    ax.grid(True)
    ax.set_xlim([0,ax.get_xlim()[1]])
    # ax.set_ylim([min(d)*0.9,max(d)*1.1])

    plt.tight_layout()
    
    # Generate export strings
    fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                               'data/gas_correlations/gas_v_d_'+exp_name+'.pdf')
    file_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                                'data/gas_correlations/gas_v_d_'+exp_name+'.npy')
    
    # Export figure
    plt.savefig(fig_exp_pth)
    
    # Export model
    print(P)
    np.save(file_exp_pth,
            {'Model':'d=P[0]*v+P[1]*np.log(v)+P[2]',
             'P':P, 'vg_min':min(v), 'vg_max':max(v)})
    
    return v, d, P

# Linear regression for height of mixing zone
def regression_mixing_height(raw_file='data/raw/raw data_height mixing.csv',
                             simple=False):
    # Append full path
    full_path = os.path.join(os.path.dirname( __file__ ),"..","..", raw_file)
    
    # Generate export strings
    fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                               'data/gas_correlations/gas_v_h_mix')
    file_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                                'data/gas_correlations/gas_v_h_mix')
    
    # Import the data from .txt
    df = pd.read_csv(full_path, delimiter=';', skiprows=1, 
                     header=None)
    
    V = df.iloc[:,:3].astype(float).values
    h = df.iloc[:,3:6].astype(float).values
    
    if simple:
        # Linear Regression without y intercept
        m = np.sum(np.sum(V, axis=1)*np.mean(h, axis=1)) / np.sum(np.sum(V, axis=1)**2)
        c = 0
                
        # Calculate R2
        h_mod = m*np.sum(V, axis=1)
        
        SS_res = np.sum((np.mean(h, axis=1) - h_mod) ** 2)
        SS_tot = np.sum(np.mean(h, axis=1)**2)
        R2 = 1 - SS_res / SS_tot 
        
        # Export model m in [m min / mL]
        np.save(file_exp_pth + '_simple.npy',
                {'Model':'h=m*np.sum(v)',
                 'm':m*1e-3})
    else:
        mod = LinearRegression(fit_intercept=False)#, positive=True)
        mod.fit(V, np.mean(h, axis=1))
        R2 = mod.score(V, np.mean(h, axis=1))
        
        # Calculate model for plot
        h_mod = mod.predict(V)
        
        print(f'R2 = {R2:.3f}')
        print(f'Model coefficients = {mod.coef_}')
        print(f'Model intercept = {mod.intercept_}')
        
        # Export model 
        np.save(file_exp_pth + '_multiple.npy',
                {'Model':'MLR, input [Vg1, Vg2, Vg3] in mL/min, output h in mm',
                 'mod': mod})

    # Create plot individual compartments
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[15.99, 6])
    
    fig, ax = plt.subplots(1,3) 
    ax[0].scatter(V[:,0], h[:,0], edgecolor='k', color=mp.green, zorder=3)
    ax[1].scatter(V[:,1], h[:,1], edgecolor='k', color=mp.green, zorder=3)
    ax[2].scatter(V[:,2], h[:,2], edgecolor='k', color=mp.green, zorder=3)
    
    # Customize axes
    ax[0].set_xlabel(r'$\dot{V}_{\mathrm{g},1}$ / $\mathrm{mL\,min^{-1}}$')
    ax[1].set_xlabel(r'$\dot{V}_{\mathrm{g},2}$ / $\mathrm{mL\,min^{-1}}$')
    ax[2].set_xlabel(r'$\dot{V}_{\mathrm{g},3}$ / $\mathrm{mL\,min^{-1}}$')
    ax[0].set_ylabel(r'$\overline{h}_{\mathrm{mix},i}$ / $\mathrm{mm}$')

    for a in ax:
        a.grid(True)
        a.set_xlim([0,a.get_xlim()[1]])
        a.set_ylim([0,a.get_ylim()[1]])

    plt.tight_layout()
    
    # Create plot integral gas flow rate
    mp.init_plot(scl_a4=2, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[1, 1])
    
    fig2, ax2 = plt.subplots() 
    ax2.scatter(np.sum(V, axis=1), np.mean(h, axis=1), edgecolor='k', color=mp.green, zorder=3, label='data')
    ax2.scatter(np.sum(V, axis=1), h_mod, edgecolor='k', color=mp.red, zorder=1, label='linear fit')
    
    # Customize axes
    ax2.set_xlabel(r'$\sum\dot{V}_{\mathrm{g}}$ / $\mathrm{mL\,min^{-1}}$')
    ax2.set_ylabel(r'$\overline{h}$ / $\mathrm{m}$')
    
    ax2.text(0.98, 0.05, r"$R^2"+f"={R2:.3f}$", 
            transform=ax2.transAxes, verticalalignment='bottom', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='w', alpha=1))

    ax2.grid(True)
    ax2.set_xlim([0,ax2.get_xlim()[1]])
    ax2.set_ylim([0,ax2.get_ylim()[1]])
    ax2.legend()

    plt.tight_layout()
    
    # Create parity plot
    mp.init_plot(scl_a4=2, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[1, 1])
    
    fig3, ax3 = plt.subplots() 
    ax3.scatter(np.mean(h, axis=1), h_mod, edgecolor='k', color=mp.green, zorder=3, label='data')
    ax3.plot([0,100],[0,100],color='k',linestyle='-.')
    
    # Customize axes
    ax3.set_xlabel(r'$\overline{h}_{\mathrm{calib}} / \mathrm{mm}$')
    ax3.set_ylabel(r'$\overline{h}_{\mathrm{mod}} / \mathrm{mm}$')
    
    ax3.text(0.98, 0.05, r"$R^2"+f"={R2:.3f}$", 
            transform=ax3.transAxes, verticalalignment='bottom', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    
    ax3.grid(True)
    ax3.set_xlim([0,20])
    ax3.set_ylim([0,20])    
    # ax3.legend()
    plt.tight_layout()
    
    # Report R2
    #print(f'R2 = {R2:.4f}')
    
    # Export figure
    if simple:
        plt.savefig(fig_exp_pth+'_simple.pdf')
    else:        
        plt.savefig(fig_exp_pth+'_multiple.pdf')
    
    return V, h, mod

# Regression for volume fraction based on gas flow rate
def regression_v_fraction(raw_file_calib='data/raw/raw data_height mixing.csv', 
                          raw_file_exp='data/exp_data/0_processed_data.npy',
                          use_exp=False, poly=False):
    
    # Append full path
    full_path_exp = os.path.join(os.path.dirname( __file__ ),"..","..", raw_file_exp)    
    full_path_calib = os.path.join(os.path.dirname( __file__ ),"..","..", raw_file_calib)
    
    # Generate export strings
    fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                               'data/gas_correlations/gas_v_vf_mix.pdf')
    file_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                                'data/gas_correlations/gas_v_vf_mix.npy')
    
    data = np.load(full_path_exp, allow_pickle=True).item()
    
    exp_names = data['exp_names']
    exp_data = data['exp_data']
    
    # Initialize arrays
    V_exp = np.zeros((len(data['exp_names']),3))
    k_exp = np.zeros((len(data['exp_names']),3))
    
    # Fill arrays
    for e in range(len(data['exp_names'])):
        num_t = exp_data[e]['kappa'].shape[0]
        idx = int(num_t/2)
        
        V_exp[e,:] = exp_data[e]['vg']
        k_exp[e,:] = np.mean(exp_data[e]['kappa'][idx:,:], axis=0)
    
    v_exp = -0.0146*k_exp+0.767   
    
    # Import the data from .txt
    df = pd.read_csv(full_path_calib, delimiter=';', skiprows=1, 
                     header=None)
    
    V_calib = df.iloc[:,:3].astype(float).values
    k_calib = df.iloc[:,6:].astype(float).values
    v_calib = -0.0146*k_calib+0.767
        
    if use_exp:
        V = V_exp
        k = k_exp
        v = v_exp
    else:
        V = V_calib
        k = k_calib
        v = v_calib
        
    V_sum = np.sum(V, axis=1)
    k_mean = np.mean(k, axis=1)
    v_mean = np.mean(v, axis=1)
    
    # Linear Regression 
    if poly:
        poly = PolynomialFeatures(2, interaction_only=True)
        V = poly.fit_transform(V)
        V_exp = poly.transform(V_exp)
        
    mod = LinearRegression(fit_intercept=False)#, positive=True)
    mod.fit(V, v_mean)
    R2 = mod.score(V, v_mean)
    
    # Calculate model for plot
    v_mod = mod.predict(V)
    v_mod_exp = mod.predict(V_exp)
    RMSE_exp = np.sqrt(np.mean((np.mean(v_exp, axis=1)-v_mod_exp)**2))
    
    # Create plot individual compartments
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[15.99, 6])
    
    fig, ax = plt.subplots(1,3) 
    ax[0].scatter(V[:,0], k[:,0], edgecolor='k', color=mp.green, zorder=3)
    ax[1].scatter(V[:,1], k[:,1], edgecolor='k', color=mp.green, zorder=3)
    ax[2].scatter(V[:,2], k[:,2], edgecolor='k', color=mp.green, zorder=3)
    
    # Customize axes
    ax[0].set_xlabel(r'$\dot{V}_{\mathrm{g},1}$ / $\mathrm{mL\,min^{-1}}$')
    ax[1].set_xlabel(r'$\dot{V}_{\mathrm{g},2}$ / $\mathrm{mL\,min^{-1}}$')
    ax[2].set_xlabel(r'$\dot{V}_{\mathrm{g},3}$ / $\mathrm{mL\,min^{-1}}$')
    ax[0].set_ylabel(r'$\overline{\kappa}_i$ / $\mathrm{mS}$')

    for a in ax:
        a.grid(True)
        a.set_xlim([0,40])
        # a.set_xlim([0,a.get_xlim()[1]])
        # a.set_ylim([0,a.get_ylim()[1]])

    plt.tight_layout()
    
    # Create plot integral gas flow rate
    mp.init_plot(scl_a4=2, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[1, 1])
    
    fig2, ax2 = plt.subplots() 
    ax2.scatter(V_sum, v_mean, edgecolor='k', color=mp.green, zorder=3, label='data')
    ax2.scatter(V_sum, v_mod, edgecolor='k', color=mp.red, zorder=3, label='model')
    # ax2.plot(V_mod, h_mod, color='k', zorder=1, label='linear fit')
    
    # Customize axes
    ax2.set_xlabel(r'$\sum\dot{V}_{\mathrm{g}}$ / $\mathrm{mL\,min^{-1}}$')
    ax2.set_ylabel(r'$\overline{v}$')
    
    # ax2.text(0.98, 0.05, r"$R^2"+f"={R2:.3f}$", 
    #         transform=ax2.transAxes, verticalalignment='bottom', 
    #         horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='w', alpha=1))

    ax2.grid(True)
    # ax2.set_xlim([0,ax2.get_xlim()[1]])
    # ax2.set_ylim([0,ax2.get_ylim()[1]])
    ax2.legend()

    plt.tight_layout()
    
    # Create parity plot
    mp.init_plot(scl_a4=2, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[1, 1])
    
    fig3, ax3 = plt.subplots() 
    ax3.scatter(v_mean, v_mod, edgecolor='k', color=mp.green, zorder=3, label='data')
    ax3.plot([0,1],[0,1],color='k',linestyle='-.')
    # ax3.plot(V_mod, h_mod, color='k', zorder=1, label='linear fit')
    
    # Customize axes
    ax3.set_xlabel(r'$\overline{v}_{\mathrm{calib}}$')
    ax3.set_ylabel(r'$\overline{v}_{\mathrm{mod}}$')
    
    ax3.text(0.98, 0.05, r"$R^2"+f"={R2:.3f}$", 
            transform=ax3.transAxes, verticalalignment='bottom', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='w', alpha=1))

    ax3.grid(True)
    ax3.set_xlim([0,0.3])
    ax3.set_ylim([0,0.3])    
    # ax3.legend()
    plt.tight_layout()
    
    # Export figure
    plt.savefig(fig_exp_pth)
    
    # Create parity plot for experimental data
    mp.init_plot(scl_a4=2, page_lnewdth_cm=13.7, fnt='Arial', mrksze=7, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[1, 1])
    
    fig4, ax4 = plt.subplots() 
    ax4.scatter(np.mean(v_exp, axis=1), v_mod_exp, edgecolor='k', color=mp.green, zorder=3, label='data')
    ax4.plot([0,1],[0,1],color='k',linestyle='-.')
    # ax4.plot(V_mod, h_mod, color='k', zorder=1, label='linear fit')
    
    # Customize axes
    ax4.set_xlabel(r'$\overline{v}_{\mathrm{exp}}$')
    ax4.set_ylabel(r'$\overline{v}_{\mathrm{mod, exp}}$')
    
    # ax4.text(0.98, 0.05, r"$R^2"+f"={R2:.3f}$", 
    #         transform=ax4.transAxes, verticalalignment='bottom', 
    #         horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    
    ax4.grid(True)
    ax4.set_xlim([0,0.3])
    ax4.set_ylim([0,0.3])    
    # ax4.legend()
    plt.tight_layout()
    
    print(f'R2 = {R2:.3f}')
    print(f'Model coefficients = {mod.coef_}')
    print(f'Model intercept = {mod.intercept_}')
    print(f'RMSE on experimental data = {RMSE_exp:.3f}')
    
    # Export model 
    np.save(file_exp_pth,
            {'Model':'MLR, input [Vg1, Vg2, Vg3] in mL/min, output v in -',
             'mod': mod})
      
    return V, v, mod

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
    # regression_bubble(raw_file='data/raw/v_d32_glass.txt')
    V_d, d, mod_d = regression_bubble(raw_file='data/raw/v_d32_twill.txt', text=False)
    # regression_bubble(raw_file='data/raw/v_d32_metal.txt')
    
    # Regression for height of mixing zone
    V_h, h, mod_h = regression_mixing_height(simple=False)
    
    # Regression conductivity
    V_v, v, mod_v = regression_v_fraction(poly=False)

    
    
    
    
    
    

