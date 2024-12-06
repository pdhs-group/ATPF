"""
ATPF Model Python package
--- UTILITY ---
"""

## IMPORT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os ,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"..",".."))

from atpf.utils import my_plotter as mp
from atpf import ATPFSolver

## Single ATPF simulation based on a set of process parameters
def atpf_sim(PP=None, KP=None, t_eval=None, init_only=False,
             fr_case='dyn', ad_case='ss', verbose=0):
    # PP contains process parameters and is a dictionary with the following keys:
    ## Q_top: Volume flow rate top phase [m³/s]
    ## Q_bot: Volume flow rate bot phase [m³/s]
    ## vg: List of gas volume flows per compartment [mL/min]
    ## w0_bot: Feed concentration bot [w/w]
    ## t_max: Process time [s]
    
    # KP contains kinetic (model) parameters and is a dictionary with the following keys:
    ## k_a: Adsorption rate [mol/m³s]
    ## k_d: Desorption rate [1/s]
    ## R_max: Maximum surface loading [mol/m²]
    
    # If P is not provided use default set
    if PP is None:
        PP = {'Q_top':1.67e-6/60, 'Q_bot':8.33e-6/60, 'vg':[30,20,10], 'w0_bot':0.02, 't_max':120*60}
    
    # If P is not provided use default set
    if KP is None:
        KP = {'k_a':2e-1, 'k_d':1e-2, 'R_max':6e-8}
        
    # Create new atpf instance
    a = ATPFSolver()
    
    # Change parameters to the ones given in P
    a.Q_top = PP['Q_top']
    a.Q_bot = PP['Q_bot']
    a.vg = PP['vg']
    a.w0_bot = PP['w0_bot']
    a.t_max = PP['t_max']
    
    # Change parameters to the ones given in KP
    a.k_a = KP['k_a']
    a.k_d = KP['k_d']
    a.K = KP['k_a']/KP['k_d']
    a.R_max = KP['R_max']
    
    # Be sure to use correct models 
    a.fr_case = fr_case
    a.ad_case = ad_case
    
    # Initialize the calculations
    a.init_calc()
    
    # Solve the ODE. Report for timesteps given in t_eval
    if init_only: 
        return None, None, None, a
    else:
        a.solve(t_eval, verbose=verbose)
    
        # Return in the following order
        ## Time array, concentration (w/w) of top right compartment as function of time,
        ## Separation efficiency array and the whole atpf instance 
        return a.t, a.w[0,-1,:], a.E, a

## Full ATPF DOE 
def atpf_doe(DOE_file='data/ATPF_DOE_241121.xlsx', KP=None, t_eval=None, verbose=1,
             fr_case='dyn', ad_case='ss'):    
    # KP contains kinetic (model) parameters and is a dictionary with the following keys:
    ## k_a: Adsorption rate [mol/m³s]
    ## k_d: Desorption rate [1/s]
    ## R_max: Maximum surface loading [mol/m²]
    
    # Read DOE data
    df = pd.read_excel(DOE_file, sheet_name=0, skiprows=3, header=None)
    no = df.iloc[:,0].astype(int).values
    w0_bot = df.iloc[:,1].astype(float).values*1e-2 # Convert from %
    Q_bot = df.iloc[:,2].astype(float).values*1e-6/60 # Convert to m³/s
    Q_top = df.iloc[:,3].astype(float).values*1e-6/60 # Convert to m³/s
    vg_1 = df.iloc[:,4].astype(float).values
    vg_2 = df.iloc[:,5].astype(float).values
    vg_3 = df.iloc[:,6].astype(float).values
    t_exp = df.iloc[:,7].astype(float).values*60 # Convert to s

    # Initialize result arrays
    E = np.zeros((len(no),len(t_eval)))
    w_top = np.zeros((len(no),len(t_eval)))
    a = np.empty(no.shape,dtype=object)
    
    # Loop through all experiments
    for i in range(len(no)):
        if verbose > 0: 
            print(f'simulating experiment no. {i+1}/{len(no)}')
        # Fill PP dictionary
        PP = {'Q_top':Q_top[i], 
              'Q_bot':Q_bot[i], 
              'vg':[vg_1[i],vg_2[i],vg_3[i]], 
              'w0_bot':w0_bot[i], 
              't_max':t_exp[i]}
        
        # Simulate experiment
        _, w_top[i,:], E[i,:], a[i] = atpf_sim(PP, KP, t_eval, fr_case=fr_case, ad_case=ad_case)
            
    return E, w_top, a, df        

## Cost Function DOE
def cost_atpf_doe(param, metric='w_RMSE_full', verbose=0,
                  DOE_file='data/ATPF_DOE_241121.xlsx', exp_folder='data/exp_data'): 
        
    # Parameters to optimize
    K = param[0]
    R_max = 10**param[1]
    
    # Import experimental data
    w_exp, E_exp, t_exp, no_exp = import_transform_exp_doe(exp_folder)
    # print(no_exp)
    
    # Calculate DOE based on current param
    KP = {'k_a':K, 'k_d':1, 'R_max':R_max}    
    E_mod, w_mod, _, _ = atpf_doe(DOE_file=DOE_file, KP=KP, t_eval=t_exp*60, verbose=0)
    
    # Calculate loss metric
    if metric == 'w_RMSE_full':
        loss = np.sqrt(np.mean((w_mod-w_exp)**2))
    else:
        loss = 1e6
        print('ERROR: Provide correct loss metric.')
        
    # Print that new iteration is happening?
    if verbose > 0:
        print(f'Current param: [{K:.2f},{R_max:.2e}], loss: {loss:.3e}')    
        
    return loss
    
## Import Functions
def import_exp_single(folder):
    filenames = os.listdir(folder)
    
    # Extract wavelength array from first file (identical for all)
    df = pd.read_csv(folder + '/' + filenames[0], 
                     delimiter='\t', skiprows=9, header=None)
    wl = df.iloc[:,0].astype(float).values
    
    # Preallocate memory for absorbance and concentration array
    A = np.ones((len(filenames),len(wl)))
    t = np.ones((len(filenames)))
    
    # Iterate through every time
    for i in range(len(filenames)):
        A[i,:] = pd.read_csv(folder + '/' + filenames[i], 
                             delimiter='\t', skiprows=9, 
                             header=None).iloc[:,1].astype(float).values
        tmp = filenames[i].split('_')[-1]
        t[i] = float(tmp.rstrip('.txt'))
    
    # Sort them by t (only for visuals)
    idx_s = np.argsort(t)

    return wl, A[idx_s,:], t[idx_s]

## Validate a set of kinetic parameters
def validate_KP(KP, DOE_file='data/ATPF_DOE_241121.xlsx', exp_folder='data/exp_data', export=True):
    # Import experimental data
    w_exp, E_exp, t_exp, no_exp = import_transform_exp_doe(exp_folder)
    
    # Calculate DOE based on current param   
    E_mod, w_mod, _, _ = atpf_doe(DOE_file=DOE_file, KP=KP, t_eval=t_exp*60, verbose=0)
    
    # Initialize
    mp.init_plot(scl_a4=1, page_lnewdth_cm=15.9, fnt='Arial', mrksze=4, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[1, 1])
    
    # Plot
    fig, ax = plt.subplots()     
    ax.scatter(w_exp, w_mod, marker='s', color=mp.green, edgecolor='k')
    ax.plot((0,1), (0,1), color='k')
    
    # Customize axes
    ax.set_xlabel(r'$w_{\,\mathrm{top,\,exp}}$ / %')
    ax.set_ylabel(r'$w_{\,\mathrm{top,\,mod}}$ / %')
    ax.set_ylim([0,0.15])
    ax.set_xlim([0,0.15])
    ax.grid(True)
    plt.tight_layout()
    
    if export:
        plt.savefig('export/KP_validation.pdf')
        plt.savefig('export/KP_validation.png',dpi=300)
    
    return ax, fig
    
## Visualize E(t):
def visualize_E_t(t, E, w_top=None, export=False, expname='E_t'):
    # Initialize
    mp.init_plot(scl_a4=1, page_lnewdth_cm=15.99, fnt='Arial', mrksze=4, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[15.99, 6])
    
    # Plot
    fig, ax = plt.subplots()     
    ax.plot(t/60, E, marker='s', color=mp.green, mec='k', label='$E$')
    if w_top is not None:
        ax2 = ax.twinx()
        ax2.plot(t/60, 100*w_top, marker='o', color=mp.red, mec='k', label=r'$w_{\,\mathrm{top}}$')
        ax2.set_ylabel(r'$w_{\,\mathrm{top}}$ / %')
        # ax2.set_ylim([0,12])
        ax.legend(loc='upper left')
        ax2.legend(loc='lower right')
    
    # Customize axes
    ax.set_xlabel('$t$ / min')
    ax.set_ylabel('$E$ / %')
    ax.set_ylim([0,100])
    ax.set_xlim([t[0]/60,t[-1]/60])
    ax.grid(True)
    plt.tight_layout()
    
    if export:
        plt.savefig('export/'+expname+'.pdf')
        plt.savefig('export/'+expname+'.png',dpi=300)
    
    return ax, fig