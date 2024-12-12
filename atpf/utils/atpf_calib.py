"""
ATPF Calibration Python package
--- CALIBRATION ---
"""

## IMPORT
import os ,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"..",".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute 
import json

from atpf.utils import my_plotter as mp
from atpf import ATPFSolver

mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Latin Modern Roman', mrksze=7, 
             fontsize = 11, labelfontsize=11, tickfontsize=9, 
             aspect_ratio=[13.7, 6])

def read_exp(filename, path='data/exp_data'):
    full_path = os.path.join(os.path.dirname( __file__ ),"..","..", path, filename)
    
    # Import the data from .json
    with open(full_path + '_meta.json', 'r') as file:
        jd = json.load(file)
    
    # Only proceed if the file corresponds to an online measurement
    if not jd['online_measurement'] or not jd['experiment_type']=='constant_u':
        return None

    else:
        w0_bot = jd['cF_bot_0']
        
        # Import the data from .csv
        df = pd.read_csv(full_path + '.csv', delimiter=',')
        
        t = df['t'].values*60
        w02 = df['c_top_3'].values*1e-2
        vg = df.loc[0,['Vd_1','Vd_2','Vd_3']].values
        kappa = np.array([df['kappa_bot_1'].values, df['kappa_bot_2'].values, df['kappa_bot_3'].values]).T
        Q_top = df['Qd_top'].values[0]*1e-6/60
        Q_bot = df['Qd_bot'].values[0]*1e-6/60
        
        exp_data = {'t': t,
                    'w02': w02,
                    'vg': vg,
                    'kappa': kappa,
                    'Q_top': Q_top,
                    'Q_bot': Q_bot,
                    'w0_bot': w0_bot}
        
        return exp_data

def read_folder(path='data/exp_data', save=True):
    full_path = os.path.join(os.path.dirname( __file__ ),"..","..", path)
    all_files = os.listdir(full_path)
    
    # Initialize result lists
    files = []
    exp_names = []
    exp_data = []
    
    # Identify unique filenames without extention
    for i in range(len(all_files)):
        # Identify current filename, continue if it is not .csv 
        if all_files[i].rsplit('.', 1)[-1] == 'csv':
            files.append(all_files[i].rsplit('.', 1)[0])
        else:
            continue
        
    # Read every file_name and store result in exp_data list (corresponding index to file_names)
    for i in range(len(files)):
        e = read_exp(files[i],path)
        
        # if None is returned (no online measurement), do not save!
        if e is not None:
            exp_names.append(files[i])
            exp_data.append(e)
    
    if save:
        # print(full_path + '/0_processed_data.npy')
        np.save(full_path + '/0_processed_data.npy',{'exp_names':exp_names,
                                                    'exp_data':exp_data})
    
    return exp_names, exp_data

def simulate_DOE(MP, case='all', path='data/exp_data', cf_file='exp_data_study_config.py', verbose=1):
    # Import processed data
    full_path = os.path.join(os.path.dirname( __file__ ),"..","..", path)
    cf_path = os.path.join(os.path.dirname( __file__ ),"..","..", 'config')
    data = np.load(full_path + '/0_processed_data.npy', allow_pickle=True).item()
    
    exp_names = data['exp_names']
    exp_data = data['exp_data']
    
    # Loop through all experiments:
    sim_data = []
    for i in range(len(exp_names)):
        if verbose > 0:
            print('Simulating experiment ', exp_names[i])
        
        # Set up class instance
        a = ATPFSolver(cf_file=cf_file, cf_pth=cf_path, verbose=0)
        
        # Set correct experimental parameters
        a.vg = exp_data[i]['vg']
        a.t = exp_data[i]['t']
        a.t_max = max(a.t)
        a.Q_bot = exp_data[i]['Q_bot']
        a.Q_top = exp_data[i]['Q_top']
        a.kappa_data = {'t': exp_data[i]['t'],
                        'kappa': exp_data[i]['kappa']}
        a.w0_bot = exp_data[i]['w0_bot']

        # Set model parameters provided by MP
        a.R_max = MP['R_max']
        a.K = MP['K']
        a.k_A = MP['k_A']
        a.k_h = MP['k_h']
        
        # If a certain effect is neglected make adjustments here
        if case == 'no_f':
            # No flotation mass transfer
            a.fr_case = 'const'
            a.fr0 = 0
        elif case == 'no_e':
            # No extraction mass transfer
            a.A_tot = 0
        
        # Initialize calculation
        a.init_calc()

        # Solve the ATPF experiment and save solution at experimental timesteps
        a.solve(t_eval=a.t)
        
        t = a.t
        w02_mod = a.w[0,-1,:]
        w02_exp = exp_data[i]['w02']
        M_e_int = a.M_e_int
        M_f_int = a.M_f_int
        
        # Calculate RMSE for this experiment
        RMSE = np.sqrt(np.mean((w02_mod-w02_exp)**2))
        
        # Append data as dict into sim_data
        sim_data.append({'t': t,
                         'w02_mod': w02_mod,
                         'w02_exp': w02_exp,
                         'M_e_int': M_e_int,
                         'M_f_int': M_f_int,                         
                         'RMSE': RMSE})
    
    return exp_names, exp_data, sim_data

def opt_MP_DOE(param0, algo='minimize', crit='RMSE_full', case='all', path='data/exp_data', 
               cf_file='exp_data_study_config.py', verbose=1):
    
    if verbose > 0:
        print('Starting model parameter optimization..')
        if case == 'all':
            print('Including all transport mechanisms')        
        elif case == 'no_f':
            print('No flotation transport')
        elif case == 'no_e':
            print('No extraction transport')
        
    if algo == 'minimize':
        opt_res = minimize(cost_MP_DOE, param0, method='Nelder-Mead', 
                           args=(crit, case, path, cf_file, 1), #tol=1e-6,
                           bounds=((-8,-2),(1,1),(0,100),(0,1)), 
                           # bounds=((0,0),(0,0),(0,100),(0,1)), 
                           options={'maxiter':100})
        MP_opt = {'R_max': 10**opt_res.x[0],
              'K': opt_res.x[1],
              'k_A': opt_res.x[2],
              'k_h': opt_res.x[3]}
    elif algo == 'brute':
        opt_res = brute(cost_MP_DOE, ranges=((-8,-2),(0,100),(0,100),(0,10)),
                        args=(crit, case, path, cf_file, 1), Ns=5)
        
        MP_opt = {'R_max': 10**opt_res.x0[0],
              'K': opt_res.x0[1],
              'k_A': opt_res.x0[2],
              'k_h': opt_res.x0[3]}
    else:
        raise ValueError('Provide correct optimization algorithm')

    print('#####################')
    print(f'The optimized model parameters are {10**opt_res.x[0]:.2e} | {opt_res.x[1]:.1f} | {opt_res.x[2]:.1f} | {opt_res.x[3]:.1f}')
    
    return MP_opt
    
def cost_MP_DOE(param, crit='RMSE_full', case='all', path='data/exp_data', cf_file='exp_data_study_config.py', verbose=1):
    ### param is a list containing all parameters that require optimization
    ## param[0]: log10(R_max)
    ## param[1]: K
    ## param[2]: k_A
    ## param[3]: k_h
    R_max = 10**param[0]
    K = param[1]
    k_A = param[2]
    k_h = param[3]
    
    MP = {'R_max': 10**param[0],
          'K': param[1],
          'k_A': param[2],
          'k_h': param[3]}
        
    # Simulate DOE for current MP
    exp_names, exp_data, sim_data = simulate_DOE(MP, case, path=path, cf_file=cf_file, verbose=0)

    # Calculate loss criterion
    loss = 0
    # Loop through all experiments
    for i in range(len(exp_names)):
        if crit == 'RMSE_full':
            loss += np.sqrt(np.mean((sim_data[i]['w02_exp']-sim_data[i]['w02_mod'])**2))/len(exp_names)
    
    if verbose > 0:
        print(f'Current MP: {R_max:.2e} | {K:.1f} | {k_A:.3e} | {k_h:.3e} || Loss: {loss:.2e}')
            
    return loss

def visualize_MP_DOE(MP, case='all', path='data/exp_data', cf_file='exp_data_study_config.py'):
    # Simulate full 
    exp_names, exp_data, sim_data = simulate_DOE(MP, case=case, path=path, cf_file=cf_file, verbose=0)
    RMSE = sum([sim_data[i]['RMSE'] for i in range(len(exp_names))])/len(exp_names)
    
    # Create plot
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 5, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7/2])
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'h']
    
    fig, ax = plt.subplots(1,2) 
    for i in range(len(exp_names)):
        ax[0].scatter(sim_data[i]['w02_exp'], sim_data[i]['w02_mod'], 
                      edgecolor='k', zorder=1, label=exp_names[i],
                      marker=np.random.choice(markers))
    ax[0].axline((0, 0), slope=1, color=mp.red, linestyle='--', zorder=3)
    
    # Customize axes
    ax[0].set_xlabel(r'$w_{\mathrm{exp}}(0,2)$ / $-$')
    ax[0].set_ylabel(r'$w_{\mathrm{mod}}(0,2)$ / $-$')
    ax[0].text(0.98, 0.05, r"$\mathrm{RMSE}"+f"={RMSE:.3e}$", 
               transform=ax[0].transAxes, verticalalignment='bottom', 
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='w', alpha=1))

    ax[0].grid(True)
    ax[1].axis('off')  # Hide the axes of the right subplot
    ax[1].legend(*ax[0].get_legend_handles_labels(), loc='center', ncol=2)

    plt.tight_layout()
    
    # Generate export strings
    fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                               'export/w_exp_mod.pdf')
    
    # Export figure
    plt.savefig(fig_exp_pth)
    
    return exp_names, exp_data, sim_data 

def visualize_MP_exp(MP, name=None, idx=None, data=None, export=False,
                     path='data/exp_data', cf_file='exp_data_study_config.py'):
    
    # If data is not provided, resimulate full DOE
    if data is None:
        # Simulate full 
        exp_names, exp_data, sim_data = simulate_DOE(MP, path=path, cf_file=cf_file, verbose=0)
    else:
        exp_names, exp_data, sim_data = data['exp_names'], data['exp_data'], data['sim_data']
    
    
    # Find index 
    if name is None:
        if idx is None:
            raise ValueError('Provide either name or idx')
    else:
        try:
            idx = exp_names.index(name)
        except ValueError:
            print(f"'{name}' is not in exp_names")
    
    # Initialize
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7/2])
    
    # Plot
    fig, ax = plt.subplots()     
    ax.plot(sim_data[idx]['t']/60, sim_data[idx]['w02_exp'], marker='s', color=mp.green, mec='k', label='exp')
    ax.plot(sim_data[idx]['t']/60, sim_data[idx]['w02_mod'], marker='s', color=mp.red, mec='k', label='mod')
    
    # Customize axes
    ax.set_xlabel('$t$ / min')
    ax.set_ylabel('$w(0,2,t)$ / $-$')
    ax.set_title(f'Experiment: {exp_names[idx]}')
    ax.set_xlim([sim_data[idx]['t'][0]/60,sim_data[idx]['t'][-1]/60])
    ax.grid(True)
    plt.tight_layout()
    
    if export:
        plt.savefig('export/'+name+'.pdf')
    
    return ax, fig

def visualize_hist_RMSE(sim_data, export=False):
    # Initialize
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7/2])
    
    RMSE_arr = np.array([sim_data[i]['RMSE'] for i in range(len(sim_data))])
    
    # Plot
    fig, ax = plt.subplots()     
    ax.hist(RMSE_arr)
    
    # Customize axes
    ax.set_xlabel('RMSE / $-$')
    ax.set_ylabel('h / $-$')
    ax.grid(True)
    plt.tight_layout()
    
    if export:
        plt.savefig('export/RMSE_hist.pdf')
    
    return RMSE_arr
    
def visualize_hist_E_F(sim_data, export=False):
    # Initialize
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7/2])
    
    E_F_arr = np.array([sim_data[i]['M_e_int'][-1]/(sim_data[i]['M_e_int'][-1]+sim_data[i]['M_f_int'][-1]) for i in range(len(sim_data))])
    
    # Plot
    fig, ax = plt.subplots()     
    ax.hist(E_F_arr)
    
    # Customize axes
    ax.set_xlabel('$\dot{M}_{\mathrm{e,int}}/(\dot{M}_{\mathrm{e,int}}+\dot{M}_{\mathrm{f,int}})$ / $-$')
    ax.set_ylabel('h / $-$')
    ax.grid(True)
    plt.tight_layout()
    
    if export:
        plt.savefig('export/E_F_hist.pdf')
    
    return E_F_arr

def visualize_corr_h(k_h):
    
    # Generate data
    vg = np.linspace(1,100,1000)
    # k = 1+vg/(vg+k_h)
    # k = 1-np.exp(-k_h*vg)
    k = 1/(1+np.exp(-k_h*(vg-30)))
    
    # Initialize
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 11, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7/2])
    
    # Plot
    fig, ax = plt.subplots()     
    ax.plot(vg, k, marker='s', color=mp.green, mec='k')

    # Customize axes
    ax.set_xlabel('$\sum \dot{V}_g$ / $\mathrm{mL\,min^{-1}}$')
    ax.set_ylabel('$k*$ / $-$')
    ax.grid(True)
    plt.tight_layout()  
        
#%%            
if __name__ == '__main__':
    plt.close('all')
    file = '240209_MG_0.2_15-15-15_1.50_90min'
    data = read_exp(file)
    exp_names, exp_data = read_folder()
    param = [np.log10(1.5e-5), 1, 6, 0.025]
    # param = [0, 0, 13.2, 0.01]
    
    case = 'all'        # ['all', 'no_f', 'no_e']
    MP = {'R_max': 10**param[0],
          'K': param[1],
          'k_A': param[2],
          'k_h': param[3]}
    loss = cost_MP_DOE(param)
    MP = opt_MP_DOE(param, algo='minimize', case=case)
    # exp_names, exp_data, sim_data = simulate_DOE(MP_opt)
    
    # %%
    exp_names, exp_data, sim_data = visualize_MP_DOE(MP, case=case)
    data_dict = {'exp_names': exp_names,
                 'exp_data': exp_data,
                 'sim_data': sim_data}
    # %%
    exp_test = '240223_MG_0.2_15-15-15_1.50_90min'
    visualize_MP_exp(MP, data=data_dict, name=exp_test)
    exp_test = '240425_MG_0.2_10-00-00_1.00'
    visualize_MP_exp(MP, data=data_dict, name=exp_test)
    
    PLT_ALL = False
    if PLT_ALL:
        for i in range(len(exp_names)):
            visualize_MP_exp(MP, idx=i, data=data_dict)
    
    #%%
    visualize_corr_h(MP['k_h'])
    RMSE = visualize_hist_RMSE(sim_data)
    E_F = visualize_hist_E_F(sim_data)