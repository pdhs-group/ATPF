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
from scipy.optimize import minimize, brute, differential_evolution 
import json
from SALib.sample import saltelli
from SALib.analyze import sobol

from atpf.utils import my_plotter as mp
from atpf import ATPFSolver

mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Latin Modern Roman', mrksze=7, 
             fontsize = 11, labelfontsize=11, tickfontsize=9, 
             aspect_ratio=[13.7, 6])

# Read a single experiment
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

# Read an experimental folder
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
        # print(files[i])
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

# Simulate a full experimental design
def simulate_DOE(MP, case='all', path='data/exp_data', cf_file='exp_data_study_config.py', 
                 verbose=1, sensitivity=False):
    # Import processed data
    full_path = os.path.join(os.path.dirname( __file__ ),"..","..", path)
    cf_path = os.path.join(os.path.dirname( __file__ ),"..","..", 'config')
    data = np.load(full_path + '/0_processed_data.npy', allow_pickle=True).item()
    
    exp_names = data['exp_names']
    exp_data = data['exp_data']
    
    # Loop through all experiments:
    sim_data = []
    for i in range(len(exp_names)):
        # print(exp_names[i])
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
        a.d_d = MP['d_d']
        if sensitivity:
            a.K = MP['K']
        
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

# Optimize model parameters to experimental data
def opt_MP_DOE(param0, algo='minimize', crit='RMSE_full', case='all', path='data/exp_data', 
               cf_file='exp_data_study_config.py', bounds=None, verbose=1):
    
    if verbose > 0:
        print('Starting model parameter optimization..')
        if case == 'all':
            print('Including all transport mechanisms')        
        elif case == 'no_f':
            print('No flotation transport')
        elif case == 'no_e':
            print('No extraction transport')
    
    # Set bounds of specific parameter equal -> Optimizer knows that this parameter is irrelevant
    if bounds is None:
        if case == 'no_f':
            bounds = ((param0[0],param0[0]),(1e-6,1e-3))
        elif case == 'no_e':
            bounds = ((-7,-3),(param0[1],param0[1]))
        else:
            bounds = ((-7,-3),(1e-6,1e-3))        

        
    if algo == 'minimize':
        print('Using scipy.minimize( )')   
        opt_res = minimize(cost_MP_DOE, param0, method='Nelder-Mead', 
                           args=(crit, case, path, cf_file, 1), #tol=1e-6,
                           bounds=bounds, 
                           options={'maxiter':100})
        sol = opt_res.x

    elif algo == 'evo':
        print('Using scipy.differential_evolution( )')  
        opt_res = differential_evolution(cost_MP_DOE, bounds, maxiter=20, popsize=4,
                                         seed = 1, disp=True, polish=True, init='sobol',
                                         args=(crit, case, path, cf_file, 1))
        sol = opt_res.x
        
    elif algo == 'brute':        
        print('Using scipy.brute( )')  
        opt_res = brute(cost_MP_DOE, ranges=bounds,
                        args=(crit, case, path, cf_file, 1), Ns=5)
        sol = opt_res.x0
        
    else:
        raise ValueError('Provide correct optimization algorithm')
        
            
    MP_opt = {'R_max': 10**sol[0],
              'd_d': sol[1]}

    print('#####################')
    print(f'The optimized model parameters are {10**sol[0]:.3e} | {sol[1]:.3e}')
    
    exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                           'export/opt_'+algo+'_'+case+'.npy')
    np.save(exp_pth, MP_opt)
    
    return MP_opt

# Cost function used during optimization    
def cost_MP_DOE(param, crit='RMSE_full', case='all', path='data/exp_data', 
                cf_file='exp_data_study_config.py', verbose=1,
                sensitivity=False):
    
    ### param is a list containing all parameters that require optimization
    ## param[0]: log10(R_max)
    ## param[1]: d_d
    R_max = 10**param[0]
    d_d = param[1]
    
    MP = {'R_max': 10**param[0],
          'd_d': param[1]}
    
    # In case of sensitivity analysis use all three possible model parameters
    if sensitivity:
        K = param[1]
        d_d = param[2]
        MP = {'R_max': 10**param[0],
              'K': param[1],
              'd_d': param[2]}
            
    # Simulate DOE for current MP
    exp_names, exp_data, sim_data = simulate_DOE(MP, case, path=path, cf_file=cf_file, 
                                                 verbose=0, sensitivity=sensitivity)

    # Calculate loss criterion
    loss = 0
    # Loop through all experiments
    for i in range(len(exp_names)):
        if crit == 'RMSE_full':
            loss += np.sqrt(np.mean((sim_data[i]['w02_exp']-sim_data[i]['w02_mod'])**2))/len(exp_names)
    
    if verbose > 0:
        if sensitivity:
            print(f'Current MP: {R_max:.2e} | {K:.3e} | {d_d:.3e} || Loss: {loss:.2e}')  
        else:
            print(f'Current MP: {R_max:.2e} | {d_d:.3e} || Loss: {loss:.2e}')
            
    return loss

# Sobol sensitivity analysis
def sensitivity_analysis(N_samples=1000, second_order=True, 
                         case='all', path='data/exp_data', cf_file='exp_data_study_config.py'):
    problem = {
    'num_vars': 3,
    'names': ['R_max', 'K', 'd_d'],
    'bounds': [[-7, -4],        # Range for R_max (log)
               [1, 200],        # Range for K (lin)
               [1e-5, 1e-3]]    # Range for d_d (lin)
    }

    param_values = saltelli.sample(problem, N_samples, calc_second_order=second_order)
    print(f'Number of samples: {param_values.shape[0]}')
    # param_values[:,3] = 10**param_values[:,3]
    Y = np.zeros(param_values.shape[0])
    for i in range(param_values.shape[0]):
        Y[i] = cost_MP_DOE(param_values[i,:], case=case, path=path, cf_file=cf_file, 
                           verbose=1, sensitivity=True)
    sobol_indices = sobol.analyze(problem, Y, calc_second_order=second_order)
    
    exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                           'export/sobol_'+str(N_samples)+'.npy')
    np.save(exp_pth, {'sobol_indices': sobol_indices,
                      'problem': problem})
    
    
    print('The Total Sobol Idices are:')
    print(sobol_indices['ST'])
    
    return sobol_indices
    
def visualize_MP_DOE(MP, case='all', path='data/exp_data', cf_file='exp_data_study_config.py', legend=True):
    # Simulate full 
    exp_names, exp_data, sim_data = simulate_DOE(MP, case=case, path=path, cf_file=cf_file, verbose=0)
    RMSE = sum([sim_data[i]['RMSE'] for i in range(len(exp_names))])/len(exp_names)
    
    # Create plot
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 5, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7/2])
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'h']
    
    if legend:
        fig, ax = plt.subplots(1,2) 
        for i in range(len(exp_names)):
            ax[0].scatter(sim_data[i]['w02_exp'], sim_data[i]['w02_mod'], 
                          edgecolor='k', zorder=1, label=exp_names[i],
                          marker=np.random.choice(markers),
                          linewidths=0.5, alpha=0.7)
        ax[0].axline((0, 0), slope=1, color=mp.red, linestyle='--', zorder=3)
        
        # Customize axes
        ax[0].set_xlabel(r'$w_{\mathrm{exp}}(0,2)$ / $-$')
        ax[0].set_ylabel(r'$w_{\mathrm{mod}}(0,2)$ / $-$')
        ax[0].text(0.98, 0.05, r"$\mathrm{RMSE}"+f"={RMSE:.2e}$", 
                   transform=ax[0].transAxes, verticalalignment='bottom', 
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    
        ax[0].grid(True)
        ax[1].axis('off')  # Hide the axes of the right subplot
        ax[1].legend(*ax[0].get_legend_handles_labels(), loc='center', ncol=2)
    
        plt.tight_layout()
        
        # Generate export strings
        fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                                   'export/w_exp_mod_leg.pdf')
    
    else:
        fig, ax = plt.subplots() 
        for i in range(len(exp_names)):
            ax.scatter(sim_data[i]['w02_exp'], sim_data[i]['w02_mod'], 
                          edgecolor='k', zorder=1, label=exp_names[i],
                          marker=np.random.choice(markers),
                          linewidths=0.5, alpha=0.7)
        ax.axline((0, 0), slope=1, color=mp.red, linestyle='--', zorder=3)
        
        # Customize axes
        ax.set_xlabel(r'$w_{\mathrm{exp}}(0,2)$ / $-$')
        ax.set_ylabel(r'$w_{\mathrm{mod}}(0,2)$ / $-$')
        ax.text(0.98, 0.05, r"$\mathrm{RMSE}"+f"={RMSE:.3e}$", 
                   transform=ax.transAxes, verticalalignment='bottom', 
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    
        ax.grid(True)
    
        plt.tight_layout()
        
        # Generate export strings
        fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                                   'export/w_exp_mod_noleg.pdf')
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
        fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                                   'export/individ_exp/'+name+'.pdf')
        plt.savefig(fig_exp_pth)
    
    return ax, fig

def visualize_vg_RMSE_EF(sim_data, exp_data, vg_case='sum', sort='vg', 
                         export=False, export_name=''):
    # Initialize
    mp.init_plot(scl_a4=1, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 12, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[13.7, 13.7/2])
    
    RMSE_arr = np.array([sim_data[i]['RMSE'] for i in range(len(sim_data))])
    E_F_arr = np.array([sim_data[i]['M_e_int'][-1]/(sim_data[i]['M_e_int'][-1]+sim_data[i]['M_f_int'][-1]) for i in range(len(sim_data))])
    if vg_case == 'sum':
        vg_arr = np.array([sum(exp_data[i]['vg']) for i in range(len(exp_data))])
    else:
        vg_arr = np.array([max(exp_data[i]['vg']) for i in range(len(exp_data))])
    N = np.arange(len(RMSE_arr))
    
    # Linear approximations
    vg_cont = np.linspace(min(vg_arr), max(vg_arr),100)
    c_RMSE = np.polyfit(vg_arr, RMSE_arr, 1)
    RMSE_cont = np.polyval(c_RMSE, vg_cont)
    c_E_F = np.polyfit(vg_arr, E_F_arr, 1)
    E_F_cont = np.polyval(c_E_F, vg_cont)
    
    # Plot
    fig, ax1 = plt.subplots()  
    ax2 = ax1.twinx() 

    ax1.scatter(vg_arr, RMSE_arr, edgecolor='k', color=mp.green, 
               marker='s', zorder=2, label=r'RMSE')
    ax2.scatter(vg_arr, E_F_arr, edgecolor='k', color=mp.red, 
               marker='s', zorder=2, label=r'Transport')

    ax1.plot(vg_cont, RMSE_cont, color=mp.green, marker=None, linestyle='-.',
             zorder=0)
    ax2.plot(vg_cont, E_F_cont, color=mp.red, marker=None, linestyle='-.',
             zorder=0)
    
    # Customize axes
    if vg_case == 'sum':
        ax1.set_xlabel(r'$\sum \dot{V}_g$ / $\mathrm{mL\,min^{-1}}$')
    else:
        ax1.set_xlabel(r'max $\dot{V}_g$ / $\mathrm{mL\,min^{-1}}$')

    ax1.set_ylabel('RMSE / $-$')
    ax2.set_ylabel(r'$\dot{M}_{\mathrm{e,int}}/(\dot{M}_{\mathrm{e,int}}+\dot{M}_{\mathrm{f,int}})$ / $-$')
    ax1.legend(loc='upper left')
    ax2.legend(loc='center right')
    # ax1.set_ylim([0,ax1.get_ylim()[1]])
    ax1.set_ylim([0,np.ceil(max(RMSE_arr)*1000)/1000+0.002])
    ax2.set_ylim([0,1])
    
    ax1.grid(True)
    plt.tight_layout()
    
    fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                               'export/vg_RMSE_EF_'+export_name+'.pdf')
    
    if export:
        plt.savefig(fig_exp_pth)
    
    return RMSE_arr, E_F_arr, vg_arr

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
    ax.set_xlabel(r'$\dot{M}_{\mathrm{e,int}}/(\dot{M}_{\mathrm{e,int}}+\dot{M}_{\mathrm{f,int}})$ / $-$')
    ax.set_ylabel('h / $-$')
    ax.grid(True)
    plt.tight_layout()
    
    if export:
        plt.savefig('export/E_F_hist.pdf')
    
    return E_F_arr


def visualize_sensitivity(sob_i=None, data='sobol_10', export=False):
    
    # If data is not provided, load default
    if sob_i is None: 
        file = os.path.join(os.path.dirname( __file__ ),"..","..",
                                'export/',data+'.npy')
        sob_data = np.load(file, allow_pickle=True).item()
        sob_i = sob_data['sobol_indices']
        prob = sob_data['problem']
     
    # Create plot
    mp.init_plot(scl_a4=2, page_lnewdth_cm=13.7, fnt='Arial', mrksze=4, 
                 fontsize = 5, labelfontsize=11, tickfontsize=9, 
                 aspect_ratio=[1, 1])
    
    fig, ax = plt.subplots() 

    ax.bar(np.arange(len(sob_i['ST'])), sob_i['ST'], color=mp.green)
    ax.set_xticks(np.arange(len(sob_i['ST'])))
    lbls = [r'$\Gamma_{\mathrm{max}}$', '$K$', r'$d_{\mathrm{d}}$']
    ax.set_xticklabels(lbls)
 
    ax.set_ylabel(r'Total Sobol Index $S_T$')

    ax.grid(axis='y')

    plt.tight_layout()
    
    # Generate export strings
    fig_exp_pth = os.path.join(os.path.dirname( __file__ ),"..","..",
                               'export/sensitivity'+data+'.pdf')
    
    # Export figure
    plt.savefig(fig_exp_pth)
    
    return sob_i

# Generate run table for latex
def generate_run_table():
    no = 1 
    rt = ''
    
    # Read experimental folder
    exp_names, exp_data = read_folder()
    
    for d in exp_data:
        rt += str(no) + '&' + str(d['w0_bot']*100) + '&' + str(max(d['t'])/60) + '&' + f"{d['Q_bot']*1e6*60:.2f}" + '&' + f"{d['Q_top']*1e6*60:.2f}" + '&'
        rt += str(d['vg'][0]) + '&' + str(d['vg'][1]) + '&' + str(d['vg'][2]) + '\\\\' + r' \hline' + '\n' 
        no += 1
    return rt
    
#%% MAIN           
if __name__ == '__main__':
    OPT = False                      # Set to true to optimize model parameters to experiments
    SENSITIVITY = False              # Set to true to perform sensitivity analysis
    READ_EXP = True                 # Only read experiments
    
    if OPT:        
        plt.close('all')
        
        # Usage example: Reading a single experiment
        # file = '240209_MG_0.2_15-15-15_1.50_90min'
        # data = read_exp(file)
        
        # Read all experimental data
        exp_names, exp_data = read_folder()
        
        # Define initial guess for optimization procedure
        param = [np.log10(1e-5), 1e-5] 
        
        # Define case
        ## 'all': flotation and extraction (case I)
        ## 'no_f': no flotation (case II)
        ## 'no_e': no extraction (case III)
        case = 'all'        # ['all', 'no_f', 'no_e']
        
        # Define optimization algorithm
        ## 'minimize': scipy.minimize
        ## 'evo': genetic algorithm
        ## 'brute': grid search
        algo = 'evo'
        MP = {'R_max': 10**param[0],
              'd_d': param[1]}
        #loss = cost_MP_DOE(param)
        
        # Call optimizer
        MP_opt = opt_MP_DOE(param, algo=algo, case=case)
        #exp_names, exp_data, sim_data = simulate_DOE(MP_opt)
        
        # Visualize results
        exp_names, exp_data, sim_data = visualize_MP_DOE(MP_opt, case=case, legend=False)
        data_dict = {'exp_names': exp_names,
                     'exp_data': exp_data,
                     'sim_data': sim_data}

        exp_test = '240425_MG_0.2_10-00-00_1.00'
        visualize_MP_exp(MP, data=data_dict, name=exp_test, export=True)
        exp_test = '241114_MG_0.2_30-20-10_1.00_45min'
        visualize_MP_exp(MP, data=data_dict, name=exp_test, export=True)
        exp_test = '241223_MG_0.2_1.5-1-1_1.0_V1'
        visualize_MP_exp(MP, data=data_dict, name=exp_test, export=True)
        exp_test = '240815_MG_0.2_30-20-10_3.00'
        visualize_MP_exp(MP, data=data_dict, name=exp_test, export=True)
        exp_test = '240826_MG_0.2_30-20-10_2.00'
        visualize_MP_exp(MP, data=data_dict, name=exp_test, export=True)
        
        RMSE_arr, E_F_arr, vg_arr = visualize_vg_RMSE_EF(sim_data, exp_data, vg_case='sum', 
                                                         export=True, export_name=case+'_sum_vg')
    
    if SENSITIVITY:
        # Perform sensitivity analysis
        # sob_i = sensitivity_analysis(N_samples=1024)
        sob_i = sensitivity_analysis(N_samples=8)
        
    if READ_EXP:
        exp_names, exp_data = read_folder()
        rt = generate_run_table()
        
        
    