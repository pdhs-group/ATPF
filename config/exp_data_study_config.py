import numpy as np

config = {
        # Define shape of result arrays
        ## Note: Add 1 to horizontal compartments, as FEED compartments
        'n_comp': 3, 
                
        # General options
        ## fr_case in {'const','dyn'}: How is flotation rate calculated [dc/(dt*c)] 
        ## 'const': Assume constant floation rate 
        ## 'dyn': Calculate based on bubble size and current concentration 
        'fr_case': 'dyn',   
        ## ad_case in {'ss','dyn'}: Adsorption model
        ## 'ss': Steady state Langmuir model 
        ## 'dyn': Dynamic (time dependent) adsorption 
        'ad_case': 'ss',                   
        
        # Geometric Parameters
        'v_top_tot': 1e-4,                   # Total volume top phase [m³]
        'v_bot_tot': 5e-4,                   # Total volume bot phase [m³]
        'A_tot': 5e-5,                       # Total interphase area [m²]
        'h_bot': 0.097,                      # Height of bot phase [m]
        
        # Process Parameters
        'vg': [30,20,10],                    # Gas volume flow per compartment [mL/min]
        'Q_top': 1.67e-6/60,                 # Volume flow top [m³/s]
        'Q_bot': 8.33e-6/60,                 # Volume flow bot [m³/s]
        't_max': 60*60,                      # Process time [s]
        'w0_top': 0,                         # Feed concentration top [w/w]
        'w0_bot': 0.02,                      # Feed concentration bot [w/w]
        'w0_case':'full',                    # 'empty': w_i(0)=0, 'full': w_i(0)=w0
        'inlet_g_medium': 'twill',           # Gassing medium used ['glass','metal',twill']
        
        # Material Parameters             
        'fr0': 5e-4,                         # Flotation rate constant [1/s]
        'rho_bot': 1192,                     # Density bot phase [kg/m³]
        'rho_top': 1091,                     # Density top phase [kg/m³]
        'rho_gas': 1.225,                    # Density gas [kg/m³]
        'eta_bot': 4.48e-3,                  # Viscosity bot phase [Pa s]
        'eta_top': 15.72e-3,                 # Viscosity top phase [Pa s]
        'M_enz': 14.5,                       # Molar mass enzyme [kg/mol]
        'g': 9.81,                           # Gravitational constant [m²/s]
        
        # Diffusion specific parameters
        'kappa_case':'exp',                  # Use experimental conductivity data? ['const','exp']
        # Conductivity data. kappa[i,:]: Conductivity at time t[i]
        'kappa_data':{'t': np.array([0]),              
                      'kappa': np.array([[50,50,50]])},
        'k_A': 1,                            # Height mixing zone h / droplet diameter d [-]
        'h_case': 'corr_sum',                # Correct h based on vg? ['const','corr_individ','corr_sum']
        'k_h': 0.1,                          # Correction factor h'=h*vg/(vg+k_h)
        'k_i': 8e-5,                         # Mass transport coefficient through interface [m/s]
        'K_p': 16.51,                        # Partition coefficient K_p=c_top/c_bot [-]
        
        # Adsorption specific parameters
        'k_a': 2e-1,                         # Adsorption rate [m³/mol s]
        'k_d': 1e-2,                         # Desorption rate [1/s]
        'K': 20,                             # Steady state Langmuir constant [m³/mol]
        'R_max': 5e-7,                       # Maximum surface loading [mol/m²]
}