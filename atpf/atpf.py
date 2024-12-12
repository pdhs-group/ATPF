"""
ATPF Model Python package
--- ATPF CLASS ---
"""

## IMPORT
import numpy as np
import math
import os, sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import importlib
import pandas as pd
from scipy.integrate import solve_ivp

class ATPFSolver():
    
    def __init__(self, cf_file=None, cf_pth=None, verbose=1):
        # Fallback config file
        if cf_file is None:
            cf_file = 'atpf_config_batch.py'
        # Fallback config path
        if cf_pth is None:
            cf_pth = os.path.join('..','config')
        
        cf_abs_pth = os.path.join(cf_pth, cf_file)
        spec = importlib.util.spec_from_file_location(os.path.splitext(cf_file)[0], cf_abs_pth)
        cf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cf_module)
        cf = cf_module.config
        
        # Assign attributes from the configuration file to the atpf instance
        for key, value in cf.items():
            setattr(self, key, value)
                    
        # Set baseline path (to parent folder)
        self.pth = os.path.dirname(os.path.dirname(__file__))
            
        ## Note: Add 1 to horizontal compartments, as FEED compartments
        self.shape = (2,self.n_comp+1)
                
        # Data files
        self.gas_v_d_file = os.path.join(self.pth,'data/gas_correlations/',
                                         'gas_v_d_'+self.inlet_g_medium+'.npy')
        
        # Consistency check
        self.check_consistency()
        
        # Calculate all other arrays
        self.init_calc()
        
        # Report that import was successful
        if verbose >= 1:
            print(f'Imported the file {cf_file}. The instance has {self.n_comp} compartment(s).')
    
    # Check if inputs are valid
    def check_consistency(self):
        if self.n_comp != len(self.vg):
            raise ValueError('(!) Number of compartments does not match input for v_g') 
        
        if self.n_comp != len(self.kappa_data['kappa'][0]):
            raise ValueError('(!) Number of compartments does not match input for kappa_data')   
            
        if len(self.kappa_data['t']) != len(self.kappa_data['kappa'][:,0]):
            raise ValueError('(!) Number of provided timesteps does not match input for kappa_data')   
        
        if not self.inlet_g_medium in ['glass','metal','twill']:
            raise ValueError("(!) Select correct gas inlet medium. Valid options are ['glass','metal','twill']")
            
    # Initial calculations
    def init_calc(self):
        # Initial concentration array c in [mol/m³] and w in [w/w]        
        self.c0_top = self.w_to_c(self.w0_top)  # Feed concentration top [mol/m³]
        self.c0_bot = self.w_to_c(self.w0_bot)  # Feed concentration bot [mol/m³]          
        if self.w0_case == 'empty':
            self.c0 = np.zeros(self.shape)
            self.c0[0,0] = self.c0_top
            self.c0[1,0] = self.c0_bot
        if self.w0_case == 'full':
            self.c0 = np.stack([np.ones(self.shape[1])*self.c0_top,
                                np.ones(self.shape[1])*self.c0_bot])
        self.w0 = self.c_to_w(self.c0)
                
        # Compartment volumes [m³]
        # Geometric Parameters
        self.A = self.A_tot/self.n_comp                        # Interface area comp [m²]
        self.v_top = self.v_top_tot/self.n_comp                # Volume of a top comp [m³]
        self.v_bot = self.v_bot_tot/self.n_comp                # Volume of a bot comp [m³]
        
        self.v = np.stack([np.ones(self.shape[1])*self.v_top,
                           np.ones(self.shape[1])*self.v_bot]) 
        
        # Volume fluxes (convection) [1/s]        
        ## Flux: Part of the volume exchanged per second
        self.f_top = self.Q_top/self.v_top      # Flux top phase [1/s]
        self.f_bot = self.Q_bot/self.v_bot      # Flux bot phase [1/s]
        self.f = np.stack([np.ones(self.shape[1])*self.f_top,
                           np.ones(self.shape[1])*self.f_bot]) 
                
    
        # Constant flotation rate array
        self.fr = np.ones(self.shape[1])*self.fr0
        self.fr[0] = 0
        
        # Calculate enzmye concentration in bot phase in equilibrium
        self.c_inf_bot_0 = (self.c0_bot*(self.v_bot/self.v_top)+self.c0_top)/(self.K_p+(self.v_bot/self.v_top))
        
        # Initialize default discrete time-step array
        self.t_eval = np.linspace(0,self.t_max,10) 
        
        # Load bubble correlation
        gas_v_d_data = np.load(self.gas_v_d_file,allow_pickle=True).item()
        self.P = gas_v_d_data['P']
        self.vg_range = [gas_v_d_data['vg_min'],gas_v_d_data['vg_max']]
        
        # Transfer kappa data into volume ratio and add to dictionary
        self.kappa_data['v'] = self.kappa_to_v(self.kappa_data['kappa'])
        
        # Consistency check
        self.check_consistency()
        
    # Right hand side function to solve the transport equations suitable for use with scipy.solve_ivp
    def transport(self, t, c_1d):
        # Reshape 1D array into 2D representation (easier to understand)
        # Last two entries are for M_e_integral and M_f_integral!
        c = c_1d[:-2].reshape(self.shape)
        M_c = np.zeros(c.shape)
        M_e = np.zeros(c.shape)
        M_f = np.zeros(c.shape)
        
        # Define whether or not a constant flotation rate should be used
        if self.fr_case == 'const':
            fr = self.fr
        else:
            fr = np.zeros(self.shape[1])
            # Calculate CURRENT flotation rate for each bot compartment
            for i in range(1,len(fr)):
                if c[1,i] == 0:
                    fr[i] = 0
                else:
                    # Note: vg[i+1] to account for feed compartment
                    fr[i] = self.fr_from_v(self.vg[i-1],c[1,i],ad_case=self.ad_case)
            
        # Get currenct correction term for mixing zone height
        if self.h_case == 'const':
            h_corr = np.ones(self.vg.shape) 
        elif self.h_case == 'corr_individ':
            # h_corr = self.vg/(self.vg+self.k_h)
            h_corr = 1/(1+np.exp(-self.k_h*(self.vg-30)))
        else:
            vg_sum = np.sum(self.vg)
            # h_corr = np.ones(self.vg.shape)*vg_sum/(vg_sum+self.k_h)
            # h_corr = np.ones(self.vg.shape)*(1-np.exp(-self.k_h*vg_sum))
            h_corr = np.ones(self.vg.shape)/(1+np.exp(-self.k_h*(vg_sum-30)))
            
        # Calculate the current effective interface
        if self.kappa_case == 'const':
            # Use first entry in kappa_data
            A_e = self.A*(1+h_corr*self.k_A*6*(1+1/self.kappa_data['v'][0,:])) 
        else:
            # Look up corresponding time
            idx = np.searchsorted(self.kappa_data['t'],t,side='right')-1
            A_e = self.A*(1+h_corr*self.k_A*6*(1+1/self.kappa_data['v'][idx,:]))
        
        # Move through all compartments horizontally (start at 1 to skip feed)
        for i in range(1,self.shape[1]):
            ## (1) Convection 
            ### IN
            M_c[:,i] += c[:,i-1]*self.f[:,i-1]*self.v[:,i-1]/self.v[:,i]
            ### OUT
            M_c[:,i] -= c[:,i]*self.f[:,i]
            
            ## (2) Flotation
            ### IN
            M_f[0,i] += c[1,i]*fr[i]*self.v[1,i]/self.v[0,i]
            ### OUT
            M_f[1,i] -= c[1,i]*fr[i]
            
            ## (3) Diffusion across bot/top interface
            ## Note: Defined as flow from bot to top (can be negative)
            ### Equilibrium concentration in bot phase based on current concentrations and K_p
            c_inf_bot = (c[1,i]*(self.v_bot/self.v_top)+c[0,i])/(self.K_p+(self.v_bot/self.v_top))
            ### IN
            M_e[0,i] += (c[1,i]-c_inf_bot)*A_e[i-1]*self.k_i/self.v[0,i]
            ### OUT
            M_e[1,i] -= (c[1,i]-c_inf_bot)*A_e[i-1]*self.k_i/self.v[1,i]
         
        dcdt = M_c + M_e + M_f
        
        return np.append(dcdt.reshape(-1),
                         np.array([np.sum(M_e[0,:]),np.sum(M_f[0,:])]))
        
    # Solver for atpf
    def solve(self, t_eval=None, verbose=0):
        # Use default time-step array if None (other) is provided
        if t_eval is None:
            t_eval = self.t_eval
            
        # Solve ODE for given parameters. Reshape c array to 1D
        res = solve_ivp(fun=self.transport, 
                        t_span=[0,self.t_max], 
                        y0=np.append(self.c0.reshape(-1),
                                  np.array([0,0])),
                        t_eval=t_eval)
            
        # Extract info
        self.t = res.t
        self.c = res.y[:-2,:].reshape((self.shape[0],self.shape[1],len(self.t)))
        self.M_e_int = res.y[-2,:]
        self.M_f_int = res.y[-1,:]
        self.w = np.zeros(self.c.shape)
        self.w[0,:,:] = self.c_to_w(self.c[0,:,:], phase='top')
        self.w[1,:,:] = self.c_to_w(self.c[1,:,:], phase='bot')
        
        # Calculate Separation efficiency 
        self.E = self.E_from_w()
    
    # Calculate separation efficiency
    def E_from_w(self):
        # Calculation depends on conti or batch (Q_bot)
        if self.Q_bot == 0:
            E = 100*(self.v_top*self.rho_top*self.w[0,-1,:])/(self.v_bot*self.rho_bot*self.w0_bot)
        else:
            E = 100*(self.Q_top*self.rho_top*self.w[0,-1,:])/(self.Q_bot*self.rho_bot*self.w0_bot)
            
        return E
    # Calculate flotation rate for given gas volume flow and concentration
    def fr_from_v(self,v_gas, c, ad_case='ss', verbose=0):
        # Calculate mean bubble diameter 
        # Do not allow extrapolation (otherwise use calibration edges)
        if v_gas > self.vg_range[1]:
            db = self.P[0]*self.vg_range[1]+self.P[1]*np.log(self.vg_range[1])+self.P[2]
        elif v_gas < self.vg_range[0]:
            db = self.P[0]*self.vg_range[0]+self.P[1]*np.log(self.vg_range[0])+self.P[2]
        else:
            db = self.P[0]*v_gas+self.P[1]*np.log(v_gas)+self.P[2]
        db *= 1e-6
  
        # Calculate mean residence time of bubble based on Stokes
        vb = self.g*(self.rho_bot-self.rho_gas)*db**2/(18*self.eta_bot)
        tb = self.h_bot/vb
                
        # Calculate bubble surface area and volume per bubble
        Ab_1 = math.pi*db**2
        Vb_1 = math.pi*db**3/6 
        
        # Calculate number of bubbles and total area per time
        v_gas_SI = v_gas*1e-6/60 # [m³/s]
        # v_gas_ss = v_gas_SI*tb # ! THINK ! is *tb correct or just use 1s (also below!)
        Nb = v_gas_SI/Vb_1  # [1/s]
     
        Ab = Ab_1*Nb        # [m²/s] of fresh interface 
        
        if ad_case == 'dyn':
            # Solve adsorption kinetic
            # Assumption: R_0 = 0 (bubble unloaded at first)
            # Assumption: c is constant throughout (simple kinetic)
            res = solve_ivp(adsorption_kinetik, [0,tb], 
                            [0], args=(self.k_a,self.k_d,self.R_max,c))
            
            # Final loading of bubbles when entering the top phase
            R = res.y[0][-1]
                        
            # Final bubble loading 
            theta = R/self.R_max
        else:
            # Bubble loading is instant in steady state
            R = self.R_max*self.K*c/(1+self.K*c)
            
            # bubble loading 
            theta = R/self.R_max
        
        # Calculate concentration change rate
        dcdt = R*Ab/self.v_bot     # [mol/m³s]
        
        # Flotation rate fr is defined as dc/dtc
        fr = dcdt/c
        
        if verbose > 0:
            print(f'Relativ surface loading bubbles: {100*theta:.1f} %')
            # print(f'Absolute surface loading (R): {R:.3e} mol/m2')
        if verbose > 1:            
            print(f'bubble diameter: {db*1e6:.3f} mum')
            print(f'adsorption time: {tb:.3f}')
            print(f'total bubble surface area: {Ab:.3e} m**2')
            print(f'number of bubbles in steady state: {Nb:.3e}')
            
        return fr
    
    # Correlation from conductivity to volume ratio mixing zone
    def kappa_to_v(self, kappa):
        # kappa in mS/cm
        return -0.0146*kappa+0.767
        
    # Converting w/w in mol/m³
    def w_to_c(self,w,phase='bot'):
        if phase == 'bot':
            return w*self.rho_bot/self.M_enz
        else:
            return w*self.rho_top/self.M_enz
        
    # Converting mol/m³ in w/w
    def c_to_w(self,c,phase='bot'):
        if phase == 'bot':
            return c*self.M_enz/self.rho_bot
        else:
            return c*self.M_enz/self.rho_top

        
# Adsorption kintetic DYNAMIC c:
# Note: This should be integrated directly in fr_from_v to account for dcdt
# def adsorption_kinetik_complex(t,y,k_a,k_d,R_max,A,V):
#     # t in s
#     # k_a in mol/m³s
#     # k_d in 1/s
#     # R_max in mol/m2
#     # A in m²
#     # V in m³
#     c = y[0]    # Bulk concentration
#     R = y[1]    # Surface loading
#     dRdt = k_a*c*(R_max-R)-k_d*R
#     dcdt = -dRdt*A/V
#     return [dcdt,dRdt]

# Adsorption kintetic CONSTANT c:
def adsorption_kinetik(t,R,k_a,k_d,R_max,c):
    # t in s
    # k_a in mol/m³s
    # k_d in 1/s
    # R_max in mol/m2
    dRdt = k_a*c*(R_max-R)-k_d*R
    return dRdt
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        