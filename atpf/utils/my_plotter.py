import matplotlib.pyplot as plt

#%% Define KIT colors
transparent = (1.0, 1.0, 1.0, 0.0)
white = (1.0, 1.0, 1.0)
black='#000000'
gray='#404040'
green='#009682'
blue='#4664aa'
red='#a22223'
yellow='#fce500'
orange='#df9b1b'
lightgreen='#8cb63c'
purple='#a3107c'
brown='#a7822e'
cyan='#23a1e0'

def init_plot(scl_a4=1, aspect_ratio=[3,2], page_lnewdth_cm=16.5, fnt='arial', mrksze=6, 
              lnewdth=1.5, fontsize=10, labelfontsize=9, tickfontsize=8):
    
    # --- Initialize defaults ---
    plt.rcdefaults()

    # --- Calculate figure size in inches ---
    # scl_a4=2: Half page figure
    if scl_a4==2:     
            fac=page_lnewdth_cm/(2.54*aspect_ratio[0]*2) #2.54: cm --> inch
            figsze=[aspect_ratio[0]*fac,aspect_ratio[1]*fac]

    # scl_a4=1: Full page figure
    elif scl_a4==1:
            fac=page_lnewdth_cm/(2.54*aspect_ratio[0]) #2.54: cm --> inch
            figsze=[aspect_ratio[0]*fac,aspect_ratio[1]*fac]

    # --- Adjust legend ---
    plt.rc('legend', fontsize=fontsize, fancybox=True, shadow=False, 
           edgecolor='k', handletextpad=0.2, handlelength=1,
           borderpad=0.2, labelspacing=0.2, columnspacing=0.2)

    # --- General plot setup ---
    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family=fnt)
    plt.rc('xtick', labelsize=tickfontsize)
    plt.rc('ytick', labelsize=tickfontsize)
    plt.rc('axes', labelsize=labelfontsize, linewidth=0.5, titlesize=labelfontsize)
    plt.rc('legend', fontsize=fontsize)
    plt.rc('axes', axisbelow=True) # Grid lines in Background
    plt.rcParams['lines.markersize']=mrksze
    plt.rcParams['hatch.linewidth']=lnewdth/2
    plt.rcParams['lines.linewidth']=lnewdth     
    plt.rcParams['figure.figsize']=figsze