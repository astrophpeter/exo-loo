
"""
Plot spectral data, retrieved models, loo score
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers
import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore",category=matplotlib.mplDeprecation)
warnings.filterwarnings("ignore",category=UserWarning)

# Set font properties
font = {'weight' : 'bold',
        'size'   : 17}
plt.rc('text', usetex=True)
plt.rc('font', **font)
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath} \boldmath']
#Smoothing properties based on telescopes
sig=0.5812547437
wfc3sig=6.48821342143


def open_model(name, model_folder):
    filenames=['_TD_median_result','_TD_high1sigma','_TD_low1sigma','_TD_high2sigma','_TD_low2sigma']
    for option in filenames:
        file=np.loadtxt(model_folder+'/fig_out_m/'+name+option+'.txt', ndmin=2)
        waves.append(file[:,0])
        model.append(file[:,1])
    waves=np.asarray(waves)
    model=np.asarray(model)
    return waves, model

def open_data(name, model_folder):
    file=np.loadtxt(model_folder+'/data/HD209458b_'+name+'.txt')
    return file[:,0], file[:,1],file[:,2], file[:,3]

def plot_data(ax, instrument, model_folder, color='green', name='Data', marker='o'):
    if name is not None:
        label=r'$\mathrm{'+name+'}$'
    else:
        label=None
    data_lam, data_bin, data_depth, data_error=open_data(instrument,model_folder)
    ax.errorbar(data_lam, data_depth*100, xerr=data_bin, yerr=data_error*100, marker=marker,markersize=6, elinewidth=2, capsize=3, capthick=1.2, zorder=-50, ls='none', color=color,markeredgecolor='black',  ecolor='black')
    ax.legend(loc=1, ncol=1, fontsize=12, frameon=False)

def plot_spectra(ax, model, model_folder, smooth_model=True,sigma=6.48821342143, colour='grey', model_label=None, zorder=0):
    if model_label==None:
        model_label=model
    model_lam, model_depth= open_model(model,model_folder)
    ax.plot(model_lam, gaussian_filter1d(model_depth, sigma=sigma, mode='nearest')*100, lw=2, color = colour, zorder=zorder, alpha=1.0, label=r'$\mathrm{'+model_label+'}$')
    ax.legend(loc=2, ncol=1, fontsize=12, frameon=False)
    ax.set_xlim(0.25,5.1)

def plot_retrieval(ax, model,model_folder, smooth_model=True,sigma=6.48821342143, plot_sigma=True, colour='blue', sigma_colour='purple', model_label=None, zorder=0):
    median=np.loadtxt(model_folder+'/fig_out_m/'+model+'_TD_median_result.txt', ndmin=2)
    median_lam=median[:,0]
    median_depth=median[:,1]
    high1=np.loadtxt(model_folder+'/fig_out_m/'+model+'_TD_high1sigma.txt', ndmin=2)
    high1_depth=high1[:,1]
    high2=np.loadtxt(model_folder+'/fig_out_m/'+model+'_TD_high2sigma.txt', ndmin=2)
    high2_depth=high2[:,1]
    low1=np.loadtxt(model_folder+'/fig_out_m/'+model+'_TD_low1sigma.txt', ndmin=2)
    low1_depth=low1[:,1]
    low2=np.loadtxt(model_folder+'/fig_out_m/'+model+'_TD_low2sigma.txt', ndmin=2)
    low2_depth=low2[:,1]

    if model_label==None:
        model_label=model
    if smooth_model:
        ax.plot(median_lam, gaussian_filter1d(median_depth, sigma=sigma, mode='nearest')*100, lw=2, color = colour, zorder=zorder, alpha=1.0, label=r'$\mathrm{'+model_label+'}$')
    else:
        ax.plot(median_lam, median_depth*100, lw=2, color = colour, zorder=zorder, alpha=1.0)

    if plot_sigma:
        ax.fill_between(median_lam, gaussian_filter1d(low1_depth, sigma=sigma, mode='nearest')*100, gaussian_filter1d(high1_depth, sigma=sigma, mode='nearest')*100, facecolor=sigma_colour, alpha=0.4, linewidth=0.0, zorder=-10, label=r"$1 \, \sigma$")
        ax.fill_between(median_lam, gaussian_filter1d(low2_depth, sigma=sigma, mode='nearest')*100, gaussian_filter1d(high2_depth, sigma=sigma, mode='nearest')*100,  facecolor=sigma_colour, alpha=0.2, linewidth=0.0, zorder=-10, label=r"$2 \, \sigma$")

    ax.set_xlim(0.3,5.2)
    ax.set_ylim(1.435,1.53)
    ax.set_xscale('log')

    ytix = ax.get_yticks()
    ax.yaxis.set_ticklabels(['$\mathbf{'+str("{:.2f}".format(l))+'}$' for l in ytix],minor=False, fontsize = 15)
    xtisminor=[0.4,0.6, 0.8, 1.2,1.6]
    ax.set_xticks(xtisminor, minor=True)
    ax.xaxis.set_ticklabels(['$\mathbf{'+str("{:.1f}".format(l))+'}$' for l in xtisminor],minor=True, fontsize=15)

    xtixmajor=[1.0,2.0,3.0,4.0,5.0]
    ax.set_xticks(xtixmajor, minor=False)
    xtisfull=[0.4,0.6, 0.8,1.0, 1.2,1.6,2.0,3.0,4.0,5.0]
    ax.xaxis.set_ticklabels(['$\mathbf{'+str("{:.1f}".format(l))+'}$' for l in xtixmajor],minor=False, fontsize=15)

    ax.tick_params(which='major', direction='in', length=8, width=2,top=True, right=True, zorder=30)
    ax.tick_params(axis='x', which='both',pad=5)
    ax.tick_params(which='minor', direction='in', length=8, width=2,top=True, right=True, zorder=30)
    ax.tick_params(axis='both', which='both', labelsize=15)

    [i.set_linewidth(2.5) for i in iter(ax.spines.values())]

    ax.legend(loc=1, ncol=1, fontsize=15, frameon=False)

def bolden(ax, fs=15):
    [i.set_linewidth(2) for i in iter(ax.spines.values())]
    ax.tick_params(which='major', direction='in', length=8, width=2,top=True, right=True, zorder=1e6)
    ax.tick_params(which='minor', direction='in', length=4, width=2,top=True, right=True, zorder=1e6)
    ax.tick_params(axis='both', which='major', labelsize=fs)

def plot_loo_comparison_gradient(ax,loo_score):
    cmap = plt.get_cmap('bwr')
    bounds = np.arange(np.min(loo_score),np.max(loo_score),0.01)
    idx=np.searchsorted(bounds,0)
    bounds= np.insert(bounds,idx,0)
    offset=mcolors.TwoSlopeNorm(vmin=np.min(loo_score), vcenter=0.0, vmax=np.max(loo_score))
    PCM=ax.scatter(all_data[:,0],all_data[:,2]*100,c=loo_score, zorder=100, cmap=cmap, norm=offset)
    cbar = plt.colorbar(PCM, ax=ax, pad=0.01,cmap=cmap, norm=offset)#, boundaries=bounds)#, ticks=[-1,0,1])#, ticks=bounds, boundaries=bounds)
    cbar.ax.set_ylabel(r'$\mathbf{LOO \, Score}$',fontsize=20)

def plot_loo_comparison_binary(ax, loo_score):
    loo_score[loo_score <0] = -1
    loo_score[loo_score >0] = 1
    cmap = plt.get_cmap('bwr')
    bounds = [-2,2]
    idx=np.searchsorted(bounds,0)
    bounds= np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)
    PCM=ax.scatter(all_data[:,0],all_data[:,2]*100,c=loo_score, zorder=100, cmap=cmap)
    cbar = plt.colorbar(PCM, ax=ax, pad=0.01,cmap=cmap, norm=norm, boundaries=bounds, ticks=[-1,0,1])#, boundaries=bounds)#, ticks=[-1,0,1])#, ticks=bounds, boundaries=bounds)
    cbar.ax.set_ylabel(r'$\mathbf{LOO \, Score}$',fontsize=20)
    cbar.ax.set_yticklabels([r'$\mathbf{No \, H_2O \, Model}$', '', r'$\mathbf{H_2O \, Model}$'],rotation = 90,verticalalignment= 'center')

def plot_loo_score(ax,loo_score):
    PCM=ax.scatter(all_data[:,0],all_data[:,2]*100,c=loo_score)
    cbar = plt.colorbar(PCM, ax=ax,pad=0.01)
    cbar.ax.set_ylabel(r'$\mathbf{LOO \, Score}$',fontsize=20)

def plot_loo_vs_wavelength(ax,loo_score):
    ax.scatter(all_data[:,0],loo_score)
    ax.set_ylabel(r'$\mathbf{LOO\, Score}$', fontsize=20)
    ax.set_xlabel(r'$\mathbf{Wavelength} \, \, \boldsymbol{(} \boldsymbol{\mu} \mathbf{m} \boldsymbol{)}$', fontsize=20)

def plot_loo_vs_data_number(ax,loo_score):
    ax.scatter(np.arange(1,len(loo_score)+1),loo_score)
    ax.set_ylabel(r'$\mathbf{LOO\, Score}$', fontsize=20)
    ax.set_xlabel(r'$\mathbf{Data} \, \, \boldsymbol{(} \boldsymbol{\#} \boldsymbol{)}$', fontsize=20)

def label_spectrum_plot(ax):
    ax.set_ylabel(r'$\mathbf{Transit\, Depth \,  \boldsymbol{(}\boldsymbol{\%}\boldsymbol{)}}$', fontsize=20)
    ax.set_xlabel(r'$\mathbf{Wavelength} \, \, \boldsymbol{(} \boldsymbol{\mu} \mathbf{m} \boldsymbol{)}$', fontsize=20)
    ax.annotate( "", xy=(0.31,1.035), xytext=(1.03,1.035),  arrowprops={'arrowstyle' : "-", 'lw':3})
    ax.annotate( "$\mathbf{ HST\\text{-}STIS}$", xy=(0.59,1.02),horizontalalignment='center',verticalalignment='center')
    ax.annotate( "", xy=(1.1,1.035), xytext=(1.8,1.035),  arrowprops={'arrowstyle' : "-", 'lw':3})
    ax.annotate( "$\mathbf{HST\\text{-}WFC3}$", xy=(1.4,1.02),horizontalalignment='center',verticalalignment='center')
    ax.annotate( "", xy=(0.31,1.035), xytext=(0.93,1.035),  arrowprops={'arrowstyle' : "-", 'lw':3})
    ax.annotate( "$\mathbf{SPITZER}$", xy=(3.15,1.02),horizontalalignment='left',verticalalignment='center')
    ax.annotate( "", xy=(3.1,1.035), xytext=(5.1,1.035),  arrowprops={'arrowstyle' : "-", 'lw':3})

def plot_score(ax, loo_scores,tick_names, data=None, range=None):
    if range is not None:
        min, max=get_indices(range,data)
    else:
        min, max= 0,len(loo_scores[0])
    for i,loo_score_array in enumerate(loo_scores):
        selected_array=loo_score_array[min:max]
        sum=np.sum(selected_array)
        error=np.sqrt(len(selected_array)*np.var(selected_array))
        ax.errorbar(i+1, sum, yerr=error,marker='o',markersize=10, elinewidth=2, capsize=5, capthick=1.2)
    ax.set_xlim(0,len(loo_scores)+1)
    ax.set_xticklabels(tick_names)
    upper_limit=np.sum(loo_scores[0][min:max])+np.sqrt(len(loo_scores[0][min:max])*np.var(loo_scores[0][min:max]))
    lower_limit=np.sum(loo_scores[0][min:max])-np.sqrt(len(loo_scores[0][min:max])*np.var(loo_scores[0][min:max]))
    ax.fill_between(np.arange(0,len(loo_scores)+2), upper_limit, lower_limit , facecolor='gold', alpha=0.4, linewidth=0.0)
    ax.set_ylabel(r'$\mathbf{LOO\, Score \, Sum}$', fontsize=20)
    ax.set_xlabel(r'$\mathbf{Model}$', fontsize=20)

def get_indices(vrange, data):
    min_index=np.argmin(np.abs(data[:,0]-vrange[0]))
    max_index=np.argmin(np.abs(data[:,0]-vrange[1]))
    return min_index, max_index


fig=plt.figure(figsize=(11, 5))
spec = plt.gca()

model_1_folder='2_cloud_fractions'
model_1_name="2f"
model_2_folder='2_cloud_fractions_xH2O'
model_2_name="2f_xH2O"
model_3_folder='2_cloud_fractions_xNa'
model_3_name="2f_xNa"
model_4_folder='2_cloud_fractions_xK'
model_4_name="2f_xK"
model_5_folder='2_cloud_fractions_xNH3'
model_5_name="2f_xNH3"
model_6_folder='2_cloud_fractions_xCH4'
model_6_name="2f_xCH4"
model_7_folder='2_cloud_fractions_xHCN'
model_7_name="2f_xHCN"
plot_name='test2'

xticklabels=['','$\mathbf{Reference}$','$\mathbf{No \, H_2O}$','$\mathbf{No \, Na}$','$\mathbf{No \, K}$','$\mathbf{No \, NH_3}$','$\mathbf{No \, CH_4}$','$\mathbf{No \, HCN}$']
all_data=np.loadtxt(model_1_folder+'/loo_analysis/'+model_1_name+'_data.txt', ndmin=2)
approximate_loo_model_1=np.load(model_1_folder+"/loo_analysis/"+model_1_name+"_loo_i.npy")
approximate_loo_model_2=np.load(model_2_folder+"/loo_analysis/"+model_2_name+"_loo_i.npy")
approximate_loo_model_3=np.load(model_3_folder+"/loo_analysis/"+model_3_name+"_loo_i.npy")
approximate_loo_model_4=np.load(model_4_folder+"/loo_analysis/"+model_4_name+"_loo_i.npy")
approximate_loo_model_5=np.load(model_5_folder+"/loo_analysis/"+model_5_name+"_loo_i.npy")
approximate_loo_model_6=np.load(model_6_folder+"/loo_analysis/"+model_6_name+"_loo_i.npy")
approximate_loo_model_7=np.load(model_7_folder+"/loo_analysis/"+model_7_name+"_loo_i.npy")
# approximate_loo_model_8=np.load(model_8_folder+"/loo_analysis/"+model_8_name+"_loo_i.npy")
loo_score=approximate_loo_model_2-approximate_loo_model_1
loo_array=[approximate_loo_model_1,approximate_loo_model_2,approximate_loo_model_3,approximate_loo_model_4,approximate_loo_model_5,approximate_loo_model_6,approximate_loo_model_7]

# print(len(all_data[:,0]))
#Plot a spectrum
# plot_retrieval(spec,model_1_name,model_1_folder, model_label='Median \, retrieved \,  model', zorder=0)
# plot_data(spec,'stis1',model_1_folder, color='yellow')
# plot_data(spec,'stis2', model_1_folder, name=None,color='yellow')
# plot_data(spec,'wfc3', model_1_folder, name=None,color='yellow')
# plot_data(spec,'spitzer1', model_1_folder, name=None,color='yellow')
# plot_data(spec,'spitzer2', model_1_folder, name=None,color='yellow')

# label_spectrum_plot(spec)

#Plot a loo score on top of the spectrum
# plot_loo_comparison_gradient(spec,loo_score)
# plt.subplots_adjust(left=0.1, right=1.05, top=0.95, bottom=0.15)

#Plot a loo score vs data or wavelength
# plot_loo_vs_data_number(spec,approximate_loo_model_1)
plot_score(spec,loo_array,xticklabels,all_data,[1.2,1.7])
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)


# plt.tight_layout()
fig.savefig(model_1_name+'_'+plot_name+".pdf")
