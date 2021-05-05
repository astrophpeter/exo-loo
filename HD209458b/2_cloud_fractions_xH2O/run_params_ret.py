import numpy as np
import scipy.constants as sc
import pathlib
import skbio.stats.composition as scb

planet = "HD209458b"

ND=[100]                                                                                    # ND = number of atmospheric layers (or depths)
NF=            [2000,500,1000]#,1000]#,1299]#,1000]                                                # NF = number of wavelength values in each range if spacing is 'R' then the value is used as the resolution, options include: 'native' and 'passed' to give it a lam array. Define the directory of passed array in atm.profile
lam_sections = [0.3,1.9,3.0,5.1]#,5.1]#,3.5,4.6]#[0.77,1.87,5.5]#4.17,5.5][1.1,1.75]                # lam_sections = boundaries of wavelength ranges
spacing = "lam" #'lam' or 'R' for fixed resolution
instruments = ["stis1","stis2","wfc3","spitzer1","spitzer2"]               # instruments = which instruments are the data from?
pecul_name1 = "2f_xH2O"                                                                       # pecul_name_1 = label for the retrieval
num_live_pts = 2000

# Which functionalities do you want to use?
sampler='MultiNest'#MultiNest or PolyChord or Dynesty
retrieve =      False                                                                        #Are you running a retrieval?
resume=True
plot_spec =     False                                                                       #Do you want to plot the PT prof, spectrum, and corner plot?
analyse_bayes = False                                                                       #Do you want to analyse bayesian evidence of models?
luis_alkali= True                                                                           #Do you want to use Luis' Na/K broadened opacities (Allard et al. 2019) rather than Voigt profiles?
run_fwd_model=False
simulate_data = False                                                                       #Are you simulating data?
leave_out_file_generator=False                                                               #Are you generating the LOO files?
leave_out_file_plot=False
pvalue=True
#Plotting  and notification options
indices=False                                                                               #Do you want a subset of parameters (of index i) only in your corner plot?
index_array=[0,4,7,8,12,25,26,27,28]
notifyme = False                                                                             #send an email when the retrieval is done
plot_T100mb=True                                                                           # Plots distribution of temperature at 100mb
plot_mu=True                                                                               # Plots distribution of mean molecular weight
sendresults=False                                                                           #sends an email with plots of TD, PT-Profile, and posteriors attached
mu_set=None                                                                                 #Do we fix the mean molecular weight?
true_values=None#[  -4.93,     -2.0,  -8.77e-3, -12.00, -12.0, -2.0, 250.0,  1.0, -1.0, 1.0]                                                                            #Do you have true values for the data you are retrieving?
# true_values=[  -3.3,  -5.70,    -7.00,   -4.0,    -8.00,   -10.00,    -5.0,  -8.0,  956.78,  1.22, 0.97, -1.65,  -4.15,  0.63, -2.56, 0.0, -4.00, -6.0,  0.0 ]                                                                            #Do you have true values for the data you are retrieving?

default_params = {}
ret_params = []
priors = {}

#Booleans. What do you want to assume in your retrieval?
He_fraction=0.17
assume_hydrogen_rich=True                                                                  #Do you assume a H2 rich atmosphere?
PT_prof = "madhu" #"madhu","madhu_100mb", "line" , "6PT"
isotherm=False                                                                               #Is the P-T profile an isotherm?
ret_Pref = True                                                                             #Do you want to retrieve Pref?
retrieve_Rp = False                                                                         #Do you want to retrieve Rp?

cloudy_atm = True                                                                          #Do you want a cloudy atmosphere?
clouds_only=False                                                                           #If you retrieve a cloudy atmosphere, do you want to retrieve cloud deck only? (i.e. NO HAZES)
clouds_and_hazes=True                                                                         #If you retrieve a cloudy atmosphere, do you want to retrieve clouds AND hazes?



#Stellar heterogeniety test
stellar_heterogeneity = False
vary_Tphot = False

retrieve_offset1=False
instruments_offset1=['tess']
instruments_1_label='TESS'                                                                   #To appear like this in corner plot
offset1_gaussian_priors=True                                                               #if not Gaussian, uniform is assumed

retrieve_offset2=False
instruments_offset2=['wfc3']
instruments_2_label='WFC3'                                                                   #To appear like this in corner plot
offset2_gaussian_priors=True                                                               #if not Gaussian, uniform is assumed

lambda_shift = False                                                                        #Do you want to retrieve wavelength shifts?
shifted_instruments=["FORS2_600b","FORS2_600ri","FORS2_600z"]
shifted_instruments_label='FORS2'

# List of molecules to be retrieved h2he should go first
mol_list = [                    "h2he",     "na",     "k",    "ch4",   "nh3",   "hcn",   "co",   "co2"]#,   "na",     "k",    "ch4",   "nh3",   "hcn",   "co"]#,     "k",   "ch4",   "nh3",    "hcn",   "co",   "co2",    "tio"]#,    "alo",   "vo"]#,     "k"]#,    "ch4",   "nh3",   "hcn",   "co"]#
mol_lower_prior = np.array([     -10.0,   -12.0,   -12.0,   -12.0,   -12.0,   -12.0,   -12.0,    -12.0,  -12.0,   -12.0,    -12.0,  -12.0,   -12.0])
mol_upper_prior = np.array([       0.0,    -1.5,   -1.5,    -1.5,    -1.5,    -1.5,    -1.5,     -1.5,   -1.5,    -1.5,    -1.5,    -1.5,     -1.5])
mixing_frac = np.array([          5e-4,    -7.05,   -5.0,    -12.0,    -12.0,    -12.0,     -3.0,    -3.0,    -3.0,    -3.0,    -3.0,    -3.0,    -3.0]) #in log mixing frac
cia = ["h2", "he"]
cia_companion= ["h2", "h2"]

#Pressure range
P_max = 1.0e2
P_min = 1.0e-6

nmols=len(mol_list)


#Priors
if assume_hydrogen_rich==False:
    Xi_priors=np.append(np.full(nmols-1,1e-12),1-(nmols-1)*1e-12)
    xi_priors=scb.clr(Xi_priors)
    xi_upper_prior=xi_priors[-1]
    xi_lower_prior=xi_priors[0]
    for i,key in enumerate(mol_list):
        priors["X_"+key] = (xi_lower_prior,xi_upper_prior)
        default_params["X_"+key] = mixing_frac[i]
        ret_params.append("X_"+key)
else:
    for i,key in enumerate(mol_list[1:]):
        priors["X_"+key] = (mol_lower_prior[i+1],mol_upper_prior[i+1])
        default_params["X_"+key] = mixing_frac[i+1]
        ret_params.append("X_"+key)

if PT_prof=="madhu":
    priors["T0"] = (800.0,1550.0)
    ret_params.append("T0")
else:
    priors["T100mb"] = (300.0,4500.0)
    ret_params.append("T100mb")
if isotherm==False:
    priors["alpha1"] = (0.02,2.0)
    priors["alpha2"] = (0.02,2.0)
    priors["P1"] = (-6.0,2.0)
    priors["P2"] = (-6.0,2.0)
    priors["P3"] = (-2.0,2.0)
    ret_params = ret_params + ["alpha1","alpha2","P1","P2","P3"]

if ret_Pref:
    priors["Pref"] = (-6.0,2.0)
    ret_params.append("Pref")

if cloudy_atm:
    if clouds_only:
        priors["cl_logPc"] = (-6.0,2.0); priors["cl_phi"] = (0.0,1.0)
        ret_params = ret_params + ["cl_logPc"]
        partial_clouds = True
        if partial_clouds:
            ret_params.append("cl_phi")
    if clouds_and_hazes:
        priors["cl_loga"] = (-4.0,10.0); priors["cl_gamma"] = (-20.0,2.0);
        ret_params = ret_params + ["cl_loga","cl_gamma"]
        partial_hazes= True
        if partial_hazes:
            priors["hz_phi"] = (0.0,1.0)
            ret_params.append("hz_phi")
        priors["cl_logPc"] = (-6.0,2.0)
        ret_params = ret_params + ["cl_logPc"]
        partial_clouds = True
        if partial_clouds:
            priors["cl_phi"] = (0.0,1.0);
            ret_params.append("cl_phi")
        partial_clouds_n_hazes = False
        if partial_clouds_n_hazes:
            priors["mix_phi"] = (0.0,1.0)
            ret_params.append("mix_phi")

if retrieve_Rp:
    priors["Rp"] = (0.10,0.30)
    ret_params.append("Rp")


if retrieve_offset1:
    if not offset1_gaussian_priors:
        priors['offset1']=(-80.0,80.0)
        ret_params.append('offset1')
    else:
        priors['offset1']=(100.0)
        ret_params.append('offset1')

if retrieve_offset2:
    if not offset2_gaussian_priors:
        priors['offset2']=(-50.0,50.0)
        ret_params.append('offset2')
    else:
        priors['offset2']=(100.0)
        ret_params.append('offset2')

if lambda_shift:
    priors["lambda_shift"] = (-0.01,0.01)
    ret_params.append("lambda_shift")





#Default parameters can be used to generate a model
default_params["offset1"] = 0.0
default_params["offset2"] = 0.0
default_params["T0"] = 2000.0
default_params["T100mb"] = 1725.0
default_params["alpha1"] = 0.32
default_params["alpha2"] = 0.6
default_params["P1"] = -0.5
default_params["P2"] = -3
default_params["P3"] = 1.0
default_params["Pref"] = -1.0
default_params["lam_shift"] = 0.0
default_params["cl_loga"] = 0.0
default_params["cl_gamma"] = -4.0
default_params["cl_logPc"] = 1.0
default_params["cl_phi"] = 0.0  #For a retrieval ensure this is 0.0
default_params["hz_phi"] = 0.0 #For a retrieval ensure this is 0.0
default_params["mix_phi"] = 0.0 #For a retrieval ensure this is 0.0

default_params["spot_fraction"] = 0.0
default_params["T_spot"] = 3500.0
default_params["T_phot"] = 3500.0

default_params["lambda_shift"] = 0.0
#System parameters like Rp, Rstar and log(g) are at the bottom

# model to compare to if analyse_bayes = true
pecul_name2 = "001_new_xK"                                                      # label indicating which molecule(s) are present/absent in the comparison (e.g. "no_vo")
mod_bayes = "K"                                                                 # what assumption changed between models? I.e. K
mod_reduc = 1                                                                   # number of molecules included/excluded in the comparison (e.g. 1 for "no_vo")


# locations of instrument data and simulated data
data_file_loc={}
sim_file_loc={}

data_file_loc["flat"] = "../../../data/actual_data/"+planet+"/"+planet+"_flat.txt"
data_file_loc["flat2"] = "../../../data/actual_data/"+planet+"/"+planet+"_flat2.txt"
data_file_loc["FORS2_600b"] = "../../../data/actual_data/"+planet+"/"+planet+"_FORS2_600b.txt"
data_file_loc["FORS2_600ri"] = "../../../data/actual_data/"+planet+"/"+planet+"_FORS2_600ri.txt"
data_file_loc["FORS2_600z"] = "../../../data/actual_data/"+planet+"/"+planet+"_FORS2_600z.txt"
data_file_loc["wfc3"] = "../../../data/actual_data/"+planet+"/"+planet+"_wfc3.txt" #Just to use the mandell data for WASP19b
data_file_loc["wfc3_102"] = "../../../data/actual_data/"+planet+"/"+planet+"_wfc3_102.txt"
data_file_loc["stis1"] = "../../../data/actual_data/"+planet+"/"+planet+"_stis1.txt"
data_file_loc["stis2"] = "../../../data/actual_data/"+planet+"/"+planet+"_stis2.txt"
data_file_loc["spitzer1"] = "../../../data/actual_data/"+planet+"/"+planet+"_spitzer1.txt"
data_file_loc["spitzer2"] = "../../../data/actual_data/"+planet+"/"+planet+"_spitzer2.txt"
data_file_loc["spitzer3"] = "../../../data/actual_data/"+planet+"/"+planet+"_spitzer3.txt"
data_file_loc["spitzer4"] = "../../../data/actual_data/"+planet+"/"+planet+"_spitzer4.txt"
data_file_loc["tess"] = "../../../data/actual_data/"+planet+"/"+planet+"_tess.txt"
data_file_loc["nirspec_prism"] = "data/"+planet+"_nirspec_prism.txt"
data_file_loc["nirspec_g140m"] = "../../../data/actual_data/"+planet+"/"+planet+"_nirspec_g140m.txt"
data_file_loc["nirspec_g140h"] = "data/"+planet+"_nirspec_g140h.txt"
data_file_loc["nirspec_g235h"] = "data/"+planet+"_nirspec_g235h.txt"
data_file_loc["nirspec_g235m"] = "../../../data/actual_data/"+planet+"/"+planet+"_nirspec_g235m.txt"
data_file_loc["nirspec_g395h"] = "data/"+planet+"_nirspec_g395h.txt"
data_file_loc["nirspec_g395m"] = "../../../data/actual_data/"+planet+"/"+planet+"_nirspec_g395m.txt"
data_file_loc["miri_lrs"] = "data/"+planet+"_miri_lrs.txt"

# output files for simulated data and multinest outputs. Creates directories and parents if they do not exist.
if sampler=='MultiNest':
    output_ret_loc = "ret_out_m/"
    pathlib.Path(output_ret_loc).mkdir(parents=True, exist_ok=True)
    output_fig_file = "fig_out_m/"
    pathlib.Path(output_fig_file).mkdir(parents=True, exist_ok=True)

    # file names to pass to multinest and plotting routine
    output_ret = "ret_out_m/"+pecul_name1
    output_figs = "ret_out_m/"+pecul_name1

elif sampler=='PolyChord':
    output_ret_loc = "ret_out_p/"
    pathlib.Path(output_ret_loc).mkdir(parents=True, exist_ok=True)
    output_fig_file = "fig_out_p/"
    pathlib.Path(output_fig_file).mkdir(parents=True, exist_ok=True)

    # file names to pass to multinest and plotting routine
    output_ret = "ret_out_p/"+pecul_name1
    output_figs = "ret_out_p/"+pecul_name1

elif sampler=='Dynesty':
    output_ret_loc = "ret_out_d/"
    pathlib.Path(output_ret_loc).mkdir(parents=True, exist_ok=True)
    output_fig_file = "fig_out_d/"
    pathlib.Path(output_fig_file).mkdir(parents=True, exist_ok=True)

    # file names to pass to multinest and plotting routine
    output_ret = "ret_out_d/"+pecul_name1
    output_figs = "ret_out_d/"+pecul_name1
# location of instrument sensitivity data
inst_file_loc={}
inst_file_loc["flat"]= "../../../../tools/instrument_sens/flat_sensitivity.dat"
inst_file_loc["flat2"]= "../../../../tools/instrument_sens/flat_sensitivity.dat"
inst_file_loc["FORS2_600b"]= "../../../../tools/instrument_sens/flat_sensitivity.dat" #For this retrieval I am using a flat sensitivity as Ryan did in his paper
inst_file_loc["FORS2_600ri"]= "../../../../tools/instrument_sens/flat_sensitivity.dat"
inst_file_loc["FORS2_600z"]= "../../../../tools/instrument_sens/flat_sensitivity.dat"
# inst_file_loc["g600b"]= "../../../../tools/instrument_sens/g600b_sensitivity.dat"
# inst_file_loc["g600ri"]= "../../../../tools/instrument_sens/g600ri_sensitivity.dat"
# inst_file_loc["g600z"]= "../../../../tools/instrument_sens/g600z_sensitivity.dat"
inst_file_loc["wfc3"]= "../../../../tools/instrument_sens/G141_sensitivity.dat"
inst_file_loc["wfc3_102"]= "../../../../tools/instrument_sens/G102_sensitivity.dat"
inst_file_loc["stis1"]= "../../../../tools/instrument_sens/G430L_sensitivity.dat"
inst_file_loc["stis2"]="../../../../tools/instrument_sens/G750L_sensitivity.dat"
inst_file_loc["spitzer1"]="../../../../tools/instrument_sens/IRAC1_sensitivity.dat"
inst_file_loc["spitzer2"]="../../../../tools/instrument_sens/IRAC2_sensitivity.dat"
inst_file_loc["spitzer3"]="../../../../tools/instrument_sens/IRAC3_sensitivity.dat"
inst_file_loc["spitzer4"]="../../../../tools/linstrument_sens/IRAC4_sensitivity.dat"
inst_file_loc["tess"]="../../../../tools/instrument_sens/tess_sensitivity_knicole.dat"

inst_file_loc["nirspec_prism"]= "../../../../tools/instrument_sens/nirspec_prism.dat"
inst_file_loc["nirspec_g140m"]= "../../../../tools/instrument_sens/nirspec_g140m.dat"
inst_file_loc["nirspec_g140h"]= "../../../../tools/instrument_sens/nirspec_g140h.dat"
inst_file_loc["nirspec_g235h"]= "../../../../tools/instrument_sens/nirspec_g235h.dat"
inst_file_loc["nirspec_g235m"]= "../../../../tools/instrument_sens/nirspec_g235m.dat"
inst_file_loc["nirspec_g395h"]= "../../../../tools/instrument_sens/nirspec_g395h.dat"
inst_file_loc["nirspec_g395m"]= "../../../../tools/instrument_sens/nirspec_g395m.dat"
inst_file_loc["miri_lrs"]= "../../../../tools/instrument_sens/miri_lrs.dat"

# instrument wavelength limits
inst_limits={}
inst_limits["flat"] = [0.2,1.3]
inst_limits["flat2"] = [0.4,0.51]
inst_limits["FORS2_600b"] = [0.330,0.620] #As per the Nature paper Sedaghati 2017
inst_limits["FORS2_600ri"] = [0.52,0.853]#inst_limits["FORS2_600ri"] = [0.536,0.853] originally
inst_limits["FORS2_600z"] = [0.740,1.051]
inst_limits["wfc3"] = [1.05,1.8]
inst_limits["wfc3_102"] = [0.75,1.2]
inst_limits["stis1"] = [0.25,0.8]
inst_limits["stis2"] = [0.52,1.04]
inst_limits["spitzer1"] = [2.87,4.165]
inst_limits["spitzer2"] = [3.704,5.27]
inst_limits["spitzer3"] = [4.615,6.896]
inst_limits["spitzer4"] = [5.618,10.31]
inst_limits["tess"] = [0.5,1.13]
inst_limits["nirspec_prism"] = [0.59,5.3]
inst_limits["nirspec_g140m"] = [0.97,1.89] #only true if used with F100 LP
inst_limits["nirspec_g140h"] = [0.8,1.89]#only true if used with F100 LP
inst_limits["nirspec_g235h"] = [1.66,3.17]
inst_limits["nirspec_g235m"] = [1.66,3.17]
inst_limits["nirspec_g395h"] = [2.87,5.27]
inst_limits["nirspec_g395m"] = [2.87,5.27]
inst_limits["miri_lrs"] = [4.5,12.5]


# instrument noise in ppm for simulating data
noise={}
noise["flat"] = 30.0
noise["flat2"] = 30.0
noise["FORS2_600b"] = 30.0 #Just making it up, no reference
noise["FORS2_600ri"] = 30.0 #Just making it up, no reference
noise["FORS2_600z"] = 30.0 #Just making it up, no reference
noise["wfc3"] = 30.0#50.0e-7#noise in ppm
noise["spitzer1"] = 60.0#200.0e-7
noise["spitzer2"] = 84.0#200.0e-7
noise["spitzer3"] = 150.0#200.0e-7
noise["spitzer4"] = 150.0#200.0e-7
noise["wfc3_102"] = 30.0#50.0e-7#noise in ppm
noise["nirspec_prism"] = 5.0 #We want this to be our floor noise even in best case scenarios. We wont use this but will come in handy in the future
noise["nirspec_g140m"] = 5.0
noise["nirspec_g140h"] =5.0
noise["nirspec_g235h"] = 5.0
noise["nirspec_g235m"] = 5.0
noise["nirspec_g395h"] = 5.0
noise["nirspec_g395m"] = 5.0
noise["miri_lrs"] = 5.0


# location of opacity file(s)
hdf5_file = "../../../../tools/luis_opacity.hdf5"
if luis_alkali==True:
    atomic_file = "../../../../tools/new_atomic_luis_revised.hdf5"
else:
    atomic_file = "../../../../tools/atomic_opacity.hdf5"



import sys
sys.path.insert(0,"../../../../transmission/data/actual_data/"+planet+"/")
import planet_prop as pl


default_params["Rstar"] = pl.Rstar
default_params["Rp"] = pl.Rp


try: pl.logg
except AttributeError:
    default_params["grav"] = np.log10(100.0*pl.M*1.898e27*sc.G/(pl.Rp*pl.Rp*7.1492e7*7.1492e7))
else:
    default_params["grav"] = pl.logg

if stellar_heterogeneity:
    default_params["Teff"] = pl.Teff
    default_params["loggstar"] = pl.loggstar
    default_params["Zmet"] = pl.Zstar

    priors["spot_fraction"] = (0.0,0.5)
    priors["T_spot"] = (0.5*default_params["Teff"],1.5*default_params["Teff"])
    ret_params.append('spot_fraction')
    ret_params.append('T_spot')
    if vary_Tphot:
        priors["T_phot"] =(100.0) #The width of the gaussian centered at Tphot
        ret_params.append('T_phot')
    else:
        default_params["T_phot"] = default_params["Teff"]


def AmIMaster():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return rank == 0

assert((clouds_and_hazes and clouds_only and cloudy_atm)==False), "Can't have clouds and hazes and clouds only simultanously selected"

if AmIMaster():
    if run_fwd_model:
        print:('Generating forward model for ',planet,' named ',pecul_name1)
    else:
        print('Run for ',planet,' retrieval named ',pecul_name1)

time_taken=None                                                                 #Variable that prints how long the retrieval took to run
