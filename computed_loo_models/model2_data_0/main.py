import sys
sys.path.insert(0,"../../../../tools/source_code/project_aurora/")
from retrieval import Retrieval
from instrument import Instrument
from atm_profile import Profile
from absorption import Absorption
from RT import RT
import run_params_ret as run_params
import time
import sendmail
from datetime import timedelta
from dynesty.dynamicsampler import stopping_function, weight_function

def get_time():
    return run_params.time_taken
def set_time(val):
    run_params.time_taken="{:0>8}".format(str(timedelta(seconds=val)))

# Call instances of the appropriate classes
prof = Profile()



if run_params.run_fwd_model:
    fast_solver=False
    save_spectrum=True
    plot_fwd_model=True
    show_plot=True
    check_resolution=True
    log_plot=False

    absorp = Absorption(prof, fast_solver=fast_solver)
    rt = RT(prof)
    inst = {}
    ret = Retrieval(prof,absorp,rt,inst)
    ret.fwd_model(run_params.default_params, save_spectrum=save_spectrum, plot=plot_fwd_model, show_plot=show_plot, check_resolution=check_resolution, fast_solver=fast_solver,log=log_plot)
    exit()

if run_params.simulate_data:
    fast_solver=False
    save_spectrum=True
    plot_fwd_model=True
    show_plots=True
    check_resolution=True
    log_plot=False
    use_actual_data_noise=True
    add_error=True #displace points by median

    absorp = Absorption(prof, fast_solver=fast_solver)
    rt = RT(prof)
    inst = {}
    for key in run_params.instruments:
        inst[key] = Instrument(prof.lam,key,run_params.data_file_loc[key],run_params.inst_file_loc[key])
    ret = Retrieval(prof,absorp,rt,inst)
    ret.fwd_model(run_params.default_params, save_spectrum=save_spectrum, plot=plot_fwd_model, show_plot=show_plots, check_resolution=check_resolution, fast_solver=fast_solver, simulate_data=True,log=log_plot)
    # ret.generate_model(run_params.default_params)
    for key in run_params.instruments:
        inst[key].sim_data(run_params.noise[key]*1.0e-6,add_error=add_error,actual_noise=use_actual_data_noise)
    ret.plot_sim_data(show_plot=show_plots, log=log_plot)
    exit()

absorp = Absorption(prof)
rt = RT(prof)
inst = {}
for key in run_params.instruments:
    inst[key] = Instrument(prof.lam,key,run_params.data_file_loc[key],run_params.inst_file_loc[key])


# Call instances of retrieval and extra_plots classes
ret = Retrieval(prof,absorp,rt,inst)

if run_params.retrieve:
    ret_time = time.time()
    if run_params.sampler=='MultiNest':
        ret.multi_nest()
    elif run_params.sampler=='PolyChord':
        ret.polychord()
    elif run_params.sampler=='Dynesty':
        static=False #Do you want to use the static sampler? alternative is dynamic

        if static:
            ret.dynesty()
            ret.sampler.run_nested(dlogz=0.5, print_progress=False)
            ret.dynesty_results(ret.sampler.results)
        else:
            ret.dynesty(static=static)
            # ret.sampler.run_nested(dlogz_init=0.5, nlive_init=run_params.num_live_pts,print_progress=False)# THIS RUN OPTIMIZES 80 vs 20 in favor of posterior run
            ret.sampler.run_nested(dlogz_init=0.5, nlive_init=run_params.num_live_pts, wt_kwargs={'pfrac': 1.0},print_progress=False)#nlive_init=run_params.num_live_pts, print_progress=False)
            ret.dynesty_results(ret.sampler.results, static=static)


    if run_params.AmIMaster():
        set_time(time.time()-ret_time)
        print('THIS RETRIEVAL TOOK',get_time())
# Plotting options
#---------------------------------------------------
if run_params.AmIMaster():

    if run_params.plot_spec:
        ret.write_models_and_sigmas()
        ret.corner()
        ret.plot_ret()
        if run_params.sampler=='MultiNest':
            ret.check_best_fit()

    if run_params.plot_T100mb:
        ret.plot_T100mb()
    if run_params.plot_mu:
        ret.plot_mean_mu_distribution()
    # Bayes analysis option
    if run_params.analyse_bayes:
        ret.asses_model(run_params.mod_bayes, run_params.output_ret, run_params.output_ret_loc+run_params.pecul_name2, ret.n_params, run_params.mod_reduc)

    if run_params.leave_out_file_generator:
        ret.get_likelihoods()
    if run_params.leave_out_file_plot:
        ret.produce_loo()

if run_params.AmIMaster():
    if run_params.notifyme:
        total_time_taken=get_time()
        sendmail.sendemail(run_params.planet+' '+run_params.pecul_name1, total_time_taken)
    if run_params.sendresults:
        total_time_taken=get_time()
        sendmail.sendemail_wattachments(run_params.planet,run_params.pecul_name1, total_time_taken)
