
from retrieval import Retrieval
from model_class import CustomModel
import run_params_ret as run_params
import numpy as np

# from celerite.modeling import Model
import celerite
from celerite import terms
from scipy.optimize import minimize
from celerite.modeling import Model

def get_time():
    return run_params.time_taken
def set_time(val):
    run_params.time_taken="{:0>8}".format(str(timedelta(seconds=val)))


data=np.loadtxt('data.txt',ndmin=2)

data_lam=data[:,0]
data_bin=data[:,1]
data_depth=data[:,2]
data_err=data[:,3]

mean_model=CustomModel(**{'X_h2o':-7.3, 'T0':1200.0, 'Pref':-1.0}) #input of different model parameters
params_dic={'X_h2o':-7.5, 'T0':1200.0, 'Pref':-1.0}

kernel = terms.Matern32Term(log_sigma=np.log(np.var(data_depth)), log_rho=np.log(20.0))
gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(data_lam, data_err)

def log_likelihood(params):
    gp.set_parameter_vector(params)
    return gp.log_likelihood(data_depth)

test_parameter_values = np.ones(5)
print("Initial log-likelihood: {0}".format(log_likelihood(test_parameter_values)))
