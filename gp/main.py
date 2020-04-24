
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


ret = Retrieval() #This is the class instance  that in the retrieval code will have the Pressure temperature grid, cross sections, and the sampling algorithms. In here, the function to get the model resides.

mean_model=CustomModel(**{'X_h2o':-7.3, 'T0':1200.0, 'Pref':-1.0}) #input of different model parameters
params_dic={'X_h2o':-7.5, 'T0':1200.0, 'Pref':-1.0}
y=mean_model.get_value(retrieval=ret)
print(y)  #By design, this should return an array of len=29 and only zeros
mean_model.set_parameter_vector([-5.4,1300,-3]) #try new input of different model parameters
y=mean_model.get_value(retrieval=ret)
print(y) #By design, this should return an array of len=29 and only zeros
kernel = terms.Matern32Term(log_sigma=np.log(np.var(y)), log_rho=-np.log(10.0))
gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(data_lam, data_err)

# print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))
